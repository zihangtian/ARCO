"""Joint RL training — GRPO for π, score regression + KL regularisation for μ.

Usage:
    python scripts/rl_train.py
    python scripts/rl_train.py --config configs/hotpotqa/arco_qwen.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# ── 必须在 import torch 之前设置，否则无效 ──────────────────────────────────
# Guard with __name__ check so vLLM's spawn subprocess doesn't re-execute this
if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default="configs/hotpotqa/arco_qwen.yaml")
    _pre_args, _ = _pre.parse_known_args()

    # 从 yaml 读 visible_gpus（在 import torch 之前）
    def _read_visible_gpus(config_path):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return str(cfg.get("data", {}).get("visible_gpus", ""))

    _gpus = _read_visible_gpus(_pre_args.config)
    if _gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
        print(f"[RL] CUDA_VISIBLE_DEVICES={_gpus}")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.misc import load_yaml
from src.environments.registry import get_dataset
from src.training import make_trainer
from src.training.dual_head_mu import DualHeadMu
from src.training.pi_eval import evaluate_pi_model


def _disable_fused_sdp_if_qwen(base_path: str) -> None:
    if "qwen" not in base_path.lower():
        return
    if hasattr(torch.backends, "cuda"):
        try:
            # Allow flash + mem_efficient (memory-cheap) for long contexts in Qwen.
            # Math SDPA materializes (B, H, S, S) and OOMs on long sequences.
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            print("[RL] SDPA backends enabled for Qwen (flash + mem_efficient + math)")
        except Exception as exc:
            print(f"[RL] Could not adjust SDPA backend: {exc}")


def load_model_for_rl(base_path: str, adapter_path: str | None,
                      device: str = "auto", use_4bit: bool = False,
                      train_adapter: bool = False, max_memory: dict | None = None,
                      attn_implementation: str | None = None):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    device_map = {"": device} if device != "auto" else "auto"
    model_kwargs = {
        "quantization_config": bnb,
        "dtype": torch.bfloat16,
        "device_map": device_map,
        "max_memory": max_memory,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if adapter_path:
        from peft import PeftModel
        mode = "trainable" if train_adapter else "frozen"
        print(f"    loading SFT adapter from {adapter_path} ({mode}) ...")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=train_adapter)

    return model, tokenizer


def apply_lora(model, lora_cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa/arco_qwen.yaml")
    args = parser.parse_args()

    rl_cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(rl_cfg["data"].get("dataset_config", "configs/hotpotqa/hotpotqa.yaml"))
    cfg = {**dataset_cfg, **rl_cfg}
    dataset = dataset_cfg.get("dataset", "hotpotqa")

    use_4bit = rl_cfg["model"].get("use_4bit", False)
    base_path = rl_cfg["model"].get("base_model_path")
    pi_base_path = rl_cfg["model"].get("pi_base_model_path", base_path)
    mu_base_path = rl_cfg["model"].get("mu_base_model_path", base_path)
    if not pi_base_path:
        raise ValueError("model.pi_base_model_path or model.base_model_path must be set")
    attn_implementation = rl_cfg["model"].get("attn_implementation")
    pi_device = rl_cfg["model"].get("pi_device", "auto")
    pi_ref_device = rl_cfg["model"].get("pi_ref_device", pi_device)
    mu_device = rl_cfg["model"].get("mu_device", "auto")
    mu_max_memory = rl_cfg["model"].get("mu_max_memory")
    use_mu = rl_cfg["model"].get("enable_mu", True)

    _disable_fused_sdp_if_qwen(pi_base_path)
    if use_mu and mu_base_path and mu_base_path != pi_base_path:
        _disable_fused_sdp_if_qwen(mu_base_path)

    prompt_dir = Path(dataset_cfg["prompts"]["dir"])
    pi_system = (prompt_dir / "policy" / "system.txt").read_text(encoding="utf-8").strip()
    pi_user_tmpl = (prompt_dir / "policy" / "turn.txt").read_text(encoding="utf-8").strip()
    pi_finish_path = prompt_dir / "policy" / "forced_finish.txt"
    pi_finish_tmpl = pi_finish_path.read_text(encoding="utf-8").strip() if pi_finish_path.exists() else ""
    max_steps = dataset_cfg["environment"]["max_steps"]

    print("[RL] Loading π (base + SFT adapter) ...")
    pi_model, pi_tok = load_model_for_rl(
        pi_base_path,
        rl_cfg["model"].get("pi_sft_adapter"),
        pi_device,
        use_4bit,
        train_adapter=True,
        attn_implementation=attn_implementation,
    )

    pi_ref_model = None
    pi_ref_adapter = rl_cfg["model"].get("pi_ref_adapter", None)
    pi_kl_coef = rl_cfg["rl"].get("pi_kl_coef", 0.0)
    sparse_pi_kl_coef = rl_cfg["rl"].get("sparse_pi_kl_coef", pi_kl_coef)
    dense_pi_kl_coef = rl_cfg["rl"].get("dense_pi_kl_coef", pi_kl_coef)
    if pi_ref_adapter and max(pi_kl_coef, sparse_pi_kl_coef, dense_pi_kl_coef) > 0:
        print("[RL] Loading π_ref (frozen reference for KL) ...")
        pi_ref_model, _ = load_model_for_rl(
            pi_base_path,
            pi_ref_adapter,
            pi_ref_device,
            use_4bit,
            train_adapter=False,
            attn_implementation=attn_implementation,
        )
        pi_ref_model = freeze(pi_ref_model)
    else:
        print("[RL] Skipping π_ref (no pi_ref_adapter or KL disabled)")

    mu_model = None
    mu_tok = None
    mu_ref_model = None
    if use_mu:
        mu_lr = rl_cfg["rl"].get("mu_lr", 1e-5)
        mu_kl = rl_cfg["rl"].get("mu_kl_coef", 0.1)
        train_mu = mu_lr > 0

        print(f"[RL] Loading μ (base + SFT adapter, {'trainable' if train_mu else 'frozen'}) ...")
        mu_model, mu_tok = load_model_for_rl(
            mu_base_path or pi_base_path,
            rl_cfg["model"].get("mu_sft_adapter"),
            mu_device,
            use_4bit,
            train_adapter=train_mu,
            max_memory=mu_max_memory,
            attn_implementation=attn_implementation,
        )
        if not rl_cfg["model"].get("mu_sft_adapter") and rl_cfg["model"].get("lora"):
            mu_model = apply_lora(mu_model, rl_cfg["model"]["lora"])

        # Only load μ_ref when KL regularisation is active and μ is being trained
        if train_mu and mu_kl > 0:
            print("[RL] Loading μ_ref (frozen reference for KL) ...")
            mu_ref_adapter = rl_cfg["model"].get("mu_ref_adapter",
                                                  rl_cfg["model"].get("mu_sft_adapter"))
            mu_ref_model, _ = load_model_for_rl(
                mu_base_path or pi_base_path,
                mu_ref_adapter,
                mu_device,
                use_4bit,
                train_adapter=False,
                max_memory=mu_max_memory,
                attn_implementation=attn_implementation,
            )
            mu_ref_model = freeze(mu_ref_model)
        else:
            print("[RL] Skipping μ_ref (μ frozen or μ_kl_coef=0)")

        # Wrap μ with dual-head (text + score)
        _mc = mu_model.config
        hidden_size = _mc.text_config.hidden_size if hasattr(_mc, 'text_config') else _mc.hidden_size
        mu_sft_path = rl_cfg["model"].get("mu_sft_adapter")
        num_scores = DualHeadMu.infer_num_scores(mu_sft_path) if mu_sft_path else DualHeadMu.NUM_SCORES
        mu_model = DualHeadMu(mu_model, hidden_size, num_scores=num_scores)
        # Load SFT-trained score_head weights if available
        if mu_sft_path:
            actual_mu_device = next(mu_model.parameters()).device
            score_state = DualHeadMu.load_score_head(mu_sft_path, hidden_size, actual_mu_device)
            if score_state:
                mu_model.score_head.load_state_dict(score_state)
                print(f"[RL] Loaded SFT score_head from {mu_sft_path}")
        # Freeze score_head too when mu is not being trained
        if not train_mu:
            for p in mu_model.score_head.parameters():
                p.requires_grad_(False)
            print("[RL] μ score_head frozen (mu_lr=0)")
    else:
        print("[RL] μ disabled: outcome-only RL without rubric model")

    retriever = None
    # Check rl_cfg first (per-GPU retriever), then fall back to dataset_cfg
    retriever_url = cfg.get("retriever", {}).get("url") or dataset_cfg.get("retriever", {}).get("url")
    if retriever_url:
        from src.utils.retriever import Retriever
        retriever = Retriever(retriever_url)
        print(f"[RL] Retriever at {retriever_url}")

    spec = get_dataset(dataset)

    examples = spec.load_data(dataset_cfg, "train")
    max_ex = rl_cfg["data"].get("max_examples")
    if max_ex:
        examples = examples[:max_ex]
    print(f"[RL] {len(examples)} training examples")

    eval_split_names = rl_cfg["data"].get("eval_splits", ["dev"])
    eval_examples = {}
    for split in eval_split_names:
        split_examples = spec.load_data(dataset_cfg, split)
        max_eval = rl_cfg["data"].get("max_eval_examples")
        if max_eval:
            split_examples = split_examples[:max_eval]
        eval_examples[split] = split_examples
        print(f"[RL] {len(split_examples)} examples for evaluation split={split}")

    if rl_cfg["rl"].get("pre_eval", True):
        print("[RL] Evaluating π SFT-adapter baseline ...")
        for split, split_examples in eval_examples.items():
            evaluate_pi_model(
                model=pi_model,
                tokenizer=pi_tok,
                examples=split_examples,
                retriever=retriever,
                max_steps=max_steps,
                pi_system=pi_system,
                pi_user_tmpl=pi_user_tmpl,
                pi_finish_tmpl=pi_finish_tmpl,
                max_new_tokens=rl_cfg["rl"].get("pi_max_new_tokens", 64),
                batch_size=rl_cfg["rl"].get("eval_batch_size", rl_cfg["rl"].get("batch_size", 64)),
                desc=f"π SFT-adapter baseline {split}",
                dataset=dataset,
            )
    else:
        print("[RL] Skipping pre-training evaluation (pre_eval=false)")

    if not rl_cfg["model"].get("pi_sft_adapter") and rl_cfg["model"].get("lora"):
        pi_model = apply_lora(pi_model, rl_cfg["model"]["lora"])

    trainer = make_trainer(
        cfg,
        pi_model=pi_model, pi_tokenizer=pi_tok,
        mu_model=mu_model, mu_tokenizer=mu_tok,
        train_examples=examples,
        retriever=retriever,
        pi_ref_model=pi_ref_model,
        mu_ref_model=mu_ref_model,
        vllm_cfg=rl_cfg.get("vllm"),
    )
    trainer.train(eval_examples=eval_examples)


if __name__ == "__main__":
    main()
