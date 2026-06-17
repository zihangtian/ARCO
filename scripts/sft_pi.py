"""SFT train π on warmup trajectories.

Usage:
    python scripts/sft_pi.py
    python scripts/sft_pi.py --config configs/hotpotqa/pi.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.misc import load_yaml, read_jsonl, write_jsonl, resolve_warmup_file
from src.utils.retriever_auto import configure_retriever_url
from src.training.sft_data_builder import build_pi_sft_data
from src.training.sft_trainer import train_sft
from src.training.pi_eval import evaluate_pi_model
from src.environments.registry import get_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa/pi.yaml")
    parser.add_argument("--base_model", default=None, help="Override model.base_model_path")
    parser.add_argument("--device", default=None, help="Override model.device")
    parser.add_argument("--output_dir", default=None, help="Override training.output_dir")
    parser.add_argument("--warmup_file", default=None, help="Override data.warmup_file")
    parser.add_argument("--train_file", default=None, help="Override path for dumped SFT jsonl")
    parser.add_argument("--retriever_gpu", default="auto",
                        help="Physical retriever GPU id, or 'auto' from CUDA_VISIBLE_DEVICES; use 'config' to keep YAML URL")
    parser.add_argument("--retriever_url", default=None,
                        help="Explicit retriever URL override")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.base_model:
        cfg["model"]["base_model_path"] = args.base_model
    if args.device:
        cfg["model"]["device"] = args.device
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.warmup_file:
        cfg["data"]["warmup_file"] = args.warmup_file
    if args.train_file:
        cfg["data"]["train_file"] = args.train_file
    data_cfg = cfg["data"]
    prompt_cfg = cfg["prompts"]
    dataset_cfg = load_yaml(data_cfg.get("dataset_config", "configs/hotpotqa/hotpotqa.yaml"))
    retriever_url = configure_retriever_url(
        dataset_cfg,
        retriever_gpu=args.retriever_gpu,
        retriever_url=args.retriever_url,
        device=cfg["model"].get("device"),
    )
    if retriever_url:
        print(f"[sft_pi] retriever_url={retriever_url}")
    dataset = dataset_cfg.get("dataset", "hotpotqa")

    warmup_path = resolve_warmup_file(
        data_cfg.get("warmup_file"),
        data_cfg.get("dataset_config", "configs/hotpotqa/hotpotqa.yaml"),
    )
    print(f"[sft_pi] warmup_file={warmup_path}")
    records = read_jsonl(warmup_path)
    prompt_dir = Path(prompt_cfg["prompt_dir"])

    pi_system = (prompt_dir / "policy" / "system.txt").read_text(encoding="utf-8").strip()
    pi_user_tmpl = (prompt_dir / "policy" / "turn.txt").read_text(encoding="utf-8").strip()
    pi_finish_path = prompt_dir / "policy" / "forced_finish.txt"
    pi_finish_tmpl = pi_finish_path.read_text(encoding="utf-8").strip() if pi_finish_path.exists() else ""

    max_steps = dataset_cfg["environment"]["max_steps"]
    min_reward = data_cfg.get("min_reward", 1.0)

    print(f"[sft_pi] Building dataset from {len(records)} warmup records "
          f"(min_reward={min_reward}, max_steps={max_steps})...")
    data = build_pi_sft_data(records, pi_system, pi_user_tmpl,
                              finish_template=pi_finish_tmpl,
                              max_steps=max_steps,
                              min_reward=min_reward,
                              dataset=dataset)
    print(f"[sft_pi] Total examples: {len(data)}")

    train_data = data
    val_data = None
    print(f"[sft_pi] Train: {len(train_data)}")

    train_file = Path(data_cfg.get("train_file") or Path(dataset_cfg["data"]["output_dir"]) / "pi_sft_train.jsonl")
    write_jsonl(train_data, train_file)
    print(f"[sft_pi] train_file={train_file}")

    eval_cfg = cfg.get("eval", {})
    pre_eval_cb = None
    epoch_eval_cb = None
    if eval_cfg.get("enabled", False):
        splits = eval_cfg.get("splits", [eval_cfg.get("split", "dev")])
        spec = get_dataset(dataset)
        eval_examples = {}
        for split in splits:
            examples = spec.load_data(dataset_cfg, split)
            max_eval = eval_cfg.get("max_examples")
            if max_eval:
                examples = examples[:max_eval]
            eval_examples[split] = examples
        retriever = None
        if dataset_cfg.get("retriever", {}).get("url"):
            from src.utils.retriever import Retriever
            retriever = Retriever(dataset_cfg["retriever"]["url"])
        eval_batch_size = eval_cfg.get("batch_size", 8)
        eval_max_steps = eval_cfg.get("max_steps", max_steps)
        max_new_tokens = eval_cfg.get(
            "max_new_tokens",
            dataset_cfg["warmup"]["policy"].get("max_tokens", 64),
        )

        def _run_eval(model, tokenizer, epoch_label_prefix):
            default_debug_steps = eval_max_steps + (1 if spec.has_forced_finish else 0)
            results = {}
            for split, examples in eval_examples.items():
                results[split] = evaluate_pi_model(
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    retriever=retriever,
                    max_steps=eval_max_steps,
                    pi_system=pi_system,
                    pi_user_tmpl=pi_user_tmpl,
                    pi_finish_tmpl=pi_finish_tmpl,
                    max_new_tokens=max_new_tokens,
                    batch_size=eval_batch_size,
                    desc=f"{epoch_label_prefix} {split}",
                    dataset=dataset,
                    debug_examples=eval_cfg.get("debug_examples", 0),
                    debug_steps=eval_cfg.get("debug_steps", default_debug_steps),
                )
            return results

        if eval_cfg.get("pre_eval", True):
            pre_eval_cb = lambda model, tokenizer: _run_eval(model, tokenizer, "eval epoch 0")
        epoch_eval_cb = lambda model, tokenizer, epoch: _run_eval(model, tokenizer, f"eval epoch {epoch}")

    train_sft(
        cfg,
        train_data,
        val_data,
        output_dir=cfg["training"]["output_dir"],
        pre_eval_callback=pre_eval_cb,
        epoch_eval_callback=epoch_eval_cb,
    )


if __name__ == "__main__":
    main()
