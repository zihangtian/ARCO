"""Collect warmup trajectories.

Warmup uses:
  - a policy backend: "api" | "local" | "vllm"
  - a rubric backend: currently "api"

For the vLLM policy backend, start the server first:
  CUDA_VISIBLE_DEVICES=<gpu_id> python -m vllm.entrypoints.openai.api_server \\
    --model <model_path> --port <port> --max-model-len 4096

Usage:
    python scripts/collect_warmup.py --config configs/hotpotqa/hotpotqa.yaml [--split train] [--max_examples N] [--output_file PATH]
"""

import argparse
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.misc import load_yaml
from src.utils.cache import DiskCache
from src.utils.api_client import APIClient
from src.agents.policy import PolicyAgent
from src.agents.rubric import RubricAgent
from src.environments.registry import get_dataset
from src.training.warmup_collect import collect_warmup_data


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("/", "-")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def _auto_output_path(cfg: dict, mode: str, split: str, max_examples: int | None) -> str:
    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup_cfg = cfg["warmup"]
    policy_cfg = warmup_cfg["policy"]
    rubric_cfg = warmup_cfg["rubric"]
    k = cfg["warmup"].get("k_samples", 3)
    temp = policy_cfg.get("temperature",
                          policy_cfg.get("sample_temperature",
                                         policy_cfg.get("api_temperature", 0.8)))
    mu_model = _slug(rubric_cfg.get("api_model", "unknown"))

    if mode == "api":
        pi_model = _slug(policy_cfg.get("api_model", "unknown"))
        prefix = f"warmup_{split}_api-{pi_model}_rubric-{mu_model}_k{k}_temp{temp}"
    elif mode == "local":
        pi_model = _slug(Path(policy_cfg["local_model_path"]).name)
        prefix = f"warmup_{split}_local-{pi_model}_mu-{mu_model}_k{k}_temp{temp}"
    elif mode == "vllm":
        pi_model = _slug(Path(policy_cfg["vllm_model"]).name)
        prefix = f"warmup_{split}_vllm-{pi_model}_rubric-{mu_model}_k{k}_temp{temp}"
    else:
        prefix = f"warmup_{split}_{mode}"

    exact = out_dir / f"{prefix}.jsonl"
    if exact.exists():
        return str(exact)

    legacy = sorted(out_dir.glob(f"{prefix}_max*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if legacy:
        return str(legacy[0])

    return str(exact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa/hotpotqa.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--policy_model", default=None,
                        help="Override warmup.policy.api_model, e.g. gpt-4o-mini")
    parser.add_argument("--rubric_model", default=None,
                        help="Override warmup.rubric.api_model, e.g. gpt-5.4")
    parser.add_argument("--policy_temperature", type=float, default=None,
                        help="Override warmup.policy.temperature")
    parser.add_argument("--k_samples", type=int, default=None,
                        help="Override warmup.k_samples")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Override warmup.n_workers")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    warmup_cfg = cfg["warmup"]
    policy_cfg = warmup_cfg["policy"]
    rubric_cfg = warmup_cfg["rubric"]
    if args.policy_model is not None:
        policy_cfg["api_model"] = args.policy_model
        policy_cfg["backend"] = "api"
    if args.rubric_model is not None:
        rubric_cfg["api_model"] = args.rubric_model
        rubric_cfg["backend"] = "api"
    if args.policy_temperature is not None:
        policy_cfg["temperature"] = args.policy_temperature
    if args.k_samples is not None:
        warmup_cfg["k_samples"] = args.k_samples
    if args.n_workers is not None:
        warmup_cfg["n_workers"] = args.n_workers
    prompt_dir = Path(cfg["prompts"]["dir"])
    shared_dir = Path(cfg["prompts"]["shared_dir"])
    dataset = cfg.get("dataset", "hotpotqa")
    policy_backend = policy_cfg.get("backend", "api")
    rubric_backend = rubric_cfg.get("backend", "api")
    policy_temperature = policy_cfg.get(
        "temperature",
        policy_cfg.get("sample_temperature", policy_cfg.get("api_temperature", 0.0)),
    )
    if rubric_backend != "api":
        raise ValueError(f"Unsupported warmup rubric backend: {rubric_backend!r}")
    output_path = args.output_file or _auto_output_path(cfg, policy_backend, args.split, args.max_examples)

    cache = DiskCache(cfg["data"]["cache_dir"])
    rubric_client = APIClient(
        model=rubric_cfg.get("api_model", "gpt-5-ca"),
        cache=cache,
        base_url=cfg["api"].get("base_url"),
        api_key=cfg["api"].get("api_key"),
    )
    policy_client = None
    if policy_backend == "api":
        policy_client = APIClient(
            model=policy_cfg.get("api_model", "gpt-4o-mini"),
            cache=cache,
            base_url=cfg["api"].get("base_url"),
            api_key=cfg["api"].get("api_key"),
        )

    spec = get_dataset(dataset)

    rubric_agent = RubricAgent.from_prompt_dirs(
        client=rubric_client,
        shared_dir=shared_dir,
        dataset_dir=prompt_dir,
        dataset=dataset,
        raw_actions=spec.display_raw_actions,
        max_tokens=rubric_cfg.get("max_tokens", 128),
    )

    examples = spec.load_data(cfg, args.split)
    max_examples = args.max_examples or warmup_cfg.get("max_examples")
    if max_examples:
        examples = examples[:max_examples]
    print(f"[warmup] {len(examples)} examples from split '{args.split}', "
          f"policy_backend={policy_backend!r} rubric_backend={rubric_backend!r}")
    print(f"[warmup] output_file={output_path}")

    retriever = None
    if cfg.get("retriever", {}).get("url"):
        from src.utils.retriever import Retriever
        retriever = Retriever(cfg["retriever"]["url"], fail_on_empty=True)
        print(f"[warmup] Retriever at {cfg['retriever']['url']}")

    # ── api mode: GPT as π ────────────────────────────────────────────────────
    if policy_backend == "api":
        policy = PolicyAgent.from_prompt_dir(
            client=policy_client,
            prompt_dir=prompt_dir,
            dataset=dataset,
            parser_fn=spec.parse_output,
            temperature=policy_temperature,
            max_tokens=policy_cfg.get("max_tokens", 128),
        )
        collect_warmup_data(
            examples=examples,
            dataset=dataset,
            policy=policy,
            rubric_agent=rubric_agent,
            max_steps=cfg["environment"]["max_steps"],
            output_path=output_path,
            retriever=retriever,
            resume=warmup_cfg.get("resume", True),
            n_workers=warmup_cfg.get("n_workers", 8),
            mode="api",
            debug=args.debug,
            k_samples=warmup_cfg.get("k_samples", 3),
        )

    # ── local_pi mode: local HF model as π, GPT as μ ─────────────────────────
    elif policy_backend == "local":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        pi_model_path = policy_cfg["local_model_path"]
        pi_device = policy_cfg.get("local_device", "cuda:0")
        print(f"[warmup] Loading local π model from {pi_model_path} on {pi_device} ...")
        pi_tokenizer = AutoTokenizer.from_pretrained(pi_model_path)
        if pi_tokenizer.pad_token is None:
            pi_tokenizer.pad_token = pi_tokenizer.eos_token

        pi_model = AutoModelForCausalLM.from_pretrained(
            pi_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": pi_device},
        )
        pi_model.eval()
        print(f"[warmup] Local π model loaded on {pi_device}.")

        # Load prompt templates for local π
        pi_system = (prompt_dir / "policy" / "system.txt").read_text(encoding="utf-8").strip()
        pi_user_tmpl = (prompt_dir / "policy" / "turn.txt").read_text(encoding="utf-8").strip()
        pi_finish_path = prompt_dir / "policy" / "forced_finish.txt"
        pi_finish_tmpl = pi_finish_path.read_text(encoding="utf-8").strip() if pi_finish_path.exists() else ""

        collect_warmup_data(
            examples=examples,
            dataset=dataset,
            rubric_agent=rubric_agent,
            max_steps=cfg["environment"]["max_steps"],
            output_path=output_path,
            pi_model=pi_model,
            pi_tokenizer=pi_tokenizer,
            pi_system=pi_system,
            pi_user_tmpl=pi_user_tmpl,
            pi_finish_tmpl=pi_finish_tmpl,
            sample_temperature=policy_temperature,
            pi_max_new_tokens=policy_cfg.get("max_tokens", 128),
            k_samples=warmup_cfg.get("k_samples", 3),
            pi_batch_size=policy_cfg.get("local_batch_size", 32),
            retriever=retriever,
            resume=warmup_cfg.get("resume", True),
            n_workers=warmup_cfg.get("n_workers", 8),
            mode="local_pi",
            debug=args.debug,
        )

    # ── vllm_pi mode: vLLM server as π, GPT as μ ─────────────────────────────
    elif policy_backend == "vllm":
        vllm_url = policy_cfg.get("vllm_url", "http://127.0.0.1:8001/v1")
        vllm_model = policy_cfg["vllm_model"]

        vllm_cache = DiskCache(str(Path(cfg["data"]["cache_dir"]) / "vllm"))
        vllm_client = APIClient(
            model=vllm_model,
            cache=vllm_cache,
            base_url=vllm_url,
            api_key="EMPTY",   # vLLM doesn't check the key
        )
        pi_policy = PolicyAgent.from_prompt_dir(
            client=vllm_client,
            prompt_dir=prompt_dir,
            dataset=dataset,
            parser_fn=spec.parse_output,
            temperature=policy_temperature,
            max_tokens=policy_cfg.get("max_tokens", 128),
        )
        print(f"[warmup] vLLM π at {vllm_url}  model={vllm_model}")

        collect_warmup_data(
            examples=examples,
            dataset=dataset,
            rubric_agent=rubric_agent,
            max_steps=cfg["environment"]["max_steps"],
            output_path=output_path,
            pi_policy=pi_policy,
            k_samples=warmup_cfg.get("k_samples", 3),
            retriever=retriever,
            resume=warmup_cfg.get("resume", True),
            n_workers=warmup_cfg.get("n_workers", 16),
            mode="vllm_pi",
            debug=args.debug,
        )

    else:
        raise ValueError(f"Unknown warmup policy backend: {policy_backend!r}  (expected 'api', 'local', 'vllm')")


if __name__ == "__main__":
    main()
