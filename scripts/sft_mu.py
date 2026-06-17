"""SFT train μ on warmup trajectories.

Usage:
    python scripts/sft_mu.py
    python scripts/sft_mu.py --config configs/hotpotqa/mu.yaml
    python scripts/sft_mu.py --rubric_mode direct  # w/o Rubric ablation
    python scripts/sft_mu.py --rubric_mode fixed   # w/o Generative Rubric ablation

Rubric modes:
  - generative (default): ARCO. μ generates per-step rubric (3 criteria) +
    regresses 3 scores. Both LM loss and score MSE are active.
  - direct: w/o Rubric ablation. μ keeps the same prompt, but learns only a
    single mean-pooled scalar score (num_scores=1). LM rubric tokens are
    dropped (score_only=True).
  - fixed: w/o Generative Rubric ablation. The 3 criteria target is replaced
    by one shared static rubric (loaded from --static_rubric_file). Score
    head still regresses 3 scores; the rubric content is held constant
    across every step.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.misc import load_yaml, read_jsonl, write_jsonl, resolve_warmup_file
from src.training.sft_data_builder import build_mu_sft_data
from src.training.sft_trainer import train_sft
from src.training.dual_head_mu import DualHeadMu


_DEFAULT_STATIC_RUBRICS = {
    "hotpotqa": "scripts/ablation/static_rubric_hotpotqa.json",
}


def _load_static_rubric(path_str: str | None, dataset: str) -> str:
    """Load the static rubric JSON for fixed-rubric ablation."""
    if path_str is None:
        path_str = _DEFAULT_STATIC_RUBRICS.get(dataset)
        if path_str is None:
            raise ValueError(
                f"No default static rubric for dataset {dataset!r}; "
                f"pass --static_rubric_file"
            )
    text = Path(path_str).read_text(encoding="utf-8").strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or "criteria" not in parsed:
        raise ValueError(
            "static rubric file must contain a JSON object with key 'criteria'"
        )
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa/mu.yaml")
    parser.add_argument("--base_model", default=None, help="Override model.base_model_path")
    parser.add_argument("--device", default=None, help="Override model.device")
    parser.add_argument("--output_dir", default=None, help="Override training.output_dir")
    parser.add_argument(
        "--rubric_mode",
        choices=["generative", "direct", "fixed"],
        default="generative",
        help="generative=ARCO; direct=w/o Rubric; fixed=w/o Generative Rubric",
    )
    parser.add_argument(
        "--static_rubric_file",
        default=None,
        help="Path to static rubric JSON (only used when rubric_mode=fixed). "
             "Defaults to scripts/ablation/static_rubric_<dataset>.json.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.base_model:
        cfg["model"]["base_model_path"] = args.base_model
    if args.device:
        cfg["model"]["device"] = args.device
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    data_cfg = cfg["data"]
    prompt_cfg = cfg["prompts"]
    dataset_cfg_path = data_cfg.get("dataset_config", "configs/hotpotqa/hotpotqa.yaml")
    dataset_cfg = load_yaml(dataset_cfg_path)
    dataset = dataset_cfg.get("dataset", "hotpotqa")

    sd = Path(prompt_cfg["shared_dir"])
    dd = Path(prompt_cfg["prompt_dir"])

    warmup_path = resolve_warmup_file(
        data_cfg.get("warmup_file"),
        dataset_cfg_path,
    )
    print(f"[sft_mu] rubric_mode={args.rubric_mode}")
    print(f"[sft_mu] warmup_file={warmup_path}")
    records = read_jsonl(warmup_path)

    # Filter: only keep records where decomposition constraint is satisfied
    # i.e. |sum of step score means - reward| < threshold
    decomp_threshold = data_cfg.get("decomp_filter_threshold", 0.05)
    if decomp_threshold > 0:
        before = len(records)
        filtered = []
        for rec in records:
            steps = rec.get("steps", [])
            if not steps:
                continue
            step_means = []
            for s in steps:
                sc = s.get("scores", [])
                if sc:
                    step_means.append(sum(sc) / len(sc))
            if not step_means:
                continue
            diff = abs(sum(step_means) - rec.get("reward", 0))
            if diff < decomp_threshold:
                filtered.append(rec)
        records = filtered
        print(f"[sft_mu] Decomposition filter: {before} → {len(records)} "
              f"(threshold={decomp_threshold}, kept {100*len(records)/before:.0f}%)")

    task_context = (dd / "task_description.txt").read_text(encoding="utf-8").strip()
    mu_system_tmpl = (sd / "rubric" / "system.txt").read_text(encoding="utf-8").strip()
    mu_system = mu_system_tmpl.format(task_context=task_context)
    mu_user_tmpl = (dd / "rubric" / "step.txt").read_text(encoding="utf-8").strip()

    print(f"[sft_mu] Building dataset from {len(records)} warmup records...")
    data = build_mu_sft_data(
        records,
        mu_system,
        mu_user_tmpl,
        drop_empty_criteria=data_cfg.get("drop_empty_criteria", True),
        dataset=dataset,
    )

    # Apply rubric_mode-specific transformation.
    if args.rubric_mode == "direct":
        # w/o Rubric: drop rubric text target, regress single scalar mean.
        transformed = []
        for item in data:
            scores = item.get("scores") or [0.0]
            mean_score = sum(scores) / len(scores)
            transformed.append({
                "example_id": item["example_id"],
                "sample_idx": item.get("sample_idx"),
                "step": item["step"],
                "messages": item["messages"],
                "target": "",
                "scores": [mean_score],
            })
        data = transformed
        num_scores = 1
        score_only = True
    elif args.rubric_mode == "fixed":
        # w/o Generative Rubric: replace the 3-criterion target with one
        # shared static rubric. Score head still regresses 3 scores.
        static_rubric_text = _load_static_rubric(args.static_rubric_file, dataset)
        print(f"[sft_mu] static_rubric_file resolved (mode=fixed)")
        transformed = []
        for item in data:
            transformed.append({
                "example_id": item["example_id"],
                "sample_idx": item.get("sample_idx"),
                "step": item["step"],
                "messages": item["messages"],
                "target": static_rubric_text,
                "scores": item["scores"],
            })
        data = transformed
        num_scores = 3
        score_only = False
    else:
        # generative: ARCO default — keep build_mu_sft_data output as-is.
        num_scores = 3
        score_only = False

    print(f"[sft_mu] Total examples: {len(data)} (num_scores={num_scores}, "
          f"score_only={score_only})")

    train_data = data
    val_data = None
    print(f"[sft_mu] Train: {len(train_data)}")

    out_dir = Path(dataset_cfg["data"]["output_dir"])
    write_jsonl(train_data, out_dir / f"mu_sft_train_{args.rubric_mode}.jsonl")

    def _get_hidden_size(m):
        cfg = m.config
        # Gemma3 nests text config under text_config
        if hasattr(cfg, 'text_config'):
            return cfg.text_config.hidden_size
        return cfg.hidden_size

    train_sft(
        cfg,
        train_data,
        val_data,
        output_dir=cfg["training"]["output_dir"],
        model_wrapper=lambda m: DualHeadMu(m, _get_hidden_size(m), num_scores=num_scores),
        score_only=score_only,
    )


if __name__ == "__main__":
    main()
