"""Miscellaneous I/O utilities."""

import json
import re
from pathlib import Path

import yaml


_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _coerce_yaml_scalars(obj):
    """Convert numeric-looking YAML strings into Python numbers.

    PyYAML treats values like ``1e-5`` as strings unless they include a decimal
    point. Our generated configs often use scientific notation for learning
    rates, so normalize them here after loading.
    """
    if isinstance(obj, dict):
        return {k: _coerce_yaml_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_yaml_scalars(v) for v in obj]
    if isinstance(obj, str) and _NUMERIC_RE.fullmatch(obj):
        try:
            if any(ch in obj for ch in ".eE"):
                return float(obj)
            return int(obj)
        except ValueError:
            return obj
    return obj


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return _coerce_yaml_scalars(yaml.safe_load(f))


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def resolve_warmup_file(
    configured_path: str | Path | None,
    dataset_config_path: str | Path = "configs/hotpotqa/hotpotqa.yaml",
) -> Path:
    """Resolve the warmup jsonl path with sensible fallbacks.

    Resolution order:
    1. `configured_path` if it exists
    2. `dataset_config["warmup"]["output_file"]` if it exists
    3. newest `warmup*.jsonl` under `dataset_config["data"]["output_dir"]`
    """
    if configured_path:
        path = Path(configured_path)
        if path.exists():
            return path

    dataset_cfg = load_yaml(dataset_config_path)

    warmup_cfg_path = Path(dataset_cfg["warmup"]["output_file"])
    if warmup_cfg_path.exists():
        return warmup_cfg_path

    out_dir = Path(dataset_cfg["data"]["output_dir"])
    candidates = sorted(
        out_dir.glob("warmup*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    attempted = [str(configured_path)] if configured_path else []
    attempted.append(str(warmup_cfg_path))
    attempted.append(str(out_dir / "warmup*.jsonl"))
    raise FileNotFoundError(
        "Could not resolve a warmup jsonl file. Tried: " + ", ".join(attempted)
    )
