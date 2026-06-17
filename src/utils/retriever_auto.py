"""Helpers for mapping training GPUs to retriever ports."""

from __future__ import annotations

import os
import re
from typing import Any


DEFAULT_RETRIEVER_HOST = "127.0.0.1"
DEFAULT_RETRIEVER_PORT_BASE = 2022


def _logical_cuda_index(device: str | None) -> int | None:
    if not device:
        return None
    match = re.fullmatch(r"cuda(?::(\d+))?", device.strip())
    if not match:
        return None
    return int(match.group(1) or 0)


def _visible_physical_gpus() -> list[int] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return None

    gpus: list[int] = []
    for item in visible.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            return None
        gpus.append(int(item))
    return gpus or None


def infer_physical_gpu(device: str | None = None) -> int | None:
    """Map a process-local CUDA device to the physical GPU id when possible."""
    logical_idx = _logical_cuda_index(device)
    visible_gpus = _visible_physical_gpus()
    if visible_gpus:
        idx = logical_idx if logical_idx is not None else 0
        if 0 <= idx < len(visible_gpus):
            return visible_gpus[idx]
        return visible_gpus[0]
    return logical_idx


def retriever_url_for_gpu(
    gpu: int,
    host: str = DEFAULT_RETRIEVER_HOST,
    port_base: int = DEFAULT_RETRIEVER_PORT_BASE,
) -> str:
    return f"http://{host}:{port_base + gpu}/retrieve"


def configure_retriever_url(
    dataset_cfg: dict[str, Any],
    retriever_gpu: str | int | None = "auto",
    retriever_url: str | None = None,
    device: str | None = None,
    host: str = DEFAULT_RETRIEVER_HOST,
    port_base: int = DEFAULT_RETRIEVER_PORT_BASE,
) -> str | None:
    """Update dataset_cfg['retriever']['url'] using an explicit or inferred GPU.

    `retriever_gpu="auto"` maps CUDA_VISIBLE_DEVICES plus the local cuda index
    to the physical GPU id, so CUDA_VISIBLE_DEVICES=1 with --device cuda:0 uses
    port 2023.
    """
    retriever_cfg = dataset_cfg.get("retriever")
    if not isinstance(retriever_cfg, dict):
        return None

    if retriever_url:
        retriever_cfg["url"] = retriever_url
        return retriever_url

    if str(retriever_gpu).lower() in {"none", "off", "config", "false"}:
        return retriever_cfg.get("url")

    if str(retriever_gpu).lower() == "auto":
        gpu = infer_physical_gpu(device)
    else:
        gpu = int(retriever_gpu) if retriever_gpu is not None else None

    if gpu is None:
        return retriever_cfg.get("url")

    url = retriever_url_for_gpu(gpu, host=host, port_base=port_base)
    retriever_cfg["url"] = url
    retriever_cfg["port"] = port_base + gpu
    return url
