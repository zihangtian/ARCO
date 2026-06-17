"""Dataset registry — one-stop lookup for dataset-specific callables.

Each dataset module (e.g., hotpotqa.py) calls ``register_dataset``
at import time.  Consumers call ``get_dataset(name)`` to obtain a
:class:`DatasetSpec` that bundles everything needed to build environments,
parse model output, compute rewards, etc.

Adding a new dataset
--------------------
1. Create ``src/environments/<name>.py`` with env class + helpers.
2. Call ``register_dataset(DatasetSpec(...))`` at the bottom of that module.
3. Add ``from . import <name>`` to ``src/environments/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from .base import BaseEnvironment


@dataclass(frozen=True)
class DatasetSpec:
    """Everything a consumer needs to work with a dataset."""

    name: str

    # (example, **kw) -> BaseEnvironment
    # kw may include retriever=, max_steps=, etc.
    make_env: Callable[..., BaseEnvironment]

    # model text -> (thought_or_empty, action)
    parse_output: Callable[[str], Tuple[str, str]]

    # (env, example) -> dict  with at least a "reward" key
    compute_reward: Callable[[BaseEnvironment, dict], dict]

    # (cfg, split_name) -> list[dict]
    load_data: Callable[[dict, str], list[dict]]

    # action -> budget cost (e.g. HotpotQA: Search=1, Finish=0; ALFWorld: always 1)
    step_cost: Callable[[str], int]

    # Whether to strip Act[…] / Finish[…] wrappers when displaying actions
    display_raw_actions: bool = False

    # Whether budget exhaustion triggers a forced-finish prompt
    has_forced_finish: bool = True


# ── registry ──────────────────────────────────────────────────────────────

_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    """Register a DatasetSpec.  Called at module import time."""
    _REGISTRY[spec.name] = spec
    return spec


def get_dataset(name: str) -> DatasetSpec:
    """Look up a registered dataset by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown dataset {name!r}. Registered: {available}")
    return _REGISTRY[name]


def registered_datasets() -> list[str]:
    """Return sorted list of registered dataset names."""
    return sorted(_REGISTRY)
