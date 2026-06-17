"""Trainer selection.

`make_trainer(cfg, **kwargs)` returns the ARCO `RLTrainer`. Baseline
trainers are not part of the minimal release.
"""
from __future__ import annotations

from .rl_trainer import RLTrainer


def make_trainer(cfg: dict, *, trainer_cls=None, **kwargs):
    if trainer_cls is None:
        trainer_cls = RLTrainer
    return trainer_cls(cfg=cfg, **kwargs)


__all__ = ["RLTrainer", "make_trainer"]
