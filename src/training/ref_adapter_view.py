"""Shared-base reference model wrapper.

When π and π_ref (or μ and μ_ref) share a single PEFT base model with
two adapters, wrap the reference view in `RefAdapterView`. On __call__
it switches the active adapter to the frozen one, runs forward under
`torch.no_grad()`, then switches back to the trainable adapter. Backward
on the train forward is unaffected — by then the LoRA outputs have
already been captured into the autograd graph.

`to(...)` is a no-op so that trainer code that offloads the reference
model to CPU does not also yank the trainable model. Set
`offloadable=True` only if the wrapper *owns* a separate base.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RefAdapterView(nn.Module):
    """Forward-only view that runs `model` under a fixed adapter."""

    def __init__(self, model: nn.Module, ref_adapter: str, train_adapter: str):
        super().__init__()
        # underscore so PyTorch does not register as submodule (avoids
        # double-counting in parameters() / state_dict())
        object.__setattr__(self, "_m", model)
        self._ref = ref_adapter
        self._train = train_adapter

    @property
    def config(self):
        return self._m.config

    @property
    def device(self):
        return next(self._m.parameters()).device

    def __call__(self, *args, **kwargs):
        self._m.set_adapter(self._ref)
        try:
            with torch.no_grad():
                return self._m(*args, **kwargs)
        finally:
            self._m.set_adapter(self._train)

    # Trainer offloads ref to CPU between phases; with a shared base,
    # those calls must not move the trainable model.
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def parameters(self, recurse: bool = True):
        return iter(())

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return iter(())
