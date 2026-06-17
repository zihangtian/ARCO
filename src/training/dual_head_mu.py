"""Dual-head μ model: shared backbone with text generation head + score head.

The backbone is a CausalLM (e.g. Qwen2.5-3B with LoRA). On top of it:
- lm_head (built-in): next-token prediction for criteria text/JSON generation
- score_head (new): 2-layer MLP that maps the last token's hidden state
  to 3 scores ∈ [-1, 1], one per criterion

The score_head provides a differentiable reward signal so that gradients
can flow back through the backbone during RL training.
"""

from pathlib import Path

import torch
import torch.nn as nn


class DualHeadMu(nn.Module):
    """Shared-backbone dual-head μ model."""

    NUM_SCORES = 3  # default: one score per criterion

    def __init__(self, backbone: nn.Module, hidden_size: int, num_scores: int | None = None):
        super().__init__()
        self.backbone = backbone
        n = num_scores if num_scores is not None else self.NUM_SCORES
        self.num_scores = n
        # Gradual reduction: H → 512 → 128 → n
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, n),
            nn.Tanh(),
        )
        # Initialise score_head on same device / dtype as backbone
        device = next(backbone.parameters()).device
        # score_head 用 float32 训练（更稳定），但放在同一设备
        self.score_head = self.score_head.to(device=device, dtype=torch.float32)

        # Disable HF kv-cache once: incompatible with gradient checkpointing
        # and we never call .generate() through forward_score paths.
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False
        inner = self._get_hf_model()
        if inner is not None and hasattr(inner, "config"):
            inner.config.use_cache = False

    # ── forward methods ───────────────────────────────────────────────────

    def _get_hf_model(self):
        """Unwrap PEFT/LoRA to reach the HF CausalLM that owns lm_head.

        Chain: PeftModel → LlamaForCausalLM (has lm_head) → LlamaModel (no lm_head)
        We want LlamaForCausalLM — one level down from PeftModel.
        """
        m = self.backbone
        if hasattr(m, 'base_model'):
            # PeftModel.base_model → LoraModel; LoraModel.model → CausalLM
            m = m.base_model
        if hasattr(m, 'model') and hasattr(m.model, 'lm_head'):
            return m.model  # CausalLM
        # Fallback: walk until we find lm_head as own attribute
        while hasattr(m, 'model'):
            if 'lm_head' in m.__dict__ or 'lm_head' in type(m).__dict__:
                return m
            m = m.model
        return m

    def _inner_backbone(self):
        """The transformer body (no lm_head). Calling this directly skips
        the language-model head — useful when only hidden states are needed
        (saves ~25 GB on large-vocab models). LoRA adapters live under here
        so they still apply."""
        hf = self._get_hf_model()
        return getattr(hf, "model", hf)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, **kwargs):
        """Standard forward compatible with HF labels interface and logits access."""
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

    def forward_score(self, input_ids, attention_mask=None, score_only=False,
                      prompt_lengths=None):
        """Differentiable score prediction + logits in one forward pass.

        Parameters
        ----------
        score_only : bool
            If True, bypass lm_head by calling the inner backbone directly
            (saves ~25 GB on large-vocab models). `out` is then the bare
            transformer output (BaseModelOutputWithPast), not a CausalLM
            output — callers that need logits must use score_only=False.
        prompt_lengths : ignored (kept for API compatibility)

        Returns
        -------
        scores : Tensor (B, num_scores)
        out : ModelOutput  (CausalLMOutput if score_only=False, else BaseModelOutput)
        """
        if score_only:
            out = self._inner_backbone()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden = out.last_hidden_state
        else:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = out.hidden_states[-1]  # (B, seq_len, H)

        # Mean pooling over all non-padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            h = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, H)
        else:
            h = last_hidden.mean(dim=1)  # (B, H)

        scores = self.score_head(h.float())  # (B, num_scores)
        return scores, out

    def generate(self, **kwargs):
        """Delegate generation to backbone."""
        return self.backbone.generate(**kwargs)

    # ── save / load ───────────────────────────────────────────────────────

    def save_pretrained(self, path):
        """Save LoRA adapter + score_head weights side by side."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.score_head.state_dict(), path / "score_head.pt")

    @staticmethod
    def load_score_head(path, hidden_size, device="cpu"):
        """Load score_head state dict from a checkpoint directory."""
        score_head_path = Path(path) / "score_head.pt"
        if score_head_path.exists():
            return torch.load(score_head_path, map_location=device, weights_only=True)
        return None

    @staticmethod
    def infer_num_scores(path) -> int:
        """Infer num_scores from a saved score_head checkpoint."""
        score_head_path = Path(path) / "score_head.pt"
        if score_head_path.exists():
            state = torch.load(score_head_path, map_location="cpu", weights_only=True)
            # Last linear layer bias shape = (num_scores,)
            for key in ("4.bias", "2.bias", "score_head.4.bias", "score_head.2.bias"):
                if key in state:
                    return state[key].shape[0]
        return DualHeadMu.NUM_SCORES  # default

    # ── proxy methods so external code works transparently ────────────────

    def parameters(self, recurse=True):
        yield from self.backbone.parameters(recurse=recurse)
        yield from self.score_head.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        bp = prefix + "backbone." if prefix else "backbone."
        sp = prefix + "score_head." if prefix else "score_head."
        yield from self.backbone.named_parameters(prefix=bp, recurse=recurse)
        yield from self.score_head.named_parameters(prefix=sp, recurse=recurse)

    def train(self, mode=True):
        self.backbone.train(mode)
        self.score_head.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.backbone, "is_gradient_checkpointing", False)

    @property
    def config(self):
        return self.backbone.config

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def print_trainable_parameters(self):
        backbone_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        score_trainable = sum(p.numel() for p in self.score_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        print(
            f"DualHeadMu: backbone_trainable={backbone_trainable / 1e6:.1f}M, "
            f"score_head={score_trainable / 1e6:.1f}M, total={total / 1e9:.2f}B"
        )
