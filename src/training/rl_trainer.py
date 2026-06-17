"""Batch RL training for π and μ.

Algorithm
─────────
Stage 1  (epoch < dense_start_epoch):
  π  ← policy gradient with sparse reward R
       loss per step: -R · log π(a_t | s_t)
  μ  ← L_score + β·KL(μ_current || μ_ref)

Stage 2  (epoch ≥ dense_start_epoch):
  π  ← policy gradient with configurable dense μ signal
       loss per step: -A_t · log π(a_t | s_t)
  μ  ← same loss

Rollout is batched: at each step, all active envs are queried together with
a single batched generate call for π and a single batched generate call for μ.
Trajectories finish independently; inactive envs are masked out each step.

Gradient update happens once per batch (batch_size trajectories).
"""

import random
import re
import json
import logging
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..environments.registry import get_dataset
from ..environments.base import BaseState
from ..utils.prompt_builders import (
    policy_fields_from_state,
    rubric_fields_from_state_action,
)
from ..utils.misc import load_yaml, write_jsonl
from .pi_eval import evaluate_pi_model


# ── helpers ───────────────────────────────────────────────────────────────────

def _token_log_probs(model, input_ids: torch.Tensor,
                     response_start: int) -> torch.Tensor:
    """Differentiable per-token log probs for response positions."""
    out = model(input_ids=input_ids)
    log_probs = F.log_softmax(out.logits[0], dim=-1)
    lps = [log_probs[i, input_ids[0, i + 1]]
           for i in range(response_start, input_ids.shape[1] - 1)]
    return torch.stack(lps) if lps else input_ids.new_zeros(1).float()


def _parse_score(text: str) -> float:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return max(-1.0, min(1.0, float(json.loads(m.group()).get("score", 0.0))))
        except (json.JSONDecodeError, ValueError):
            pass
    nums = re.findall(r"-?(?:0\.\d+|1\.0+|0|1)\b", text)
    return float(nums[0]) if nums else 0.0


# ── RLTrainer ─────────────────────────────────────────────────────────────────

def _module_device(m):
    """Device of a Module, or its `.device` attr (for shared-base wrappers
    that own no parameters of their own)."""
    if hasattr(m, "device"):
        return m.device
    try:
        return next(m.parameters()).device
    except StopIteration:
        # Fallback: try to walk a wrapped underlying module
        inner = getattr(m, "_m", None)
        if inner is not None:
            return next(inner.parameters()).device
        raise


class RLTrainer:
    def __init__(self, cfg: dict,
                 pi_model, pi_tokenizer,
                 mu_model, mu_tokenizer,
                 train_examples: list[dict],
                 retriever=None,
                 pi_ref_model=None,
                 mu_ref_model=None,
                 vllm_cfg: dict | None = None):
        self.cfg = cfg
        self.rl_cfg = cfg["rl"]

        self.pi = pi_model
        self.pi_tok = pi_tokenizer
        self.pi_ref = pi_ref_model  # frozen warmup checkpoint for KL regularisation

        self.mu = mu_model
        self.mu_tok = mu_tokenizer
        self.mu_ref = mu_ref_model  # frozen warmup checkpoint for KL regularisation
        self.has_mu = (self.mu is not None and self.mu_tok is not None)

        self.examples = train_examples
        self.dataset = cfg.get("dataset", "hotpotqa")
        self.spec = get_dataset(self.dataset)
        self.retriever = retriever
        self.pi_device = next(pi_model.parameters()).device
        self.pi_ref_device = (
            _module_device(pi_ref_model) if pi_ref_model is not None else self.pi_device
        )
        self.mu_device = next(mu_model.parameters()).device if self.has_mu else None
        self.max_steps = cfg["environment"]["max_steps"]
        self.batch_size = self.rl_cfg.get("train_batch_size", self.rl_cfg.get("batch_size", 8))
        self.eval_batch_size = self.rl_cfg.get("eval_batch_size", self.batch_size)
        self.dense_start = self.rl_cfg.get("dense_start_epoch", 3)
        self.rollout_temperature = self.rl_cfg.get("rollout_temperature", 1.0)
        self.debug_rollout = self.rl_cfg.get("debug_rollout", False)
        self.dense_signal_mode = self.rl_cfg.get("dense_signal_mode", "sum_return")
        self.dense_baseline_mode = self.rl_cfg.get("dense_baseline_mode", "global_batch")
        self.dense_bucket_mode = self.rl_cfg.get("dense_bucket_mode", "search_terminal")
        self.dense_gamma = self.rl_cfg.get("dense_gamma", 0.9)
        self.dense_bucket_leave_one_out = self.rl_cfg.get("dense_bucket_leave_one_out", False)
        self.dense_bucket_fallback_to_global = self.rl_cfg.get(
            "dense_bucket_fallback_to_global", True)
        self.mu_score_mode = self.rl_cfg.get("mu_score_mode", "rubric_generate")
        if self.mu_score_mode not in {"rubric_generate", "direct"}:
            raise ValueError(f"Unknown mu_score_mode: {self.mu_score_mode}")

        # Sparse/dense π learning-rate aliases.
        # Prefer the explicit sparse_* key, but keep backward compatibility with pi_lr.
        self.sparse_pi_lr = self.rl_cfg.get("sparse_pi_lr", self.rl_cfg.get("pi_lr", 0.0))
        self.dense_pi_lr = self.rl_cfg.get("dense_pi_lr")

        # μ only needs logits when KL regularisation is active (mu_lr > 0 and mu_kl > 0)
        mu_lr = self.rl_cfg.get("mu_lr", 0)
        mu_kl = self.rl_cfg.get("mu_kl_coef", 0)
        self._mu_needs_logits = (mu_lr > 0 and mu_kl > 0)

        # ── static criteria mode (predefined rubric baseline) ────────────
        static_file = self.rl_cfg.get("static_criteria_file")
        if static_file and self.has_mu:
            self.static_criteria_text = Path(static_file).read_text(encoding="utf-8").strip()
            print(f"[RL] Static criteria mode: {static_file}")
        else:
            self.static_criteria_text = None

        # ── reward mode (baseline variants) ──────────────────────────────
        self.reward_mode = self.rl_cfg.get("reward_mode", "default")
        self.reward_label = "EM" if self.reward_mode == "default" else "R"
        self.rubric_outcome_reward = self.rl_cfg.get("rubric_outcome_reward", False)

        # Hook for baseline subclasses to attach extra components / GPT clients.
        # Default: no-op.
        self._init_baseline_components(cfg)

        prompt_dir = Path(cfg["prompts"]["dir"])
        shared_dir = Path(cfg["prompts"]["shared_dir"])
        task_context = (prompt_dir / "task_description.txt").read_text(encoding="utf-8").strip()
        mu_sys_tmpl = (shared_dir / "rubric" / "system.txt").read_text(encoding="utf-8").strip()

        self.pi_system = (prompt_dir / "policy" / "system.txt").read_text(encoding="utf-8").strip()
        self.pi_user_tmpl = (prompt_dir / "policy" / "turn.txt").read_text(encoding="utf-8").strip()
        _ff_path = prompt_dir / "policy" / "forced_finish.txt"
        self.pi_finish_tmpl = _ff_path.read_text(encoding="utf-8").strip() if _ff_path.exists() else ""
        self.mu_system = mu_sys_tmpl.format(task_context=task_context) if self.has_mu else None
        self.mu_user_tmpl = (
            (prompt_dir / "rubric" / "step.txt").read_text(encoding="utf-8").strip()
            if self.has_mu else None
        )
        if self.has_mu and cfg.get("data", {}).get("mu_use_prefix_free_prompt", False):
            pf_path = prompt_dir / "rubric" / "step_prefix_free.txt"
            self.mu_user_tmpl = pf_path.read_text(encoding="utf-8").strip()
            print(f"[RL] mu_user_tmpl: prefix-free version → {pf_path}")

        # Rubric width K (matches mu's score_head). Templates with {K}/{K_schema}
        # placeholders rely on these; legacy cfgs without num_criteria default to 3.
        self.K = int(cfg.get("data", {}).get("num_criteria", 3))
        self.K_schema = ", ".join(f'"<criterion {i+1}>"' for i in range(self.K))

        # Gradient checkpointing: trade compute for memory
        if self.rl_cfg.get("gradient_checkpointing", True):
            self.pi.gradient_checkpointing_enable()
            self.pi.enable_input_require_grads()
            if self.has_mu and self.rl_cfg.get("mu_lr", 1e-5) > 0:
                self.mu.backbone.gradient_checkpointing_enable()
                self.mu.backbone.enable_input_require_grads()
            # Suppress harmless transformers warnings about use_cache + gradient checkpointing
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("transformers.models.qwen2.modeling_qwen2").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")
            print("[RL] Gradient checkpointing enabled for π" + (" and μ" if self.has_mu else ""))

        print("[RL] Dense signal: "
              f"mode={self.dense_signal_mode}, "
              f"baseline={self.dense_baseline_mode}, "
              f"bucket={self.dense_bucket_mode}, "
              f"gamma={self.dense_gamma}")
        if self.has_mu:
            print(f"[RL] μ score mode: {self.mu_score_mode}")

        pi_trainable = [p for p in self.pi.parameters() if p.requires_grad]
        print(f"[RL] π trainable params: {sum(p.numel() for p in pi_trainable)/1e6:.1f}M")
        if self.has_mu:
            mu_trainable = [p for p in self.mu.parameters() if p.requires_grad]
            print(f"[RL] μ trainable params: {sum(p.numel() for p in mu_trainable)/1e6:.1f}M")

        self.pi_optim = torch.optim.AdamW(
            pi_trainable,
            lr=self.sparse_pi_lr,
            weight_decay=self.rl_cfg.get("weight_decay", 0.01),
        )
        self.mu_optim = None
        self.mu_score_optim = None
        if self.has_mu and mu_trainable:
            # Separate optimizer for score_head (needs higher lr)
            score_head_params = list(self.mu.score_head.parameters())
            backbone_params = [p for p in mu_trainable
                               if not any(p is sp for sp in score_head_params)]
            # Phase-aware mu / score_head lr. New keys win; legacy mu_lr/score_head_lr
            # serve as the default for both phases when phase-specific keys are absent.
            mu_lr_legacy = self.rl_cfg["mu_lr"]
            score_head_lr_legacy = self.rl_cfg.get("score_head_lr", mu_lr_legacy)
            self.sparse_mu_lr = self.rl_cfg.get("sparse_mu_lr", mu_lr_legacy)
            self.dense_mu_lr = self.rl_cfg.get("dense_mu_lr", mu_lr_legacy)
            self.sparse_score_head_lr = self.rl_cfg.get(
                "sparse_score_head_lr", score_head_lr_legacy)
            self.dense_score_head_lr = self.rl_cfg.get(
                "dense_score_head_lr", score_head_lr_legacy)
            if backbone_params:
                self.mu_optim = torch.optim.AdamW(
                    backbone_params, lr=self.sparse_mu_lr,
                    weight_decay=self.rl_cfg.get("weight_decay", 0.01),
                )
            if score_head_params:
                self.mu_score_optim = torch.optim.AdamW(
                    score_head_params, lr=self.sparse_score_head_lr,
                    weight_decay=self.rl_cfg.get("weight_decay", 0.01),
                )
            print(f"[RL] π sparse lr={self.sparse_pi_lr}, "
                  f"π dense lr={self.dense_pi_lr}, "
                  f"μ sparse lr={self.sparse_mu_lr}, μ dense lr={self.dense_mu_lr}, "
                  f"score_head sparse lr={self.sparse_score_head_lr}, "
                  f"score_head dense lr={self.dense_score_head_lr}")

        # ── vLLM rollout engine (optional) ────────────────────────────────
        self.vllm_cfg = vllm_cfg
        self._vllm_engine = None
        self._vllm_lora_dir = None
        if vllm_cfg and vllm_cfg.get("enabled", False) and not self.has_mu:
            raise ValueError("vLLM rollout with enable_mu=false is not supported")
        if vllm_cfg and vllm_cfg.get("enabled", False):
            from .vllm_engine import VllmRolloutEngine
            # Auto-detect max LoRA rank from saved SFT adapters
            max_lora_rank = 64  # safe default
            for adapter_key in ("pi_sft_adapter", "mu_sft_adapter"):
                adapter_path = cfg["model"].get(adapter_key)
                if adapter_path:
                    ac_file = Path(adapter_path) / "adapter_config.json"
                    if ac_file.exists():
                        ac = json.load(open(ac_file))
                        max_lora_rank = max(max_lora_rank, ac.get("r", 0))
            # Cross-family detection: if π and μ have different base models,
            # vLLM cannot host both. Auto-fall-back to pi-only mode (μ runs on HF).
            pi_base = cfg["model"].get("pi_base_model_path") or cfg["model"].get("base_model_path")
            mu_base = cfg["model"].get("mu_base_model_path") or pi_base
            pi_only_mode = (pi_base != mu_base) and self.has_mu
            if pi_only_mode:
                print(f"[RL] WARNING: π and μ have different base models "
                      f"({pi_base} vs {mu_base}). vLLM will host π only; "
                      f"μ will use HF generate (slower but supports cross-family).")
            self._vllm_engine = VllmRolloutEngine(
                base_model_path=pi_base,
                max_model_len=vllm_cfg.get("max_model_len", 4096),
                gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.85),
                max_lora_rank=max_lora_rank,
                device=vllm_cfg.get("device", "cuda:0"),
                pi_only=pi_only_mode,
            )
            self._vllm_lora_dir = Path(tempfile.mkdtemp(prefix="vllm_lora_"))
            self._vllm_lora_version = -1
            print(f"[RL] vLLM rollout engine enabled")

    # ── prompt building ───────────────────────────────────────────────────────

    def _build_pi_text(self, state: BaseState, searches_remaining: int) -> str:
        user = self.pi_user_tmpl.format(**policy_fields_from_state(state, self.dataset, searches_remaining))
        return self.pi_tok.apply_chat_template(
            [{"role": "system", "content": self.pi_system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    def _build_pi_finish_text(self, state: BaseState) -> str:
        user = self.pi_finish_tmpl.format(**policy_fields_from_state(state, self.dataset, 0))
        return self.pi_tok.apply_chat_template(
            [{"role": "system", "content": self.pi_system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    def _build_mu_text(self, state: BaseState, action: str) -> str:
        if not self.has_mu:
            raise RuntimeError("μ is disabled")
        searches_done = sum(
            self.spec.step_cost(h.get("action", ""))
            for h in getattr(state, "action_history", [])
        )
        searches_remaining = max(self.max_steps - searches_done, 0)
        user = self.mu_user_tmpl.format(
            **rubric_fields_from_state_action(
                state, action, self.dataset, searches_remaining=searches_remaining),
            K=self.K, K_schema=self.K_schema)
        return self.mu_tok.apply_chat_template(
            [{"role": "system", "content": self.mu_system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    def _build_mu_trace_payload(
        self,
        example: dict,
        state: BaseState,
        action: str,
        step_idx: int,
        searches_remaining: int,
        epoch: int | None = None,
        split: str | None = None,
    ) -> dict:
        fields = rubric_fields_from_state_action(
            state,
            action,
            self.dataset,
            searches_remaining=searches_remaining,
        )
        mu_prompt_text = ""
        if self.has_mu:
            user = self.mu_user_tmpl.format(**fields, K=self.K, K_schema=self.K_schema)
            mu_prompt_text = self.mu_tok.apply_chat_template(
                [{"role": "system", "content": self.mu_system},
                 {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        phase = None
        if epoch is not None:
            phase = "dense" if (epoch - 1) >= self.dense_start else "sparse"
        return {
            "example_id": example["_id"],
            "question": example.get("question", ""),
            "gold_answer": example.get("answer", example.get("gold_answer", "")),
            "step_idx": step_idx,
            "action": action,
            "observation": "",
            "history": fields["history"],
            "source_text": (
                f"Question: {fields['question']}\n"
                f"History:\n{fields['history']}\n"
                f"Action: {fields['action']}"
            ),
            "mu_prompt_text": mu_prompt_text,
            "searches_remaining_before": searches_remaining,
            "forced_finish": searches_remaining <= 0,
            "valid_parse": True,
            "split": split,
            "epoch": epoch,
            "phase": phase,
        }

    def _generate_mu_hf_batches(self, mu_texts: list[str], mu_max_new: int):
        mu_generate_batch_size = self.rl_cfg.get("mu_generate_batch_size", len(mu_texts))
        mu_generate_batch_size = max(1, int(mu_generate_batch_size or len(mu_texts)))
        mu_prompt_ids_list = []
        mu_resp_ids_list = []
        mu_resp_texts = []
        full_seqs = []

        for start in range(0, len(mu_texts), mu_generate_batch_size):
            batch_texts = mu_texts[start:start + mu_generate_batch_size]
            mu_enc = self.mu_tok(
                batch_texts, return_tensors="pt", padding=True
            ).to(self.mu_device)
            mu_padded_plen = mu_enc.input_ids.shape[1]

            mu_out = self.mu.generate(
                **mu_enc,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=mu_max_new,
                pad_token_id=self.mu_tok.pad_token_id,
            )

            for j in range(len(batch_texts)):
                resp_ids = mu_out[j][mu_padded_plen:]
                mu_resp_ids_list.append(resp_ids)
                mu_resp_texts.append(
                    self.mu_tok.decode(resp_ids, skip_special_tokens=True)
                )
                real_mask = mu_enc.attention_mask[j].bool()
                prompt_ids = mu_enc.input_ids[j][real_mask]
                mu_prompt_ids_list.append(prompt_ids)
                full_seqs.append(torch.cat([prompt_ids, resp_ids], dim=0))

            del mu_enc, mu_out
            if self.rl_cfg.get("empty_cache_after_batch", False):
                torch.cuda.empty_cache()

        return mu_prompt_ids_list, mu_resp_ids_list, mu_resp_texts, full_seqs

    @torch.no_grad()
    def _score_eval_trace_steps(self, step_records: list[dict]) -> None:
        if not step_records:
            return
        if not self.has_mu:
            for row in step_records:
                row["criteria_text"] = ""
                row["scores"] = [0.0, 0.0, 0.0]
                row["score"] = 0.0
            return

        mu_texts = [row.get("mu_prompt_text", "") for row in step_records]
        score_batch = None
        criteria_texts: list[str] = []

        if self.mu_score_mode == "direct":
            score_batch = self._score_direct_mu_prompts(mu_texts)
            criteria_texts = [""] * len(mu_texts)
        else:
            mu_max_new = self.rl_cfg.get("mu_max_new_tokens", 256)

            if self.static_criteria_text is not None:
                criteria_texts = [self.static_criteria_text] * len(mu_texts)
                full_seqs = []
                static_resp_ids = self.mu_tok.encode(
                    self.static_criteria_text,
                    add_special_tokens=False,
                )
                static_resp_ids_t = torch.tensor(static_resp_ids, dtype=torch.long)
                for text in mu_texts:
                    prompt_enc = self.mu_tok(
                        text,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    prompt_ids = prompt_enc.input_ids.squeeze(0)
                    full_seqs.append(torch.cat([prompt_ids, static_resp_ids_t], dim=0))
            else:
                use_vllm_mu = (
                    self._vllm_engine is not None
                    and self._vllm_engine.llm is not None
                    and self._vllm_engine.serves_mu
                )
                if use_vllm_mu:
                    mu_vllm_results = self._vllm_engine.generate_mu(
                        mu_texts,
                        max_tokens=mu_max_new,
                    )
                    full_seqs = []
                    for vr in mu_vllm_results:
                        criteria_texts.append(vr.resp_text)
                        prompt_ids_t = torch.tensor(vr.prompt_token_ids, dtype=torch.long)
                        resp_ids_t = torch.tensor(vr.resp_token_ids, dtype=torch.long)
                        full_seqs.append(torch.cat([prompt_ids_t, resp_ids_t], dim=0))
                else:
                    _, _, criteria_texts, full_seqs = self._generate_mu_hf_batches(
                        mu_texts, mu_max_new
                    )

            score_mini_batch = self.rl_cfg.get("score_mini_batch", 16)
            all_scores = []
            for mb_start in range(0, len(full_seqs), score_mini_batch):
                mb_seqs = full_seqs[mb_start:mb_start + score_mini_batch]
                max_len = max(seq.size(0) for seq in mb_seqs)
                pad_id = self.mu_tok.pad_token_id or 0
                padded_ids = torch.full(
                    (len(mb_seqs), max_len),
                    pad_id,
                    dtype=torch.long,
                    device=self.mu_device,
                )
                score_attn_mask = torch.zeros(
                    (len(mb_seqs), max_len),
                    dtype=torch.long,
                    device=self.mu_device,
                )
                for j, seq in enumerate(mb_seqs):
                    padded_ids[j, max_len - seq.size(0):] = seq.to(self.mu_device)
                    score_attn_mask[j, max_len - seq.size(0):] = 1
                mb_scores, _ = self.mu.forward_score(
                    padded_ids,
                    attention_mask=score_attn_mask,
                    score_only=True,
                )
                all_scores.append(mb_scores)
                del padded_ids, score_attn_mask
                if self.rl_cfg.get("empty_cache_after_rollout", False):
                    torch.cuda.empty_cache()
            score_batch = torch.cat(all_scores, dim=0)

        for row, scores_tensor, criteria_text in zip(step_records, score_batch, criteria_texts):
            step_scores = scores_tensor.tolist()
            row["criteria_text"] = criteria_text
            row["scores"] = step_scores
            row["score"] = sum(step_scores) / len(step_scores) if step_scores else 0.0

    def _score_direct_mu_prompts(self, mu_texts: list[str]) -> torch.Tensor:
        """Direct score-only μ ablation: prompt -> score without rubric generation."""
        if not mu_texts:
            return torch.empty(0, 1, device=self.mu_device)
        score_mini_batch = self.rl_cfg.get("score_mini_batch", 16)
        all_scores = []
        for start in range(0, len(mu_texts), score_mini_batch):
            batch_texts = mu_texts[start:start + score_mini_batch]
            enc = self.mu_tok(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.mu_device)
            mb_scores, _ = self.mu.forward_score(
                enc.input_ids,
                attention_mask=enc.attention_mask,
                score_only=True,
            )
            all_scores.append(mb_scores)
            del enc
            if self.rl_cfg.get("empty_cache_after_rollout", False):
                torch.cuda.empty_cache()
        return torch.cat(all_scores, dim=0)

    def _traj_had_search(self, steps: list[dict]) -> bool:
        return any(str(step.get("action", "")).startswith("Search") for step in steps)

    def _traj_has_format_error(self, steps: list[dict]) -> bool:
        for step in steps:
            if not step.get("valid_parse", True):
                return True
            obs = str(step.get("observation", ""))
            if obs.startswith("Forced finish violation:"):
                return True
        return False

    def _apply_reward_mode(self, reward_info: dict, steps: list[dict]) -> float:
        """Compute trajectory reward from environment feedback.

        Default: pass through EM as the reward. Baseline subclasses override
        this to apply reward shaping (search bonuses, format penalties, etc).
        """
        return float(reward_info["reward"])

    # ── baseline hooks ─────────────────────────────────────────────────────
    # Default implementations are no-ops. Baseline subclasses override these
    # to attach extra components (GPT clients, rubric buffers), pre-train
    # phases (offline rubric generation), or post-rollout reward overrides.

    def _init_baseline_components(self, cfg: dict) -> None:
        """Hook called at end of __init__. Override to attach baseline state."""
        return

    def _pre_train_setup(self, examples: list[dict]) -> None:
        """Hook called at start of train(). Override for offline phases."""
        return

    def _post_rollout_reward_override(self, results: list[dict]) -> None:
        """Hook called at end of _rollout_batch. Override to mutate
        results[i]["reward"] in batch (e.g. GPT-judged rewards)."""
        return

    def _post_rollout_finalize(self, results: list[dict]) -> None:
        """Hook called after rewards are finalized. Override to tag
        steps or rewrite per-step fields (e.g. terminal-split)."""
        return

    # ── vLLM phase transitions ─────────────────────────────────────────────

    def _save_lora_for_vllm(self):
        """Save current π and μ LoRA adapters to uniquely-named dirs for vLLM.

        Uses an incrementing counter so vLLM doesn't serve stale cached weights.
        Under shared-base / multi-adapter PEFT, only the active adapter is
        written (vLLM rejects safetensors that contain weights for adapters
        other than the one it's loading). In pi_only mode, μ is not served by
        vLLM, so its adapter is not written.
        """
        self._vllm_lora_version += 1
        pi_only = self._vllm_engine is not None and self._vllm_engine.pi_only
        pi_root = self._vllm_lora_dir / f"pi_v{self._vllm_lora_version}"

        def _active_name(m):
            a = getattr(m, "active_adapter", None) or getattr(m, "active_adapters", None)
            if isinstance(a, (list, tuple)):
                a = a[0] if a else None
            return a or "default"
        pi_active = _active_name(self.pi)
        self.pi.save_pretrained(pi_root, selected_adapters=[pi_active])
        pi_path = pi_root / pi_active if (pi_root / pi_active).exists() else pi_root
        self._strip_adapter_suffixed_keys(pi_path / "adapter_model.safetensors")

        mu_path_str = None
        if not pi_only:
            mu_root = self._vllm_lora_dir / f"mu_v{self._vllm_lora_version}"
            mu_backbone = self.mu.backbone if hasattr(self.mu, "backbone") else self.mu
            mu_active = _active_name(mu_backbone)
            mu_backbone.save_pretrained(mu_root, selected_adapters=[mu_active])
            mu_path = mu_root / mu_active if (mu_root / mu_active).exists() else mu_root
            self._strip_adapter_suffixed_keys(mu_path / "adapter_model.safetensors")
            mu_path_str = str(mu_path)

        # Clean up old version to save disk
        old = self._vllm_lora_version - 1
        if old >= 0:
            import shutil
            for prefix in ("pi_v", "mu_v"):
                old_path = self._vllm_lora_dir / f"{prefix}{old}"
                if old_path.exists():
                    shutil.rmtree(old_path, ignore_errors=True)
        return str(pi_path), mu_path_str

    @staticmethod
    def _strip_adapter_suffixed_keys(safetensors_path):
        """Rewrite a PEFT adapter_model.safetensors to drop keys of the form
        `...lora_A.{adapter_name}.weight`, keeping only the canonical
        `...lora_A.weight` keys that vLLM expects."""
        import re
        from pathlib import Path
        from safetensors import safe_open
        from safetensors.torch import save_file

        if not Path(safetensors_path).exists():
            return
        kept = {}
        # `lora_A.weight` / `lora_B.weight` are kept;
        # `lora_A.<name>.weight` / `lora_B.<name>.weight` are dropped.
        suffixed = re.compile(r"\.lora_[AB]\.[^.]+\.weight$")
        with safe_open(str(safetensors_path), framework="pt") as f:
            for k in f.keys():
                if suffixed.search(k):
                    continue
                kept[k] = f.get_tensor(k)
        save_file(kept, str(safetensors_path))

    def _start_vllm_epoch(self):
        """Start vLLM engine once at the beginning of an epoch.

        Idempotent: if vLLM is already running (persistent across epochs),
        only refresh the LoRA adapter paths. Frequent start/stop fragments
        GPU 0 (each cycle leaks ~5-10 GB), so we keep the engine alive for
        the whole training run.
        """
        if self._vllm_engine is None:
            return
        # Already running? Just push fresh LoRA weights.
        if self._vllm_engine.llm is not None:
            self._update_vllm_lora()
            return
        # Save initial LoRA weights
        pi_lora_path, mu_lora_path = self._save_lora_for_vllm()
        # Offload models not needed during rollout to CPU.
        # In pi_only mode, μ runs on HF during rollout, so μ/μ_ref stay on GPU.
        self.pi.to("cpu")
        if self.pi_ref is not None:
            self.pi_ref.to("cpu")
        if self.mu_ref is not None and self._vllm_engine.serves_mu:
            self.mu_ref.to("cpu")
        torch.cuda.empty_cache()
        self._vllm_engine.start(pi_lora_path=pi_lora_path,
                                mu_lora_path=mu_lora_path)

    def _stop_vllm_epoch(self):
        """End-of-epoch hook. Keep vLLM alive across epochs to avoid
        fragmentation; only restore HF models that were offloaded for rollout.
        """
        if self._vllm_engine is None:
            return
        # Don't actually stop vLLM — restore HF models for training/save/eval
        self.pi.to(self.pi_device)
        if self.pi_ref is not None:
            self.pi_ref.to(self.pi_ref_device)
        if self.mu_ref is not None and self._vllm_engine.serves_mu:
            self.mu_ref.to(self.mu_device)
        torch.cuda.empty_cache()

    def _update_vllm_lora(self):
        """Save updated LoRA weights and update engine paths (no restart)."""
        if self._vllm_engine is None:
            return
        pi_lora_path, mu_lora_path = self._save_lora_for_vllm()
        self._vllm_engine.update_lora(pi_lora_path=pi_lora_path,
                                      mu_lora_path=mu_lora_path)

    def _enter_rollout_phase(self):
        """Prepare for rollout — offload HF models if vLLM is persistent."""
        if self._vllm_engine is None:
            return
        # If vLLM is already running (persistent mode), just offload HF models
        if self._vllm_engine.llm is not None:
            self.pi.to("cpu")
            if self.pi_ref is not None:
                self.pi_ref.to("cpu")
            if self.mu_ref is not None:
                self.mu_ref.to("cpu")
            torch.cuda.empty_cache()
            return
        # Fallback: start fresh (shouldn't normally happen)
        self._start_vllm_epoch()

    def _exit_rollout_phase(self):
        """After rollout — restore HF models to GPU (vLLM stays alive)."""
        if self._vllm_engine is None:
            return
        # Don't stop vLLM — just restore HF models for training
        self.pi.to(self.pi_device)
        if self.pi_ref is not None:
            self.pi_ref.to(self.pi_ref_device)
        if self.mu_ref is not None:
            self.mu_ref.to(self.mu_device)

    # ── batched rollout ───────────────────────────────────────────────────────

    # ── rollout ───────────────────────────────────────────────────────────────

    @staticmethod
    def _inner_hf_model(model):
        """Traverse PEFT / DualHead wrappers to reach the innermost HF model.

        Layer order:  DualHeadMu  →  PeftModel  →  LoraModel  →  HF model
        """
        if hasattr(model, "backbone"):    # DualHeadMu
            model = model.backbone
        if hasattr(model, "base_model"):  # PeftModel → LoraModel
            model = model.base_model
        if hasattr(model, "model"):       # LoraModel → Qwen2ForCausalLM
            model = model.model
        return model

    def _should_score_mu_step(
        self,
        batch_index: int,
        action: str,
        forced_finish: bool,
        searches_remaining_before: int | None,
    ) -> bool:
        """Hook for subclasses that skip μ on selected rollout steps."""
        return True

    @torch.no_grad()
    def _rollout_batch(self, examples: list[dict], skip_mu: bool = False) -> list[dict]:
        """Roll out all examples in parallel.

        At each step, all active envs are batched together for a single
        π generate call and a single μ generate call.
        skip_mu=True skips μ scoring (used during evaluation).
        """
        # Switch to eval mode for generation (disables dropout, fixes
        # gradient-checkpointing / use_cache conflicts).
        self.pi.eval()
        if self.has_mu:
            self.mu.eval()

        # Rollout is under @torch.no_grad() — no computation graph, no GC needed.
        _debug_logged = False  # print first parse/step failure for diagnosis

        n = len(examples)
        envs = [self.spec.make_env(ex, retriever=self.retriever, max_steps=self.max_steps)
                for ex in examples]
        active = [True] * n          # False when env done or parse fails
        searches_done = [0] * n      # step budget used per example
        all_steps: list[list] = [[] for _ in range(n)]

        # Left-padding required for batched generation with decoder-only models
        self.pi_tok.padding_side = "left"
        if self.has_mu:
            self.mu_tok.padding_side = "left"

        while True:
            active_idx = [i for i in range(n) if active[i] and not envs[i].done]
            if not active_idx:
                break

            # Partition: normal prompt vs forced-finish prompt
            finish_idx = [i for i in active_idx if searches_done[i] >= self.max_steps]

            # ── π: build texts (mix of normal and finish prompts) ─────────
            pre_states = {i: envs[i].state for i in active_idx}
            searches_remaining_before = {
                i: max(self.max_steps - searches_done[i], 0)
                for i in active_idx
            }
            pi_texts = []
            for i in active_idx:
                if self.spec.has_forced_finish and searches_done[i] >= self.max_steps:
                    pi_texts.append(self._build_pi_finish_text(pre_states[i]))
                else:
                    pi_texts.append(self._build_pi_text(pre_states[i],
                                                        self.max_steps - searches_done[i]))

            # ── π generate ──────────────────────────────────────────────────
            greedy = (self.rollout_temperature <= 0.0)
            pi_max_new = self.rl_cfg.get("pi_max_new_tokens", 256)

            if self._vllm_engine is not None and self._vllm_engine.llm is not None:
                # vLLM path
                vllm_results = self._vllm_engine.generate_pi(
                    pi_texts,
                    temperature=0.0 if greedy else self.rollout_temperature,
                    max_tokens=pi_max_new,
                )
                # Build pi_ids_store and parse actions
                actions: dict[int, str] = {}
                pi_ids_store: dict[int, dict] = {}
                invalid_idx: set[int] = set()
                for j, i in enumerate(active_idx):
                    vr = vllm_results[j]
                    pi_ids_store[i] = {
                        "prompt": vr.prompt_token_ids,
                        "resp":   vr.resp_token_ids,
                    }
                    resp_text = vr.resp_text
                    action_text = resp_text.strip() or "<EMPTY>"
                    try:
                        _, action = self.spec.parse_output(resp_text)
                    except ValueError:
                        if self.debug_rollout and not _debug_logged:
                            print(f"\n[DEBUG rollout] π parse fail — resp_text={resp_text[:300]!r}")
                            _debug_logged = True
                        actions[i] = action_text
                        invalid_idx.add(i)
                        continue
                    actions[i] = action
            else:
                # HF generate path (fallback)
                pi_enc = self.pi_tok(
                    pi_texts, return_tensors="pt", padding=True
                ).to(self.pi_device)
                padded_plen = pi_enc.input_ids.shape[1]

                pi_out = self.pi.generate(
                    **pi_enc,
                    do_sample=not greedy,
                    temperature=None if greedy else self.rollout_temperature,
                    top_p=None,
                    top_k=None,
                    max_new_tokens=pi_max_new,
                    pad_token_id=self.pi_tok.pad_token_id,
                )

                # ── parse responses ─────────────────────────────────────────
                actions: dict[int, str] = {}
                pi_ids_store: dict[int, dict] = {}
                invalid_idx: set[int] = set()

                for j, i in enumerate(active_idx):
                    resp_ids = pi_out[j][padded_plen:]
                    pad_id = self.pi_tok.pad_token_id
                    if pad_id is not None:
                        real_len = resp_ids.shape[0]
                        while real_len > 0 and resp_ids[real_len - 1].item() == pad_id:
                            real_len -= 1
                        resp_ids = resp_ids[:real_len]
                    resp_text = self.pi_tok.decode(resp_ids, skip_special_tokens=True)
                    action_text = resp_text.strip() or "<EMPTY>"
                    real_mask = pi_enc.attention_mask[j].bool()
                    pi_ids_store[i] = {
                        "prompt": pi_enc.input_ids[j][real_mask].tolist(),
                        "resp":   resp_ids.tolist(),
                    }
                    try:
                        _, action = self.spec.parse_output(resp_text)
                    except ValueError:
                        if self.debug_rollout and not _debug_logged:
                            print(f"\n[DEBUG rollout] π parse fail — resp_text={resp_text[:300]!r}")
                            _debug_logged = True
                        actions[i] = action_text
                        invalid_idx.add(i)
                        continue
                    actions[i] = action
                del pi_enc

            observations: dict[int, str] = {}
            valid_parse_flags: dict[int, bool] = {}
            # Handle invalid parses synchronously (no env interaction)
            valid_actions: dict[int, str] = {}
            for i in list(actions):
                if i in invalid_idx:
                    valid_parse_flags[i] = False
                    parse_err_obs = (
                        f"Invalid action: {actions[i]}. "
                        "Action parsing error: output did not match the required format. "
                        "No retrieval was executed."
                    )
                    observations[i] = parse_err_obs
                    envs[i].record_invalid_action(
                        actions[i], parse_err_obs, done=i in finish_idx)
                    if i not in finish_idx:
                        searches_done[i] += 1
                    if i in finish_idx or (
                        searches_done[i] >= self.max_steps and not self.spec.has_forced_finish
                    ):
                        active[i] = False
                elif i in finish_idx and not str(actions[i]).strip().startswith("Finish"):
                    valid_parse_flags[i] = True
                    observations[i] = (
                        f"Forced finish violation: {actions[i]}. "
                        "Search budget exhausted. No answer submitted."
                    )
                    envs[i].record_invalid_action(actions[i], observations[i], done=True)
                    active[i] = False
                else:
                    valid_parse_flags[i] = True
                    valid_actions[i] = actions[i]

            # Debug: print first trajectory's first 3 steps
            if self.debug_rollout and 0 in actions and searches_done[0] <= 3:
                admissible = getattr(envs[0].state, "admissible_commands", [])
                print(f"\n[DEBUG traj0 step={searches_done[0]}] action={actions[0]!r}")
                print(f"[DEBUG traj0] admissible={admissible[:5]}{'...' if len(admissible) > 5 else ''}")
                print(f"[DEBUG traj0] invalid={0 in invalid_idx}")
                if 0 in valid_actions:
                    in_admissible = valid_actions[0] in admissible
                    print(f"[DEBUG traj0] action_in_admissible={in_admissible}")
                    if not in_admissible and admissible:
                        print(f"[DEBUG traj0] action_repr={valid_actions[0]!r} vs first_cmd={admissible[0]!r}")

            # Step valid environments concurrently (retriever HTTP is the bottleneck)
            def _step_env(idx):
                return idx, envs[idx].step(valid_actions[idx])

            with ThreadPoolExecutor(max_workers=max(min(len(valid_actions), 32), 1)) as pool:
                futures = {pool.submit(_step_env, i): i for i in valid_actions}
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        _, obs = fut.result()
                        observations[i] = obs
                    except ValueError as e:
                        if self.debug_rollout and not _debug_logged:
                            print(f"\n[DEBUG rollout] env.step fail — action={actions[i]!r} err={e}")
                            _debug_logged = True
                        observations[i] = (
                            f"Invalid action: {actions[i]}. "
                            "Action execution error: invalid action format. "
                            "No retrieval was executed."
                        )
                        valid_parse_flags[i] = False
                        envs[i].record_invalid_action(
                            actions[i], observations[i], done=i in finish_idx)
                        if i not in finish_idx:
                            searches_done[i] += 1
                        if i in finish_idx or (
                            searches_done[i] >= self.max_steps and not self.spec.has_forced_finish
                        ):
                            active[i] = False
                        continue
                    searches_done[i] += self.spec.step_cost(actions[i])
                    if i in finish_idx or (
                        searches_done[i] >= self.max_steps and not self.spec.has_forced_finish
                    ):
                        active[i] = False

            # Debug: print first trajectory's observation after step
            if self.debug_rollout and 0 in observations and searches_done[0] <= 3:
                obs_preview = observations[0][:200]
                print(f"[DEBUG traj0] obs={obs_preview!r}")

            # ── μ: single batched generate ────────────────────────────────
            mu_candidates = [i for i in active_idx if i in observations]
            mu_active = [
                i for i in mu_candidates
                if self._should_score_mu_step(
                    i,
                    actions.get(i, ""),
                    i in finish_idx,
                    searches_remaining_before.get(i),
                )
            ]
            mu_active_set = set(mu_active)
            mu_skipped = [i for i in mu_candidates if i not in mu_active_set]

            # Some subclasses intentionally skip μ on selected steps. Keep the
            # π/action trace and use a neutral score placeholder; subclasses can
            # overwrite it later once the trajectory reward is known.
            for i in mu_skipped:
                all_steps[i].append({
                    "action":      actions[i],
                    "observation": observations[i],
                    "searches_remaining_before": searches_remaining_before.get(i),
                    "forced_finish": i in finish_idx,
                    "valid_parse": valid_parse_flags.get(i, True),
                    "pi_input_ids":    pi_ids_store[i]["prompt"],
                    "pi_response_ids": pi_ids_store[i]["resp"],
                    "mu_input_ids": [], "mu_response_ids": [],
                    "scores": [0.0, 0.0, 0.0], "score": 0.0,
                })

            if not self.has_mu or skip_mu or not mu_active:
                # no μ scoring: store step with score=0 placeholder
                for i in mu_active:
                    all_steps[i].append({
                        "action":      actions[i],
                        "observation": observations[i],
                        "searches_remaining_before": searches_remaining_before.get(i),
                        "forced_finish": i in finish_idx,
                        "valid_parse": valid_parse_flags.get(i, True),
                        "pi_input_ids":    pi_ids_store[i]["prompt"],
                        "pi_response_ids": pi_ids_store[i]["resp"],
                        "mu_input_ids": [], "mu_response_ids": [],
                        "scores": [0.0, 0.0, 0.0], "score": 0.0,
                    })
                continue

            mu_texts = [self._build_mu_text(pre_states[i], actions[i]) for i in mu_active]

            if self.mu_score_mode == "direct":
                score_batch = self._score_direct_mu_prompts(mu_texts)
                mu_prompt_ids_list = []
                mu_resp_ids_list = []
                mu_resp_texts = []
                for text in mu_texts:
                    prompt_ids = self.mu_tok.encode(text, add_special_tokens=False)
                    mu_prompt_ids_list.append(torch.tensor(prompt_ids, dtype=torch.long))
                    mu_resp_ids_list.append(torch.empty(0, dtype=torch.long))
                    mu_resp_texts.append("")
            else:
                mu_max_new = self.rl_cfg.get("mu_max_new_tokens", 256)

                # ── static criteria: skip generate, use fixed response text ──
                if self.static_criteria_text is not None:
                    mu_resp_ids_list = []
                    mu_resp_texts = []
                    mu_prompt_ids_list = []
                    full_seqs = []
                    static_resp_ids = self.mu_tok.encode(
                        self.static_criteria_text, add_special_tokens=False)
                    static_resp_ids_t = torch.tensor(static_resp_ids, dtype=torch.long)
                    for j, i in enumerate(mu_active):
                        prompt_enc = self.mu_tok(
                            mu_texts[j], return_tensors="pt", add_special_tokens=False)
                        prompt_ids = prompt_enc.input_ids.squeeze(0)
                        mu_prompt_ids_list.append(prompt_ids)
                        mu_resp_ids_list.append(static_resp_ids_t)
                        mu_resp_texts.append(self.static_criteria_text)
                        full_seqs.append(torch.cat([prompt_ids, static_resp_ids_t], dim=0))

                elif (self._vllm_engine is not None
                      and self._vllm_engine.llm is not None
                      and self._vllm_engine.serves_mu):
                    # vLLM path for μ generate
                    mu_vllm_results = self._vllm_engine.generate_mu(
                        mu_texts, max_tokens=mu_max_new)
                    mu_resp_ids_list = []
                    mu_resp_texts = []
                    mu_prompt_ids_list = []
                    full_seqs = []
                    for j, i in enumerate(mu_active):
                        vr = mu_vllm_results[j]
                        prompt_ids = torch.tensor(vr.prompt_token_ids, dtype=torch.long)
                        resp_ids = torch.tensor(vr.resp_token_ids, dtype=torch.long)
                        mu_prompt_ids_list.append(prompt_ids)
                        mu_resp_ids_list.append(resp_ids)
                        mu_resp_texts.append(vr.resp_text)
                        full_seqs.append(torch.cat([prompt_ids, resp_ids], dim=0))
                else:
                    # HF generate path (fallback)
                    (
                        mu_prompt_ids_list,
                        mu_resp_ids_list,
                        mu_resp_texts,
                        full_seqs,
                    ) = self._generate_mu_hf_batches(mu_texts, mu_max_new)

                # ── batched forward_score (shared by rubric paths) ───────────
                score_mini_batch = self.rl_cfg.get("score_mini_batch", 16)
                all_scores = []
                for mb_start in range(0, len(full_seqs), score_mini_batch):
                    mb_seqs = full_seqs[mb_start:mb_start + score_mini_batch]
                    max_len = max(s.size(0) for s in mb_seqs)
                    pad_id = self.mu_tok.pad_token_id or 0
                    padded_ids = torch.full(
                        (len(mb_seqs), max_len), pad_id,
                        dtype=torch.long, device=self.mu_device)
                    score_attn_mask = torch.zeros(
                        (len(mb_seqs), max_len),
                        dtype=torch.long, device=self.mu_device)
                    for j, seq in enumerate(mb_seqs):
                        padded_ids[j, max_len - seq.size(0):] = seq.to(self.mu_device)
                        score_attn_mask[j, max_len - seq.size(0):] = 1
                    mb_scores, _ = self.mu.forward_score(
                        padded_ids, attention_mask=score_attn_mask,
                        score_only=True)
                    all_scores.append(mb_scores)
                    del padded_ids, score_attn_mask
                    if self.rl_cfg.get("empty_cache_after_rollout", False):
                        torch.cuda.empty_cache()
                score_batch = torch.cat(all_scores, dim=0)

            for j, i in enumerate(mu_active):
                step_scores = score_batch[j].tolist()  # list of 3 floats
                all_steps[i].append({
                    "action":         actions[i],
                    "observation":    observations[i],
                    "searches_remaining_before": searches_remaining_before.get(i),
                    "forced_finish":   i in finish_idx,
                    "valid_parse":     valid_parse_flags.get(i, True),
                    "pi_input_ids":   pi_ids_store[i]["prompt"],
                    "pi_response_ids": pi_ids_store[i]["resp"],
                    "mu_input_ids":   mu_prompt_ids_list[j].tolist(),
                    "mu_response_ids": mu_resp_ids_list[j].tolist(),
                    "scores":         step_scores,
                    "score":          sum(step_scores) / len(step_scores),  # mean for π dense signal
                    "criteria_text":  mu_resp_texts[j],
                })

        # Restore padding side
        self.pi_tok.padding_side = "right"
        if self.has_mu:
            self.mu_tok.padding_side = "right"
        if self.rl_cfg.get("empty_cache_after_rollout", False):
            torch.cuda.empty_cache()

        # Restore train mode for gradient updates
        self.pi.train()
        if self.has_mu:
            self.mu.train()

        results = []
        for i, example in enumerate(examples):
            reward_info = self.spec.compute_reward(envs[i], example)
            reward = self._apply_reward_mode(reward_info, all_steps[i])
            if self.rubric_outcome_reward and all_steps[i]:
                step_scores = [s.get("score", 0.0) for s in all_steps[i]]
                rubric_reward = sum(step_scores) / len(step_scores)
                # Normalize to [0, 1] from [-1, 1]
                reward = max(0.0, min(1.0, (rubric_reward + 1.0) / 2.0))
            results.append({
                "example_id": example["_id"],
                "reward":     reward,
                "steps":      all_steps[i],
                "_question":  example.get("question", ""),
                "_answer":    example.get("answer", example.get("gold_answer", "")),
            })

        # Hook for subclasses to override rewards in batch. Default: no-op.
        self._post_rollout_reward_override(results)

        # Hook for variants that need to post-process trajectories after
        # rewards are finalized (e.g. terminal-split tags step roles and
        # rewrites the last-step μ payload). Default: no-op.
        self._post_rollout_finalize(results)

        return results

    # ── π loss ────────────────────────────────────────────────────────────────

    def _dense_step_buckets(self, traj: dict) -> list[str]:
        """Return per-step bucket labels for dense baselines."""
        steps = traj["steps"]
        if not steps:
            return []

        mode = self.dense_bucket_mode
        if mode == "global":
            return ["global"] * len(steps)
        if mode == "step_index":
            return [f"step_{idx + 1}" for idx in range(len(steps))]
        if mode != "search_terminal":
            raise ValueError(f"Unknown dense_bucket_mode: {mode}")

        buckets = []
        search_rank = 0
        last_idx = len(steps) - 1
        for idx, step in enumerate(steps):
            action = str(step.get("action", "")).strip()
            is_search = action.startswith("Search")
            is_finish = action.startswith("Finish")

            # Treat the final unfinished step as terminal so failed trajectories
            # are compared against other terminal decisions instead of searches.
            if idx == last_idx and not is_finish:
                buckets.append("terminal")
                continue
            if is_search:
                search_rank += 1
                buckets.append(f"search_{search_rank}")
            else:
                buckets.append("terminal")
        return buckets

    def _dense_raw_signals(self, traj: dict) -> list[float]:
        """Return raw per-step dense returns before subtracting baselines."""
        steps = traj["steps"]
        if not steps:
            return []

        scores = [float(step.get("score", 0.0)) for step in steps]
        mode = self.dense_signal_mode

        if mode == "sum_return":
            returns = []
            running = 0.0
            for score in reversed(scores):
                running += score
                returns.append(running)
            returns.reverse()
            return returns

        if mode == "avg_return":
            returns = []
            running = 0.0
            count = 0
            for score in reversed(scores):
                running += score
                count += 1
                returns.append(running / count)
            returns.reverse()
            return returns

        if mode == "discounted_return":
            returns = []
            running = 0.0
            gamma = float(self.dense_gamma)
            for score in reversed(scores):
                running = score + gamma * running
                returns.append(running)
            returns.reverse()
            return returns

        raise ValueError(f"Unknown dense_signal_mode: {mode}")

    def _dense_advantages_batch(self, batch: list[dict]) -> list[list[float]]:
        """Return per-step dense learning signals for π for a whole batch."""
        raw_signals = [self._dense_raw_signals(traj) for traj in batch]
        scale = self.rl_cfg.get("dense_reward_scale", 1.0)
        baseline_mode = self.dense_baseline_mode

        all_values = [value for values in raw_signals for value in values]
        global_baseline = (sum(all_values) / len(all_values)) if all_values else 0.0

        if baseline_mode == "global_batch":
            return [[(value - global_baseline) * scale for value in values]
                    for values in raw_signals]

        if baseline_mode != "position_bucket":
            raise ValueError(f"Unknown dense_baseline_mode: {baseline_mode}")

        from collections import defaultdict

        bucket_sums = defaultdict(float)
        bucket_counts = defaultdict(int)
        bucket_labels = [self._dense_step_buckets(traj) for traj in batch]

        for labels, values in zip(bucket_labels, raw_signals):
            for label, value in zip(labels, values):
                bucket_sums[label] += value
                bucket_counts[label] += 1

        dense_advantages = []
        for labels, values in zip(bucket_labels, raw_signals):
            traj_advantages = []
            for label, value in zip(labels, values):
                count = bucket_counts[label]
                if self.dense_bucket_leave_one_out and count > 1:
                    baseline = (bucket_sums[label] - value) / (count - 1)
                elif count > 0:
                    baseline = bucket_sums[label] / count
                    if count == 1 and self.dense_bucket_fallback_to_global:
                        baseline = global_baseline
                else:
                    baseline = global_baseline
                traj_advantages.append((value - baseline) * scale)
            dense_advantages.append(traj_advantages)
        return dense_advantages

    def _pi_loss_traj(self, traj: dict, use_dense: bool,
                      reward_baseline: float = 0.0,
                      dense_signals: list[float] | None = None) -> torch.Tensor:
        """Policy gradient loss for ONE trajectory (caller accumulates grads).

        Each step is backward-ed independently to avoid accumulating activation
        memory across the entire trajectory.  Returns a detached scalar for
        logging only — gradients are already accumulated in .grad buffers.
        reward_baseline: batch-mean reward, subtracted in sparse mode so that
                         below-average trajectories receive negative signal.
        """
        if not traj["steps"]:
            return torch.zeros(1, requires_grad=True, device=self.pi_device)
        if self.rl_cfg.get("binary_reward", False):
            R = (2.0 * traj["reward"] - 1.0) - reward_baseline
        else:
            R = traj["reward"] - reward_baseline
        default_beta = self.rl_cfg.get("pi_kl_coef", 0.0)
        beta = self.rl_cfg.get(
            "dense_pi_kl_coef" if use_dense else "sparse_pi_kl_coef",
            default_beta,
        )
        total_val = 0.0  # detached accumulator for logging
        valid_steps = [(idx, step) for idx, step in enumerate(traj["steps"])
                       if step.get("valid", True)]
        n_valid = len(valid_steps)
        if n_valid == 0:
            return torch.zeros(1, requires_grad=True, device=self.pi_device)
        scale = 1.0 / n_valid

        # ── REINFORCE mode (original) ────────────────────────────────────
        if use_dense and dense_signals is None:
            dense_signals = self._dense_advantages_batch([traj])[0]
        for idx, step in valid_steps:
            signal = dense_signals[idx] if use_dense else R
            ids = torch.tensor(
                step["pi_input_ids"] + step["pi_response_ids"],
                dtype=torch.long).unsqueeze(0).to(self.pi_device)
            plen = len(step["pi_input_ids"])
            n_resp = ids.shape[1] - plen
            if n_resp <= 0:
                continue
            out = self.pi(input_ids=ids)
            log_probs = F.log_softmax(out.logits[0], dim=-1)
            lp_sum = torch.stack([
                log_probs[i, ids[0, i + 1]]
                for i in range(plen - 1, ids.shape[1] - 1)
            ]).sum()

            pg_loss = -signal * lp_sum
            kl_loss = torch.zeros(1, device=self.pi_device)
            if self.pi_ref is not None and step["pi_response_ids"] and beta > 0:
                logits_current = out.logits[0, plen - 1:-1]
                logp_current = F.log_softmax(logits_current, dim=-1)
                with torch.no_grad():
                    ref_ids = ids.to(self.pi_ref_device)
                    ref_out = self.pi_ref(input_ids=ref_ids)
                    logits_ref = ref_out.logits[0, plen - 1:-1].to(logits_current.device)
                    logp_ref = F.log_softmax(logits_ref, dim=-1)
                kl_per_token = F.kl_div(
                    logp_current,
                    logp_ref,
                    log_target=True,
                    reduction="none",
                ).sum(dim=-1)
                kl_loss = kl_per_token.mean()
                del ref_out
            # No per-trajectory scaling — standard REINFORCE / GRPO
            step_loss = pg_loss + beta * kl_loss
            step_loss.backward()  # free activation memory immediately
            total_val += step_loss.item()
            del out, step_loss
        # Return detached scalar for logging (grads already in .grad buffers)
        return torch.tensor(total_val, device=self.pi_device)

    # ── μ loss ────────────────────────────────────────────────────────────────

    def _mu_loss_traj(self, traj: dict, score_weight: float = 1.0) -> torch.Tensor:
        """μ loss for ONE trajectory (caller accumulates grads).

        score_loss: differentiable MSE via score_head.
                    Each step produces 3 scores; the step mean is the step's
                    contribution.  Constraint: sum of step means = R.
        kl_loss:    KL(μ_current || μ_ref) on criteria tokens (prevents drift)

        All losses accumulated into a single total_loss, one backward at the end.
        Returns detached scalar for logging.
        """
        if not traj["steps"]:
            return torch.zeros(1, requires_grad=True, device=self.mu_device)
        beta = self.rl_cfg.get("mu_kl_coef", self.rl_cfg.get("mu_text_coef", 0.1))
        R = torch.tensor(traj["reward"], dtype=torch.float32, device=self.mu_device)
        step_means = []  # mean of 3 scores per step
        kl_losses = []
        for step in traj["steps"]:
            if not step.get("mu_input_ids"):
                continue
            full_ids = step["mu_input_ids"] + step.get("mu_response_ids", [])
            if not full_ids:
                continue
            ids = torch.tensor(
                full_ids,
                dtype=torch.long).unsqueeze(0).to(self.mu_device)
            plen = len(step["mu_input_ids"])

            # One forward pass: score_head + logits (logits only if KL needed)
            scores_3, out = self.mu.forward_score(
                ids, score_only=not self._mu_needs_logits)  # (1, 3)
            step_means.append(scores_3.mean(dim=-1))    # (1,) → scalar

            if self.mu_ref is not None and step.get("mu_response_ids") and beta > 0:
                logits_current = out.logits[0, plen - 1:-1]
                logp_current = F.log_softmax(logits_current, dim=-1)

                with torch.no_grad():
                    ref_out = self.mu_ref(input_ids=ids)
                    logits_ref = ref_out.logits[0, plen - 1:-1]
                    logp_ref = F.log_softmax(logits_ref, dim=-1)

                kl_per_token = F.kl_div(
                    logp_current,
                    logp_ref,
                    log_target=True,
                    reduction="none",
                ).sum(dim=-1)
                kl_losses.append(kl_per_token.mean())
                del ref_out

        if not step_means:
            return torch.zeros(1, requires_grad=True, device=self.mu_device)

        # Constraint: sum of step means = R
        sum_of_means = torch.stack(step_means).sum()
        score_loss = F.mse_loss(sum_of_means, R)
        # Dynamic class balancing: score_weight is higher for minority class
        score_loss = score_loss * score_weight
        kl_loss = sum(kl_losses) * beta if kl_losses else torch.zeros(1, device=self.mu_device)
        total_loss = score_loss + kl_loss
        total_loss.backward()

        total_val = total_loss.item()
        del total_loss
        return torch.tensor(total_val, device=self.mu_device)

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        dev_examples: list[dict],
        desc: str = "eval",
        split: str | None = None,
        epoch: int | None = None,
    ) -> dict:
        """Greedy rollout on eval examples, with optional per-step trace dumps."""
        save_traces = bool(self.rl_cfg.get("save_eval_traces", False))
        if not save_traces:
            result = evaluate_pi_model(
                model=self.pi,
                tokenizer=self.pi_tok,
                examples=dev_examples,
                retriever=self.retriever,
                max_steps=self.max_steps,
                pi_system=self.pi_system,
                pi_user_tmpl=self.pi_user_tmpl,
                pi_finish_tmpl=self.pi_finish_tmpl,
                max_new_tokens=self.rl_cfg.get("pi_max_new_tokens", 64),
                batch_size=self.eval_batch_size,
                desc=desc,
                dataset=self.dataset,
            )
            if self.has_mu:
                self.mu.train()
            return result

        split_name = split or desc.split()[0]
        all_results = []
        all_step_records: list[dict] = []

        pi_orig_padding = self.pi_tok.padding_side
        mu_orig_padding = self.mu_tok.padding_side if self.has_mu else None
        self.pi_tok.padding_side = "left"
        if self.has_mu:
            self.mu_tok.padding_side = "left"

        pi_was_training = self.pi.training
        mu_was_training = self.mu.training if self.has_mu else False
        self.pi.eval()
        if self.has_mu:
            self.mu.eval()

        total_batches = (len(dev_examples) + self.eval_batch_size - 1) // self.eval_batch_size
        pbar = tqdm(
            range(0, len(dev_examples), self.eval_batch_size),
            total=total_batches,
            desc=desc,
            dynamic_ncols=True,
            leave=False,
        )

        for start in pbar:
            batch_ex = dev_examples[start:start + self.eval_batch_size]
            envs = [
                self.spec.make_env(ex, retriever=self.retriever, max_steps=self.max_steps)
                for ex in batch_ex
            ]
            active = [True] * len(batch_ex)
            steps_used = [0] * len(batch_ex)
            step_records_by_example: list[list[dict]] = [[] for _ in batch_ex]
            batch_results = []

            while True:
                active_idx = [i for i in range(len(batch_ex)) if active[i] and not envs[i].state.done]
                if not active_idx:
                    break

                finish_idx = {
                    i for i in active_idx
                    if self.spec.has_forced_finish and steps_used[i] >= self.max_steps
                }
                pre_states = {i: envs[i].state for i in active_idx}
                searches_remaining = {
                    i: max(self.max_steps - steps_used[i], 0)
                    for i in active_idx
                }

                texts = [
                    self._build_pi_finish_text(pre_states[i])
                    if i in finish_idx
                    else self._build_pi_text(pre_states[i], searches_remaining[i])
                    for i in active_idx
                ]
                pi_max_new = self.rl_cfg.get("pi_max_new_tokens", 64)
                use_vllm_eval = (
                    self._vllm_engine is not None and self._vllm_engine.llm is not None
                )
                if use_vllm_eval:
                    vllm_results = self._vllm_engine.generate_pi(
                        texts,
                        temperature=0.0,
                        max_tokens=pi_max_new,
                    )
                    resp_texts = [vr.resp_text for vr in vllm_results]
                    enc = None
                else:
                    enc = self.pi_tok(texts, return_tensors="pt", padding=True).to(self.pi_device)
                    prompt_len = enc.input_ids.shape[1]
                    out = self.pi.generate(
                        **enc,
                        max_new_tokens=pi_max_new,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=self.pi_tok.pad_token_id,
                    )
                    resp_texts = [
                        self.pi_tok.decode(out[j][prompt_len:], skip_special_tokens=True)
                        for j in range(len(texts))
                    ]
                    del out

                valid_actions: dict[int, str] = {}
                pending_records: dict[int, dict] = {}
                scored_this_round: list[dict] = []

                for j, i in enumerate(active_idx):
                    resp_text = resp_texts[j]
                    try:
                        _, action = self.spec.parse_output(resp_text)
                    except ValueError:
                        action = resp_text.strip() or "<EMPTY>"
                        row = self._build_mu_trace_payload(
                            batch_ex[i],
                            pre_states[i],
                            action,
                            step_idx=len(step_records_by_example[i]),
                            searches_remaining=searches_remaining[i],
                            epoch=epoch,
                            split=split_name,
                        )
                        row["valid_parse"] = False
                        step_records_by_example[i].append(row)
                        all_step_records.append(row)
                        obs = (
                            f"Invalid action: {action}. "
                            "Action parsing error: output did not match the required format. "
                            "No retrieval was executed."
                        )
                        row["observation"] = obs
                        self._score_eval_trace_steps([row])
                        envs[i].record_invalid_action(action, obs, done=i in finish_idx)
                        if i not in finish_idx:
                            steps_used[i] += 1
                        if i in finish_idx or (
                            steps_used[i] >= self.max_steps and not self.spec.has_forced_finish
                        ):
                            active[i] = False
                        continue

                    row = self._build_mu_trace_payload(
                        batch_ex[i],
                        pre_states[i],
                        action,
                        step_idx=len(step_records_by_example[i]),
                        searches_remaining=searches_remaining[i],
                        epoch=epoch,
                        split=split_name,
                    )
                    step_records_by_example[i].append(row)
                    all_step_records.append(row)
                    pending_records[i] = row

                    if i in finish_idx and not action.startswith("Finish"):
                        obs = (
                            f"Forced finish violation: {action}. "
                            "Search budget exhausted. No answer submitted."
                        )
                        envs[i].record_invalid_action(action, obs, done=True)
                        row["observation"] = obs
                        scored_this_round.append(row)
                        active[i] = False
                        continue

                    valid_actions[i] = action

                if enc is not None:
                    del enc

                def _step_env(idx):
                    return idx, envs[idx].step(valid_actions[idx])

                with ThreadPoolExecutor(max_workers=max(min(len(valid_actions), 32), 1)) as pool:
                    futures = {pool.submit(_step_env, i): i for i in valid_actions}
                    for fut in as_completed(futures):
                        i = futures[fut]
                        row = pending_records[i]
                        try:
                            _, obs = fut.result()
                        except ValueError as e:
                            obs = (
                                f"Invalid action: {valid_actions[i]}. "
                                "Action execution error: invalid action format. "
                                f"No retrieval was executed. ({e})"
                            )
                            row["valid_parse"] = False
                            envs[i].record_invalid_action(valid_actions[i], obs, done=i in finish_idx)
                            if i not in finish_idx:
                                steps_used[i] += 1
                            if i in finish_idx or (
                                steps_used[i] >= self.max_steps and not self.spec.has_forced_finish
                            ):
                                active[i] = False
                        else:
                            steps_used[i] += self.spec.step_cost(valid_actions[i])
                            if i in finish_idx or (
                                steps_used[i] >= self.max_steps and not self.spec.has_forced_finish
                            ):
                                active[i] = False
                        row["observation"] = obs
                        scored_this_round.append(row)

                self._score_eval_trace_steps(scored_this_round)

            for i, ex in enumerate(batch_ex):
                reward_info = self.spec.compute_reward(envs[i], ex)
                traj_reward = float(reward_info.get("reward", 0.0))
                traj_em = float(reward_info.get("em", traj_reward))
                traj_f1 = float(reward_info.get("f1", traj_reward))
                submitted_answer = envs[i].state.submitted_answer
                for row in step_records_by_example[i]:
                    row["reward"] = traj_reward
                    row["em"] = traj_em
                    row["f1"] = traj_f1
                    row["submitted_answer"] = submitted_answer
                batch_results.append(reward_info)
                all_results.append(reward_info)

            batch_em = sum(1 for r in batch_results if r["reward"] >= 1.0) / len(batch_results)
            pbar.set_postfix({"EM": f"{batch_em:.3f}"})

        self.pi_tok.padding_side = pi_orig_padding
        if self.has_mu and mu_orig_padding is not None:
            self.mu_tok.padding_side = mu_orig_padding
        if pi_was_training:
            self.pi.train()
        if self.has_mu and mu_was_training:
            self.mu.train()
        torch.cuda.empty_cache()

        n_correct = sum(1 for r in all_results if r["reward"] >= 1.0)
        exact_match = n_correct / len(all_results)
        avg_f1 = sum(r.get("f1", r["reward"]) for r in all_results) / len(all_results)
        print(
            f"[eval] {desc}  n={len(all_results)}  reward: EM={exact_match:.4f}  "
            f"F1={avg_f1:.4f}  ({n_correct}/{len(all_results)} correct)"
        )

        if epoch is not None:
            trace_root = self.rl_cfg.get("eval_trace_dir")
            trace_dir = Path(trace_root) if trace_root else Path(self.rl_cfg["output_dir"]) / "dev_traces"
            trace_path = trace_dir / split_name / f"epoch_{epoch}.jsonl"
            write_jsonl(all_step_records, trace_path)
            print(
                f"[eval-trace] saved {split_name} epoch {epoch} -> {trace_path} "
                f"({len(all_step_records)} step records)"
            )

        return {"exact_match": exact_match, "f1": avg_f1}

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self, eval_examples: dict[str, list[dict]] | None = None) -> None:
        output_dir = Path(self.rl_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        examples = list(self.examples)

        # Hook for subclasses to set up persistent state before training.
        # Default: no-op.
        self._pre_train_setup(examples)

        # Ensure both models are in train mode before training starts
        self.pi.train()
        if self.has_mu:
            self.mu.train()

        start_epoch = int(self.rl_cfg.get("start_epoch", 0))
        total_epochs = int(self.rl_cfg["epochs"])
        if start_epoch < 0 or start_epoch > total_epochs:
            raise ValueError(f"Invalid start_epoch={start_epoch}; expected 0 <= start_epoch <= epochs={total_epochs}")
        if start_epoch > 0:
            print(f"[RL] Resuming epoch loop at epoch {start_epoch + 1}/{total_epochs} (optimizer/RNG state is freshly initialized)")

        for epoch in range(start_epoch, total_epochs):
            use_dense = (epoch >= self.dense_start)
            dense_tag = "dense" if use_dense else "sparse"

            # Adjust π lr for dense phase (dense has per-step gradients → lower lr)
            dense_pi_lr = self.dense_pi_lr
            if dense_pi_lr is not None and use_dense:
                for pg in self.pi_optim.param_groups:
                    pg["lr"] = dense_pi_lr
            elif dense_pi_lr is not None and not use_dense:
                for pg in self.pi_optim.param_groups:
                    pg["lr"] = self.sparse_pi_lr

            # Phase-aware μ / score_head lr (defaults to mu_lr/score_head_lr in legacy cfgs).
            if self.mu_optim is not None:
                target_mu_lr = self.dense_mu_lr if use_dense else self.sparse_mu_lr
                for pg in self.mu_optim.param_groups:
                    pg["lr"] = target_mu_lr
            if self.mu_score_optim is not None:
                target_score_lr = self.dense_score_head_lr if use_dense else self.sparse_score_head_lr
                for pg in self.mu_score_optim.param_groups:
                    pg["lr"] = target_score_lr

            random.shuffle(examples)
            pi_losses, mu_losses, rewards = [], [], []

            # Start vLLM once for the entire epoch
            self._start_vllm_epoch()

            grpo_k = self.rl_cfg.get("grpo_group_size", 1)
            # GRPO only in sparse phase; dense phase uses full batch (faster)
            if not use_dense and grpo_k > 1:
                effective_batch = self.batch_size // grpo_k
                step_size = effective_batch
            else:
                step_size = self.batch_size

            n_batches = (len(examples) + step_size - 1) // step_size
            pbar = tqdm(range(0, len(examples), step_size),
                        total=n_batches,
                        desc=f"Epoch {epoch+1}/{self.rl_cfg['epochs']} [{dense_tag}]",
                        dynamic_ncols=True)

            for batch_idx, batch_start in enumerate(pbar):
                if not use_dense and grpo_k > 1:
                    # GRPO: repeat each query K times for group comparison
                    unique_examples = examples[batch_start:batch_start + effective_batch]
                    batch_examples = unique_examples * grpo_k
                else:
                    # Dense phase or no GRPO: normal batching
                    batch_examples = examples[batch_start:batch_start + self.batch_size]

                # Update LoRA weights in vLLM (skip first batch — already loaded at epoch start)
                if batch_idx > 0:
                    self._update_vllm_lora()

                self._enter_rollout_phase()
                batch = self._rollout_batch(batch_examples)
                self._exit_rollout_phase()
                if self.rl_cfg.get("empty_cache_after_batch", False):
                    torch.cuda.empty_cache()

                n_traj = sum(1 for t in batch if t["steps"])
                avg_r  = sum(t["reward"] for t in batch) / len(batch)
                avg_steps = sum(len(t["steps"]) for t in batch) / len(batch)
                rewards.append(avg_r)

                # ── π 和 μ 并行 backward（各自在独立 GPU 上）───────────────
                pi_loss_val = 0.0
                mu_loss_val = 0.0

                # reward baseline for sparse REINFORCE / GRPO
                binary = self.rl_cfg.get("binary_reward", False)
                raw_rewards = [t["reward"] for t in batch]
                if binary:
                    batch_rewards = [2.0 * r - 1.0 for r in raw_rewards]
                else:
                    batch_rewards = raw_rewards

                # Compute per-trajectory baselines / dense step-wise signals
                dense_advantages = None
                if not use_dense and grpo_k > 1:
                    # Sparse + GRPO: per-query group mean baseline
                    from collections import defaultdict as _dd
                    _grp = _dd(list)
                    for i, t in enumerate(batch):
                        _grp[t.get("_question", t["example_id"])].append(i)
                    per_traj_baseline = [0.0] * len(batch)
                    for _q, _idxs in _grp.items():
                        grp_mean = sum(batch_rewards[i] for i in _idxs) / len(_idxs)
                        for i in _idxs:
                            per_traj_baseline[i] = grp_mean
                elif not use_dense:
                    # Sparse without GRPO: batch mean baseline
                    all_same = (min(batch_rewards) == max(batch_rewards))
                    bl = 0.0 if all_same else (sum(batch_rewards) / len(batch_rewards))
                    per_traj_baseline = [bl] * len(batch)
                else:
                    dense_advantages = self._dense_advantages_batch(batch)
                    per_traj_baseline = [0.0] * len(batch)

                def _run_pi():
                    nonlocal pi_loss_val
                    self.pi_optim.zero_grad()
                    n_batch = max(len(batch), 1)
                    for i, traj in enumerate(batch):
                        tl = self._pi_loss_traj(
                            traj,
                            use_dense,
                            reward_baseline=per_traj_baseline[i],
                            dense_signals=(dense_advantages[i] if use_dense else None),
                        )
                        pi_loss_val += tl.item()
                    for p in self.pi.parameters():
                        if p.grad is not None:
                            p.grad.div_(n_batch)
                    all_pi_params = list(self.pi.parameters())
                    torch.nn.utils.clip_grad_norm_(all_pi_params, 1.0)
                    self.pi_optim.step()

                t_pi = threading.Thread(target=_run_pi)
                t_pi.start()
                if self.has_mu and (self.mu_optim is not None or self.mu_score_optim is not None):
                    def _run_mu():
                        nonlocal mu_loss_val
                        if self.mu_optim:
                            self.mu_optim.zero_grad()
                        if self.mu_score_optim:
                            self.mu_score_optim.zero_grad()
                        n_batch = max(len(batch), 1)
                        # Dynamic class balancing for score_loss
                        n_pos = max(sum(1 for t in batch if t["reward"] > 0.5), 1)
                        n_neg = max(sum(1 for t in batch if t["reward"] <= 0.5), 1)
                        w_pos = n_batch / (2.0 * n_pos)  # upweight minority
                        w_neg = n_batch / (2.0 * n_neg)
                        for traj in batch:
                            sw = w_pos if traj["reward"] > 0.5 else w_neg
                            tl = self._mu_loss_traj(traj, score_weight=sw)
                            mu_loss_val += tl.item()
                        for p in self.mu.parameters():
                            if p.grad is not None:
                                p.grad.div_(n_batch)
                        torch.nn.utils.clip_grad_norm_(self.mu.parameters(), 1.0)
                        if self.mu_optim:
                            self.mu_optim.step()
                        if self.mu_score_optim:
                            self.mu_score_optim.step()

                    t_mu = threading.Thread(target=_run_mu)
                    t_mu.start()
                    t_pi.join()
                    t_mu.join()
                else:
                    t_pi.join()

                n_batch = max(len(batch), 1)
                pi_losses.append(pi_loss_val / n_batch)
                mu_losses.append(mu_loss_val / n_batch)

                # Collect μ score statistics per batch.
                # of the trajectory reward), so summing all steps double-counts
                # EM. Restrict the sum to prefix steps where μ actually predicts.
                scores_r1, scores_r0 = [], []
                for traj in batch:
                    if not traj["steps"]:
                        continue
                    traj_sum = sum(
                        float(s["score"]) for s in traj["steps"]
                        if s.get("split_role") != "terminal"
                    )
                    if traj["reward"] > 0.5:
                        scores_r1.append(traj_sum)
                    else:
                        scores_r0.append(traj_sum)
                avg_s1 = sum(scores_r1) / max(len(scores_r1), 1)
                avg_s0 = sum(scores_r0) / max(len(scores_r0), 1)

                postfix = {
                    "π": f"{pi_loss_val / n_batch:.2f}",
                    self.reward_label: f"{avg_r:.3f}",
                    "active": n_traj,
                }
                if self.has_mu:
                    postfix["μ"] = f"{mu_loss_val / n_batch:.3f}"
                    postfix["sR1"] = f"{avg_s1:.2f}"
                    postfix["sR0"] = f"{avg_s0:.2f}"
                pbar.set_postfix(postfix)

            avg_pi = sum(pi_losses) / len(pi_losses)
            avg_mu = sum(mu_losses) / len(mu_losses)
            avg_R  = sum(rewards)   / len(rewards)

            # Epoch-level μ score statistics
            all_step_scores_r1, all_step_scores_r0 = [], []
            all_traj_sums_r1, all_traj_sums_r0 = [], []
            for batch in []:  # can't access old batches, use last batch
                pass
            # Use rewards to estimate R distribution
            n_r1 = sum(1 for r in rewards if r > 0.3)  # approximate
            print(f"[RL] Epoch {epoch+1} ── "
                  f"π_loss={avg_pi:.4f}  μ_loss={avg_mu:.4f}  "
                  f"reward: {self.reward_label}={avg_R:.3f}  n_batches={len(pi_losses)}")

            # Stop vLLM at epoch end, restore HF models for eval/save
            self._stop_vllm_epoch()

            self.pi.save_pretrained(output_dir / f"pi_epoch_{epoch+1}")
            if self.has_mu:
                self.mu.save_pretrained(output_dir / f"mu_epoch_{epoch+1}")

            if eval_examples:
                # Reuse the persistent vLLM engine for π eval. HF μ stays
                # resident for forward_score on cuda:1.
                vllm_eval_active = False
                if self._vllm_engine is not None and self._vllm_engine.llm is not None:
                    try:
                        # Push fresh LoRA weights to vLLM (no restart).
                        self._update_vllm_lora()
                        # Offload HF gen weights so vLLM has room on cuda:0.
                        # In pi_only mode, μ stays on GPU (HF rollout target).
                        self.pi.to("cpu")
                        if self.pi_ref is not None:
                            self.pi_ref.to("cpu")
                        if self.mu_ref is not None and self._vllm_engine.serves_mu:
                            self.mu_ref.to("cpu")
                        torch.cuda.empty_cache()
                        vllm_eval_active = True
                    except Exception as e:
                        print(f"[eval] vLLM lora refresh failed, falling back to HF: {e}")
                        # Restore HF models to GPU for HF eval path
                        self.pi.to(self.pi_device)
                        if self.pi_ref is not None:
                            self.pi_ref.to(self.pi_ref_device)
                        if self.mu_ref is not None:
                            self.mu_ref.to(self.mu_device)

                for split, split_examples in eval_examples.items():
                    self.evaluate(
                        split_examples,
                        desc=f"{split} epoch {epoch+1}",
                        split=split,
                        epoch=epoch + 1,
                    )

                if vllm_eval_active:
                    # Don't stop vLLM — keep it alive for next epoch.
                    # Reload HF models to GPU so the next epoch's grad updates work.
                    self.pi.to(self.pi_device)
                    if self.pi_ref is not None:
                        self.pi_ref.to(self.pi_ref_device)
                    if self.mu_ref is not None and self._vllm_engine.serves_mu:
                        self.mu_ref.to(self.mu_device)
                    torch.cuda.empty_cache()
