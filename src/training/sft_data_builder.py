"""Build SFT datasets from warmup records for π and μ.

Dataset-specific prompt fields differ, so builders are registered per dataset.
To add a new dataset, implement:
  - build_pi_sft_data_<dataset>()
  - build_mu_sft_data_<dataset>()
and register them in PI_SFT_BUILDERS / MU_SFT_BUILDERS.
"""

import json
import torch
from torch.utils.data import Dataset
from ..utils.prompt_builders import (
    display_action,
    format_history_from_lines,
    policy_fields_from_record_step,
    rubric_fields_from_record_step,
)

# ── Data builder functions ─────────────────────────────────────────────────

def _display_action(step: dict) -> str:
    """Return the display form of an action from a warmup record step."""
    return step.get("action_display", step.get("env_action", step["action"]))


def _searches_remaining_before(step: dict, max_steps: int,
                               searches_done: int) -> int:
    """Use recorded rollout budget when available.

    New warmup data records invalid actions as budget-consuming steps. Replaying
    prompts from action strings alone would miss that because malformed actions
    need not start with ``Search``.
    """
    if "searches_remaining_before" in step:
        return max(int(step["searches_remaining_before"]), 0)
    return max(max_steps - searches_done, 0)


def build_pi_sft_data_hotpotqa(records: list[dict], pi_system_prompt: str,
                               user_template: str,
                               finish_template: str | None = None,
                               max_steps: int = 3,
                               min_reward: float = 1.0,
                               dataset: str = "hotpotqa") -> list[dict]:
    data = []
    for rec in records:
        if not rec["steps"]:
            continue
        if rec.get("reward", 0.0) < min_reward:
            continue
        history_lines: list[str] = []
        searches_done = 0
        for step in rec["steps"]:
            searches_remaining = _searches_remaining_before(
                step, max_steps, searches_done)
            if searches_remaining <= 0 and finish_template is not None:
                user_content = finish_template.format(
                    question=rec["question"],
                    history=format_history_from_lines(history_lines),
                )
            else:
                user_content = user_template.format(**{
                    "question": rec["question"],
                    "history": format_history_from_lines(history_lines),
                    "searches_remaining": searches_remaining,
                })
            target = _display_action(step)
            data.append({
                "example_id": rec["example_id"],
                "step": step["step"],
                "messages": [
                    {"role": "system", "content": pi_system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "target": target,
            })
            history_lines.append(f"Action: {step['action']}")
            history_lines.append(f"Observation: {step['observation']}")
            if step["action"].startswith("Search"):
                searches_done += 1
    return data


def build_mu_sft_data_hotpotqa(records: list[dict], mu_system_prompt: str,
                               user_template: str,
                               drop_empty_criteria: bool = True,
                               dataset: str = "hotpotqa",
                               max_steps: int = 3,
                               num_criteria: int = 3) -> list[dict]:
    data = []
    K = num_criteria
    for rec in records:
        history_lines: list[str] = []
        searches_done = 0
        for step in rec["steps"]:
            searches_remaining = _searches_remaining_before(
                step, max_steps, searches_done)
            if drop_empty_criteria and not step.get("criteria"):
                history_lines.append(f"Action: {step['action']}")
                history_lines.append(f"Observation: {step['observation']}")
                if step["action"].startswith("Search") and searches_remaining > 0:
                    searches_done += 1
                continue
            user_content = user_template.format(
                K=K,
                K_schema=", ".join(f'"<criterion {i+1}>"' for i in range(K)),
                **rubric_fields_from_record_step(
                    rec, step, history_lines, dataset, raw_actions=False,
                    searches_remaining=searches_remaining
                ))
            # Target: K criteria as JSON
            criteria = step["criteria"][:K]
            while len(criteria) < K:
                criteria.append("")
            target = json.dumps({"criteria": criteria}, ensure_ascii=False)
            # Scores: list of K floats (rubric) or single float (AgentPRM)
            if "scores" in step:
                scores = step["scores"][:K]
                while len(scores) < K:
                    scores.append(0.0)
            else:
                scores = [step.get("score", 0.0)]
            data.append({
                "example_id": rec["example_id"],
                "sample_idx": rec.get("sample_idx"),
                "step": step["step"],
                "messages": [
                    {"role": "system", "content": mu_system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "target": target,
                "scores": scores,
            })
            history_lines.append(f"Action: {step['action']}")
            history_lines.append(f"Observation: {step['observation']}")
            if step["action"].startswith("Search") and searches_remaining > 0:
                searches_done += 1
    return data


PI_SFT_BUILDERS = {
    "hotpotqa": build_pi_sft_data_hotpotqa,
}

MU_SFT_BUILDERS = {
    "hotpotqa": build_mu_sft_data_hotpotqa,
}


def build_pi_sft_data(records: list[dict], pi_system_prompt: str,
                      user_template: str,
                      finish_template: str | None = None,
                      max_steps: int = 3,
                      min_reward: float = 1.0,
                      dataset: str = "hotpotqa") -> list[dict]:
    # Drop 1-step records: direct finish without reasoning, useless for training
    before = len(records)
    records = [r for r in records if len(r.get("steps", [])) > 1]
    if before != len(records):
        print(f"[sft_pi] Dropped {before - len(records)} single-step records "
              f"({before} → {len(records)})")
    try:
        builder = PI_SFT_BUILDERS[dataset]
    except KeyError as e:
        raise ValueError(f"Unsupported dataset for π SFT builder: {dataset!r}") from e
    kwargs = {
        "finish_template": finish_template,
        "max_steps": max_steps,
        "min_reward": min_reward,
    }
    if builder is build_pi_sft_data_hotpotqa:
        kwargs["dataset"] = dataset
    return builder(records, pi_system_prompt, user_template, **kwargs)


def build_mu_sft_data(records: list[dict], mu_system_prompt: str,
                      user_template: str,
                      drop_empty_criteria: bool = True,
                      dataset: str = "hotpotqa",
                      max_steps: int = 3,
                      num_criteria: int = 3) -> list[dict]:
    # Drop 1-step records: direct finish without reasoning, rubric can't train on these
    before = len(records)
    records = [r for r in records if len(r.get("steps", [])) > 1]
    if before != len(records):
        print(f"[sft_mu] Dropped {before - len(records)} single-step records "
              f"({before} → {len(records)})")
    try:
        builder = MU_SFT_BUILDERS[dataset]
    except KeyError as e:
        raise ValueError(f"Unsupported dataset for μ SFT builder: {dataset!r}") from e
    kwargs = {"drop_empty_criteria": drop_empty_criteria}
    if builder is build_mu_sft_data_hotpotqa:
        kwargs["dataset"] = dataset
        kwargs["max_steps"] = max_steps
        kwargs["num_criteria"] = num_criteria
    return builder(records, mu_system_prompt, user_template, **kwargs)


def split_by_example(data: list[dict], val_ratio: float = 0.1,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split by example_id to avoid leakage."""
    import random
    rng = random.Random(seed)
    ids = sorted({d["example_id"] for d in data})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train = [d for d in data if d["example_id"] not in val_ids]
    val = [d for d in data if d["example_id"] in val_ids]
    return train, val


# ── PyTorch Dataset ────────────────────────────────────────────────────────

class ChatSFTDataset(Dataset):
    """Tokenize chat messages + target. Masks prompt tokens in labels."""

    def __init__(self, data: list[dict], tokenizer, max_seq_length: int = 2048,
                 apply_chat_template: bool = True, score_only: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.apply_chat_template = apply_chat_template
        self.score_only = score_only
        self.examples = self._process(data)

    def _process(self, data: list[dict]) -> list[dict]:
        processed = []
        dropped_no_target = 0
        for item in data:
            if self.apply_chat_template:
                prompt_text = self.tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            else:
                # Fallback: concatenate text manually
                text = ""
                for m in item["messages"]:
                    text += f"{m['role'].upper()}: {m['content']}\n"
                prompt_ids = self.tokenizer.encode(text, add_special_tokens=True)

            if self.score_only:
                # Score-head-only: backbone never sees a text target as loss.
                # If target is non-empty (e.g. rubric_mode=fixed: static rubric
                # appended to prompt to match RL inference layout), we still
                # tokenize and concatenate it onto input_ids — but labels are
                # fully -100 so no LM loss flows back. EOS is *not* appended,
                # mirroring rl_trainer.py:957–973 which concatenates raw
                # static_criteria_text with no special tokens.
                target_text = item.get("target") or ""
                if target_text:
                    target_ids = self.tokenizer.encode(
                        target_text, add_special_tokens=False)
                else:
                    target_ids = []
                # Truncate prompt from the left if needed (preserve target tail).
                max_prompt_len = self.max_seq_length - len(target_ids)
                if max_prompt_len <= 0:
                    target_ids = target_ids[-self.max_seq_length:]
                    prompt_ids = []
                elif len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]
                input_ids = prompt_ids + target_ids
                labels = [-100] * len(input_ids)
            else:
                target_ids = self.tokenizer.encode(
                    item["target"] + self.tokenizer.eos_token,
                    add_special_tokens=False,
                )
                # Preserve supervision by truncating the prompt from the left first.
                if len(target_ids) >= self.max_seq_length:
                    target_ids = target_ids[: self.max_seq_length]
                    prompt_ids = []
                else:
                    max_prompt_len = self.max_seq_length - len(target_ids)
                    if len(prompt_ids) > max_prompt_len:
                        prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
                input_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids
                if not any(tok != -100 for tok in labels):
                    dropped_no_target += 1
                    continue

            processed.append({
                "input_ids": input_ids,
                "labels": labels,
                "scores": item.get("scores"),      # list[float] or None (μ SFT has scores)
                "prompt_len": len(prompt_ids),      # needed by score_head
            })
        if dropped_no_target:
            print(
                f"[ChatSFTDataset] Dropped {dropped_no_target} examples with no "
                "target tokens after truncation"
            )
        return processed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_sft(batch: list[dict], pad_token_id: int) -> dict:
    """Pad batch to same length."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(x["input_ids"]) + [0] * pad_len)
    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    # Include score regression targets when available (μ SFT with score_head)
    if batch[0].get("scores") is not None:
        result["scores"] = torch.tensor(
            [x["scores"] for x in batch], dtype=torch.float32)  # (B, 3)
        result["prompt_lens"] = torch.tensor(
            [x["prompt_len"] for x in batch], dtype=torch.long)
    return result
