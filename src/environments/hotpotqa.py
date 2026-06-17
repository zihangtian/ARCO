"""HotpotQA environment, reward, parsing, and data loading."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from .base import BaseEnvironment, BaseState
from ..utils.retriever import Retriever
from ..utils.metrics import em, f1_score


@dataclass
class HotpotState(BaseState):
    q_type: str = ""
    level: str = ""
    paragraphs: list[str] = field(default_factory=list)


class HotpotEnv(BaseEnvironment):
    def __init__(self, example: dict, retriever: Retriever | None = None):
        self.retriever = retriever
        super().__init__(example)

    def _init_state(self, example: dict) -> HotpotState:
        paragraphs = ["".join(s) for _, s in example.get("context", [])]
        return HotpotState(
            question=example["question"],
            gold_answer=example["answer"],
            example_id=example["_id"],
            q_type=example.get("type", ""),
            level=example.get("level", ""),
            paragraphs=paragraphs,
        )

    def step(self, action: str) -> str:
        action_type, argument = hotpotqa_parse_action(action)
        self.state.step_count += 1

        if action_type == "Search":
            observation = (self.retriever.retrieve(argument, data=self.state.paragraphs)
                           if self.retriever else f"[no retriever; searched: {argument}]")
            self.state.context += f"\n\nObservation: {observation}"
            self.state.action_history.append({"action": action, "observation": observation})
            return observation

        elif action_type == "Finish":
            self.state.submitted_answer = argument
            self.state.done = True
            obs = f"Submitted answer: {argument}"
            self.state.action_history.append({"action": action, "observation": obs})
            return obs

        raise ValueError(f"Unknown action type: {action_type}")

    def had_search(self) -> bool:
        return any(h["action"].startswith("Search") for h in self.state.action_history)


# ── Parsing ──────────────────────────────────────────────────────────────────

def hotpotqa_parse_output(output: str) -> Tuple[str, str]:
    # Strip Qwen3-style <think>...</think> blocks before parsing
    stripped = re.sub(r"<think>.*?</think>\s*", "", output, flags=re.DOTALL | re.IGNORECASE).strip()

    # Preferred format: a single bare action line.
    if re.match(r"^(Search|Finish)\[[^\]\n]*[\]\)]\s*$", stripped, re.DOTALL):
        return "", stripped

    t = re.search(r"\bThought:\s*(.*?)(?=\n\s*Action:|\Z)", stripped, re.DOTALL | re.IGNORECASE)
    a = re.search(r"\bAction:\s*(.*)", stripped, re.DOTALL | re.IGNORECASE)
    thought = t.group(1).strip() if t else ""
    action = a.group(1).strip() if a else ""

    # Fallback: some generations omit the "Action:" prefix but still place a
    # valid action on the final line after the thought text.
    if not action:
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines:
            candidate = lines[-1]
            if re.match(r"^(Search|Finish)\[[^\]\n]*[\]\)]\s*$", candidate, re.DOTALL):
                action = candidate
                if thought:
                    thought = re.sub(rf"\n?\s*{re.escape(candidate)}\s*$", "", thought, flags=re.DOTALL).strip()

    # Fallback: if the model emitted one or more valid actions anywhere in the
    # text, use the first one and ignore trailing junk / extra actions.
    if not action:
        matches = re.findall(r"(Search\[[^\]\n]*[\]\)]|Finish\[[^\]\n]*[\]\)])", stripped, re.DOTALL)
        if matches:
            action = matches[0].strip()

    if not action:
        raise ValueError(f"Cannot parse action from: {output!r}")
    return thought, action


def hotpotqa_parse_action(action: str) -> Tuple[str, str]:
    # Accept both ] and ) as closing bracket (model occasionally mismatches)
    m = re.match(r"(Search|Finish)\[(.+?)[\]\)]", action, re.DOTALL)
    if not m:
        raise ValueError(f"Invalid action format: {action!r}")
    return m.group(1), m.group(2).strip()


# ── Reward ───────────────────────────────────────────────────────────────────

def hotpotqa_reward(submitted_answer: str | None, gold_answer: str,
                    had_search: bool) -> dict:
    if submitted_answer is None:
        return {"em": 0, "f1": 0.0, "reward": 0.0, "detail": "no_answer"}
    em_val = em(submitted_answer, gold_answer)
    f1_val, prec, recall = f1_score(submitted_answer, gold_answer)
    reward = float(em_val)
    detail = "normal"
    return {"em": em_val, "f1": f1_val, "precision": prec, "recall": recall,
            "reward": max(0.0, min(1.0, reward)), "detail": detail}


# ── Data loading ──────────────────────────────────────────────────────────────

def hotpotqa_load_data(cfg: dict, split: str) -> list[dict]:
    split_dir = Path(cfg["data"]["split_dir"])
    split_path = split_dir / f"{split}.jsonl"
    if split_path.exists():
        from ..utils.misc import read_jsonl
        return read_jsonl(split_path)
    key = f"{split}_file"
    raw_path = cfg["data"].get(key)
    if raw_path and Path(raw_path).exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"No data for split '{split}' in {split_dir}")


# backward-compat alias
hotpotqa_parse_thought_action = hotpotqa_parse_output


# ── Unified wrappers for registry ────────────────────────────────────────

def _compute_reward(env: HotpotEnv, example: dict) -> dict:
    return hotpotqa_reward(
        submitted_answer=env.state.submitted_answer,
        gold_answer=example["answer"],
        had_search=env.had_search(),
    )


def _step_cost(action: str) -> int:
    return 1 if action.startswith("Search") else 0


# ── Registration ─────────────────────────────────────────────────────────

from .registry import DatasetSpec, register_dataset  # noqa: E402

register_dataset(DatasetSpec(
    name="hotpotqa",
    make_env=lambda example, retriever=None, **kw: HotpotEnv(example, retriever=retriever),
    parse_output=hotpotqa_parse_output,
    compute_reward=_compute_reward,
    load_data=hotpotqa_load_data,
    step_cost=_step_cost,
    display_raw_actions=False,
    has_forced_finish=True,
))
