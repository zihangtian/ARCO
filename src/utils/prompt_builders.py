"""Dataset-aware prompt field builders.

Adding a new dataset should mostly require registering its formatting behavior
here, then reusing the shared helpers across warmup / SFT / eval / RL.
"""

from __future__ import annotations

import re
from typing import Iterable


def display_action(action: str, dataset: str = "", raw_actions: bool = False) -> str:
    """Strip Act[…]/Finish[…] wrappers when *raw_actions* is True."""
    if not raw_actions:
        return action
    m = re.match(r"(?:Act|Finish)\[(.*?)\]\s*$", action.strip(), re.DOTALL)
    return m.group(1).strip() if m else action


def format_history_from_action_history(action_history: list[dict], dataset: str,
                                       raw_actions: bool = False) -> str:
    if not action_history:
        return "(none)"
    lines = []
    for h in action_history:
        lines.append(f"Action: {display_action(h['action'], dataset, raw_actions)}")
        lines.append(f"Observation: {h['observation']}")
    return "\n".join(lines)


def format_history_from_lines(history_lines: list[str]) -> str:
    return "\n".join(history_lines) if history_lines else "(none)"


def format_admissible_commands(commands: Iterable[str] | None) -> str:
    commands = list(commands or [])
    if not commands:
        return "(not provided)"
    return "\n".join(f"- {cmd}" for cmd in commands)


def current_observation_from_state(state) -> str:
    if getattr(state, "action_history", None):
        return state.action_history[-1]["observation"]
    if getattr(state, "context", None):
        return str(state.context).strip()
    return "(none)"


def policy_fields_from_state(state, dataset: str, steps_remaining: int) -> dict:
    return {
        "question": state.question,
        "history": format_history_from_action_history(
            state.action_history,
            dataset,
            raw_actions=getattr(state, "display_raw_actions", False),
        ),
        "current_observation": current_observation_from_state(state),
        "admissible_commands": format_admissible_commands(
            getattr(state, "admissible_commands", None)
        ),
        "searches_remaining": steps_remaining,
        "steps_remaining": steps_remaining,
    }


def policy_fields_from_record_step(rec: dict, step: dict, history_lines: list[str],
                                   dataset: str, steps_remaining: int) -> dict:
    return {
        "question": rec["question"],
        "history": format_history_from_lines(history_lines),
        "current_observation": step.get("current_observation", rec["question"]),
        "admissible_commands": format_admissible_commands(
            step.get("admissible_commands_before")
        ),
        "searches_remaining": steps_remaining,
        "steps_remaining": steps_remaining,
    }


def rubric_fields_from_state_action(state, action: str, dataset: str,
                                    searches_remaining=None) -> dict:
    if searches_remaining is None:
        searches_remaining = "(not provided)"
    return {
        "question": state.question,
        "history": format_history_from_action_history(
            state.action_history,
            dataset,
            raw_actions=getattr(state, "display_raw_actions", False),
        ),
        "searches_remaining": searches_remaining,
        "steps_remaining": searches_remaining,
        "action": display_action(action, dataset, raw_actions=getattr(state, "display_raw_actions", False)),
    }


def rubric_fields_from_record_step(rec: dict, step: dict, history_lines: list[str],
                                   dataset: str, raw_actions: bool = False,
                                   searches_remaining=None) -> dict:
    shown_action = step.get("action_display", display_action(step["action"], dataset, raw_actions=raw_actions))
    if searches_remaining is None:
        searches_remaining = "(not provided)"
    return {
        "question": rec["question"],
        "history": format_history_from_lines(history_lines),
        "searches_remaining": searches_remaining,
        "steps_remaining": searches_remaining,
        "action": shown_action,
    }


def format_trajectory_steps(steps: list[dict], dataset: str,
                            raw_actions: bool = False) -> str:
    parts = []
    for s in steps:
        shown_action = s.get("action_display", display_action(s["action"], dataset, raw_actions=raw_actions))
        parts.append(f"Step {s['step']}:")
        if "searches_remaining_before" in s:
            parts.append(f"  Searches remaining before action: {s['searches_remaining_before']}")
        if s.get("forced_finish"):
            parts.append("  Forced finish prompt: yes")
        if s.get("valid_parse") is False:
            parts.append("  Invalid action: yes")
        parts.append(f"  Action: {shown_action}")
        parts.append(f"  Observation: {s['observation']}")
    return "\n".join(parts)
