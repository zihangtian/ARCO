"""Rubric agent μ: (state, action) → {"criteria": list[str]}.

Two modes:
  Per-step  (RL + SFT replay): evaluate one action causally (no future info).
  Trajectory (warmup only):    see the full trajectory, score all steps at once.

Prompt files:
  shared/rubric/system.txt           shared rubric system prompt, {task_context}
  hotpotqa/task_description.txt       dataset task description
  hotpotqa/rubric/step.txt            per-step template
                                      {question}, {history}, {action}
  hotpotqa/rubric/trajectory.txt      full-trajectory template
                                      {question}, {trajectory_text}, {reward}, {n_steps}
"""

import json
import math
import re
from pathlib import Path

from ..utils.api_client import APIClient
from ..utils.prompt_builders import format_trajectory_steps, rubric_fields_from_state_action
from ..environments.base import BaseState


class RubricAgent:
    def __init__(self, client: APIClient,
                 system_prompt: str,
                 user_template: str,
                 warmup_annotate_template: str,
                 dataset: str = "hotpotqa",
                 raw_actions: bool = False,
                 max_tokens: int = 128,
                 K: int = 3):
        self.client = client
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.warmup_annotate_template = warmup_annotate_template
        self.dataset = dataset
        self.raw_actions = raw_actions
        self.max_tokens = max_tokens
        self.K = K
        # Schema strings injected into prompts
        self.K_schema = ", ".join(f'"<criterion {i+1}>"' for i in range(K))
        self.K_score_schema = ", ".join(["<float>"] * K)

    def evaluate(self, state: BaseState, action: str,
                 use_cache: bool = True) -> dict:
        """Causal per-step evaluation."""
        fields = rubric_fields_from_state_action(state, action, self.dataset)
        user_content = self.user_template.format(
            **fields, K=self.K, K_schema=self.K_schema)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        output = self.client.chat(messages, temperature=0.0,
                                  max_tokens=self.max_tokens, use_cache=use_cache)
        return parse_rubric_output(output, K=self.K)

    # ── full-trajectory annotation (warmup only) ─────────────────────────────

    def evaluate_trajectory(self, question: str, steps: list[dict],
                            reward: float, use_cache: bool = True) -> list[dict]:
        """One API call for the full trajectory. Returns list of {criteria, score}.

        Prompt tells GPT to assign step scores whose summed means equal `reward`.
        Parsed scores are projected to the same constraint after parsing.
        """
        if not steps:
            return []
        trajectory_text = format_trajectory_steps(steps, self.dataset,
                                                   raw_actions=self.raw_actions)
        user_content = self.warmup_annotate_template.format(
            question=question,
            trajectory_text=trajectory_text,
            reward=reward,
            n_steps=len(steps),
            K=self.K,
            K_schema=self.K_schema,
            K_score_schema=self.K_score_schema,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        output = self.client.chat(
            messages, temperature=0.0,
            max_tokens=self.max_tokens * len(steps),
            use_cache=use_cache,
        )
        return parse_trajectory_output(output, n_steps=len(steps),
                                       fallback_score=reward, K=self.K)

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_prompt_dirs(cls, client: APIClient,
                         shared_dir: str | Path,
                         dataset_dir: str | Path,
                         raw_actions: bool = False,
                         K: int = 3,
                         **kwargs) -> "RubricAgent":
        sd, dd = Path(shared_dir), Path(dataset_dir)
        task_context = (dd / "task_description.txt").read_text(encoding="utf-8").strip()
        system_prompt = (sd / "rubric" / "system.txt").read_text(
            encoding="utf-8").strip().format(task_context=task_context)
        user_tmpl = (dd / "rubric" / "step.txt").read_text(encoding="utf-8").strip()
        warmup_tmpl = (dd / "rubric" / "trajectory.txt").read_text(encoding="utf-8").strip()
        return cls(client=client, system_prompt=system_prompt,
                   user_template=user_tmpl,
                   warmup_annotate_template=warmup_tmpl,
                   raw_actions=raw_actions, K=K, **kwargs)


# ── output parsers ────────────────────────────────────────────────────────────

def _parse_criteria(obj: dict) -> list[str]:
    """Parse 2-4 criteria; tolerate a single string by wrapping it."""
    raw = obj.get("criteria")
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
        if items:
            return items
    raw_single = obj.get("criterion")
    if raw_single is not None:
        single = str(raw_single).strip()
        if single:
            return [single]
    return []


def _strip_think_and_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Fix trailing commas before ] or } (invalid JSON but common GPT output)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _fix_latex_backslashes(text: str) -> str:
    """Fix invalid JSON backslashes from LaTeX (e.g. \\(, \\frac, \\cdot).

    Only called as a fallback when initial json.loads fails, to avoid
    double-escaping already-valid JSON strings like ``\\\\(``."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def _clamp_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(-1.0, min(1.0, score))


def _project_scores_to_reward(results: list[dict], reward: float, K: int = 3) -> list[dict]:
    """Enforce sum(step mean scores) == reward under per-score box bounds.

    GPT often follows the local scoring semantics but misses the exact
    trajectory-level decomposition. This projects the parsed scores onto the
    constraint set with minimal uniform distortion:

      sum_i mean(scores_i) == reward,  scores_ij in [-1, 1].

    Because each step has K scores, the equivalent flat-score target is
    sum_ij scores_ij == K * reward.
    """
    if not results:
        return results

    flat_scores: list[float] = []
    locations: list[tuple[list[float], int]] = []
    for item in results:
        scores = [_clamp_score(s) for s in item.get("scores", [])[:K]]
        while len(scores) < K:
            scores.append(0.0)
        item["scores"] = scores
        for i in range(K):
            flat_scores.append(scores[i])
            locations.append((scores, i))

    if not flat_scores:
        return results

    try:
        target_reward = float(reward)
    except (TypeError, ValueError):
        target_reward = 0.0
    if not math.isfinite(target_reward):
        target_reward = 0.0
    target_total = float(K) * target_reward
    lower_total = -float(len(flat_scores))
    upper_total = float(len(flat_scores))
    target_total = max(lower_total, min(upper_total, target_total))

    if abs(sum(flat_scores) - target_total) <= 1e-6:
        return results

    # L2 projection onto {x: sum(x)=target_total, -1<=x<=1}.
    # The solution has x_i = clamp(v_i + lambda, -1, 1).
    lo, hi = -2.0, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        shifted_sum = sum(max(-1.0, min(1.0, s + mid)) for s in flat_scores)
        if shifted_sum < target_total:
            lo = mid
        else:
            hi = mid
    projected = [max(-1.0, min(1.0, s + hi)) for s in flat_scores]

    # Clean up tiny numerical residuals so downstream filters see an exact match.
    residual = target_total - sum(projected)
    if abs(residual) > 1e-9:
        if residual > 0:
            for i, score in enumerate(projected):
                delta = min(residual, 1.0 - score)
                projected[i] += delta
                residual -= delta
                if abs(residual) <= 1e-9:
                    break
        else:
            for i, score in enumerate(projected):
                delta = max(residual, -1.0 - score)
                projected[i] += delta
                residual -= delta
                if abs(residual) <= 1e-9:
                    break

    for value, (scores, idx) in zip(projected, locations):
        scores[idx] = value
    return results


def parse_rubric_output(text: str, K: int = 3) -> dict:
    """Parse {"criteria": [...]} from per-step output.

    Pads or truncates to exactly K criteria.
    """
    text = _strip_think_and_fences(text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            criteria = _parse_criteria(obj)
            if criteria:
                while len(criteria) < K:
                    criteria.append("")
                criteria = criteria[:K]
                return {"criteria": criteria}
        except (json.JSONDecodeError, ValueError):
            pass
    fallback = text[:300]
    pad = [fallback] + [""] * (K - 1) if fallback else [""] * K
    return {"criteria": pad[:K], "parse_error": True}


def parse_trajectory_output(text: str, n_steps: int,
                             fallback_score: float, K: int = 3) -> list[dict]:
    """Parse JSON array from full-trajectory annotation output.

    Each step has K criteria and K scores. Returns list of
    {"criteria": [...K], "scores": [...K]}.
    """
    text = _strip_think_and_fences(text)

    def _parse_scores(obj: dict) -> list[float]:
        """Extract exactly K scores from a step object."""
        raw = obj.get("scores")
        if isinstance(raw, list) and len(raw) >= K:
            return [_clamp_score(s) for s in raw[:K]]
        if isinstance(raw, list) and raw:
            scores = [_clamp_score(s) for s in raw]
            while len(scores) < K:
                scores.append(scores[-1])
            return scores[:K]
        # Fallback: single "score" field → replicate to K
        single = obj.get("score")
        if single is not None:
            s = _clamp_score(single)
            return [s] * K
        return [0.0] * K

    def _try_parse_array(t: str):
        m = re.search(r"\[.*\]", t, re.DOTALL)
        if not m:
            return None
        arr = json.loads(m.group())
        if isinstance(arr, list) and len(arr) >= n_steps:
            results = []
            for obj in arr[:n_steps]:
                criteria = _parse_criteria(obj)
                while len(criteria) < K:
                    criteria.append("")
                criteria = criteria[:K]
                scores = _parse_scores(obj)
                results.append({"criteria": criteria, "scores": scores})
            return _project_scores_to_reward(results, fallback_score, K=K)
        return None

    def _try_parse_single(t: str):
        if n_steps != 1:
            return None
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            return None
        obj = json.loads(m.group())
        if isinstance(obj, dict):
            criteria = _parse_criteria(obj)
            while len(criteria) < K:
                criteria.append("")
            criteria = criteria[:K]
            scores = _parse_scores(obj)
            return _project_scores_to_reward(
                [{"criteria": criteria, "scores": scores}], fallback_score, K=K
            )
        return None

    # Try parsing as-is first, then retry with LaTeX backslash fix
    for t in (text, _fix_latex_backslashes(text)):
        for parser in (_try_parse_array, _try_parse_single):
            try:
                result = parser(t)
                if result is not None:
                    return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    print(f"  [rubric] parse error on trajectory output, using fallback score {fallback_score:.3f}")
    print(f"  [rubric] raw trajectory output:\n{text}")
    per_step = fallback_score / n_steps if n_steps > 0 else 0.0
    fallback = [{"criteria": [""] * K, "scores": [per_step] * K,
                 "parse_error": True}
                for _ in range(n_steps)]
    return _project_scores_to_reward(fallback, fallback_score, K=K)
