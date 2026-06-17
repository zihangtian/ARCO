"""Warmup data collection.

Three modes:

  "api"      OpenAI-compatible API model as π, GPT as μ.
             K trajectories per example (k_samples), concurrent via threads.

  "local_pi" Local HuggingFace model as π (sampled), GPT as μ.
             K trajectories per example (k_samples).  Rollout is batched:
             at each step all active examples are forwarded together.
             μ annotation is still threaded GPT calls.

  "vllm_pi"  Local model served by vLLM (OpenAI-compatible API) as π,
             GPT as μ.  K trajectories per example, concurrent via threads.

Output record format (same for all modes):
{
  "example_id": str,
  "sample_idx": int,       # 0..K-1
  "question":   str,
  "gold_answer": str,
  "reward":     float,
  "reward_info": dict,
  "submitted_answer": str | None,
  "steps": [
    {"step": int, "action": str,
     "observation": str, "criteria": list[str], "scores": list[float]},
    ...
  ]
}
"""

import json
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

from ..agents.policy import PolicyAgent
from ..agents.rubric import RubricAgent
from ..environments.registry import get_dataset
from ..utils.prompt_builders import format_history_from_action_history, policy_fields_from_state
from ..utils.misc import read_jsonl


def _existing_sample_map(records: list[dict]) -> dict[str, set[int]]:
    sample_map: dict[str, set[int]] = defaultdict(set)
    for r in records:
        sample_map[r["example_id"]].add(int(r.get("sample_idx", 0)))
    return sample_map


# ── API mode ──────────────────────────────────────────────────────────────────

def _process_one_api(
    example: dict,
    policy: PolicyAgent,
    rubric_agent: RubricAgent,
    max_steps: int,
    retriever,
    dataset: str = "hotpotqa",
    sample_idx: int = 0,
    debug: bool = False,
) -> dict:
    """Single example, API π (greedy). Safe to call from any thread."""
    spec = get_dataset(dataset)
    env = spec.make_env(example, retriever=retriever)
    raw_steps = []
    steps_used = 0

    while not env.done:
        force_finish = spec.has_forced_finish and steps_used >= max_steps
        if steps_used >= max_steps and not spec.has_forced_finish:
            break
        steps_remaining = max_steps - steps_used
        history_before = list(env.state.action_history)
        admissible_before = list(getattr(env.state, "admissible_commands", []) or [])
        current_observation = history_before[-1]["observation"] if history_before else env.state.context
        if force_finish:
            action, model_output, messages, valid_parse = policy.act_finish_with_trace_safe(env.state)
        else:
            action, model_output, messages, valid_parse = policy.act_with_trace_safe(
                env.state, steps_remaining)
        if not valid_parse:
            observation = (
                f"Invalid action: {action}. "
                "Action parsing error: output did not match the required format. "
                "No retrieval was executed."
            )
            env.record_invalid_action(action, observation, done=force_finish)
        elif force_finish and not action.startswith("Finish"):
            observation = (
                f"Forced finish violation: {action}. "
                "Search budget exhausted. No answer submitted."
            )
            env.record_invalid_action(action, observation, done=True)
        else:
            try:
                observation = env.step(action)
            except ValueError as e:
                valid_parse = False
                observation = (
                    f"Invalid action: {action}. "
                    "Action execution error: invalid action format. "
                    "No retrieval was executed."
                )
                env.record_invalid_action(action, observation, done=force_finish)
        step_record = {
            "step": len(raw_steps),
            "action": action,
            "env_action": action,
            "action_display": action,
            "observation": observation,
            "history_before": history_before,
            "current_observation": current_observation,
            "admissible_commands_before": admissible_before,
            "searches_remaining_before": max(steps_remaining, 0),
            "forced_finish": force_finish,
            "valid_parse": valid_parse,
            "model_output": model_output,
            "prompt_messages": messages,
            "env_done": env.state.done,
            "env_won": getattr(env.state, "won", False),
            "env_score": getattr(env.state, "score", None),
            "env_max_score": getattr(env.state, "max_score", None),
        }
        raw_steps.append(step_record)
        if debug:
            print(f"[debug] id={example['_id']} sample={sample_idx} step={step_record['step']}")
            if messages:
                print(f"[debug] system_prompt:\n{messages[0]['content']}")
                if len(messages) > 1:
                    print(f"[debug] user_prompt:\n{messages[1]['content']}")
            print(f"[debug] model_output: {model_output!r}")
            print(f"[debug] parsed_action: {action}")
            print(f"[debug] env_observation:\n{observation}")
            print(f"[debug] env_status: done={env.state.done} won={getattr(env.state, 'won', False)} "
                  f"score={getattr(env.state, 'score', None)}/{getattr(env.state, 'max_score', None)}")
        if not force_finish:
            steps_used += spec.step_cost(action) if valid_parse else 1
        if force_finish:
            break

    reward_info = spec.compute_reward(env, example)
    R = reward_info["reward"]

    # For datasets like math where one action = full solution,
    # split into reasoning steps before rubric annotation.
    if (len(raw_steps) == 1 and raw_steps[0].get("valid_parse", True)
            and "\n" in raw_steps[0]["action"]):
        import re
        full_action = raw_steps[0]["action"]
        # Split by --- separators or markdown step headers
        parts = re.split(r"\n-{3,}\n|\n(?=#{1,3}\s+Step\s|\*\*Step\s)", full_action)
        parts = [p.strip() for p in parts if p.strip()]
        # Fallback 1: split by double newline (paragraphs)
        if len(parts) <= 1:
            parts = [p.strip() for p in full_action.split("\n\n") if p.strip()]
        # Fallback 2: split by single newline (lines, skip empty)
        if len(parts) <= 1:
            parts = [l.strip() for l in full_action.split("\n") if l.strip()]
        if len(parts) > 1:
            expanded = []
            for i, part in enumerate(parts):
                expanded.append({
                    **raw_steps[0],
                    "step": i,
                    "action": part,
                    "action_display": part,
                    "observation": "",
                })
            raw_steps = expanded

    annotations = rubric_agent.evaluate_trajectory(
        question=env.state.question, steps=raw_steps, reward=R, use_cache=True,
    )
    steps = [{**s, "criteria": a["criteria"], "scores": a["scores"]}
             for s, a in zip(raw_steps, annotations)]

    return {
        "example_id": example["_id"],
        "sample_idx": sample_idx,
        "question": env.state.question,
        "gold_answer": example.get("answer", "success"),
        "q_type": example.get("type", ""),
        "level": example.get("level", ""),
        "reward": R,
        "reward_info": reward_info,
        "submitted_answer": getattr(env.state, "submitted_answer", None),
        "steps": steps,
    }


# ── Local π mode ──────────────────────────────────────────────────────────────

def _build_pi_text(state, searches_remaining, pi_system, pi_user_tmpl,
                   pi_finish_tmpl, tokenizer, dataset: str):
    fields = policy_fields_from_state(state, dataset, searches_remaining)
    if searches_remaining <= 0 and pi_finish_tmpl:
        user = pi_finish_tmpl.format(**fields)
    else:
        user = pi_user_tmpl.format(**fields)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": pi_system},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def _rollout_local_batch(
    examples: list[dict],
    model,
    tokenizer,
    max_steps: int,
    pi_system: str,
    pi_user_tmpl: str,
    pi_finish_tmpl: str,
    sample_temperature: float,
    pi_max_new_tokens: int,
    retriever,
    dataset: str = "hotpotqa",
) -> list[dict]:
    """Batched rollout with a local HuggingFace model.

    Returns a list of partial records (no rubric annotation yet):
      {"example_id", "sample_idx", "question", "gold_answer", ...,
       "reward", "reward_info", "submitted_answer", "raw_steps": [...]}
    """
    n = len(examples)
    spec = get_dataset(dataset)
    envs = [spec.make_env(ex, retriever=retriever) for ex in examples]
    active = [True] * n
    steps_used = [0] * n
    all_steps: list[list] = [[] for _ in range(n)]

    orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    while True:
        active_idx = [i for i in range(n) if active[i] and not envs[i].done]
        if not active_idx:
            break

        finish_idx = {
            i for i in active_idx
            if spec.has_forced_finish and steps_used[i] >= max_steps
        }
        pre_states = {i: envs[i].state for i in active_idx}
        pi_texts = [
            _build_pi_text(pre_states[i], max_steps - steps_used[i],
                           pi_system, pi_user_tmpl, pi_finish_tmpl, tokenizer, dataset)
            for i in active_idx
        ]
        enc = tokenizer(pi_texts, return_tensors="pt", padding=True).to(model.device)
        padded_plen = enc.input_ids.shape[1]

        out = model.generate(
            **enc,
            max_new_tokens=pi_max_new_tokens,
            do_sample=(sample_temperature > 0),
            temperature=sample_temperature if sample_temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

        for j, i in enumerate(active_idx):
            resp_ids = out[j][padded_plen:]
            resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            try:
                _, action = spec.parse_output(resp_text)
            except ValueError:
                action = resp_text.strip() or "<EMPTY>"
                observation = (
                    f"Invalid action: {action}. "
                    "Action parsing error: output did not match the required format. "
                    "No retrieval was executed."
                )
                envs[i].record_invalid_action(action, observation, done=i in finish_idx)
                all_steps[i].append({
                    "step": len(all_steps[i]),
                    "action": action,
                    "action_display": action,
                    "observation": observation,
                    "searches_remaining_before": max(max_steps - steps_used[i], 0),
                    "forced_finish": i in finish_idx,
                    "valid_parse": False,
                    "model_output": resp_text,
                })
                if i not in finish_idx:
                    steps_used[i] += 1
                if i in finish_idx or (
                    steps_used[i] >= max_steps and not spec.has_forced_finish
                ):
                    active[i] = False
                continue

            step_valid_parse = True
            if i in finish_idx and not action.startswith("Finish"):
                observation = (
                    f"Forced finish violation: {action}. "
                    "Search budget exhausted. No answer submitted."
                )
                envs[i].record_invalid_action(action, observation, done=True)
            else:
                try:
                    observation = envs[i].step(action)
                except ValueError as e:
                    step_valid_parse = False
                    observation = (
                        f"Invalid action: {action}. "
                        "Action execution error: invalid action format. "
                        "No retrieval was executed."
                    )
                    envs[i].record_invalid_action(action, observation, done=i in finish_idx)
            all_steps[i].append({
                "step": len(all_steps[i]),
                "action": action,
                "observation": observation,
                "searches_remaining_before": max(max_steps - steps_used[i], 0),
                "forced_finish": i in finish_idx,
                "valid_parse": step_valid_parse,
                "model_output": resp_text,
            })
            if i not in finish_idx:
                steps_used[i] += spec.step_cost(action) if step_valid_parse else 1
            if i in finish_idx or (
                steps_used[i] >= max_steps and not spec.has_forced_finish
            ):
                active[i] = False

    tokenizer.padding_side = orig_padding

    results = []
    for i, example in enumerate(examples):
        reward_info = spec.compute_reward(envs[i], example)
        results.append({
            "example_id": example["_id"],
            "question": envs[i].state.question,
            "gold_answer": example.get("answer", "success"),
            "q_type": example.get("type", ""),
            "level": example.get("level", ""),
            "reward": reward_info["reward"],
            "reward_info": reward_info,
            "submitted_answer": getattr(envs[i].state, "submitted_answer", None),
            "raw_steps": all_steps[i],
        })
    return results


def _annotate_with_rubric(partial: dict, rubric_agent: RubricAgent,
                          sample_idx: int) -> dict:
    """Call GPT-μ to annotate a trajectory. Thread-safe."""
    raw_steps = partial["raw_steps"]

    # For datasets like math where one action = full solution,
    # split into per-line steps before rubric annotation.
    if (len(raw_steps) == 1 and raw_steps[0].get("valid_parse", True)
            and "\n" in raw_steps[0]["action"]):
        full_action = raw_steps[0]["action"]
        lines = [l.strip() for l in full_action.split("\n") if l.strip()]
        raw_steps = [{**raw_steps[0], "step": i, "action": line,
                      "action_display": line, "observation": ""}
                     for i, line in enumerate(lines)]

    R = partial["reward"]
    annotations = rubric_agent.evaluate_trajectory(
        question=partial["question"],
        steps=raw_steps,
        reward=R,
        use_cache=True,
    )
    steps = [{**s, "criteria": a["criteria"], "scores": a["scores"]}
             for s, a in zip(raw_steps, annotations)]
    return {**{k: v for k, v in partial.items() if k != "raw_steps"},
            "sample_idx": sample_idx,
            "steps": steps}


# ── Main entry point ──────────────────────────────────────────────────────────

def collect_warmup_data(
    examples: list[dict],
    rubric_agent: RubricAgent,
    max_steps: int,
    output_path: str | Path,
    dataset: str = "hotpotqa",
    # api mode
    policy: PolicyAgent | None = None,
    # vllm_pi mode (PolicyAgent backed by vLLM server)
    pi_policy: PolicyAgent | None = None,
    # local_pi mode
    pi_model=None,
    pi_tokenizer=None,
    pi_system: str = "",
    pi_user_tmpl: str = "",
    pi_finish_tmpl: str = "",
    sample_temperature: float = 0.8,
    pi_max_new_tokens: int = 128,
    k_samples: int = 3,
    pi_batch_size: int = 32,
    # shared
    retriever=None,
    resume: bool = True,
    n_workers: int = 8,
    mode: str = "api",
    debug: bool = False,
) -> list[dict]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    if resume and output_path.exists():
        records = read_jsonl(output_path)
        print(f"[warmup] Resuming — {len(records)} records already done.")

    if mode == "api":
        sample_map = _existing_sample_map(records)
        todo = [ex for ex in examples if len(sample_map.get(ex["_id"], set())) < k_samples]
        total_missing = sum(k_samples - len(sample_map.get(ex["_id"], set())) for ex in todo)
        print(f"[warmup] api mode — {len(todo)} examples, {total_missing} trajectories remaining, "
              f"{n_workers} workers.")
        _collect_api(todo, policy, rubric_agent, max_steps, retriever,
                     k_samples, n_workers, records, output_path, sample_map, dataset, debug)

    elif mode == "local_pi":
        sample_map = _existing_sample_map(records)
        todo = [ex for ex in examples if len(sample_map.get(ex["_id"], set())) < k_samples]
        total_missing = sum(k_samples - len(sample_map.get(ex["_id"], set())) for ex in todo)
        print(f"[warmup] local_pi mode — {len(todo)} examples, {total_missing} trajectories remaining.")
        _collect_local(todo, pi_model, pi_tokenizer, pi_system, pi_user_tmpl,
                       pi_finish_tmpl, sample_temperature, pi_max_new_tokens,
                       k_samples, pi_batch_size, rubric_agent, max_steps, retriever,
                       n_workers, records, output_path, sample_map, dataset)

    elif mode == "vllm_pi":
        sample_map = _existing_sample_map(records)
        todo = [ex for ex in examples if len(sample_map.get(ex["_id"], set())) < k_samples]
        total_missing = sum(k_samples - len(sample_map.get(ex["_id"], set())) for ex in todo)
        print(f"[warmup] vllm_pi mode — {len(todo)} examples, {total_missing} trajectories remaining, "
              f"{n_workers} workers.")
        _collect_vllm(todo, pi_policy, rubric_agent, max_steps, retriever,
                      k_samples, n_workers,
                      records, output_path, sample_map, dataset, debug)

    else:
        raise ValueError(f"Unknown warmup mode: {mode!r}  (expected 'api', 'local_pi', 'vllm_pi')")

    print(f"[warmup] Done. {len(records)} total records saved to {output_path}")
    return records


# ── api mode implementation ───────────────────────────────────────────────────

def _collect_api(todo, policy, rubric_agent, max_steps, retriever,
                 k_samples, n_workers, records, output_path, sample_map, dataset, debug):
    if not todo:
        return
    file_lock = threading.Lock()
    total = sum(k_samples - len(sample_map.get(ex["_id"], set())) for ex in todo)
    counter = {"done": 0, "total": total}

    def _append(record):
        with file_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            counter["done"] += 1
            print(f"[warmup] {counter['done']}/{counter['total']}  "
                  f"id={record['example_id']}  R={record['reward']:.2f}  "
                  f"steps={len(record['steps'])}")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for ex in todo:
            existing = sample_map.get(ex["_id"], set())
            for k in range(k_samples):
                if k in existing:
                    continue
                f = pool.submit(
                    _process_one_api, ex, policy, rubric_agent, max_steps,
                    retriever, dataset, k, debug,
                )
                futures[f] = (ex["_id"], k)
        for future in as_completed(futures):
            ex_id, k = futures[future]
            try:
                _append(future.result())
            except Exception as e:
                print(f"[warmup] ERROR id={ex_id} sample={k}: {e}")
                traceback.print_exc()


# ── local_pi mode implementation ──────────────────────────────────────────────

def _collect_local(todo, model, tokenizer, pi_system, pi_user_tmpl, pi_finish_tmpl,
                   sample_temperature, pi_max_new_tokens, k_samples, pi_batch_size,
                   rubric_agent, max_steps, retriever, n_workers, records, output_path,
                   sample_map, dataset):
    if not todo:
        return

    file_lock = threading.Lock()
    counter = {"done": 0}

    def _append(record):
        with file_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            counter["done"] += 1
            print(f"[warmup] {counter['done']}  id={record['example_id']}  "
                  f"sample={record['sample_idx']}  R={record['reward']:.2f}  "
                  f"steps={len(record['steps'])}")

    # Build full list of (example, sample_idx) to run
    expanded = []  # (example, sample_idx)
    for ex in todo:
        existing = sample_map.get(ex["_id"], set())
        for k in range(k_samples):
            if k not in existing:
                expanded.append((ex, k))

    total = len(expanded)
    print(f"[warmup] Phase 1 — GPU rollout: {total} trajectories "
          f"in batches of {pi_batch_size}.")

    # Phase 1: GPU rollout — all batches, no annotation yet
    all_partials = []  # list of (partial_dict, sample_idx)
    for batch_start in range(0, total, pi_batch_size):
        batch = expanded[batch_start: batch_start + pi_batch_size]
        batch_examples = [ex for ex, _ in batch]
        sample_idxs = [k for _, k in batch]
        print(f"[warmup]   rollout {batch_start+1}–{batch_start+len(batch)}/{total}")
        partials = _rollout_local_batch(
            batch_examples, model, tokenizer, max_steps,
            pi_system, pi_user_tmpl, pi_finish_tmpl,
            sample_temperature, pi_max_new_tokens, retriever, dataset,
        )
        all_partials.extend(zip(partials, sample_idxs))

    # Phase 2: Annotate all with GPT-μ in parallel
    print(f"[warmup] Phase 2 — GPT-μ annotation: {len(all_partials)} trajectories "
          f"with {n_workers} workers.")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_annotate_with_rubric, p, rubric_agent, k): i
            for i, (p, k) in enumerate(all_partials)
        }
        for future in as_completed(futures):
            try:
                _append(future.result())
            except Exception as e:
                print(f"[warmup] annotation ERROR: {e}")
                traceback.print_exc()


# ── vllm_pi mode implementation ───────────────────────────────────────────────

def _collect_vllm(todo, pi_policy, rubric_agent, max_steps, retriever,
                  k_samples, n_workers,
                  records, output_path, sample_map, dataset, debug):
    """vLLM-pi mode: PolicyAgent backed by vLLM server, k_samples per question.

    Identical logic to _collect_api but submits k_samples jobs per example.
    All rollout + annotation tasks run concurrently in the thread pool.
    """
    if not todo:
        return

    file_lock = threading.Lock()
    counter = {"done": 0}
    total = sum(k_samples - len(sample_map.get(ex["_id"], set())) for ex in todo)

    def _append(record):
        with file_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            counter["done"] += 1
            print(f"[warmup] {counter['done']}/{total}  "
                  f"id={record['example_id']}  sample={record['sample_idx']}  "
                  f"R={record['reward']:.2f}  steps={len(record['steps'])}")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for ex in todo:
            existing = sample_map.get(ex["_id"], set())
            for k in range(k_samples):
                if k in existing:
                    continue
                f = pool.submit(
                    _process_one_api, ex, pi_policy, rubric_agent,
                    max_steps, retriever, dataset, k, debug,
                )
                futures[f] = (ex["_id"], k)

        for future in as_completed(futures):
            ex_id, k = futures[future]
            try:
                _append(future.result())
            except Exception as e:
                print(f"[warmup] ERROR id={ex_id} sample={k}: {e}")
                traceback.print_exc()
