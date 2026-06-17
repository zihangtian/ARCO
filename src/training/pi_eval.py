"""Greedy local rollout evaluation for policy models."""

import torch
from tqdm import tqdm

from ..environments.registry import get_dataset
from ..utils.prompt_builders import display_action, policy_fields_from_state


def _build_pi_text(state, searches_remaining, pi_system, pi_user_tmpl,
                   pi_finish_tmpl, tokenizer, dataset: str) -> str:
    fields = policy_fields_from_state(state, dataset, searches_remaining)
    if searches_remaining <= 0:
        user = pi_finish_tmpl.format(**fields)
    else:
        user = pi_user_tmpl.format(**fields)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": pi_system},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

@torch.no_grad()
def evaluate_pi_model(
    model,
    tokenizer,
    examples: list[dict],
    retriever,
    max_steps: int,
    pi_system: str,
    pi_user_tmpl: str,
    pi_finish_tmpl: str,
    max_new_tokens: int,
    batch_size: int = 8,
    desc: str = "dev",
    dataset: str = "hotpotqa",
    debug_examples: int = 0,
    debug_steps: int = 50,
    vllm_engine=None,
    trace_out=None,
) -> dict:
    all_results = []
    _trace_fh = None
    if trace_out is not None:
        import json as _json
        from pathlib import Path as _Path
        _Path(trace_out).parent.mkdir(parents=True, exist_ok=True)
        _trace_fh = open(trace_out, "w", encoding="utf-8")
    orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    was_training = model.training
    model.eval()
    spec = get_dataset(dataset)

    pbar = tqdm(range(0, len(examples), batch_size), total=(len(examples) + batch_size - 1) // batch_size,
                desc=desc, dynamic_ncols=True, leave=False)
    for start in pbar:
        batch_ex = examples[start:start + batch_size]
        envs = [spec.make_env(ex, retriever=retriever, max_steps=max_steps) for ex in batch_ex]
        active = [True] * len(batch_ex)
        steps_used = [0] * len(batch_ex)

        while True:
            active_idx = [i for i in range(len(batch_ex)) if active[i] and not envs[i].state.done]
            if not active_idx:
                break
            finish_idx = {
                i for i in active_idx
                if spec.has_forced_finish and steps_used[i] >= max_steps
            }

            texts = [
                _build_pi_text(
                    envs[i].state,
                    max_steps - steps_used[i],
                    pi_system,
                    pi_user_tmpl,
                    pi_finish_tmpl,
                    tokenizer,
                    dataset,
                )
                for i in active_idx
            ]
            if vllm_engine is not None and vllm_engine.llm is not None:
                # vLLM path (greedy)
                vllm_results = vllm_engine.generate_pi(
                    texts, temperature=0.0, max_tokens=max_new_tokens)
                for j, i in enumerate(active_idx):
                    resp_text = vllm_results[j].resp_text
                    try:
                        _, action = spec.parse_output(resp_text)
                    except ValueError:
                        action = resp_text.strip() or "<EMPTY>"
                        obs = (
                            f"Invalid action: {action}. "
                            "Action parsing error: output did not match the required format. "
                            "No retrieval was executed."
                        )
                        envs[i].record_invalid_action(action, obs, done=i in finish_idx)
                        if i not in finish_idx:
                            steps_used[i] += 1
                        if i in finish_idx or (
                            steps_used[i] >= max_steps and not spec.has_forced_finish
                        ):
                            active[i] = False
                        continue
                    if i in finish_idx and not action.startswith("Finish"):
                        obs = (
                            f"Forced finish violation: {action}. "
                            "Search budget exhausted. No answer submitted."
                        )
                        envs[i].record_invalid_action(action, obs, done=True)
                        active[i] = False
                        continue
                    try:
                        envs[i].step(action)
                    except Exception as e:
                        obs = (
                            f"Invalid action: {action}. "
                            "Action execution error: invalid action format. "
                            "No retrieval was executed."
                        )
                        envs[i].record_invalid_action(action, obs, done=i in finish_idx)
                        if i not in finish_idx:
                            steps_used[i] += 1
                        if i in finish_idx or (
                            steps_used[i] >= max_steps and not spec.has_forced_finish
                        ):
                            active[i] = False
                        continue
                    steps_used[i] += spec.step_cost(action)
                    if i in finish_idx or (
                        steps_used[i] >= max_steps and not spec.has_forced_finish
                    ):
                        active[i] = False
            else:
                enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
                prompt_len = enc.input_ids.shape[1]
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )

                for j, i in enumerate(active_idx):
                    resp_ids = out[j][prompt_len:]
                    resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                    try:
                        _, action = spec.parse_output(resp_text)
                    except ValueError:
                        action = resp_text.strip() or "<EMPTY>"
                        obs = (
                            f"Invalid action: {action}. "
                            "Action parsing error: output did not match the required format. "
                            "No retrieval was executed."
                        )
                        envs[i].record_invalid_action(action, obs, done=i in finish_idx)
                        if i not in finish_idx:
                            steps_used[i] += 1
                        if i in finish_idx or (
                            steps_used[i] >= max_steps and not spec.has_forced_finish
                        ):
                            active[i] = False
                        continue
                    if i in finish_idx and not action.startswith("Finish"):
                        obs = (
                            f"Forced finish violation: {action}. "
                            "Search budget exhausted. No answer submitted."
                        )
                        envs[i].record_invalid_action(action, obs, done=True)
                        active[i] = False
                        continue
                    try:
                        envs[i].step(action)
                    except Exception as e:
                        obs = (
                            f"Invalid action: {action}. "
                            "Action execution error: invalid action format. "
                            "No retrieval was executed."
                        )
                        envs[i].record_invalid_action(action, obs, done=i in finish_idx)
                        if i not in finish_idx:
                            steps_used[i] += 1
                        if i in finish_idx or (
                            steps_used[i] >= max_steps and not spec.has_forced_finish
                        ):
                            active[i] = False
                        continue
                    steps_used[i] += spec.step_cost(action)
                    if i in finish_idx or (
                        steps_used[i] >= max_steps and not spec.has_forced_finish
                    ):
                        active[i] = False

        batch_results = []
        for i, ex in enumerate(batch_ex):
            reward_info = spec.compute_reward(envs[i], ex)
            global_idx = start + i
            if global_idx < debug_examples:
                print(f"[eval-debug] split={desc} idx={global_idx} id={ex.get('_id', ex.get('example_id', 'unknown'))}")
                for step_idx, step in enumerate(envs[i].state.action_history[:debug_steps]):
                    print(f"[eval-debug] step={step_idx} command={display_action(step['action'], raw_actions=spec.display_raw_actions)}")
                    print(f"[eval-debug] obs={step['observation']}")
                print(f"[eval-debug] reward_info={reward_info}")
            batch_results.append(reward_info)
            all_results.append(reward_info)
            if _trace_fh is not None:
                ex_id = ex.get("_id", ex.get("example_id", str(global_idx)))
                question = ex.get("question", "")
                gold = ex.get("answer", ex.get("gold_answer", ""))
                hist = envs[i].state.action_history
                running = []
                for step_idx, step in enumerate(hist):
                    act = display_action(step["action"], raw_actions=spec.display_raw_actions)
                    history_text = "\n".join(running) if running else "(none)"
                    _trace_fh.write(_json.dumps({
                        "example_id": ex_id,
                        "question": question,
                        "gold_answer": gold,
                        "step_idx": step_idx,
                        "action": act,
                        "observation": step.get("observation", ""),
                        "history": history_text,
                    }, ensure_ascii=False) + "\n")
                    running.append(f"Action: {act}\nObservation: {step.get('observation','')}")
        batch_em = sum(1 for r in batch_results if r["reward"] >= 1.0) / len(batch_results)
        pbar.set_postfix({"EM": f"{batch_em:.3f}"})

    tokenizer.padding_side = orig_padding
    if was_training:
        model.train()
    torch.cuda.empty_cache()
    if _trace_fh is not None:
        _trace_fh.close()

    n_correct = sum(1 for r in all_results if r["reward"] >= 1.0)
    exact_match = n_correct / len(all_results)
    avg_f1 = sum(r.get("f1", r["reward"]) for r in all_results) / len(all_results)
    print(f"[eval] {desc}  n={len(all_results)}  reward: EM={exact_match:.4f}  "
          f"F1={avg_f1:.4f}  ({n_correct}/{len(all_results)} correct)")
    return {"exact_match": exact_match, "f1": avg_f1}
