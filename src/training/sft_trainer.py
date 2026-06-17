"""SFT training with LoRA for π and μ.

Usage:
  train_sft(cfg_path, role="pi")   # or role="mu"

Both share the same loop; μ uses identical LM loss over the full output
(criteria JSON), while score_head regresses the scalar score separately.
"""

from pathlib import Path
from functools import partial

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

from .sft_data_builder import ChatSFTDataset, collate_sft
from ..utils.misc import read_jsonl, load_yaml, write_jsonl


def load_model_and_tokenizer(model_path: str, lora_cfg: dict,
                              use_lora: bool = True,
                              use_4bit: bool = False, device: str = "auto"):
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    device_map = {"": device} if device != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.gradient_checkpointing_enable()

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"[SFT] Full fine-tune: {total/1e9:.2f}B parameters")

    return model, tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def causal_lm_loss_or_zero(logits: torch.Tensor, labels: torch.Tensor,
                           ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy that returns a graph-connected zero when no labels are valid."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    valid = flat_labels.ne(ignore_index)
    if not torch.any(valid):
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits[valid], flat_labels[valid])


def train_sft(cfg: dict, train_data: list[dict], val_data: list[dict] | None,
              output_dir: str | Path, model_wrapper=None,
              pre_eval_callback=None, epoch_eval_callback=None,
              score_only: bool = False) -> None:
    """Generic SFT training loop for π or μ.

    Parameters
    ----------
    model_wrapper : callable, optional
        If provided, called as model_wrapper(model) after loading to wrap
        the model (e.g. with DualHeadMu).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    set_seed(train_cfg.get("seed", 42))
    print(f"[SFT] seed={train_cfg.get('seed', 42)}")

    model, tokenizer = load_model_and_tokenizer(
        model_cfg["base_model_path"],
        model_cfg.get("lora", {}),
        use_lora=model_cfg.get("use_lora", True),
        use_4bit=model_cfg.get("use_4bit", False),
        device=model_cfg.get("device", "auto"),
    )

    if model_wrapper is not None:
        model = model_wrapper(model)

    device = next(model.parameters()).device

    max_seq = cfg["data"].get("max_seq_length", 2048)
    train_ds = ChatSFTDataset(train_data, tokenizer, max_seq, score_only=score_only)
    val_ds = ChatSFTDataset(val_data, tokenizer, max_seq, score_only=score_only) if val_data else None

    collate_fn = partial(collate_sft, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

    # Separate optimizer for score_head if present (needs higher lr)
    score_loss_coef = train_cfg.get("score_loss_coef", 0.0)
    score_head_lr = train_cfg.get("score_head_lr", train_cfg["lr"])
    _has_score_head = hasattr(model, "score_head") and hasattr(model, "forward_score")
    if _has_score_head and score_loss_coef > 0:
        score_head_params = list(model.score_head.parameters())
        score_head_ids = {id(p) for p in score_head_params}
        backbone_params = [p for p in model.parameters()
                           if id(p) not in score_head_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": train_cfg["lr"]},
            {"params": score_head_params, "lr": score_head_lr},
        ], weight_decay=train_cfg.get("weight_decay", 0.01))
        print(f"[SFT] backbone lr={train_cfg['lr']}, score_head lr={score_head_lr}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
    total_steps = len(train_loader) * train_cfg["epochs"] // train_cfg.get("grad_accum", 1)
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    grad_accum = train_cfg.get("grad_accum", 1)
    has_score_head = hasattr(model, "forward_score")
    if has_score_head and score_loss_coef > 0:
        print(f"[SFT] Dual-head mode: score_loss_coef={score_loss_coef}")
    global_step = 0
    recent_losses = []
    best_metric = -1.0
    best_epoch = -1

    if pre_eval_callback is not None:
        pre_eval_callback(model, tokenizer)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{train_cfg['epochs']}", dynamic_ncols=True)
        for batch_idx, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Dual-head: single forward pass for both text loss and score loss
            if has_score_head and score_loss_coef > 0 and "scores" in batch:
                pred_scores, outputs = model.forward_score(input_ids, attention_mask)
                if score_only:
                    text_loss = torch.zeros((), device=device, dtype=pred_scores.dtype)
                else:
                    text_loss = causal_lm_loss_or_zero(outputs.logits, labels)
                score_loss = F.mse_loss(pred_scores, batch["scores"].to(device))
                loss = text_loss + score_loss_coef * score_loss
                _last_text_loss = text_loss.item()
                _last_score_loss = score_loss.item()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = causal_lm_loss_or_zero(outputs.logits, labels)

            loss = loss / grad_accum
            loss.backward()
            recent_losses.append(loss.item() * grad_accum)
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = sum(recent_losses[-grad_accum:]) / len(recent_losses[-grad_accum:])
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                }
                if has_score_head and score_loss_coef > 0:
                    postfix["txt"] = f"{_last_text_loss:.3f}"
                    postfix["scr"] = f"{_last_score_loss:.4f}"
                pbar.set_postfix(postfix)

        train_loss = sum(recent_losses) / len(recent_losses)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="  val", dynamic_ncols=True, leave=False):
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labs = batch["labels"].to(device)
                    if has_score_head and score_loss_coef > 0 and "scores" in batch:
                        ps, vout = model.forward_score(ids, mask)
                        if score_only:
                            tl = torch.zeros((), device=device, dtype=ps.dtype)
                        else:
                            tl = causal_lm_loss_or_zero(vout.logits, labs)
                        scl = F.mse_loss(ps, batch["scores"].to(device))
                        val_loss += (tl + score_loss_coef * scl).item()
                    else:
                        out = model(input_ids=ids, attention_mask=mask)
                        val_loss += causal_lm_loss_or_zero(out.logits, labs).item()
            val_loss /= len(val_loader)
            print(f"[SFT] Epoch {epoch+1}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        else:
            print(f"[SFT] Epoch {epoch+1}  train_loss={train_loss:.4f}")
        recent_losses.clear()

        # Eval and track best model
        epoch_metric = None
        if epoch_eval_callback is not None:
            eval_results = epoch_eval_callback(model, tokenizer, epoch + 1)
            # eval_results: {split: {"exact_match": float, ...}} or single dict
            if isinstance(eval_results, dict):
                if "dev" in eval_results:
                    epoch_metric = eval_results["dev"].get("exact_match")
                elif "exact_match" in eval_results:
                    epoch_metric = eval_results["exact_match"]

        if epoch_metric is not None and epoch_metric > best_metric:
            best_metric = epoch_metric
            best_epoch = epoch + 1
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
            print(f"[SFT] New best at epoch {best_epoch}: EM={best_metric:.4f}")

    # Final: copy best if available, otherwise save last epoch
    import shutil
    best_dir = output_dir / "best"
    final_dir = output_dir / "final"
    if best_dir.exists():
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(best_dir, final_dir)
        print(f"[SFT] Saved best model (epoch {best_epoch}, EM={best_metric:.4f}) "
              f"as {final_dir} (kept {best_dir} as well)")
    else:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"[SFT] No eval metric tracked; saved last epoch as {final_dir}")
