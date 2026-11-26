"""Unified supervised (B1/B2/B3/B4) and SSL-as-RL (I1/I2) training & CLI.

Usage examples:
    # Supervised NTP baseline (B1 / B2)
    python -m phase1_pilot.train --mode supervised --preset micro-small --variant ntp --max-steps 100

    # Continuous latent BPTT (B4)
    python -m phase1_pilot.train --mode supervised --preset micro-small --variant continuous_latent --max-steps 100

    # SSL-as-RL (Dr. GRPO + DAPO Clip-Higher + Frozen-Anchor R3_brier)
    python -m phase1_pilot.train --mode rl --preset micro-small --reward-type R3_brier --filter-type T2_self --max-steps 50
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from phase1_pilot.config import (
    CorpusConfig,
    DataConfig,
    ModelConfig,
    RLConfig,
    TrainConfig,
    VALID_FILTERS,
    VALID_INTERFACES,
    VALID_REWARDS,
    VALID_VARIANTS,
)
from phase1_pilot.data.tokenizer import PILOT_TOKENIZER_SPEC
from phase1_pilot.rl.grpo import (
    SSLRLTrainer,
    _all_reduce_model_grads,
    _broadcast_model_state,
    _extract_logits,
)


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_distributed(device: torch.device) -> bool:
    """Initialize torch.distributed when launched via torchrun (WORLD_SIZE > 1).

    Returns True if this call initialized the process group (so caller should
    call ``cleanup_distributed(True)`` on exit).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not dist.is_available() or dist.is_initialized():
        return False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True


def cleanup_distributed(did_init: bool) -> None:
    """Tear down torch.distributed cleanly if initialized by this process."""
    if did_init and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve 'auto' to CUDA (respecting LOCAL_RANK if set) or CPU."""
    if device == "auto":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Set deterministic RNG seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_lr_lambda(step: int, max_steps: int, warmup_frac: float = 0.05, min_ratio: float = 0.1) -> float:
    """Linear warmup followed by cosine decay to ``min_ratio`` of peak LR."""
    warmup_steps = max(1, int(max_steps * warmup_frac))
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _normalize_supervised_batch(
    batch: Any, device: torch.device, pad_id: int = 0
) -> dict[str, Tensor]:
    """Convert dict or tuple batch into a standard dict on ``device``."""
    if isinstance(batch, dict):
        out: dict[str, Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, Tensor):
                out[k] = v.to(device)
        if "labels" not in out and "input_ids" in out:
            lbl = out["input_ids"].clone()
            lbl[lbl == pad_id] = -100
            out["labels"] = lbl
        return out
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        input_ids = batch[0].to(device).long()
        labels = batch[1].to(device).long()
        return {"input_ids": input_ids, "labels": labels}
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _compute_supervised_loss(
    model: nn.Module,
    batch: dict[str, Tensor],
    pad_id: int = 0,
) -> Tensor:
    """Compute next-token cross-entropy loss across LlamaLM, PauseTokenWrapper, and ContinuousLatentWrapper."""
    input_ids = batch["input_ids"].long()
    labels = batch.get("labels", input_ids.clone()).long()
    attention_mask = batch.get("attention_mask", None)
    prompt_lens = batch.get("prompt_lens", batch.get("prompt_len", None))

    unwrapped = model.module if hasattr(model, "module") else model
    sig = inspect.signature(unwrapped.forward)
    params = sig.parameters

    # Check if wrapper handles pause/latent insertion internally with prompt_lens & labels
    kwargs: dict[str, Any] = {"input_ids": input_ids}
    if "attention_mask" in params and attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if "prompt_lens" in params and prompt_lens is not None:
        kwargs["prompt_lens"] = prompt_lens
    elif "prompt_len" in params and prompt_lens is not None:
        kwargs["prompt_len"] = prompt_lens
    if "labels" in params:
        kwargs["labels"] = labels

    out = model(**kwargs)

    # If the model/wrapper returned a precomputed loss (e.g., masked for pause/latent steps), use it!
    if hasattr(out, "loss") and out.loss is not None:
        return out.loss

    logits = _extract_logits(out)
    # Standard next-token shift when logits length matches input_ids
    if logits.shape[1] == labels.shape[1]:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
    else:
        # Fallback if wrapper prepended/inserted extra tokens without returning loss
        min_len = min(logits.shape[1], labels.shape[1])
        shift_logits = logits[:, -min_len:-1, :].contiguous()
        shift_labels = labels[:, -min_len + 1 :].contiguous()

    if not torch.any(shift_labels != -100):
        return shift_logits.sum() * 0.0

    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate_nll(
    model: nn.Module,
    val_loader: Iterable[Any],
    device: torch.device,
    *,
    pad_id: int = 0,
    bf16: bool = False,
    max_batches: int | None = None,
) -> float:
    """Evaluate mean validation NLL over ``val_loader``."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_batches = 0
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (bf16 and device.type == "cuda")
        else nullcontext()
    )
    try:
        for idx, raw_batch in enumerate(val_loader):
            if max_batches is not None and idx >= max_batches:
                break
            batch = _normalize_supervised_batch(raw_batch, device, pad_id=pad_id)
            with ctx:
                loss = _compute_supervised_loss(model, batch, pad_id=pad_id)
            total_loss += float(loss.item())
            n_batches += 1
    finally:
        if was_training:
            model.train()
    return total_loss / max(1, n_batches)


def _infinite_batches(loader: Iterable[Any]):
    """Yield batches endlessly from a finite DataLoader or list (updating DistributedSampler epochs)."""
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("DataLoader / batch iterable is empty.")
        epoch += 1


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _is_main_process() -> bool:
    return _get_rank() == 0


def train_supervised(
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any] | None = None,
    cfg: TrainConfig | None = None,
    *,
    pad_id: int = 0,
    step_callback: Callable[[int, dict[str, Any]], None] | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a supervised baseline (B1 NTP, B2 Large NTP, B3 Pause, B4 Continuous Latent).

    Supports BF16 autocast on CUDA, torch.distributed gradient averaging,
    cosine LR schedule, gradient clipping, JSONL telemetry logging, and
    checkpoint saving.
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed + _get_rank())
    device = resolve_device(cfg.device)
    model = model.to(device)
    _broadcast_model_state(model, src=0)
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _cosine_lr_lambda(s, cfg.max_steps)
    )

    out_dir = Path(cfg.output_dir) if cfg.output_dir else None
    metrics_path = out_dir / "metrics.jsonl" if out_dir is not None else None
    if metrics_path is not None and _is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            metrics_path.unlink()

    history: list[dict[str, Any]] = []
    batch_iter = _infinite_batches(train_loader)
    model.train()

    for step in range(1, cfg.max_steps + 1):
        raw_batch = next(batch_iter)
        batch = _normalize_supervised_batch(raw_batch, device, pad_id=pad_id)

        optimizer.zero_grad(set_to_none=True)
        ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with ctx:
            loss = _compute_supervised_loss(model, batch, pad_id=pad_id)

        loss.backward()
        _all_reduce_model_grads(model)
        if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        record: dict[str, Any] = {
            "step": step,
            "train_loss": float(loss.detach().item()),
            "lr": float(scheduler.get_last_lr()[0]),
        }

        is_log_step = (cfg.log_every > 0 and step % cfg.log_every == 0) or step == cfg.max_steps
        if is_log_step and val_loader is not None:
            record["val_loss"] = evaluate_nll(
                model, val_loader, device, pad_id=pad_id, bf16=use_bf16
            )

        history.append(record)
        if metrics_path is not None and _is_main_process() and is_log_step:
            _append_jsonl(metrics_path, record)
        if step_callback is not None:
            step_callback(step, record)

    final_val_loss = history[-1].get("val_loss", None)
    if final_val_loss is None and val_loader is not None:
        final_val_loss = evaluate_nll(model, val_loader, device, pad_id=pad_id, bf16=use_bf16)

    ckpt_path: str | None = None
    if out_dir is not None and _is_main_process():
        ckpt_file = out_dir / "checkpoint.pt"
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "train_config": cfg.to_dict(),
                "steps": cfg.max_steps,
                "final_train_loss": history[-1]["train_loss"],
                "final_val_loss": final_val_loss,
                **(checkpoint_extra or {}),
            },
            ckpt_file,
        )
        ckpt_path = str(ckpt_file)

    return {
        "steps": cfg.max_steps,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": final_val_loss,
        "history": history,
        "checkpoint_path": ckpt_path,
    }


def train_ssl_rl(
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any] | None = None,
    train_cfg: TrainConfig | None = None,
    rl_cfg: RLConfig | None = None,
    *,
    teacher_model: nn.Module | None = None,
    pad_id: int = 0,
    think_start_id: int = 3,
    think_end_id: int = 4,
    temperature: float = 1.0,
    continuation_sft_weight: float = 0.1,
    step_callback: Callable[[int, dict[str, Any]], None] | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a model with SSL-as-RL (Dr. GRPO + DAPO Clip-Higher + Frozen Anchor)."""
    train_cfg = train_cfg or TrainConfig()
    rl_cfg = rl_cfg or RLConfig()
    set_seed(train_cfg.seed + _get_rank())
    device = resolve_device(train_cfg.device)
    model = model.to(device)
    _broadcast_model_state(model, src=0)
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        _broadcast_model_state(teacher_model, src=0)
    use_bf16 = bool(train_cfg.bf16) and device.type == "cuda"

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=rl_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _cosine_lr_lambda(s, train_cfg.max_steps)
    )

    trainer = SSLRLTrainer(
        model=model,
        rl_cfg=rl_cfg,
        optimizer=optimizer,
        teacher_model=teacher_model,
        pad_id=pad_id,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        grad_clip=train_cfg.grad_clip,
        temperature=temperature,
        continuation_sft_weight=continuation_sft_weight,
        bf16=use_bf16,
        device=device,
    )

    out_dir = Path(train_cfg.output_dir) if train_cfg.output_dir else None
    metrics_path = out_dir / "metrics.jsonl" if out_dir is not None else None
    if metrics_path is not None and _is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            metrics_path.unlink()

    history: list[dict[str, Any]] = []
    batch_iter = _infinite_batches(train_loader)

    for step in range(1, train_cfg.max_steps + 1):
        raw_batch = next(batch_iter)
        metrics = trainer.train_step(raw_batch)
        scheduler.step()
        metrics["lr"] = float(scheduler.get_last_lr()[0])

        is_log_step = (
            train_cfg.log_every > 0 and step % train_cfg.log_every == 0
        ) or step == train_cfg.max_steps
        if is_log_step and val_loader is not None:
            metrics["val_loss"] = evaluate_nll(
                model, val_loader, device, pad_id=pad_id, bf16=use_bf16
            )

        history.append(metrics)
        if metrics_path is not None and _is_main_process() and is_log_step:
            _append_jsonl(metrics_path, metrics)
        if step_callback is not None:
            step_callback(step, metrics)

    final_val_loss = history[-1].get("val_loss", None)
    if final_val_loss is None and val_loader is not None:
        final_val_loss = evaluate_nll(model, val_loader, device, pad_id=pad_id, bf16=use_bf16)

    ckpt_path: str | None = None
    if out_dir is not None and _is_main_process():
        ckpt_file = out_dir / "checkpoint.pt"
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "train_config": train_cfg.to_dict(),
                "rl_config": rl_cfg.to_dict(),
                "steps": train_cfg.max_steps,
                "final_metrics": history[-1],
                "final_val_loss": final_val_loss,
                **(checkpoint_extra or {}),
            },
            ckpt_file,
        )
        ckpt_path = str(ckpt_file)

    return {
        "steps": train_cfg.max_steps,
        "final_metrics": history[-1],
        "final_val_loss": final_val_loss,
        "history": history,
        "checkpoint_path": ckpt_path,
        "trainer": trainer,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for supervised & SSL-as-RL training."""
    p = argparse.ArgumentParser(
        prog="python -m phase1_pilot.train",
        description="Phase 1 Small-Model Latent-CoT RL Pilot — Supervised & SSL-as-RL Trainer",
    )
    p.add_argument("--mode", choices=("supervised", "rl"), default="supervised")
    p.add_argument("--preset", default="micro-small", choices=("micro-small", "micro-large", "160m", "410m", "1b"))
    p.add_argument("--variant", default="ntp", choices=VALID_VARIANTS)
    p.add_argument("--n-pause-tokens", type=int, default=4)
    p.add_argument("--n-latent-steps", type=int, default=4)
    p.add_argument("--max-thought-len", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--output-dir", default="runs/default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-checkpoint", type=str, default=None)
    p.add_argument("--teacher-checkpoint", type=str, default=None)
    p.add_argument("--teacher-preset", type=str, default="micro-large")

    # RL hyperparameters (Track 1 §5.4)
    p.add_argument("--reward-type", default="R3_brier", choices=VALID_REWARDS)
    p.add_argument("--filter-type", default="T2_self", choices=VALID_FILTERS)
    p.add_argument("--interface", default="I1_discrete", choices=VALID_INTERFACES)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--n-ahead", type=int, default=4)
    p.add_argument("--thought-len", type=int, default=8)
    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument("--clip-high", type=float, default=0.28)
    p.add_argument("--reward-clip-tau", type=float, default=5.0)
    p.add_argument("--lambda-ntp", type=float, default=0.2)
    p.add_argument("--top-p-tokens", type=float, default=0.2)
    p.add_argument("--top-p-thought-entropy", type=float, default=0.2)
    p.add_argument("--ref-ema-decay", type=float, default=1.0)
    p.add_argument("--no-filter-zero-var", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--continuation-sft-weight", type=float, default=0.1)

    # Data hyperparameters
    p.add_argument("--n-train-samples", type=int, default=1000)
    p.add_argument("--n-eval-samples", type=int, default=200)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--regime", choices=("regime1", "regime2", "regime3", "all"), default="regime1")
    p.add_argument(
        "--tokenizer",
        default=PILOT_TOKENIZER_SPEC,
        help="'pilot' (built-in byte+symbol tokenizer) or a HuggingFace tokenizer name or path.",
    )
    p.add_argument(
        "--corpus",
        default=None,
        help="JSON Lines file of real-text documents (string fields key, text, source). "
        "Replaces the synthetic Regime 2 stream in `train` and `eval_natural`.",
    )
    corpus_defaults = CorpusConfig()
    p.add_argument("--corpus-eval-fraction", type=float, default=corpus_defaults.eval_fraction)
    p.add_argument("--corpus-headroom-tokens", type=int, default=corpus_defaults.headroom_tokens)
    return p


def corpus_config_from_args(args: argparse.Namespace) -> CorpusConfig | None:
    """The real-text windowing settings of this run, or `None` for synthetic Regime 2."""
    if args.corpus is None:
        return None
    return CorpusConfig(
        eval_fraction=args.corpus_eval_fraction,
        headroom_tokens=args.corpus_headroom_tokens,
    )


def build_dataloaders(args: argparse.Namespace, tokenizer: Any):
    """Build deterministic DataLoaders from phase1_pilot.data (sharded when distributed).

    Returns `(train_loader, val_loader, corpus_stats)`; `corpus_stats` is `None` unless
    `--corpus` supplies Regime 2.
    """
    from functools import partial

    from torch.utils.data import DataLoader, DistributedSampler

    from phase1_pilot.data import build_stratified_datasets, collate_batch
    from phase1_pilot.data.corpus import build_corpus_splits, read_corpus_jsonl
    from phase1_pilot.data.generators import natural_train_count

    data_cfg = DataConfig(
        seed=args.seed,
        n_train_samples=args.n_train_samples,
        n_eval_samples=args.n_eval_samples,
        max_seq_len=args.max_seq_len,
    )
    natural = None
    if args.corpus is not None:
        natural = build_corpus_splits(
            read_corpus_jsonl(args.corpus),
            tokenizer,
            max_seq_len=args.max_seq_len,
            n_train=natural_train_count(args.n_train_samples),
            n_eval=args.n_eval_samples,
            seed=args.seed,
            cfg=corpus_config_from_args(args),
        )
    datasets = build_stratified_datasets(data_cfg, tokenizer, natural=natural)

    train_ds = datasets["train"]
    if args.regime == "regime2":
        val_ds = datasets["eval_natural"]
    elif args.regime == "regime3":
        val_ds = datasets["eval_aleatoric"]
    else:
        val_ds = datasets["eval_id"]

    collate_fn = partial(collate_batch, pad_id=tokenizer.pad_id)
    world_size = _get_world_size()
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=_get_rank(),
            shuffle=True,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, (natural.stats if natural is not None else None)


def resolve_model_config(args: argparse.Namespace, tokenizer: Any, **overrides: Any) -> ModelConfig:
    """The model this run builds. Its vocabulary is the tokenizer's, for every preset."""
    fields: dict[str, Any] = {
        "variant": args.variant,
        "n_pause_tokens": args.n_pause_tokens,
        "n_latent_steps": args.n_latent_steps,
        "max_thought_len": args.max_thought_len,
        "max_seq_len": args.max_seq_len,
        "vocab_size": tokenizer.vocab_size,
        **overrides,
    }
    return ModelConfig.preset(fields.pop("preset", args.preset), **fields)


def data_protocol(
    tokenizer_spec: str,
    tokenizer: Any,
    corpus_cfg: CorpusConfig | None,
    max_seq_len: int,
) -> dict[str, Any]:
    """What a checkpoint must share with a run that loads it.

    `corpus` is the recipe that turns documents into windows (`None` for synthetic
    Regime 2). Arms that start from one another must use the same recipe: a different
    `eval_fraction` moves training documents into the eval split, and any other
    difference means the arms no longer train or evaluate on the same windows.
    """
    return {
        "tokenizer": tokenizer_spec,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": tokenizer.protocol_ids(),
        "corpus": None if corpus_cfg is None else {**corpus_cfg.to_dict(), "max_seq_len": max_seq_len},
    }


def _load_compatible_state(path: str, protocol: dict[str, Any], role: str) -> dict[str, Tensor]:
    """Load a checkpoint's weights after checking it was trained under `protocol`.

    Weights are only meaningful with the token IDs they were trained on, and an arm
    must not evaluate on corpus documents its starting checkpoint trained on.
    """
    from phase1_pilot.data.tokenizer import PilotTokenizer

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    saved = ckpt.get("data_protocol") if isinstance(ckpt, dict) else None
    if saved is None:
        # Checkpoints written before the protocol was recorded all used PilotTokenizer.
        saved = data_protocol(PILOT_TOKENIZER_SPEC, PilotTokenizer(), None, max_seq_len=0)
    if saved["tokenizer_fingerprint"] != protocol["tokenizer_fingerprint"]:
        raise ValueError(
            f"{role} checkpoint {path} was trained with tokenizer {saved['tokenizer']!r} "
            f"(vocab {saved['vocab_size']}), but this run uses {protocol['tokenizer']!r} "
            f"(vocab {protocol['vocab_size']}): their token IDs differ"
        )
    saved_corpus, corpus = saved.get("corpus"), protocol["corpus"]
    if saved_corpus is not None and corpus is not None and saved_corpus != corpus:
        differing = ", ".join(
            f"{k}: {saved_corpus.get(k)} vs {corpus.get(k)}"
            for k in sorted(set(saved_corpus) | set(corpus))
            if saved_corpus.get(k) != corpus.get(k)
        )
        raise ValueError(
            f"{role} checkpoint {path} was trained on corpus windows cut differently from this "
            f"run's ({differing}). The arms would not share windows, and a different "
            f"eval_fraction lets this run evaluate on documents the checkpoint trained on"
        )
    return ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt


def _describe_corpus(stats: Any) -> str:
    parts = []
    for split in ("train", "eval"):
        sources = ", ".join(
            f"{src} {n} windows / {stats.tokens[split][src]} tokens"
            for src, n in sorted(stats.windows[split].items())
        )
        parts.append(f"{split}: {sources}")
    parts.append(
        f"read {stats.documents_read} documents, skipped {stats.skipped_duplicate} duplicate, "
        f"{stats.skipped_protocol_token} with protocol tokens, {stats.skipped_short} too short"
    )
    return "[corpus] " + " | ".join(parts)


def run_training(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    """Train one arm as `args` describes, with `tokenizer` as the source of IDs and vocab size."""
    device = resolve_device(args.device)
    did_init_dist = init_distributed(device)

    try:
        from phase1_pilot.models import build_model, default_extra_steps_for_config

        corpus_cfg = corpus_config_from_args(args)
        if (
            args.mode == "rl"
            and corpus_cfg is not None
            and args.n_ahead > corpus_cfg.min_target_tokens
        ):
            raise ValueError(
                f"n_ahead={args.n_ahead} exceeds the corpus min_target_tokens="
                f"{corpus_cfg.min_target_tokens}: some windows would be scored on a shorter horizon"
            )

        train_loader, val_loader, corpus_stats = build_dataloaders(args, tokenizer)
        if corpus_stats is not None and _is_main_process():
            print(_describe_corpus(corpus_stats))
        protocol = data_protocol(args.tokenizer, tokenizer, corpus_cfg, args.max_seq_len)

        set_seed(args.seed)
        model_cfg = resolve_model_config(args, tokenizer)
        if corpus_cfg is not None:
            # An upper bound on the positions this arm inserts at the boundary.
            inserted = default_extra_steps_for_config(model_cfg) + (
                args.thought_len + 2 if args.mode == "rl" else 0
            )
            if inserted > corpus_cfg.headroom_tokens:
                raise ValueError(
                    f"this arm can insert {inserted} positions at the boundary, more than the "
                    f"corpus headroom_tokens={corpus_cfg.headroom_tokens}: a full window would "
                    f"exceed max_seq_len={args.max_seq_len}"
                )
        model = build_model(model_cfg, pause_token_id=tokenizer.pause_id)
        checkpoint_extra = {"model_config": model_cfg.to_dict(), "data_protocol": protocol}

        if args.init_checkpoint:
            state = _load_compatible_state(args.init_checkpoint, protocol, role="init")
            model.load_state_dict(state, strict=False)

        train_cfg = TrainConfig(
            preset=args.preset,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            lr=args.lr if args.lr is not None else (1e-3 if args.mode == "supervised" else 3e-4),
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            bf16=args.bf16,
            device=args.device,
            log_every=args.log_every,
            output_dir=args.output_dir,
            seed=args.seed,
        )

        if args.mode == "supervised":
            res = train_supervised(
                model,
                train_loader,
                val_loader,
                train_cfg,
                pad_id=tokenizer.pad_id,
                checkpoint_extra=checkpoint_extra,
            )
            if _is_main_process():
                print(
                    f"[supervised] steps={res['steps']} "
                    f"train_loss={res['final_train_loss']:.4f} "
                    f"val_loss={res['final_val_loss']:.4f}"
                )
            return {**res, "model": model, "corpus_stats": corpus_stats}

        teacher_model = None
        if args.filter_type == "T2_scale":
            teacher_cfg = resolve_model_config(
                args, tokenizer, preset=args.teacher_preset, variant="ntp"
            )
            teacher_model = build_model(teacher_cfg, pause_token_id=tokenizer.pause_id)
            if args.teacher_checkpoint:
                t_state = _load_compatible_state(args.teacher_checkpoint, protocol, role="teacher")
                teacher_model.load_state_dict(t_state, strict=False)

        rl_cfg = RLConfig(
            reward_type=args.reward_type,
            filter_type=args.filter_type,
            interface=args.interface,
            group_size=args.group_size,
            n_ahead=args.n_ahead,
            thought_len=args.thought_len,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            reward_clip_tau=args.reward_clip_tau,
            lambda_ntp=args.lambda_ntp,
            top_p_tokens=args.top_p_tokens,
            top_p_thought_entropy=args.top_p_thought_entropy,
            lr=args.lr if args.lr is not None else 3e-4,
            ref_ema_decay=args.ref_ema_decay,
            filter_zero_var=not args.no_filter_zero_var,
        )
        res = train_ssl_rl(
            model,
            train_loader,
            val_loader,
            train_cfg,
            rl_cfg,
            teacher_model=teacher_model,
            pad_id=tokenizer.pad_id,
            think_start_id=tokenizer.think_start_id,
            think_end_id=tokenizer.think_end_id,
            temperature=args.temperature,
            continuation_sft_weight=args.continuation_sft_weight,
            checkpoint_extra=checkpoint_extra,
        )
        if _is_main_process():
            fm = res["final_metrics"]
            print(
                f"[rl:{rl_cfg.reward_type}/{rl_cfg.filter_type}/{rl_cfg.interface}] "
                f"steps={res['steps']} loss_pg={fm['loss_pg']:.4f} loss_ntp={fm['loss_ntp']:.4f} "
                f"mean_reward={fm['mean_reward']:.4f} zero_var={fm['zero_var_frac']:.2f} "
                f"drift={fm['nothought_nll_drift']:+.4f}"
            )
        return {**res, "model": model, "corpus_stats": corpus_stats}
    finally:
        cleanup_distributed(did_init_dist)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for Phase 1 pilot training (single-process or torchrun DDP)."""
    from phase1_pilot.data.tokenizer import build_tokenizer

    args = build_arg_parser().parse_args(argv)
    return run_training(args, build_tokenizer(args.tokenizer))


if __name__ == "__main__":
    main()
