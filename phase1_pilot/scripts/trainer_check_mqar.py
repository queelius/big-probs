"""Trainer check (BL-12 step 1): can `train_supervised` teach a micro model Zoology's MQAR?

Multi-query associative recall (Arora et al. 2023, "Zoology", HazyResearch/zoology
`zoology/data/multiquery_ar.py`): a context of `num_kv_pairs` interleaved key/value tokens,
then a query region where every key appears once, at a power-law-distributed gap, and the
next position is labelled with that key's value. Filler positions hold random tokens. Only the
labelled positions carry loss. Defaults follow Zoology's `basic.py` example: 256 task tokens
(127 keys, 128 values), sequence length 64, 4 pairs, 10k train / 1k test examples, lr 1e-3,
batch 32.

Ours runs the pilot's own `LlamaLM` and `train_supervised` (AdamW, 5% warmup, cosine to 10%,
clip 1.0, weight decay 0.01), so a model that learns here clears the trainer and backbone of
EXP-006's single-lookup plateau. Task tokens are offset by 8 so they never hit protocol IDs.

`--supervised-queries 1` keeps the inputs and labels one random query per row, which matches
the pointer task's one answer label per sequence. `--n-keys 16 --n-values 19
--num-kv-pairs 3` matches its key and value vocabulary.

    uv run python scripts/trainer_check_mqar.py --steps 10000 --output runs/mqar.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from phase1_pilot.config import ModelConfig, TrainConfig
from phase1_pilot.models import build_model
from phase1_pilot.train import set_seed, train_supervised

ID_OFFSET = 8  # first ID after the eight protocol tokens
TEST_SEED_OFFSET = 1_000_003


def generate_mqar(
    n: int,
    *,
    n_keys: int = 127,
    n_values: int = 128,
    seq_len: int = 64,
    num_kv_pairs: int = 4,
    power_a: float = 0.01,
    random_non_queries: bool = True,
    supervised_queries: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(input_ids, labels)`, both `[n, seq_len]`; labels are -100 except after query keys.

    Labels are aligned with `input_ids` (the loss shifts them), so `labels[t]` is the value
    bound to the key at `input_ids[t - 1]`. Keys are `ID_OFFSET + 1 .. ID_OFFSET + n_keys`,
    values the next `n_values` IDs, and filler is drawn from all of them plus `ID_OFFSET`.
    """
    context = 2 * num_kv_pairs
    space = (seq_len - context) // 2
    if num_kv_pairs > min(n_keys, n_values, space):
        raise ValueError(f"{num_kv_pairs} pairs do not fit n_keys={n_keys}, n_values={n_values}, seq_len={seq_len}")
    rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng([seed, 1])  # own stream: label density never changes inputs
    weights = power_a * np.arange(1, space + 1) ** (power_a - 1)
    weights /= weights.sum()
    key_ids = ID_OFFSET + 1 + np.arange(n_keys)
    value_ids = ID_OFFSET + 1 + n_keys + np.arange(n_values)

    x = np.zeros((n, seq_len), dtype=np.int64)
    y = np.full((n, seq_len), -100, dtype=np.int64)
    if random_non_queries:
        x[:] = rng.integers(ID_OFFSET, ID_OFFSET + 1 + n_keys + n_values, size=(n, seq_len))
    else:
        x[:] = ID_OFFSET
    for i in range(n):
        keys = rng.choice(key_ids, num_kv_pairs, replace=False)
        values = rng.choice(value_ids, num_kv_pairs, replace=False)
        x[i, 0:context:2], x[i, 1:context:2] = keys, values
        gaps = rng.choice(space, num_kv_pairs, replace=False, p=weights)
        query_pos = context + 2 * gaps
        x[i, query_pos] = keys
        keep = np.arange(num_kv_pairs)
        if supervised_queries is not None:
            keep = label_rng.choice(num_kv_pairs, supervised_queries, replace=False)
        y[i, query_pos[keep] + 1] = values[keep]
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def answer_accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 250) -> float:
    """Greedy accuracy over labelled positions: logits at `t - 1` against `labels[t]`."""
    was_training = getattr(model, "training", False)
    model.eval()
    correct = total = 0
    for s in range(0, x.shape[0], batch_size):
        logits = model(x[s : s + batch_size]).logits[:, :-1]
        target = y[s : s + batch_size, 1:]
        mask = target != -100
        correct += int((logits.argmax(-1)[mask] == target[mask]).sum())
        total += int(mask.sum())
    if was_training:
        model.train()
    return correct / total


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    data_kw = dict(n_keys=args.n_keys, n_values=args.n_values, seq_len=args.seq_len,
                   num_kv_pairs=args.num_kv_pairs, supervised_queries=args.supervised_queries)
    x_tr, y_tr = generate_mqar(args.n_train, seed=args.seed, **data_kw)
    # Test rows are scored on every query, whatever the training label density.
    x_te, y_te = generate_mqar(args.n_test, seed=args.seed + TEST_SEED_OFFSET, **{**data_kw, "supervised_queries": None})
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=250)

    vocab = ID_OFFSET + 1 + args.n_keys + args.n_values
    model = build_model(ModelConfig.preset(args.preset, vocab_size=vocab, max_seq_len=args.seq_len, variant="ntp"))
    curve: list[dict[str, float]] = []

    def on_step(step: int, record: dict[str, Any]) -> None:
        if not math.isfinite(record["train_loss"]):  # preregistered abort
            raise FloatingPointError(f"non-finite train loss {record['train_loss']} at step {step}")
        if step % args.eval_every == 0 or step == args.steps:
            curve.append({"step": step, "train_loss": record["train_loss"],
                          "test_acc": answer_accuracy(model, x_te, y_te)})
            print(f"step {step:6d}  loss {record['train_loss']:.4f}  test_acc {curve[-1]['test_acc']:.4f}", flush=True)

    t0 = time.perf_counter()
    res = train_supervised(
        model, train_loader, None,
        TrainConfig(preset=args.preset, batch_size=args.batch_size, max_steps=args.steps, lr=args.lr,
                    log_every=0, output_dir=None, seed=args.seed, device="cpu"),
        step_callback=on_step,
    )
    reached = next((p["step"] for p in curve if p["test_acc"] >= 0.99), None)
    return {
        "config": vars(args),
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.perf_counter() - t0,
        "final_train_loss": res["final_train_loss"],
        "final_test_acc": curve[-1]["test_acc"],
        "first_step_test_acc_ge_0.99": reached,
        "curve": curve,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default="micro-small")
    p.add_argument("--n-keys", type=int, default=127)
    p.add_argument("--n-values", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--num-kv-pairs", type=int, default=4)
    p.add_argument("--supervised-queries", type=int, default=None, help="labelled queries per train row (default all)")
    p.add_argument("--n-train", type=int, default=10_000)
    p.add_argument("--n-test", type=int, default=1_000)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = run(args)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
