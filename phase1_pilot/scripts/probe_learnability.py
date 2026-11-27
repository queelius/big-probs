"""Scoping probe: does one supervised NTP model leave the prompt-blind floor on Regime 1?

Trains a single `ntp` model and reports, per (split, task, depth) cell, greedy accuracy and
mean NLL of the first target token next to the prompt-blind reference (each task's answer
marginal from the training split, `eval/reference.py`). A cell whose NLL is not below the
reference has not learned to use the prompt.

Two task sets:
  - `stratified`: the pilot's own mix (`build_stratified_datasets`: 75% Regime 1, 15%
    Regime 2, 10% Regime 3) at the requested hop depths.
  - `pointer`: pure variable-lookup chains with no arithmetic, `v3=v7; v7=v12; v12=4`,
    three chains of equal depth per prompt (copying a random literal scores 1/3).
    Names come from `v0`..`v<pool-1>`; keep `--pool-size` at 16 so every name is one token.

`--answer-only` masks every label before `prompt_len`, so only target tokens are supervised.
The default loss is the pilot's: every next token of prompt and target.

Pointer runs also report `key_swap`: each chain head of fixed bindings is queried in turn
(`key_swap_probe`), which separates resolving the query from copying any listed literal.
`--eval-checkpoint` scores a saved probe model with the same flags instead of training one.

    uv run python scripts/probe_learnability.py --steps 3000 --output runs/probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from phase1_pilot.config import DataConfig, ModelConfig, TrainConfig
from phase1_pilot.data import PilotTokenizer, TokenizedDataset, build_stratified_datasets, collate_batch
from phase1_pilot.data.generators import tokenize_sample
from phase1_pilot.data.records import Sample, TokenizedSample
from phase1_pilot.eval.reference import prompt_blind_reference
from phase1_pilot.models import build_model
from phase1_pilot.train import set_seed, train_supervised

SEED = 42
POINTER_EVAL_SEED = 4242
N_TRAIN = 50000
MAX_SEQ_LEN = 192
LR = {"micro-small": 1.5e-3, "micro-large": 1.2e-3}

POINTER_CHAINS = 3
POINTER_MOD = 19
# `v0`..`v15` are single PilotTokenizer tokens. From `v16` on a name splits into two tokens
# whose first (`v1`, `v2`, `v3`) is itself a variable, so a larger pool confounds lookup with
# multi-token key matching. EXP-006's first pointer probes used 32 names; keep 16 otherwise.
DEFAULT_POINTER_POOL = 16


def generate_pointer_chains(
    n: int, depths: tuple[int, ...], seed: int, pool_size: int = DEFAULT_POINTER_POOL
) -> list[Sample]:
    """`n` prompts of `POINTER_CHAINS` equal-depth copy chains; the query is chain 0's head."""
    pool = [f"v{i}" for i in range(pool_size)]
    if POINTER_CHAINS * (max(depths) + 1) > pool_size:
        raise ValueError(f"depth {max(depths)} needs {POINTER_CHAINS * (max(depths) + 1)} names; pool has {pool_size}")
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d = depths[i % len(depths)]
        heads, literals, eqs = _pointer_bindings(rng, d, pool)
        prompt = _pointer_prompt(eqs, heads[0])
        out.append(
            Sample(prompt=prompt, target=str(literals[0]), full_text=prompt + str(literals[0]),
                   regime="algorithmic", task="pointer_chain", depth=d, answer_span_start=len(prompt))
        )
    return out


def _pointer_bindings(rng: random.Random, d: int, pool: list[str]) -> tuple[list[str], list[int], list[str]]:
    """Chain heads, their literals, and the shuffled equations of one depth-`d` prompt."""
    names = rng.sample(pool, POINTER_CHAINS * (d + 1))
    literals = rng.sample(range(POINTER_MOD), POINTER_CHAINS)
    eqs: list[str] = []
    for c in range(POINTER_CHAINS):
        chain = names[c * (d + 1) : (c + 1) * (d + 1)]
        eqs.append(f"{chain[-1]}={literals[c]}")
        eqs.extend(f"{chain[j]}={chain[j + 1]}" for j in range(d))
    rng.shuffle(eqs)
    return names[:: d + 1], literals, eqs


def _pointer_prompt(eqs: list[str], query: str) -> str:
    return f"Ptr: {'; '.join(eqs)}; Query: {query}<ans>"


@torch.no_grad()
def key_swap_probe(
    model: torch.nn.Module, tok: PilotTokenizer, depth: int, n: int = 200, seed: int = 777,
    pool_size: int = DEFAULT_POINTER_POOL,
) -> dict[str, float]:
    """Query each chain head of `n` fixed binding sets; report what the answer depends on.

    A model that resolves the query is right for all three keys. One that copies a listed
    literal without matching the key keeps its answer when the key changes and scores 1/3.
    """
    rng = random.Random(seed)
    pool = [f"v{i}" for i in range(pool_size)]
    mass = p_right = right = changed = all_right = 0.0
    for _ in range(n):
        heads, literals, eqs = _pointer_bindings(rng, depth, pool)
        cand = [tok.encode(str(x))[0] for x in literals]
        preds = []
        for head, target in zip(heads, cand):
            ids = [tok.bos_id] + tok.encode(_pointer_prompt(eqs, head))
            probs = torch.softmax(model(torch.tensor([ids])).logits[0, -1].float(), -1)
            mass += float(probs[cand].sum())
            p_right += float(probs[target])
            preds.append(int(probs.argmax()))
        hits = [p == t for p, t in zip(preds, cand)]
        right += sum(hits)
        changed += len(set(preds)) > 1
        all_right += all(hits)
    queries = n * POINTER_CHAINS
    return {
        "n_prompts": n, "acc": right / queries, "mean_p_correct": p_right / queries,
        "mean_candidate_mass": mass / queries,
        "frac_prediction_changes_with_key": changed / n, "frac_all_keys_right": all_right / n,
    }


def answer_only_collate(items: list[TokenizedSample], pad_id: int) -> dict[str, Any]:
    """`collate_batch` with labels before each sample's `prompt_len` set to the ignore index."""
    batch = collate_batch(items, pad_id=pad_id)
    pos = torch.arange(batch["labels"].shape[1]).unsqueeze(0)
    batch["labels"] = batch["labels"].masked_fill(pos < batch["prompt_lens"].unsqueeze(1), -100)
    return batch


def _build_data(args: argparse.Namespace, tok: PilotTokenizer):
    """Train samples, the validation subset, and eval cells keyed `split|task|d<depth>`."""
    cells: dict[str, list[TokenizedSample]] = defaultdict(list)
    if args.tasks == "stratified":
        ds = build_stratified_datasets(
            DataConfig(seed=SEED, train_depths=args.train_depths, eval_ood_depths=args.ood_depths,
                       n_train_samples=N_TRAIN, n_eval_samples=600, max_seq_len=MAX_SEQ_LEN),
            tok,
        )
        train, val = ds["train"].tokenized, ds["eval_id"].tokenized
        for split in ("eval_id", "eval_ood", "eval_aleatoric"):
            for s in ds[split].tokenized:
                cells[f"{split}|{s.task}|d{s.depth}"].append(s)
        return train, val, cells

    train = [tokenize_sample(s, tok, MAX_SEQ_LEN) for s in generate_pointer_chains(N_TRAIN, args.train_depths, SEED, args.pool_size)]
    seen = {tuple(t.input_ids) for t in train}
    depths = tuple(args.train_depths) + tuple(args.ood_depths)
    ev = [
        t for t in (tokenize_sample(s, tok, MAX_SEQ_LEN) for s in generate_pointer_chains(1500, depths, POINTER_EVAL_SEED, args.pool_size))
        if tuple(t.input_ids) not in seen
    ]
    val = [e for e in ev if e.depth in args.train_depths][:256]
    for s in ev:
        split = "eval_id" if s.depth in args.train_depths else "eval_ood"
        cells[f"{split}|{s.task}|d{s.depth}"].append(s)
    return train, val, cells


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    set_seed(SEED)
    tok = PilotTokenizer()
    train, val, cells = _build_data(args, tok)
    collate = partial(answer_only_collate if args.answer_only else collate_batch, pad_id=tok.pad_id)
    train_loader = DataLoader(TokenizedDataset(train), batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(TokenizedDataset(val), batch_size=64, collate_fn=collate)
    model = build_model(ModelConfig.preset(args.preset, vocab_size=tok.vocab_size, max_seq_len=MAX_SEQ_LEN, variant="ntp"))

    t0 = time.perf_counter()
    if args.eval_checkpoint:
        state = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
        model.load_state_dict(state)  # strict: a preset mismatch raises
        res = {"final_train_loss": None, "final_val_loss": None, "history": []}
    else:
        res = train_supervised(
            model, train_loader, val_loader,
            TrainConfig(preset=args.preset, batch_size=32, max_steps=args.steps, lr=LR[args.preset],
                        log_every=max(1, args.steps // 10), output_dir=args.output + ".d", seed=SEED, device="cpu"),
            pad_id=tok.pad_id,
        )
    train_seconds = time.perf_counter() - t0

    model.eval()
    out_cells: dict[str, Any] = {}
    with torch.no_grad():
        for key, samples in sorted(cells.items()):
            correct, nll = 0, 0.0
            for s in samples:
                logits = model(torch.tensor([s.input_ids[: s.prompt_len]])).logits[0, -1]
                correct += int(int(logits.argmax()) == s.target_ids[0])
                nll -= float(torch.log_softmax(logits, -1)[s.target_ids[0]])
            ref = prompt_blind_reference(train, [s for s in samples])
            out_cells[key] = {
                "n": len(samples), "acc": correct / len(samples), "mean_nll": nll / len(samples),
                "prompt_blind_acc": ref.accuracy, "prompt_blind_nll": ref.mean_nll,
            }
        key_swap = (
            {f"d{d}": key_swap_probe(model, tok, d, pool_size=args.pool_size)
             for d in (*args.train_depths, *args.ood_depths)}
            if args.tasks == "pointer" else {}
        )
    return {
        "config": {**vars(args), "seed": SEED, "n_train": N_TRAIN, "lr": LR[args.preset], "batch_size": 32},
        "train_seconds": train_seconds,
        "final_train_loss": res["final_train_loss"],
        "final_val_loss": res["final_val_loss"],
        "val_curve": [[h["step"], h["val_loss"]] for h in res["history"] if h.get("val_loss")],
        "cells": out_cells,
        "key_swap": key_swap,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", choices=("stratified", "pointer"), default="stratified")
    p.add_argument("--preset", choices=tuple(LR), default="micro-small")
    p.add_argument("--train-depths", type=lambda s: tuple(int(x) for x in s.split(",")), default=(2, 4))
    p.add_argument("--ood-depths", type=lambda s: tuple(int(x) for x in s.split(",")), default=(6, 8))
    p.add_argument("--answer-only", action="store_true")
    p.add_argument("--pool-size", type=int, default=DEFAULT_POINTER_POOL, help="pointer variable names v0..v<N-1>")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-checkpoint", default=None,
                   help="score this saved probe checkpoint (`<output>.d/checkpoint.pt`) instead of training")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if set(args.train_depths) & set(args.ood_depths):
        p.error("--train-depths and --ood-depths must be disjoint")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = run_probe(args)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, c in report["cells"].items():
        print(f"{key:42s} acc {c['acc'] * 100:5.1f}% (blind {c['prompt_blind_acc'] * 100:5.1f}%)  "
              f"nll {c['mean_nll']:.3f} (blind {c['prompt_blind_nll']:.3f})  n={c['n']}")
    return report


if __name__ == "__main__":
    main()
