"""Combine per-seed `run_micro_pilot.py` metrics into per-arm mean, std, and paired differences.

Each input is one seed's `micro_pilot_metrics.json`. Arms are paired by seed: within a
seed every arm sees the same data, so a candidate-minus-control difference is computed
per seed before averaging.

    uv run python scripts/aggregate_seeds.py runs/a/micro_pilot_metrics.json \
        runs/b/micro_pilot_metrics.json --output runs/agg.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# Accuracy averaged over a group of hop depths, from `pareto.accuracy_by_depth`.
DEPTH_GROUPS: dict[str, tuple[int, ...]] = {"id_acc": (2, 4), "ood_acc": (6, 8)}

# Scalars read from a fixed path in each arm's record.
SCALAR_PATHS: dict[str, tuple[str, ...]] = {
    "mean_nll": ("pareto", "mean_nll"),
    "hard_bpb": ("pareto", "hard_bpb"),
    "inference_flops_per_token": ("pareto", "inference_flops_per_token"),
    "total_params": ("pareto", "total_params"),
    "ood_pass_at_max_k": ("pass_at_k", "ood_pass_at_max_k"),
    "delta_nll_swap": ("faithfulness", "delta_nll_swap"),
    "delta_nll_trunc": ("faithfulness", "delta_nll_trunc"),
    "reg3_ece": ("calibration_aleatoric", "ece"),
    "nothought_nll_drift": ("nothought_nll_drift",),
}

# Prompt-blind reference NLL minus the arm's Regime 1 NLL (positive: the arm uses the prompt).
# Needs the run's top-level `prompt_blind_reference`, so it is computed per run, not per arm.
RELATIVE_TO_REFERENCE = "nll_below_prompt_blind"

METRIC_NAMES: tuple[str, ...] = (*DEPTH_GROUPS, *SCALAR_PATHS, RELATIVE_TO_REFERENCE)


def _get(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def arm_metrics(arm: dict[str, Any], reference_nll: float | None = None) -> dict[str, float | None]:
    """The scalar metrics of one arm in one seed (`None` where the arm has no value)."""
    by_depth = _get(arm, ("pareto", "accuracy_by_depth")) or {}
    out: dict[str, float | None] = {}
    for name, depths in DEPTH_GROUPS.items():
        vals = [by_depth[str(d)] for d in depths if str(d) in by_depth]
        out[name] = sum(vals) / len(vals) if vals else None
    for name, path in SCALAR_PATHS.items():
        value = _get(arm, path)
        out[name] = float(value) if isinstance(value, (int, float)) else None
    nll = out["mean_nll"]
    out[RELATIVE_TO_REFERENCE] = reference_nll - nll if reference_nll is not None and nll is not None else None
    return out


def load_runs(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Read per-seed metrics files, sorted by seed. Two files with one seed are an error."""
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    seen: dict[int, str] = {}
    for path, run in zip(paths, runs):
        seed = int(run["config"]["seed"])
        if seed in seen:
            raise ValueError(f"seed {seed} appears twice: {seen[seed]} and {path}")
        seen[seed] = str(path)
    return sorted(runs, key=lambda r: int(r["config"]["seed"]))


def _stats(values: list[float | None]) -> dict[str, Any]:
    present = [v for v in values if v is not None]
    return {
        "n": len(present),
        "mean": statistics.fmean(present) if present else None,
        "std": statistics.stdev(present) if len(present) > 1 else None,
        "values": values,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm statistics of every metric in `METRIC_NAMES`, across seeds."""
    arm_sets = [set(r["arms"]) for r in runs]
    if any(s != arm_sets[0] for s in arm_sets):
        raise ValueError(f"runs have different arms: {[sorted(s) for s in arm_sets]}")
    per_seed = [
        {
            name: arm_metrics(arm, _get(r, ("prompt_blind_reference", "mean_nll")))
            for name, arm in r["arms"].items()
        }
        for r in runs
    ]
    arms = {
        name: {m: _stats([seed[name][m] for seed in per_seed]) for m in METRIC_NAMES}
        for name in sorted(arm_sets[0])
    }
    return {
        "seeds": [int(r["config"]["seed"]) for r in runs],
        "verdicts": [r.get("verdict", {}).get("status") for r in runs],
        "arms": arms,
    }


def paired_difference(summary: dict[str, Any], arm: str, control: str, metric: str) -> dict[str, Any]:
    """Per-seed `arm - control` on `metric`, with mean, std, and whether every seed is positive."""
    a = summary["arms"][arm][metric]["values"]
    c = summary["arms"][control][metric]["values"]
    diffs = [x - y if x is not None and y is not None else None for x, y in zip(a, c)]
    out = _stats(diffs)
    present = [d for d in diffs if d is not None]
    out["all_positive"] = bool(present) and all(d > 0 for d in present)
    return out


def format_table(summary: dict[str, Any], metrics: tuple[str, ...] = METRIC_NAMES) -> str:
    def cell(stat: dict[str, Any]) -> str:
        if stat["mean"] is None:
            return "—"
        if stat["std"] is None:
            return f"{stat['mean']:.4g}"
        return f"{stat['mean']:.4g} ± {stat['std']:.2g}"

    lines = [
        f"seeds: {summary['seeds']}  verdicts: {summary['verdicts']}",
        "| arm | " + " | ".join(metrics) + " |",
        "| :--- |" + " ---: |" * len(metrics),
    ]
    for name, stats in summary["arms"].items():
        lines.append(f"| {name} | " + " | ".join(cell(stats[m]) for m in metrics) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("metrics", nargs="+", help="one micro_pilot_metrics.json per seed")
    p.add_argument("--output", help="write the summary JSON here")
    args = p.parse_args(argv)
    summary = aggregate(load_runs(args.metrics))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(format_table(summary))
    return summary


if __name__ == "__main__":
    main()
