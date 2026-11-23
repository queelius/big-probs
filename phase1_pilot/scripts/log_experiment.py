#!/usr/bin/env python3
"""Ingest Phase 1 / Stage 1 metrics JSON into research/experiments.jsonl and render Pareto tables.

Usage examples (run from phase1_pilot/):
  # Inspect CLI flags
  uv run python scripts/log_experiment.py --help

  # Render the Pareto-frontier Markdown table from the existing ledger
  uv run python scripts/log_experiment.py --render-only

  # Dry-run ingesting micro_pilot_metrics.json without writing any files
  uv run python scripts/log_experiment.py \
      --metrics results/micro_pilot_metrics.json \
      --exp-id EXP-001-DRYRUN \
      --title "Dry-run check" \
      --dry-run

  # Append a completed Stage 1 run and refresh the Pareto table in research/EXPERIMENTS.md
  uv run python scripts/log_experiment.py \
      --metrics runs/h100_stage1/stage1_metrics.json \
      --exp-id EXP-002 \
      --title "Phase 1 Stage 1 H100 Core Pilot" \
      --rq-ids RQ-01,H1,H1a,H1b,H1c \
      --hardware "4x NVIDIA H100 80GB" \
      --local-report runs/h100_stage1/stage1_report.md \
      --update-markdown
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = REPO_ROOT / "research" / "experiments.jsonl"
DEFAULT_EXPERIMENTS_MD = REPO_ROOT / "research" / "EXPERIMENTS.md"
PARETO_START = "<!-- PARETO_TABLE_START -->"
PARETO_END = "<!-- PARETO_TABLE_END -->"

DEFAULT_GITHUB_REPO = "https://github.com/queelius/big-probs"
DEFAULT_HF_MODELS = "https://huggingface.co/queelius/big-probs-phase1-models"
DEFAULT_HF_DATASETS = "https://huggingface.co/datasets/queelius/big-probs-phase1-data"


def _round_opt(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _rel_repo_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path_str


def _extract_arms(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized per-arm records from a phase1_pilot metrics JSON."""
    raw_arms: dict[str, Any] = metrics.get("arms", {})
    telemetry: dict[str, Any] = metrics.get("training_telemetry", {})

    # Determine B2 FLOPs denominator (supports B2 or B2a naming).
    b2_flops: float | None = None
    summary = metrics.get("verdict", {}).get("metrics_summary", {})
    if summary.get("b2_flops_per_token"):
        b2_flops = float(summary["b2_flops_per_token"])
    else:
        for key in ("B2", "B2a", "b2", "b2a"):
            if key in raw_arms:
                cand = raw_arms[key].get("pareto", {}).get("inference_flops_per_token")
                if cand:
                    b2_flops = float(cand)
                    break

    arms_out: list[dict[str, Any]] = []
    for arm_name, payload in raw_arms.items():
        pareto = payload.get("pareto", {})
        pass_k = payload.get("pass_at_k", {})
        faith = payload.get("faithfulness", {})
        cal_r3 = payload.get("calibration_aleatoric", {})
        tel = telemetry.get(arm_name, {})
        final_metrics = tel.get("final_metrics", {}) if isinstance(tel, dict) else {}

        flops = pareto.get("inference_flops_per_token")
        flops_ratio = (float(flops) / b2_flops) if (flops is not None and b2_flops) else None

        mode = faith.get("mode")
        has_thought = mode not in (None, "ntp", "pause")

        sigma_zero = final_metrics.get("zero_var_frac")
        drift = payload.get("nothought_nll_drift")
        if drift is None and "nothought_nll_drift" in final_metrics:
            drift = final_metrics["nothought_nll_drift"]

        arms_out.append(
            {
                "arm": arm_name,
                "variant": pareto.get("variant"),
                "params": int(pareto["total_params"]) if pareto.get("total_params") is not None else None,
                "flops_ratio": _round_opt(flops_ratio, 6),
                "exact_acc": _round_opt(pareto.get("exact_accuracy"), 6),
                "hard_bpb": _round_opt(pareto.get("hard_bpb"), 6),
                "ood_pass_at_k": _round_opt(pass_k.get("ood_pass_at_max_k"), 6),
                "delta_nll_swap": _round_opt(faith.get("delta_nll_swap"), 6) if has_thought else None,
                "delta_nll_trunc": _round_opt(faith.get("delta_nll_trunc"), 6) if has_thought else None,
                "reg3_ece": _round_opt(cal_r3.get("ece"), 6),
                "reg3_brier": _round_opt(cal_r3.get("mean_brier_score"), 6),
                "sigma_zero_frac": _round_opt(sigma_zero, 4),
                "nothought_nll_drift": _round_opt(drift, 6),
            }
        )
    return arms_out


def build_experiment_record(args: argparse.Namespace, metrics: dict[str, Any]) -> dict[str, Any]:
    """Construct an experiments.jsonl record from parsed metrics JSON and CLI metadata."""
    cfg = metrics.get("config", {})
    verdict_block = metrics.get("verdict", {})

    elapsed_s = metrics.get("elapsed_seconds")
    compute_hours = args.compute_hours
    if compute_hours is None and elapsed_s is not None:
        compute_hours = round(float(elapsed_s) / 3600.0, 4)

    scale = args.scale
    if not scale:
        scope = metrics.get("scope", "phase1_pilot run")
        sup = cfg.get("sup_steps")
        rl = cfg.get("rl_steps")
        seed = cfg.get("seed")
        scale = f"{scope} (sup={sup}, rl={rl}, seed={seed})"

    rq_ids = [item.strip() for item in args.rq_ids.split(",") if item.strip()]

    artifacts: dict[str, str] = {
        "local_metrics": _rel_repo_path(str(args.metrics)) or str(args.metrics),
        "github_repo": args.github_repo,
        "hf_models": args.hf_models,
        "hf_datasets": args.hf_datasets,
    }
    if args.local_report:
        artifacts["local_report"] = _rel_repo_path(args.local_report) or args.local_report
    if cfg.get("output_dir"):
        rel_out = _rel_repo_path(cfg["output_dir"])
        if rel_out:
            artifacts["local_output_dir"] = rel_out

    record: dict[str, Any] = {
        "exp_id": args.exp_id,
        "date": args.date,
        "title": args.title,
        "rq_ids": rq_ids,
        "scale": scale,
        "hardware": args.hardware,
        "compute_hours": compute_hours,
        "status": args.status,
        "verdict": verdict_block.get("status", "UNKNOWN"),
        "pareto_axes": {
            "cost": "flops_ratio_vs_b2 (lower is better)",
            "quality": ["exact_acc (higher)", "hard_bpb (lower)", "ood_pass_at_k (higher)"],
            "calibration": "reg3_ece (lower)",
        },
        "arms": _extract_arms(metrics),
        "artifacts": artifacts,
        "bugs_caught": list(args.bugs_caught or []),
        "lessons": list(args.lessons or []),
    }
    if verdict_block.get("rationale") and not record["lessons"]:
        record["lessons"].append(verdict_block["rationale"])
    return record


def load_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {ledger_path}:{line_no}: {exc}") from exc
    return records


def compute_pareto_flags(arms: list[dict[str, Any]]) -> list[bool]:
    """Identify non-dominated arms on (flops_ratio ↓, hard_bpb ↓, exact_acc ↑, ood_pass_at_k ↑)."""
    flags: list[bool] = []
    for i, a in enumerate(arms):
        if a.get("flops_ratio") is None or a.get("hard_bpb") is None or a.get("exact_acc") is None:
            flags.append(False)
            continue
        a_ood = a.get("ood_pass_at_k") if a.get("ood_pass_at_k") is not None else -1.0
        dominated = False
        for j, b in enumerate(arms):
            if i == j:
                continue
            if b.get("flops_ratio") is None or b.get("hard_bpb") is None or b.get("exact_acc") is None:
                continue
            b_ood = b.get("ood_pass_at_k") if b.get("ood_pass_at_k") is not None else -1.0
            no_worse = (
                b["flops_ratio"] <= a["flops_ratio"]
                and b["hard_bpb"] <= a["hard_bpb"]
                and b["exact_acc"] >= a["exact_acc"]
                and b_ood >= a_ood
            )
            strictly_better = (
                b["flops_ratio"] < a["flops_ratio"]
                or b["hard_bpb"] < a["hard_bpb"]
                or b["exact_acc"] > a["exact_acc"]
                or b_ood > a_ood
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def _fmt_pct(val: float | None) -> str:
    return "—" if val is None else f"{val * 100:.1f}%"


def _fmt_float(val: float | None, digits: int = 3, signed: bool = False) -> str:
    if val is None:
        return "—"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(val)


def render_pareto_markdown(records: list[dict[str, Any]]) -> str:
    """Render a Pareto-frontier Markdown summary table across all completed experiments with arms."""
    header = (
        "| Exp ID | Arm | Params | FLOPs (vs B2) | Exact Acc | Hard-BPB | OOD pass@k "
        "| ΔNLL_swap | ΔNLL_trunc | Reg3 ECE | σ=0 Frac | Pareto? |\n"
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"
    )
    rows: list[str] = [header]
    for rec in records:
        arms = rec.get("arms") or []
        if not arms:
            continue
        pareto_flags = compute_pareto_flags(arms)
        exp_id = rec.get("exp_id", "EXP-???")
        for arm, is_pareto in zip(arms, pareto_flags):
            params_str = "—" if arm.get("params") is None else f"{int(arm['params']):,}"
            flops_str = "—" if arm.get("flops_ratio") is None else f"{float(arm['flops_ratio']):.2f}×"
            row = (
                f"| `{exp_id}` | `{arm['arm']}` | {params_str} | {flops_str} "
                f"| {_fmt_pct(arm.get('exact_acc'))} | {_fmt_float(arm.get('hard_bpb'), 3)} "
                f"| {_fmt_pct(arm.get('ood_pass_at_k'))} "
                f"| {_fmt_float(arm.get('delta_nll_swap'), 3, signed=True)} "
                f"| {_fmt_float(arm.get('delta_nll_trunc'), 3, signed=True)} "
                f"| {_fmt_float(arm.get('reg3_ece'), 3)} "
                f"| {_fmt_float(arm.get('sigma_zero_frac'), 2)} "
                f"| {'Yes' if is_pareto else 'No'} |"
            )
            rows.append(row)
    return "\n".join(rows)


def update_experiments_markdown(md_path: Path, table_md: str) -> None:
    content = md_path.read_text(encoding="utf-8")
    if PARETO_START not in content or PARETO_END not in content:
        raise ValueError(f"Markers {PARETO_START} / {PARETO_END} not found in {md_path}")
    before, rest = content.split(PARETO_START, maxsplit=1)
    _, after = rest.split(PARETO_END, maxsplit=1)
    updated = f"{before}{PARETO_START}\n{table_md}\n{PARETO_END}{after}"
    md_path.write_text(updated, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log Phase 1 / Stage 1 experiment metrics JSON into research/experiments.jsonl "
        "and render/update the Pareto-frontier Markdown table."
    )
    parser.add_argument("--metrics", type=Path, help="Path to micro_pilot_metrics.json or stage1_metrics.json")
    parser.add_argument("--exp-id", type=str, help="Experiment ID (e.g., EXP-002)")
    parser.add_argument("--title", type=str, default="Phase 1 Experiment Run", help="Human-readable experiment title")
    parser.add_argument(
        "--rq-ids",
        type=str,
        default="RQ-01,H1,H1a,H1b",
        help="Comma-separated research question / hypothesis IDs",
    )
    parser.add_argument("--scale", type=str, default="", help="Optional scale summary override")
    parser.add_argument("--hardware", type=str, default="CPU / unspecified", help="Hardware description")
    parser.add_argument(
        "--compute-hours",
        type=float,
        default=None,
        help="Compute hours (defaults to elapsed_seconds / 3600 from metrics JSON)",
    )
    parser.add_argument("--status", type=str, default="completed", help="Experiment status (default: completed)")
    parser.add_argument(
        "--date",
        type=str,
        default=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        help="Run date YYYY-MM-DD (default: today UTC)",
    )
    parser.add_argument("--local-report", type=str, default=None, help="Optional path to Markdown report")
    parser.add_argument("--github-repo", type=str, default=DEFAULT_GITHUB_REPO, help="GitHub repo URL")
    parser.add_argument("--hf-models", type=str, default=DEFAULT_HF_MODELS, help="HF models repo URL")
    parser.add_argument("--hf-datasets", type=str, default=DEFAULT_HF_DATASETS, help="HF datasets repo URL")
    parser.add_argument("--bugs-caught", action="append", default=[], help="Bug caught during audit (repeatable)")
    parser.add_argument("--lessons", action="append", default=[], help="Lesson learned (repeatable)")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help=f"Path to experiments.jsonl ledger (default: {DEFAULT_LEDGER})",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_EXPERIMENTS_MD,
        help=f"Path to EXPERIMENTS.md for Pareto table updates (default: {DEFAULT_EXPERIMENTS_MD})",
    )
    parser.add_argument(
        "--update-markdown",
        action="store_true",
        help="Rewrite the Pareto table between PARETO_TABLE_START/END markers in --markdown",
    )
    parser.add_argument(
        "--allow-update",
        action="store_true",
        help="Replace an existing record with the same --exp-id instead of refusing duplicates",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Only read --ledger and render the Pareto-frontier Markdown table (no ingestion)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print JSONL record + Pareto table without modifying any files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    existing = load_ledger(args.ledger)

    if args.render_only:
        table_md = render_pareto_markdown(existing)
        print(table_md)
        if args.update_markdown and not args.dry_run:
            update_experiments_markdown(args.markdown, table_md)
            print(f"\nUpdated Pareto table in {args.markdown}", file=sys.stderr)
        return 0

    if not args.metrics or not args.exp_id:
        print("Error: --metrics and --exp-id are required unless --render-only is set.", file=sys.stderr)
        return 2

    if not args.metrics.exists():
        print(f"Error: metrics file not found: {args.metrics}", file=sys.stderr)
        return 2

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    record = build_experiment_record(args, metrics)

    existing_ids = [r.get("exp_id") for r in existing]
    if args.exp_id in existing_ids and not args.allow_update:
        print(
            f"Error: exp_id '{args.exp_id}' already exists in {args.ledger}. "
            "Pass --allow-update to replace it or choose a new EXP-XXX ID.",
            file=sys.stderr,
        )
        return 1

    updated_records = [r for r in existing if r.get("exp_id") != args.exp_id]
    updated_records.append(record)
    table_md = render_pareto_markdown(updated_records)

    if args.dry_run:
        print("=== DRY RUN: JSONL RECORD ===")
        print(json.dumps(record, ensure_ascii=False))
        print("\n=== DRY RUN: PARETO FRONTIER TABLE ===")
        print(table_md)
        return 0

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    if args.allow_update and args.exp_id in existing_ids:
        args.ledger.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in updated_records) + "\n",
            encoding="utf-8",
        )
    else:
        with args.ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Logged {args.exp_id} ({len(record['arms'])} arms, verdict={record['verdict']}) -> {args.ledger}")
    print("\n" + table_md)

    if args.update_markdown:
        update_experiments_markdown(args.markdown, table_md)
        print(f"\nUpdated Pareto table in {args.markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
