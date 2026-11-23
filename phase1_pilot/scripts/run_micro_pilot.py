"""End-to-end CPU micro-scale Phase 1 Latent-CoT RL Pilot (proof-of-life & pipeline validation).

Trains and evaluates all 7 core Track 1 §5 arms on CPU (~1-3 minutes on 12 cores):
  1. B1         — micro-small Standard NTP (~0.9M params)
  2. B2         — micro-large Standard NTP (~2.5x params ceiling)
  3. B3         — micro-small + Pause Tokens (Goyal et al. control)
  4. B4         — micro-small + Continuous Latent BPTT (PonderLM-2 / Coconut)
  5. RL-R1      — micro-small (from B1) + Discrete RL (T2_self + R1_exact)
  6. RL-R3      — micro-small (from B1) + Discrete RL (T2_self + R3_brier, frozen anchor)
  7. I2-Hybrid  — micro-small (from B4) + Continuous Latent -> HRPO Hybrid RL (R3_brier)

Evaluates all arms on:
  - Regime 1 Algorithmic & Compositional (ID d in {2, 4} + OOD d in {6, 8})
  - Regime 3 Aleatoric Negative Control (irreducible entropy calibration probe)

Runs all 5 mandatory Track 1 §5.5 diagnostics and writes:
  - results/micro_pilot_metrics.json
  - results/micro_pilot_report.md
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from phase1_pilot.config import DataConfig, ModelConfig, RLConfig, TrainConfig
from phase1_pilot.data import PilotTokenizer, build_stratified_datasets, collate_batch
from phase1_pilot.eval.diagnostics import (
    ArmEvalBundle,
    compute_pilot_verdict,
    evaluate_calibration,
    evaluate_pareto_point,
    evaluate_pass_at_k,
    evaluate_thought_faithfulness,
)
from phase1_pilot.models import build_model
from phase1_pilot.train import set_seed, train_ssl_rl, train_supervised


def _select_balanced_by_depth(samples: list[Any], per_depth: int) -> list[Any]:
    """Deterministically take up to `per_depth` samples per hop depth."""
    by_d: dict[int, list[Any]] = {}
    for s in samples:
        d = int(getattr(s, "depth", 0))
        by_d.setdefault(d, []).append(s)
    out: list[Any] = []
    for d in sorted(by_d.keys()):
        out.extend(by_d[d][:per_depth])
    return out


def _check_finite_history(history: list[dict[str, Any]], label: str) -> None:
    for row in history:
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                raise RuntimeError(f"Non-finite metric {k}={v} detected during {label}")


def _evaluate_arm(
    arm_name: str,
    model: torch.nn.Module,
    tokenizer: PilotTokenizer,
    eval_r1_all: list[Any],
    eval_r1_passk: list[Any],
    eval_r3: list[Any],
    *,
    use_thought: bool,
    thought_len: int,
    faithfulness_mode: str,
    pass_k_n: int,
    is_rl_arm: bool,
    is_teacher_free: bool = True,
    is_continuous_or_hybrid: bool = False,
    nothought_nll_drift: float | None = None,
    seed: int = 42,
) -> ArmEvalBundle:
    """Run all 4 per-arm diagnostics (Pareto, pass@k, Faithfulness, Calibration)."""
    pareto = evaluate_pareto_point(
        model=model,
        samples=eval_r1_all,
        tokenizer=tokenizer,
        arm_name=arm_name,
        use_thought=use_thought,
        thought_len=thought_len,
    )
    pass_k = evaluate_pass_at_k(
        model=model,
        samples=eval_r1_passk,
        tokenizer=tokenizer,
        arm_name=arm_name,
        ks=(1, 4, 16, 64),
        n_samples=pass_k_n,
        temperature=0.8,
        top_p=0.95,
        use_thought=use_thought,
        thought_len=thought_len,
        ood_depths=(6, 8),
        seed=seed,
    )
    faith = evaluate_thought_faithfulness(
        model=model,
        samples=eval_r1_all,
        tokenizer=tokenizer,
        arm_name=arm_name,
        mode=faithfulness_mode,
        thought_len=thought_len,
        seed=seed,
    )
    cal_r1 = evaluate_calibration(
        model=model,
        samples=eval_r1_all,
        tokenizer=tokenizer,
        arm_name=arm_name,
        regime="regime1_algorithmic",
        n_bins=10,
        use_thought=use_thought,
        thought_len=thought_len,
    )
    cal_r3 = evaluate_calibration(
        model=model,
        samples=eval_r3,
        tokenizer=tokenizer,
        arm_name=arm_name,
        regime="regime3_aleatoric",
        n_bins=10,
        use_thought=use_thought,
        thought_len=thought_len,
    )
    return ArmEvalBundle(
        arm_name=arm_name,
        pareto=pareto,
        pass_at_k=pass_k,
        faithfulness=faith,
        calibration_primary=cal_r1,
        calibration_aleatoric=cal_r3,
        is_rl_arm=is_rl_arm,
        is_teacher_free=is_teacher_free,
        is_continuous_or_hybrid=is_continuous_or_hybrid,
        is_warm_start=False,
        nothought_nll_drift=nothought_nll_drift,
    )


def _build_report_markdown(
    *,
    verdict: Any,
    bundles: dict[str, ArmEvalBundle],
    train_telemetry: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    elapsed_sec: float,
) -> str:
    """Render a comprehensive, clearly scoped Markdown micro-pilot report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    arm_order = ["B1", "B2", "B3", "B4", "RL-R1", "RL-R3", "I2-Hybrid"]

    lines: list[str] = [
        "# Phase 1 Small-Model Latent-CoT RL Pilot — CPU Micro-Scale Report",
        "",
        "> **Scope Notice (Proof-of-Life & Pipeline Validation):** These results are from a "
        "live, reproducible **CPU micro-scale run** (`micro-small` ~0.9M params vs `micro-large` "
        "~2.5× params, trained for "
        f"{args.sup_steps} supervised steps + {args.rl_steps} RL steps on synthetic 3-regime data in "
        f"{elapsed_sec:.1f}s on CPU). This validates end-to-end training stability, anti-sabotage "
        "invariants, and all 5 Track 1 §5.5 diagnostics. It is **distinct from** the full "
        "160M / 410M / 1.0B H100 Stage 1 run (`scripts/launch_h100_stage1.sh`, ~30–45 H100 GPU-hrs).",
        "",
        f"- **Generated:** {now}",
        f"- **Seed:** `{args.seed}` | **Train samples:** `{args.n_train}` | "
        f"**Eval samples (Regime 1 / Regime 3):** `{args.eval_per_depth * 4}` / `{args.eval_aleatoric}`",
        f"- **Supervised steps:** `{args.sup_steps}` (`batch_size={args.batch_size}`) | "
        f"**RL steps:** `{args.rl_steps}` (`batch_size={args.rl_batch_size}`, `G={args.group_size}`, "
        f"`thought_len={args.thought_len}`, `n_ahead={args.n_ahead}`)",
        f"- **Support-Boundary Probe (`pass@k`):** `n={args.pass_k_n}` rollouts/prompt across `d in {{2, 4, 6, 8}}`",
        "",
        "---",
        "",
        "## 1. Track 1 §5.5 Falsification Verdict (CPU Micro-Scale)",
        "",
        f"**Verdict Status:** `{verdict.status}`",
        "",
        f"{verdict.rationale}",
        "",
        "### Summary Comparison Table (All 7 Arms)",
        "",
        verdict.markdown_table,
        "",
        "---",
        "",
        "## 2. Diagnostic 1 & 2 — Accuracy & `pass@k` by Hop Depth (ID `d={2,4}` vs OOD `d={6,8}`)",
        "",
        "| Arm | Greedy Acc `d=2` | Greedy Acc `d=4` | Greedy Acc `d=6` (OOD) | Greedy Acc `d=8` (OOD) | "
        f"ID `pass@1` | OOD `pass@1` | OOD `pass@{args.pass_k_n}` |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name in arm_order:
        b = bundles[name]
        abd = b.pareto.accuracy_by_depth
        pk = b.pass_at_k
        assert pk is not None
        max_k = max(pk.ks) if pk.ks else 1
        id_p1_vals = [pk.by_depth[d][1] for d in (2, 4) if d in pk.by_depth and 1 in pk.by_depth[d]]
        ood_p1_vals = [pk.by_depth[d][1] for d in (6, 8) if d in pk.by_depth and 1 in pk.by_depth[d]]
        id_p1 = sum(id_p1_vals) / max(1, len(id_p1_vals))
        ood_p1 = sum(ood_p1_vals) / max(1, len(ood_p1_vals))
        lines.append(
            f"| `{name}` | {abd.get(2, 0.0)*100:.1f}% | {abd.get(4, 0.0)*100:.1f}% | "
            f"{abd.get(6, 0.0)*100:.1f}% | {abd.get(8, 0.0)*100:.1f}% | "
            f"{id_p1*100:.1f}% | {ood_p1*100:.1f}% | {pk.ood_pass_at_max_k*100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Diagnostic 3 — Causal Thought Faithfulness (Swap, Format-Matched Truncation & Depth Scaling)",
            "",
            "| Arm | Mode | NLL(thought) | ΔNLL_swap (nats) | ΔNLL_trunc (nats) | "
            "Swap `d=2` | Swap `d=4` | Swap `d=6` | Swap `d=8` | Monotonic? |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )

    for name in arm_order:
        f = bundles[name].faithfulness
        assert f is not None
        sbd = f.delta_nll_swap_by_depth
        lines.append(
            f"| `{name}` | `{f.mode}` | {f.nll_with_thought:.3f} | {f.delta_nll_swap:+.4f} | "
            f"{f.delta_nll_trunc:+.4f} | {sbd.get(2, 0.0):+.4f} | {sbd.get(4, 0.0):+.4f} | "
            f"{sbd.get(6, 0.0):+.4f} | {sbd.get(8, 0.0):+.4f} | "
            f"{'Yes' if f.monotonic_depth_scaling else 'No'} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Diagnostic 4 — Distributional Calibration (Regime 1 vs Regime 3 Aleatoric Negative Control)",
            "",
            "| Arm | Reg1 ECE | Reg1 Brier (`[-2,0]`) | Reg3 ECE (Aleatoric) | "
            "Reg3 Brier (`[-2,0]`) | Reg3 Mean Conf | Reg3 Acc | Reg3 Overconf Gap |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for name in arm_order:
        c1 = bundles[name].calibration_primary
        c3 = bundles[name].calibration_aleatoric
        assert c1 is not None and c3 is not None
        lines.append(
            f"| `{name}` | {c1.ece:.4f} | {c1.mean_brier_score:.4f} | {c3.ece:.4f} | "
            f"{c3.mean_brier_score:.4f} | {c3.mean_confidence:.4f} | {c3.accuracy:.4f} | "
            f"{c3.overconfidence_gap:+.4f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Diagnostic 5 — Training Stability Telemetry",
            "",
            "| Arm | Steps | Final Train Loss / `loss_pg` | Final Val NLL / `loss_ntp` | "
            "Mean Reward | Policy Entropy | `sigma=0` Frac | No-Thought NLL Drift |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for name in arm_order:
        t = train_telemetry[name]
        if t["type"] == "supervised":
            lines.append(
                f"| `{name}` | {t['steps']} | {t['final_train_loss']:.4f} | "
                f"{t['final_val_loss']:.4f} | — | — | — | — |"
            )
        else:
            fm = t["final_metrics"]
            lines.append(
                f"| `{name}` | {t['steps']} | {fm['loss_pg']:+.4f} | {fm['loss_ntp']:.4f} | "
                f"{fm['mean_reward']:+.4f} | {fm['policy_entropy']:.3f} | "
                f"{fm['zero_var_frac']:.2f} | {fm['nothought_nll_drift']:+.4f} |"
            )

    lines.extend(
        [
            "",
            "### Headline Criterion Checklist (`" + str(verdict.best_rl_arm) + "`)",
            "",
        ]
    )
    for k, passed in verdict.checks.items():
        mark = "PASS" if passed else "FAIL"
        lines.append(f"- `{k}`: **{mark}**")

    lines.append("")
    return "\n".join(lines)


def run_micro_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the complete 7-arm CPU micro-pilot and persist JSON + Markdown reports."""
    t0 = time.perf_counter()
    set_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    tokenizer = PilotTokenizer()
    data_cfg = DataConfig(
        seed=args.seed,
        train_depths=(2, 4),
        eval_ood_depths=(6, 8),
        n_train_samples=args.n_train,
        n_eval_samples=max(args.eval_per_depth * 3, args.eval_aleatoric),
        max_seq_len=args.max_seq_len,
    )
    datasets = build_stratified_datasets(data_cfg, tokenizer)

    collate = partial(collate_batch, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    rl_train_loader = DataLoader(
        datasets["train"],
        batch_size=args.rl_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        datasets["eval_id"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    # Balanced Regime 1 eval set covering ID d in {2,4} + OOD d in {6,8}, plus Regime 3
    eval_r1_id = _select_balanced_by_depth(datasets["eval_id"].tokenized, per_depth=args.eval_per_depth)
    eval_r1_ood = _select_balanced_by_depth(datasets["eval_ood"].tokenized, per_depth=args.eval_per_depth)
    eval_r1_all = eval_r1_id + eval_r1_ood
    eval_r1_passk = _select_balanced_by_depth(eval_r1_all, per_depth=args.pass_k_per_depth)
    eval_r3 = datasets["eval_aleatoric"].tokenized[: args.eval_aleatoric]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "micro_runs"

    train_telemetry: dict[str, dict[str, Any]] = {}
    models: dict[str, torch.nn.Module] = {}

    # 1. B1 — micro-small NTP
    print("[1/7] Training B1 (micro-small NTP)...")
    cfg_b1 = ModelConfig.preset(
        "micro-small", vocab_size=tokenizer.vocab_size, max_seq_len=args.max_seq_len, variant="ntp"
    )
    m_b1 = build_model(cfg_b1)
    res_b1 = train_supervised(
        m_b1,
        train_loader,
        val_loader,
        TrainConfig(
            preset="micro-small",
            batch_size=args.batch_size,
            max_steps=args.sup_steps,
            lr=1.5e-3,
            log_every=max(1, args.sup_steps // 4),
            output_dir=str(runs_dir / "b1"),
            seed=args.seed,
            device="cpu",
        ),
        pad_id=tokenizer.pad_id,
    )
    _check_finite_history(res_b1["history"], "B1")
    models["B1"] = m_b1
    train_telemetry["B1"] = {
        "type": "supervised",
        "steps": res_b1["steps"],
        "final_train_loss": res_b1["final_train_loss"],
        "final_val_loss": res_b1["final_val_loss"],
    }

    # 2. B2 — micro-large (~2.5x) NTP
    print("[2/7] Training B2 (micro-large ~2.5x NTP)...")
    cfg_b2 = ModelConfig.preset(
        "micro-large", vocab_size=tokenizer.vocab_size, max_seq_len=args.max_seq_len, variant="ntp"
    )
    m_b2 = build_model(cfg_b2)
    res_b2 = train_supervised(
        m_b2,
        train_loader,
        val_loader,
        TrainConfig(
            preset="micro-large",
            batch_size=args.batch_size,
            max_steps=args.sup_steps,
            lr=1.2e-3,
            log_every=max(1, args.sup_steps // 4),
            output_dir=str(runs_dir / "b2"),
            seed=args.seed + 1,
            device="cpu",
        ),
        pad_id=tokenizer.pad_id,
    )
    _check_finite_history(res_b2["history"], "B2")
    models["B2"] = m_b2
    train_telemetry["B2"] = {
        "type": "supervised",
        "steps": res_b2["steps"],
        "final_train_loss": res_b2["final_train_loss"],
        "final_val_loss": res_b2["final_val_loss"],
    }

    # 3. B3 — micro-small + Pause Tokens
    print("[3/7] Training B3 (micro-small + Pause tokens)...")
    cfg_b3 = ModelConfig.preset(
        "micro-small",
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        variant="pause",
        n_pause_tokens=args.n_latent_steps,
    )
    m_b3 = build_model(cfg_b3, pause_token_id=tokenizer.pause_id)
    res_b3 = train_supervised(
        m_b3,
        train_loader,
        val_loader,
        TrainConfig(
            preset="micro-small",
            batch_size=args.batch_size,
            max_steps=args.sup_steps,
            lr=1.5e-3,
            log_every=max(1, args.sup_steps // 4),
            output_dir=str(runs_dir / "b3"),
            seed=args.seed + 2,
            device="cpu",
        ),
        pad_id=tokenizer.pad_id,
    )
    _check_finite_history(res_b3["history"], "B3")
    models["B3"] = m_b3
    train_telemetry["B3"] = {
        "type": "supervised",
        "steps": res_b3["steps"],
        "final_train_loss": res_b3["final_train_loss"],
        "final_val_loss": res_b3["final_val_loss"],
    }

    # 4. B4 — micro-small + Continuous Latent BPTT
    print("[4/7] Training B4 (micro-small + Continuous Latent BPTT)...")
    cfg_b4 = ModelConfig.preset(
        "micro-small",
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        variant="continuous_latent",
        n_latent_steps=args.n_latent_steps,
    )
    m_b4 = build_model(cfg_b4)
    res_b4 = train_supervised(
        m_b4,
        train_loader,
        val_loader,
        TrainConfig(
            preset="micro-small",
            batch_size=args.batch_size,
            max_steps=args.sup_steps,
            lr=1.5e-3,
            log_every=max(1, args.sup_steps // 4),
            output_dir=str(runs_dir / "b4"),
            seed=args.seed + 3,
            device="cpu",
        ),
        pad_id=tokenizer.pad_id,
    )
    _check_finite_history(res_b4["history"], "B4")
    models["B4"] = m_b4
    train_telemetry["B4"] = {
        "type": "supervised",
        "steps": res_b4["steps"],
        "final_train_loss": res_b4["final_train_loss"],
        "final_val_loss": res_b4["final_val_loss"],
    }

    # 5. RL-R1 — micro-small (from B1) + Discrete RL T2_self + R1_exact
    print("[5/7] Training RL-R1 (T2_self + R1_exact, Dr. GRPO + DAPO)...")
    m_rl_r1 = copy.deepcopy(m_b1)
    res_rl_r1 = train_ssl_rl(
        m_rl_r1,
        rl_train_loader,
        val_loader,
        TrainConfig(
            preset="micro-small",
            batch_size=args.rl_batch_size,
            max_steps=args.rl_steps,
            lr=3e-4,
            log_every=max(1, args.rl_steps // 4),
            output_dir=str(runs_dir / "rl_r1"),
            seed=args.seed + 4,
            device="cpu",
        ),
        RLConfig(
            reward_type="R1_exact",
            filter_type="T2_self",
            interface="I1_discrete",
            group_size=args.group_size,
            n_ahead=args.n_ahead,
            thought_len=args.thought_len,
            clip_low=0.2,
            clip_high=0.28,
            lambda_ntp=0.2,
            top_p_tokens=0.2,
            top_p_thought_entropy=0.2,
            lr=3e-4,
            ref_ema_decay=1.0,
            filter_zero_var=True,
        ),
        pad_id=tokenizer.pad_id,
        think_start_id=tokenizer.think_start_id,
        think_end_id=tokenizer.think_end_id,
    )
    _check_finite_history(res_rl_r1["history"], "RL-R1")
    models["RL-R1"] = m_rl_r1
    train_telemetry["RL-R1"] = {
        "type": "rl",
        "steps": res_rl_r1["steps"],
        "final_metrics": res_rl_r1["final_metrics"],
        "final_val_loss": res_rl_r1["final_val_loss"],
    }

    # 6. RL-R3 — micro-small (from B1) + Discrete RL T2_self + R3_brier (frozen anchor)
    print("[6/7] Training RL-R3 (T2_self + R3_brier frozen anchor, Dr. GRPO + DAPO)...")
    m_rl_r3 = copy.deepcopy(m_b1)
    res_rl_r3 = train_ssl_rl(
        m_rl_r3,
        rl_train_loader,
        val_loader,
        TrainConfig(
            preset="micro-small",
            batch_size=args.rl_batch_size,
            max_steps=args.rl_steps,
            lr=3e-4,
            log_every=max(1, args.rl_steps // 4),
            output_dir=str(runs_dir / "rl_r3"),
            seed=args.seed + 5,
            device="cpu",
        ),
        RLConfig(
            reward_type="R3_brier",
            filter_type="T2_self",
            interface="I1_discrete",
            group_size=args.group_size,
            n_ahead=args.n_ahead,
            thought_len=args.thought_len,
            clip_low=0.2,
            clip_high=0.28,
            lambda_ntp=0.2,
            top_p_tokens=0.2,
            top_p_thought_entropy=0.2,
            lr=3e-4,
            ref_ema_decay=1.0,
            filter_zero_var=True,
        ),
        pad_id=tokenizer.pad_id,
        think_start_id=tokenizer.think_start_id,
        think_end_id=tokenizer.think_end_id,
    )
    _check_finite_history(res_rl_r3["history"], "RL-R3")
    models["RL-R3"] = m_rl_r3
    train_telemetry["RL-R3"] = {
        "type": "rl",
        "steps": res_rl_r3["steps"],
        "final_metrics": res_rl_r3["final_metrics"],
        "final_val_loss": res_rl_r3["final_val_loss"],
    }

    # 7. I2-Hybrid — B4 Continuous Latent -> HRPO Hybrid RL (R3_brier)
    print("[7/7] Training I2-Hybrid (B4 continuous latent -> HRPO Hybrid RL + R3_brier)...")
    cfg_i2 = ModelConfig.preset(
        "micro-small",
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        variant="hybrid_latent",
        n_latent_steps=args.n_latent_steps,
    )
    m_i2 = build_model(cfg_i2)
    # Warm-initialize base_lm + latent_proj + latent_norm from trained B4 checkpoint
    m_i2.load_state_dict(m_b4.state_dict(), strict=False)
    res_i2 = train_ssl_rl(
        m_i2,
        rl_train_loader,
        val_loader,
        TrainConfig(
            preset="micro-small",
            batch_size=args.rl_batch_size,
            max_steps=args.rl_steps,
            lr=3e-4,
            log_every=max(1, args.rl_steps // 4),
            output_dir=str(runs_dir / "i2_hybrid"),
            seed=args.seed + 6,
            device="cpu",
        ),
        RLConfig(
            reward_type="R3_brier",
            filter_type="T2_self",
            interface="I2_hybrid",
            group_size=args.group_size,
            n_ahead=args.n_ahead,
            thought_len=args.thought_len,
            clip_low=0.2,
            clip_high=0.28,
            lambda_ntp=0.2,
            top_p_tokens=0.2,
            top_p_thought_entropy=0.2,
            lr=3e-4,
            ref_ema_decay=1.0,
            filter_zero_var=True,
        ),
        pad_id=tokenizer.pad_id,
        think_start_id=tokenizer.think_start_id,
        think_end_id=tokenizer.think_end_id,
    )
    _check_finite_history(res_i2["history"], "I2-Hybrid")
    models["I2-Hybrid"] = m_i2
    train_telemetry["I2-Hybrid"] = {
        "type": "rl",
        "steps": res_i2["steps"],
        "final_metrics": res_i2["final_metrics"],
        "final_val_loss": res_i2["final_val_loss"],
    }

    # -------------------------------------------------------------------------
    # Run all 5 Diagnostics across all 7 arms
    # -------------------------------------------------------------------------
    print("Running 5 mandatory Track 1 §5.5 diagnostics across all 7 arms...")
    bundles: dict[str, ArmEvalBundle] = {}
    arm_specs = [
        ("B1", False, 0, "ntp", False, False, None),
        ("B2", False, 0, "ntp", False, False, None),
        ("B3", False, 0, "pause", False, False, None),
        ("B4", False, 0, "continuous", False, True, None),
        (
            "RL-R1",
            True,
            args.thought_len,
            "discrete",
            True,
            False,
            float(res_rl_r1["final_metrics"]["nothought_nll_drift"]),
        ),
        (
            "RL-R3",
            True,
            args.thought_len,
            "discrete",
            True,
            False,
            float(res_rl_r3["final_metrics"]["nothought_nll_drift"]),
        ),
        (
            "I2-Hybrid",
            True,
            args.thought_len,
            "hybrid",
            True,
            True,
            float(res_i2["final_metrics"]["nothought_nll_drift"]),
        ),
    ]
    for arm_name, use_th, th_len, f_mode, is_rl, is_cont, drift in arm_specs:
        print(f"  Evaluating {arm_name}...")
        bundles[arm_name] = _evaluate_arm(
            arm_name=arm_name,
            model=models[arm_name],
            tokenizer=tokenizer,
            eval_r1_all=eval_r1_all,
            eval_r1_passk=eval_r1_passk,
            eval_r3=eval_r3,
            use_thought=use_th,
            thought_len=th_len,
            faithfulness_mode=f_mode,
            pass_k_n=args.pass_k_n,
            is_rl_arm=is_rl,
            is_teacher_free=True,
            is_continuous_or_hybrid=is_cont,
            nothought_nll_drift=drift,
            seed=args.seed,
        )

    verdict = compute_pilot_verdict(
        b1=bundles["B1"],
        b2=bundles["B2"],
        b3=bundles["B3"],
        b4=bundles["B4"],
        candidate_arms=[bundles["RL-R1"], bundles["RL-R3"], bundles["I2-Hybrid"]],
    )
    elapsed = time.perf_counter() - t0

    payload = {
        "scope": "CPU micro-scale proof-of-life & pipeline validation (distinct from 160M/410M/1.0B H100 Stage 1)",
        "elapsed_seconds": elapsed,
        "config": vars(args),
        "verdict": verdict.to_dict(),
        "arms": {k: v.to_dict() for k, v in bundles.items()},
        "training_telemetry": train_telemetry,
    }
    json_path = out_dir / "micro_pilot_metrics.json"
    md_path = out_dir / "micro_pilot_report.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_report = _build_report_markdown(
        verdict=verdict,
        bundles=bundles,
        train_telemetry=train_telemetry,
        args=args,
        elapsed_sec=elapsed,
    )
    md_path.write_text(md_report, encoding="utf-8")

    print(f"\nMicro-pilot complete in {elapsed:.1f}s.")
    print(f"Verdict: {verdict.status} | Best arm: {verdict.best_rl_arm}")
    print(f"Saved JSON metrics: {json_path}")
    print(f"Saved Markdown report: {md_path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CPU micro-scale Phase 1 Latent-CoT RL pilot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--eval-per-depth", type=int, default=16)
    p.add_argument("--eval-aleatoric", type=int, default=48)
    p.add_argument("--pass-k-per-depth", type=int, default=8)
    p.add_argument("--pass-k-n", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rl-batch-size", type=int, default=8)
    p.add_argument("--sup-steps", type=int, default=80)
    p.add_argument("--rl-steps", type=int, default=20)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--n-ahead", type=int, default=4)
    p.add_argument("--thought-len", type=int, default=6)
    p.add_argument("--n-latent-steps", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=192)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results"),
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run_micro_pilot(parse_args())
