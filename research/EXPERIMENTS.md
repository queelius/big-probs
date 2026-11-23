# Experiment Ledger & Pareto Frontier Tracker

> **Machine-Readable Source of Truth:** [`research/experiments.jsonl`](experiments.jsonl) (append-only NDJSON).
> **Logging Tool:** `cd phase1_pilot && uv run python scripts/log_experiment.py --help`
> **Preregistration Template:** [`research/templates/EXP_PREREG.md`](templates/EXP_PREREG.md)

---

## 1. Master Experiment Index

| Exp ID | Date | Title | RQs / Hypotheses | Scale | Compute | Status | Verdict |
| :--- | :--- | :--- | :--- | :--- | ---: | :---: | :---: |
| **`EXP-001`** | `2026-11-23` | Phase 1 CPU Micro-Scale 7-Arm Proof-of-Life | `RQ-01`, `H1`, `H1a`, `H1b`, `H1c` | `0.98M` vs `2.85M` (80+20 steps) | `0.026` CPU-hrs (`93.1s`) | `completed` | `FALSIFIED` *(micro-scale)* |
| **`EXP-002`** | `planned` | Phase 1 Stage 1 H100 Core Pilot | `RQ-01`, `H1`, `H1a`, `H1b`, `H1c` | `160M` vs `410M`/`1.0B` (2B tok) | `~35–40` H100-hrs | `planned` | `PENDING` |
| **`EXP-003`** | `planned` | Phase 2 Factorized Matrix & H1 Transfer Audit | `RQ-01`, `H1`, `H1a`, `H1c` | `Qwen2.5-0.5B/1.5B` vs `3B/7B` | `~250–400` H100-hrs | `planned` | `PENDING` |
| **`EXP-004`** | `planned` | Phase 3 Cross-Domain Tier 0B–2 Stack & H2 Test | `RQ-02`, `RQ-03`, `H2` | PDE / Contract PoV + PAJAMA + PP | `~500–1000` H100-hrs | `planned` | `PENDING` |
| **`EXP-005`** | `planned` | Phase 4 Causal Interventional Replay & H3 Loop | `RQ-03`, `H3` | AIPW + Co-Adaptive Sim + Tier 3 | Domain-specific | `planned` | `PENDING` |

---

## 2. Current Pareto Frontier Summary

> Auto-generated from [`research/experiments.jsonl`](experiments.jsonl) via `scripts/log_experiment.py --render-only --update-markdown`. Pareto-efficient arms (`Pareto = Yes`) are non-dominated on `(FLOPs ratio ↓, Hard-BPB ↓, Exact Acc ↑, OOD pass@k ↑)` within their experiment scale.

<!-- PARETO_TABLE_START -->
| Exp ID | Arm | Params | FLOPs (vs B2) | Exact Acc | Hard-BPB | OOD pass@k | ΔNLL_swap | ΔNLL_trunc | Reg3 ECE | σ=0 Frac | Pareto? |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `EXP-001` | `B1` | 984,192 | 0.34× | 3.1% | 2.318 | 31.2% | — | — | 0.005 | — | Yes |
| `EXP-001` | `B2` | 2,853,312 | 1.00× | 6.2% | 2.266 | 31.2% | — | — | 0.040 | — | Yes |
| `EXP-001` | `B3` | 984,704 | 1.71× | 0.0% | 2.317 | 31.2% | — | — | 0.032 | — | No |
| `EXP-001` | `B4` | 1,000,704 | 1.74× | 3.1% | 2.272 | 18.8% | -0.001 | +2.909 | 0.045 | — | No |
| `EXP-001` | `RL-R1` | 984,192 | 3.10× | 3.1% | 2.326 | 31.2% | -0.002 | +0.001 | 0.026 | 1.00 | No |
| `EXP-001` | `RL-R3` | 984,192 | 3.10× | 1.6% | 2.309 | 25.0% | +0.000 | -0.001 | 0.040 | 0.00 | No |
| `EXP-001` | `I2-Hybrid` | 1,033,600 | 2.51× | 3.1% | 2.271 | 37.5% | -0.003 | +2.732 | 0.032 | 0.00 | Yes |
<!-- PARETO_TABLE_END -->

---

## 3. Completed Experiment Post-Mortems

### `EXP-001` — Phase 1 CPU Micro-Scale 7-Arm Proof-of-Life & Diagnostic Validation
- **Date:** `2026-11-23` (22:39 UTC) | **Runtime:** `93.11s` (`0.0259` CPU-hrs on 12-core AMD Ryzen 5 7430U, 62 GiB RAM, CPU-only)
- **Config:** `seed=42`, `n_train=600`, `eval_per_depth=16` (`64` Regime 1 samples across $d\in\{2,4,6,8\}$), `eval_aleatoric=48` (`48` Regime 3 samples), `pass_k_n=16` (`k in {1,4,16}`), `sup_steps=80` (`bs=16`), `rl_steps=20` (`bs=8`, `G=4`, `thought_len=6`, `n_ahead=4`, `n_latent_steps=4`, `max_seq_len=192`).
- **Verdict:** `FALSIFIED` *(at micro-scale)* — No small-model arm simultaneously beat pause control `B3` (`0.0%`) and FLOPs-matched `B2` (`6.2%`). Best candidate `I2-Hybrid` reached `3.1%` exact acc at `2.51×` `B2` FLOPs.
- **Artifacts:**
  - Local report: [`phase1_pilot/results/micro_pilot_report.md`](../phase1_pilot/results/micro_pilot_report.md)
  - Local raw JSON: [`phase1_pilot/results/micro_pilot_metrics.json`](../phase1_pilot/results/micro_pilot_metrics.json)
  - Local checkpoints: `phase1_pilot/results/micro_runs/{b1,b2,b3,b4,rl_r1,rl_r3,i2_hybrid}`
  - HuggingFace Models: [`queelius/big-probs-phase1-models`](https://huggingface.co/queelius/big-probs-phase1-models) (`micro-pilot-v1/`)
  - HuggingFace Datasets: [`queelius/big-probs-phase1-data`](https://huggingface.co/datasets/queelius/big-probs-phase1-data)

#### Full 7-Arm Telemetry & Diagnostics

| Arm | Variant | Params | FLOPs/tok (vs B2) | Exact Acc | Hard-BPB | OOD `pass@16` | $\Delta\text{NLL}_{\text{swap}}$ | $\Delta\text{NLL}_{\text{trunc}}$ | Reg3 ECE | Reg3 Brier | `σ=0` Frac | No-Thought Drift |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1` | `ntp` | 984,192 | 0.34× | 3.1% | 2.318 | 31.2% | — | — | 0.005 | -0.994 | — | — |
| `B2` | `ntp` | 2,853,312 | 1.00× | 6.2% | 2.266 | 31.2% | — | — | 0.040 | -1.006 | — | — |
| `B3` | `pause` | 984,704 | 1.71× | 0.0% | 2.317 | 31.2% | — | — | 0.032 | -1.003 | — | — |
| `B4` | `continuous_latent` | 1,000,704 | 1.74× | 3.1% | 2.272 | 18.8% | -0.001 | +2.909 | 0.045 | -0.989 | — | — |
| `RL-R1` | `ntp+discrete_thought` | 984,192 | 3.10× | 3.1% | 2.326 | 31.2% | -0.002 | +0.001 | 0.026 | -0.997 | 1.00 | -0.0686 |
| `RL-R3` | `ntp+discrete_thought` | 984,192 | 3.10× | 1.6% | 2.309 | 25.0% | +0.000 | -0.001 | 0.040 | -1.005 | 0.00 | +0.0559 |
| `I2-Hybrid` | `hybrid_latent+discrete_thought` | 1,033,600 | 2.51× | 3.1% | 2.271 | 37.5% | -0.003 | +2.732 | 0.032 | -0.998 | 0.00 | -0.5746 |

#### Key Scientific & Engineering Takeaways
1. **Confirmed `H1a` Starvation Mechanism (`RL-R1` vs. `RL-R3`/`I2-Hybrid`):** Binary exact-match `RL-R1` hit **`sigma=0` fraction = `1.00`** (`active_group_frac = 0.0`, `reward_std = 0.0`), yielding zero policy-gradient learning signal. Strictly proper Brier (`RL-R3` and `I2-Hybrid`) eliminated starvation completely (**`sigma=0` fraction = `0.00`**, `active_group_frac = 0.25`, `reward_std ≈ 0.095–0.099`).
2. **`I2-Hybrid` Dominates Small-Model Arms on Quality & OOD Support:** `I2-Hybrid` matched `B2` (2.85M) Hard-BPB (**`2.271`** vs. **`2.266`**; `B1 = 2.318`), outperformed `B1`/`B2`/`B3` on OOD `pass@16` (**`37.5%`** vs. **`31.2%`**), beat `B3` (`0.0%`) by `+3.1` pp, and kept Regime 3 calibration clean (`ECE = 0.0325`, `Brier = -0.9982`).
3. **Anti-Sabotage Wiring Held Under Live RL:** Frozen reference anchor $p_{\theta_{\text{ref}}}$ + $\lambda_{\text{ntp}} = 0.2$ prevented no-thought baseline degradation (`RL-R3` drift `+0.0559` nats; `I2-Hybrid` drift `-0.5746` nats, meaning no-thought NLL actually *improved* from `2.9716` to `2.3970`).
4. **Why `ΔNLL_trunc` Was Large (`+2.73`–`+2.91`) While `ΔNLL_swap` Was `~0`:** Both `B4` and `I2-Hybrid` decoders depend strongly on the learned continuous latent subspace (norm-matched orthogonal perturbation costs `+2.73` to `+2.91` nats), but 80 supervised + 20 RL steps at 0.98M params is too short to learn prompt-specific multi-hop intermediate states (`ΔNLL_swap ≈ 0`).
5. **FLOPs Budget Lesson for Stage 1:** Running `thought_len=6` on *every* target token puts `I2-Hybrid` at `2.51×` `B2` FLOPs (`RL-R1`/`RL-R3` at `3.10×`). To test the strict `≤ 1.0× B2` isoFLOPs gate in `EXP-002`, small models must either use shorter latent horizons or sparse/adaptive inference triggering (`top-20%` compute-elastic tokens).

#### Bugs Caught During Audit & Verified Fixed
1. **`I2-Hybrid` Train-vs-Eval Interface Skew (`CRITICAL`, caught by `verifier`):** `diagnostics.py` initially prepended `4` pure continuous latents AND wrapped the `6` HRPO-blended steps in `<think>`/`</think>` (`12` extra steps total), whereas `ContinuousLatentWrapper.rollout_hybrid` trains with `thought_len` blended steps directly after `prefix_ids` and no delimiters. That mismatch produced spurious `NLL=7.68`, `Hard-BPB=5.030`, `ECE=0.844`, and `4.71×` `B2` FLOPs! Fixed to exact `< 1e-4` parity with `recompute_hybrid_logprobs_and_continuation` (`extra_steps == thought_len`), locked with `test_hybrid_latent_eval_matches_recompute_hybrid_parity`, and regenerated `EXP-001` cleanly.
2. **Multi-GPU `torchrun` Distributed Sync (`MEDIUM`, caught by `verifier`):** `train.py` read `LOCAL_RANK`/`RANK` without initializing `torch.distributed` or all-reducing gradients, which would have run 4 uncoordinated duplicates on 4× H100. Added `init_distributed`/`cleanup_distributed`, rank-0 broadcast, rank-sharded sampling, and `dist.all_reduce(..., op=AVG)` in `train.py` & `grpo.py`, verified by a 2-rank Gloo unit test.

---

## 4. Planned Experiments (`EXP-002` – `EXP-005`)

### `EXP-002` — Phase 1 Stage 1 H100 Core Pilot (`~35–40` H100 GPU-hrs)
- **Hypotheses:** `RQ-01`, `H1`, `H1a`, `H1b`, `H1c`
- **Arms (8):** `B1` (`160M` NTP, ~1.5h), `B2a` (`410M` NTP, ~3.5h), `B2b` (`1.0B` NTP, ~9.0h), `B3` (`160M` Pause, ~3.0h), `B4` (`160M` Continuous Latent `L=4`, ~4.5h), `RL-R2` (`160M` `T2-self` + clipped `R2_log`, ~6.0h), `RL-R3` (`160M` `T2-self` + `R3_brier`, ~6.0h), `I2-Hybrid` (`B4` $\to$ HRPO + `R3_brier`, ~6.5h).
- **Data:** 2B tokens stratified across Regime 1 (DAGs $d\in\{2,4,6,8\}$, Python traces, modular var-bind), Regime 2 (OpenWebMath 40% / Stack v2 40% / FineWeb-Edu 20%), and Regime 3 (TriviaQA / aleatoric negative control).
- **Go / No-Go Gate:** `160M` + latent CoT matches/beats `410M` (`B2a`) at $\le 1.0\times$ inference FLOPs, beats `B3` by $\ge 3$ pp on Regime 1, passes $\Delta\text{NLL}_{\text{swap}} > 0.15$ nats + $\Delta\text{NLL}_{\text{trunc}} > 0$ + monotonic OOD depth scaling, maintains Regime 3 calibration, and strictly increases OOD `pass@64` ($d=6,8$).
- **Launcher:** [`phase1_pilot/scripts/launch_h100_stage1.sh`](../phase1_pilot/scripts/launch_h100_stage1.sh)

### `EXP-003` — Phase 2 Factorized Matrix & `H1` Transfer Audit (`~250–400` H100 GPU-hrs)
- **Hypotheses:** `RQ-01`, `H1` (decisive test), `H1a`, `H1c`
- **Arms:** Continued-pretraining RL on `Qwen2.5-0.5B-Base` and `Qwen2.5-1.5B-Base` vs. `Qwen2.5-3B/7B-Base` ceilings (50M–100M filtered tokens, `G=8`, Dr. GRPO + DAPO Clip-Higher $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$, $\lambda_{\text{ntp}}=0.2$). Full factorized matrix: (`T0` / `T1` / `T2-self` / `T2-scale`) × (`R1` / `R2` / `R3`) × (`I1` discrete vs. `I2` hybrid HRPO) + **`B5` off-policy 7B-distilled SFT control** + 5k-example CoT warm-start ablation.
- **Go / No-Go Gate:** Confirm `H1` iff `I2-Hybrid` strictly beats raw discrete `I1` on OOD `pass@64` ($d=6,8$) AND strictly beats `B5` off-policy SFT on non-target transfer (coding, general QA) and PCA latent-geometry preservation at matched inference FLOPs.

### `EXP-004` — Phase 3 Cross-Domain Tier 0B–2 Stack & `H2` Stealth-Goodhart Test (`~500–1,000` H100 GPU-hrs)
- **Hypotheses:** `RQ-02`, `RQ-03`, `H2` (decisive test)
- **Target Domain:** Scientific PDE Solver Synthesis (`RLVP` continuous residuals + `IPT` metamorphic invariance + grounded inversion) OR Executable Contract/Tax PoV Auditing (`Law & Order` rules + Legal PoV loophole finder + `RcRL` rubrics).
- **Arms:** (a) $K$-model same-corpus LLM WCO/UWO ensemble vs. (b) modality-decorrelated stack (`PAJAMA` compiled programmatic committees + `IPT` + isolated surrogates + Peer Prediction + 2% randomized spot-audits) across GRPO KL-sweeps (`Gao et al.` protocol).
- **Go / No-Go Gate:** Sustain rising held-out ground-truth accuracy over $10^3+$ RL steps while cutting stealth-Goodhart FP escape rate by $\ge 50\%$ vs. (a) at matched gold reward and flagging drift before gold reward turns negative.

### `EXP-005` — Phase 4 Causal Interventional Replay & `H3` Causal Latency Floor
- **Hypotheses:** `RQ-03`, `H3` (decisive test)
- **Arms:** (a) Naive masked observational replay, (b) propensity-weighted AIPW quasi-experimental replay, (c) AIPW replay + co-adaptive Tier 2 multi-agent simulator + budgeted Tier 3 randomized micro-pilots (validated first in structural causal sim, then coupled to live `ForecastBench` or autonomous lab anchor).
- **Go / No-Go Gate:** (b) eliminates (a)'s in-support ordering-leakage/confounding collapse; (c) achieves strictly higher out-of-support interventional lift with a smooth shift-vs-Tier-3-budget Pareto curve; across $\ge 3$ live slow recalibration epochs, demonstrates automatic CCMV auto-decay of a Goodharted proxy milestone and higher real hit-rate per dollar-day than naive top-$K$ promotion.
