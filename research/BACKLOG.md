# Frontier Exploration Queue, Missing Pieces & Dead-Ends Graveyard

> **Prioritization Rule:** Rank work by **Value of Information per Compute-Dollar** ($\text{VoI} / (\$ \cdot \tau)$). De-risk mechanisms cheaply on local CPU (12 cores / 62 GiB RAM) before committing H100 GPU-hours.

---

## 1. Prioritized Exploration Queue

### Tier A — Runnable Immediately on Local CPU ($0 GPU Cost, Highest VoI/$)

| Priority | ID | Task | Target RQs | Est. Cost | Concrete Deliverable & Decision Unlocked |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **P0** | `BL-01` | **Longer Multi-Seed CPU Micro-Pilot (`3` seeds, `500–2000` sup + `100–300` RL steps)** | `RQ-01`, `H1a`, `H1b` | ~15–45 min CPU | Tests whether `ΔNLL_swap > 0` and non-trivial `d=2,4` accuracy (`>30%`) emerge at `0.98M`–`2.85M` scale once training exits the near-uniform logit regime, plus adds 3-seed CIs (`mean ± std`). |
| **P0** | `BL-02` | **CPU Filter × Reward Ablation (`T0`/`T1`/`T2-self`/`T2-scale` × `R1`/`R2`/`R3`)** | `H1a`, `H1c` | ~10–20 min CPU | First head-to-head comparison of `R2_log` (clipped $\tau=5.0$) vs. `R3_brier` and teacher-free `T2-self` vs. `T0`/`T1`/`T2-scale` on `micro-small` before spending H100 hours in `EXP-002`/`EXP-003`. |
| **P1** | `BL-03` | **Synthetic `H2` Correlated-Prior Stealth-Goodhart CPU Testbed** | `RQ-02`, `H2` | ~5 min CPU + ~1 day code | Runnable Python/PyTorch testbed with hidden gold reward, injected decoy traps, $K$-model shared-prior WCO/UWO ensemble vs. modality-decorrelated programmatic + Peer Prediction + 2% spot-audit stack across KL-sweeps (`Gao et al.` protocol). De-risks `EXP-004` before 500+ H100 hrs! |
| **P1** | `BL-04` | **Synthetic `H3` Causal Interventional Replay & CCMV CPU Testbed** | `RQ-03`, `H3` | ~5 min CPU + ~1 day code | Structural causal simulator with clinician ordering leakage, logged propensities, and strategic co-adaptation; compares naive masked replay vs. doubly robust AIPW replay vs. co-adaptive Tier 2 + budgeted Tier 3 micro-pilots + CCMV auto-decay. De-risks `EXP-005`! |
| **P2** | `BL-05` | **Automated Pareto Frontier & Reliability Plotting Script** | Program Ops | ~2 hrs code | Reads `research/experiments.jsonl` & `*_metrics.json` to emit FLOPs-vs-Accuracy/Hard-BPB Pareto scatter plots, `pass@k` depth curves, and Regime 3 calibration diagrams into `research/figures/`. |

### Tier B — Requires H100 GPU Allocation (35–400 H100 GPU-Hrs)

| Priority | ID | Task | Target RQs | Est. Cost | Concrete Deliverable & Decision Unlocked |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **P0** | `BL-11` | **Choose and Wire the Stage 1 Regime 2 Data Source (blocks `BL-06` as specified)** | `RQ-01`, `H1b` | Design decision + code | *Synthetic half done `2026-11-24` (branch `stage1-data-path`):* `generate_natural_stream` now widens a template's parameter domain once its original domain is used up, so the launcher defaults build (~12 s on CPU) and the data at every previously supported size is byte-identical (pinned by `test_generated_datasets_match_committed_artifacts`; `EXP-001` reproduced exactly). *Still open:* `BL-06` specifies real stratified tokens, but `load_h100_web_mixture` returns raw HF document streams and nothing turns documents into `Sample`s with one `<ans>` boundary. Decide (a) how documents become prompt/target samples, (b) which tokenizer the 32k presets use and how Regimes 1 and 3 are tokenized with it, (c) data access: `bigcode/the-stack-v2-dedup` is gated (the `queelius` HF account had not been granted access as of `2026-11-24`) and stores only blob IDs; the file contents sit in Software Heritage's S3 bucket and need AWS credentials. *Decided `2026-11-26`:* Stage 1 uses real Regime 2 text, and (c) is settled: `codeparrot/codeparrot-clean` (open, Python only, file contents inline) replaces Stack v2, which drops the planned C++ share; OpenWebMath and FineWeb-Edu `sample-10BT` are open with a plain `text` column. (a) and (b) are still open. |
| **P0** | `BL-07` | **One-Command Stage 1 Checkpoint Evaluator + Adaptive Inference FLOPs Mode** | `RQ-01`, `H1b` | ~3 hrs code (CPU-testable) | Build `phase1_pilot/scripts/eval_stage1.py` (loads all 8 `launch_h100_stage1.sh` checkpoints, runs all 5 diagnostics with `n_samples=64` for `pass@64`, outputs `stage1_metrics.json` directly consumable by `scripts/log_experiment.py`). Add sparse/entropy-gated inference triggering so `I2-Hybrid` / `RL-R3` can target `≤ 1.0× B2a` inference FLOPs. |
| **P0** | `BL-06` | **Execute `EXP-002` — Phase 1 Stage 1 H100 Core Pilot (`160M` vs. `410M`/`1.0B`)** | `RQ-01`, `H1`, `H1a`, `H1b` | **35–40 H100 GPU-hrs** | Decisive from-scratch test of Parametric De-Congestion (`H1b`), `R2` vs. `R3` at large vocab, and `I2-Hybrid` vs. `B2a`/`B3`/`B4` on 2B real stratified tokens (Regimes 1–3). Gate for Phase 2. |
| **P1** | `BL-08` | **Execute `EXP-003` — Phase 2 Factorized Matrix & `H1` Transfer Audit** | `RQ-01`, `H1`, `H1c` | **250–400 H100 GPU-hrs** | Continued-pretraining RL on `Qwen2.5-0.5B/1.5B-Base` vs. `3B/7B-Base`, full `T×R×I` matrix, plus **`B5` off-policy 7B teacher SFT distillation** + non-target transfer (coding/QA) + PCA latent-drift audit decisively testing Tension 1 (`H1`). |

### Tier C — Medium / Long-Term Frontier (500+ H100 GPU-Hrs / Live Anchors)

| Priority | ID | Task | Target RQs | Est. Cost | Concrete Deliverable & Decision Unlocked |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **P2** | `BL-09` | **Execute `EXP-004` — Phase 3 Cross-Domain Tier 0B–2 Stack Prototype** | `RQ-02`, `H2` | **500–1,000 H100 GPU-hrs** | End-to-end PDE Solver (`RLVP`+`IPT`) or Contract/Tax PoV stack with `PAJAMA` programmatic committees + Peer Prediction + 2% randomized spot-audits. |
| **P3** | `BL-10` | **Execute `EXP-005` — Phase 4 Live Tier 3 Reality Loop** | `RQ-03`, `H3` | Domain-specific | Couple Tier 0–2 stack to contamination-free live forecasting (`ForecastBench`-style) or closed-loop lab/assay partner across $\ge 3$ slow recalibration epochs. |

---

## 2. Explicit "What We're Missing" Audit

Honest inventory of every methodological, statistical, and engineering gap currently open in `big-probs`:

| Gap | Current State | Why It Matters | Closing Action |
| :--- | :--- | :--- | :--- |
| **Multi-Seed Confidence Intervals** | `EXP-001` ran a single seed (`seed=42`) on `64` Regime 1 / `48` Regime 3 samples. | Single-sample swings change accuracy by `±1.56` pp (`1/64`); cannot separate small differences (`3.1%` vs `1.6%`) from noise. | `BL-01` (3 seeds on CPU) + `≥ 2` seeds & `N ≥ 2000` eval samples in `EXP-002`. |
| **160M+ Capacity Regime** | Only `0.98M` vs `2.85M` tested live. | `ΔNLL_swap ≈ 0` and `≤ 6.2%` greedy accuracy at `0.98M`/80 steps cannot test `H1b` Parametric De-Congestion. | `BL-06` (`EXP-002` H100 Stage 1). |
| **Real Web-Text Regime 2 Eval** | Synthetic Regime 2 text only in `EXP-001`. | Need real OpenWebMath / Stack v2 / FineWeb-Edu (`40/40/20`) Hard-BPB + zero-shot GSM8K/MBPP/CommonsenseQA. | `BL-06` (`EXP-002` with `--extra h100` streaming loader). |
| **IsoFLOPs Adaptive Inference (`≤ 1.0× B2`)** | `I2-Hybrid` used `thought_len=6` on every target token (`2.51× B2` FLOPs). | Fails the strict `≤ 1.0× B2` FLOPs Pareto gate by construction even if accuracy improves. | `BL-07` (sparse top-20% adaptive triggering or shorter `thought_len` at eval). |
| **`B5` Off-Policy SFT Baseline & PCA Drift Probe** | Not yet implemented in `phase1_pilot/`. | Required to decisively test Tension 1 (`H1`: on-policy `I2-Hybrid` vs. off-policy distillation latent drift). | `BL-08` (`EXP-003`). |
| **Real-Text Regime 2 Data Path** | Found `2026-11-24` at commit `f03ff41`: the synthetic generator capped datasets at ~1,080 natural samples, so the launcher defaults (`200000` / `2000`) failed in job 1. The cap is fixed on branch `stage1-data-path`. `HFTokenizerAdapter` and `load_h100_web_mixture` are still not called by `train.py`. | `EXP-002` now runs at its default sizes, but it trains the 32k-vocab presets on the 512-ID synthetic data, not the real stratified tokens `BL-06` specifies. | `BL-11` (remaining half). |
| **One-Command H100 Stage 1 Evaluator** | `launch_h100_stage1.sh` trains all 8 models but leaves diagnostic evaluation manual. | Eliminates manual post-run friction and ensures automatic logging to `research/experiments.jsonl`. | `BL-07` (`scripts/eval_stage1.py`). |
| **Runnable `H2` & `H3` Code Testbeds** | Theoretical specifications only (`00-unified-framework.md §4`). | Tracks 2 & 3 currently have zero runnable falsification harnesses. | `BL-03` & `BL-04` (CPU-runnable synthetic prototypes). |
| **Automated Pareto & Calibration Plotting** | Markdown tables only (`EXPERIMENTS.md`). | Visual Pareto frontiers make frontier shifts vs. movement along a known curve immediately obvious. | `BL-05`. |

---

## 3. Dead Ends & Anti-Pattern Graveyard

> **Rule (see [`research/PROCEDURE.md`](PROCEDURE.md)):** Never silently delete a failed idea or caught bug. Log every dead end here with its exact failure signature and mandatory defense so no future human or AI session wastes compute rediscovering it.

### `DE-001` — Binary Exact-Match Reward (`R1`) on Low-Accuracy Small Models
- **Observed In:** `EXP-001` (`RL-R1`, `micro-small` 0.98M) & theoretical analysis ([`01-ssl-as-rl-latent-cot.md §1.3`](../01-ssl-as-rl-latent-cot.md#L56-L79)).
- **Failure Signature:** `zero_var_frac = 1.00`, `active_group_frac = 0.00`, `reward_std = 0.000`, `loss_pg = 0.000`. All $G$ rollouts in every group produce identical binary rewards, starving GRPO of gradient signal; also non-strictly-proper (mode-collapsing on high-entropy tokens).
- **Mandatory Defense:** Default to bounded strictly proper **Brier-score gain (`R3_brier`)** or clipped log-likelihood gain (`R2_log`, $\tau=5.0$) with dynamic $\sigma=0$ group filtering; keep `R1_exact` strictly as a negative-control ablation arm.

### `DE-002` — Un-Anchored Live No-Thought Baseline in Shared-$\theta$ SSL-as-RL
- **Observed In:** Theoretical derivation ([`00-unified-framework.md §3.3`](../00-unified-framework.md#L134-L135)) & locked via anti-sabotage unit tests (`tests/test_rl.py`).
- **Failure Signature:** When the same parameters $\theta$ compute both $p_\theta(y_t \mid x_{<t}, z_t)$ and the live baseline $p_\theta(y_t \mid x_{<t}, \varnothing)$, optimization inflates advantage by *degrading* no-thought prediction (positive `nothought_nll_drift` blowup) instead of improving thought quality.
- **Mandatory Defense:** Always compute counterfactual gain against a **frozen, stop-gradiented reference anchor** $p_{\theta_{\text{ref}}}$ (`requires_grad=False`, `ref_ema_decay=1.0`) AND retain auxiliary no-thought NTP loss $\lambda_{\text{ntp}} \mathcal{L}_{\text{NTP}}(\theta)$ (`lambda_ntp = 0.2`), monitoring `nothought_nll_drift` in telemetry.

### `DE-003` — Train-vs-Eval Latent Interface & Delimiter Skew (`I2-Hybrid` Bug)
- **Observed In:** `EXP-001` initial pre-audit run (caught by `verifier` in `phase1_pilot/eval/diagnostics.py`).
- **Failure Signature:** Training (`ContinuousLatentWrapper.rollout_hybrid`) ran `thought_len` HRPO-blended steps directly after `prefix_ids` with *no* prepended pure continuous latents and *no* `<think>`/`</think>` delimiters, while eval initially prepended `4` continuous latents + `<think>`/`</think>` (`12` extra steps). This produced catastrophic fake failure metrics (`NLL=7.68`, `Hard-BPB=5.030`, `Reg3 ECE=0.844`, `4.71× B2` FLOPs) even though the checkpoint actually had `NLL=3.369`, `Hard-BPB=2.271`, `ECE=0.0325`, `2.51×` FLOPs!
- **Mandatory Defense:** Every new latent/thought interface MUST ship with a unit test asserting `< 1e-4` continuation-logit parity between the evaluation path and the exact training recomputation path (see `test_hybrid_latent_eval_matches_recompute_hybrid_parity` in `phase1_pilot/tests/test_eval.py`).

### `DE-004` — Uncoordinated Multi-GPU `torchrun` Without Process-Group Sync
- **Observed In:** Pre-audit `phase1_pilot/train.py` & `grpo.py` (caught by `verifier`).
- **Failure Signature:** Reading `RANK`/`LOCAL_RANK` without calling `torch.distributed.init_process_group`, rank-0 weight broadcast, rank-sharded sampling, and gradient `all_reduce(..., op=AVG)` silently runs $N$ duplicate single-GPU jobs under `torchrun`, burning $N\times$ H100-hours for $1\times$ learning.
- **Mandatory Defense:** Require explicit 2-rank Gloo distributed unit test covering gradient synchronization whenever modifying `train.py` or `SSLRLTrainer`.

### `DE-005` — Multiplicative Zero-FP Hard Gating Without Cold-Start Shaping
- **Observed In:** Track 2 analysis ([`00-unified-framework.md §3.2`](../00-unified-framework.md#L127-L128); [`02-verification-asymmetry-and-inversion.md §4`](../02-verification-asymmetry-and-inversion.md#L247-L323)).
- **Failure Signature:** Multiplying $m$ binary zero-FP structural gates $\prod_{j=1}^m \mathbb{I}[I_j(x,y)]$ drives reward identically to zero at initialization ($\sigma=0$ collapse) whenever any single gate has low initial pass rate.
- **Mandatory Defense:** Use **lexicographic staged gating** (Transform 3): bounded partial-credit shaping $\frac{\lambda_{\text{gate}}}{m}\sum_j s_j(x,y)$ unlocked strictly below the product-gated continuous residual / invariance bonus.

### `DE-006` — Raw $\Delta\text{NLL}_{\text{swap}}$ Alone or Uncontrolled Pause-Token Compute Comparisons
- **Observed In:** Track 1 §5.3 & §5.5 + `EXP-001` (`B3` pause control `0.0%` acc at `1.71×` `B2` FLOPs).
- **Failure Signature:** Comparing latent-CoT against `B1` alone conflates extra forward-pass compute with learned reasoning; conversely, $\Delta\text{NLL}_{\text{swap}} > 0$ alone can be spoofed by delimiter/checksum artifacts.
- **Mandatory Defense:** Always include FLOPs-matched `B3` (pause) and `B2` ($2.5\times$ dense NTP) controls, AND pair $\Delta\text{NLL}_{\text{swap}}$ with format-matched / norm-matched $\Delta\text{NLL}_{\text{trunc}}$ and monotonic OOD hop-depth scaling ($d\in\{2,4,6,8\}$).
