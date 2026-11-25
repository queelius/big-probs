# Cold-Restart Briefing — `big-probs` Research Program

> **Read This First (<60 seconds):** This document is the authoritative entry point for any human researcher or fresh AI session resuming the `big-probs` program from zero context. Follow [`research/PROCEDURE.md`](PROCEDURE.md) for operating discipline.

- **Last Updated:** `2026-11-24`
- **Current Phase:** **Phase 1 (Track 1 Small-Model Latent-CoT RL Pilot)**: CPU micro-pilot (`EXP-001`) complete, audited, and published; H100 Stage 1 (`EXP-002`) launcher is blocked at its default dataset sizes (§5 item 9, `BL-11`) and needs a data-path fix before GPU allocation.
- **Repositories & Artifacts:**
  - **GitHub:** [`https://github.com/queelius/big-probs`](https://github.com/queelius/big-probs)
  - **HuggingFace Models:** [`https://huggingface.co/queelius/big-probs-phase1-models`](https://huggingface.co/queelius/big-probs-phase1-models)
  - **HuggingFace Datasets:** [`https://huggingface.co/datasets/queelius/big-probs-phase1-data`](https://huggingface.co/datasets/queelius/big-probs-phase1-data)

---

## 1. Overarching Thesis (2 Sentences)

Standard self-supervised learning (next-token prediction) scaled because it is the **degenerate zero-shift, zero-latency ($\tau=0$) boundary case of automatic verification**. Generalizing SSL-as-RL to arbitrary domains means leaving that boundary systematically along three axes—**latent compute depth** (Track 1 / Tier 0A), **structural verification asymmetry** (Track 2 / Tier 0B), and **verification latency compression** (Track 3 / Tiers 1–3)—while explicitly quarantining irreducible normative/causal remainders.

---

## 2. What Has Been Accomplished & Verified

| Component | Status | Exact Verification / Provenance |
| :--- | :---: | :--- |
| **Theoretical Synthesis (4 Docs)** | **Done** | [`00-unified-framework.md`](../00-unified-framework.md) integrating [`01-ssl-as-rl-latent-cot.md`](../01-ssl-as-rl-latent-cot.md), [`02-verification-asymmetry-and-inversion.md`](../02-verification-asymmetry-and-inversion.md), and [`03-slow-and-unknown-verification.md`](../03-slow-and-unknown-verification.md). Establishes the 4-Tier Verification Stack, 4 Cross-Cutting Primitives, and falsifiable hypotheses `H1`–`H3`. |
| **Phase 1 Codebase (`phase1_pilot/`)** | **Verified** | **50/50 pytest passed** (`10` data, `7` models, `13` RL incl. 2-rank Gloo distributed sync, `20` eval incl. `I2-Hybrid` parity). All Track 1 §5.4 math & anti-sabotage invariants audited (`R3_brier` exact $O(\lvert\mathcal{V}\rvert)$, frozen $p_{\theta_{\text{ref}}}$ + $\lambda_{\text{ntp}}=0.2$, Dr. GRPO + DAPO Clip-Higher, teacher-free `T2-self`). |
| **`EXP-001` CPU Micro-Pilot** | **Completed** | Live 7-arm run (`micro-small` 0.98M vs. `micro-large` 2.85M, 80 sup + 20 RL steps, `seed=42`, **93.1s**, zero NaNs). Report: [`phase1_pilot/results/micro_pilot_report.md`](../phase1_pilot/results/micro_pilot_report.md); raw JSON: [`phase1_pilot/results/micro_pilot_metrics.json`](../phase1_pilot/results/micro_pilot_metrics.json). |
| **Audit Bugs Caught & Fixed** | **Resolved** | (1) Fixed critical `I2-Hybrid` train/eval interface mismatch in `diagnostics.py` (now `< 1e-4` parity with `recompute_hybrid_logprobs_and_continuation`, `extra_steps == thought_len`); (2) Added `torch.distributed` process-group init/broadcast/sharding/grad all-reduce for multi-GPU `torchrun` in `train.py` & `grpo.py`. |
| **H100 Stage 1 Launcher** | **Blocked** | [`phase1_pilot/scripts/launch_h100_stage1.sh`](../phase1_pilot/scripts/launch_h100_stage1.sh) passes static checks (`bash -n` clean; 8/8 CLI invocations match `phase1_pilot.train`), but at its default dataset sizes job 1 exits during dataset construction (found `2026-11-24` at commit `f03ff41`; see §5 item 9 and `BL-11`). Targets ~35–40 H100 GPU-hrs (`160M`/`410M`/`1.0B`) once unblocked. |
| **GitHub + HuggingFace Publishing** | **Wired** | [`phase1_pilot/scripts/publish_hf.py`](../phase1_pilot/scripts/publish_hf.py) publishes checkpoints + Parquet splits to `queelius/big-probs-phase1-models` and `queelius/big-probs-phase1-data`. |
| **Research Operating System (`research/`)** | **Active** | Cold-restart registries (`STATE.md`, `PROCEDURE.md`, `QUESTIONS.md`, `EXPERIMENTS.md`, `experiments.jsonl`, `BACKLOG.md`, `templates/EXP_PREREG.md`) + [`phase1_pilot/scripts/log_experiment.py`](../phase1_pilot/scripts/log_experiment.py). |

---

## 3. Current Frontier Snapshot (`EXP-001` CPU Micro-Scale)

> **Scope Warning:** `EXP-001` is a **0.98M vs. 2.85M CPU proof-of-life** (80 supervised + 20 RL steps, 600 train samples). Its `FALSIFIED` verdict validates the diagnostic pipeline and surfaces real small-model RL dynamics, but does **not** falsify the 160M / 410M / 1.0B H100 hypothesis (`EXP-002`).

- **Track 1 §5.5 Verdict (`EXP-001`):** `FALSIFIED` — best small-model candidate `I2-Hybrid` reached `3.1%` exact acc at `2.51×` `B2` FLOPs vs. `B2` (`6.2%`, `1.00×`) and `B3` (`0.0%`, `1.71×`).
- **Two Positive Micro-Scale Signals Already Observed:**
  1. **Strictly Proper Brier Eliminates Variance Starvation (`H1a`):** Binary exact-match `RL-R1` hit `sigma=0` fraction = **`1.00`** (total gradient starvation), whereas Brier-anchored `RL-R3` and `I2-Hybrid` maintained `sigma=0` fraction = **`0.00`** with bounded no-thought NLL drift (`+0.0559` and `-0.5746` nats) and calibrated Regime 3 ECE (`0.040` and `0.032`).
  2. **Continuous$\to$Hybrid (`I2-Hybrid`) Leads Small-Model Arms:** Matched `B2` Hard-BPB (`2.271` vs. `2.266`; `B1 = 2.318`), beat `B1`/`B2`/`B3` on OOD `pass@16` (`37.5%` vs. `31.2%`), and showed strong format-matched truncation sensitivity (`ΔNLL_trunc = +2.732` nats), though prompt-specific swap sensitivity has not yet emerged (`ΔNLL_swap = -0.003` nats) at 80+20 micro-steps.

See [`research/EXPERIMENTS.md`](EXPERIMENTS.md) for the full 7-arm table and post-mortem.

---

## 4. Local Hardware & Environment Constraints

| Resource | Local Machine (Observed) | Required for Full Roadmap |
| :--- | :--- | :--- |
| **CPU** | AMD Ryzen 5 7430U, **12 logical CPUs** | Sufficient for tests, micro-pilots (`EXP-001` ~93s), and synthetic H2/H3 testbeds |
| **RAM** | **62 GiB** total (~50 GiB available) | Sufficient for CPU micro-runs and dataset export |
| **GPU** | **None** (`nvidia-smi` not present; CPU-only PyTorch in `phase1_pilot/.venv`) | **1–4× H100** (~35–40 GPU-hrs) for `EXP-002` Stage 1; **8× H100** (~250–400 GPU-hrs) for `EXP-003` Stage 2 |
| **Env Tooling** | `uv` (`phase1_pilot/uv.lock` pinned) | On H100 node: `uv sync --extra h100` + CUDA PyTorch (`cu124`) |

---

## 5. What Is Currently Missing / Open Blind Spots

1. **No H100 Capacity-Regime Data Yet:** At 0.98M / 80 steps, greedy accuracy is `0.0%–6.2%` and `ΔNLL_swap ≈ 0`. We cannot confirm or falsify `H1`/`H1b` until models reach non-trivial Regime 1 accuracy (`>30%` ID) on `160M` / `410M` / `1.0B` (`EXP-002`).
2. **Single-Seed Only (`seed=42`):** Zero multi-seed confidence intervals on `EXP-001`.
3. **No Real Web-Text Regime 2 Evaluation:** `EXP-001` uses synthetic Regime 2 streams; real OpenWebMath / Stack v2 / FineWeb-Edu (`40/40/20`) has not been evaluated yet.
4. **No `T0`/`T1`/`T2-self`/`T2-scale` Comparative Ablation Yet:** `EXP-001` only exercised `T2-self` (testing `R1` vs. `R3` and `I1` vs. `I2`).
5. **No `B5` Off-Policy Teacher Distillation Baseline or PCA Latent-Drift Probe:** Required to test Tension 1 (`H1`) transfer geometry.
6. **FLOPs-Budget Mismatch on `I2-Hybrid`:** In `EXP-001`, `I2-Hybrid` ran at `2.51×` `B2` FLOPs (`thought_len=6` on every target token); Stage 1 needs sparse triggering (`top-20%` adaptive inference) or shorter `thought_len` to test the strict `≤ 1.0× B2 FLOPs` Pareto gate.
7. **No One-Command Stage 1 Checkpoint Evaluator:** `launch_h100_stage1.sh` trains all 8 Stage 1 checkpoints, but a dedicated `scripts/eval_stage1.py` (emitting `stage1_metrics.json` for `log_experiment.py`) is not yet written.
8. **No Runnable Code Testbeds for `H2` or `H3`:** Tracks 2 & 3 hypotheses (`H2` stealth Goodharting, `H3` causal ordering-leakage / AIPW replay) are specified theoretically in `00-unified-framework.md` §4, with zero runnable code yet. Both can be prototyped on CPU!
9. **Stage 1 Data Path Cannot Build Launcher-Scale Datasets (`BL-11`):** `generate_natural_stream` in `phase1_pilot/phase1_pilot/data/generators.py` yields only ~1,080 unique samples (every third sample is an arithmetic-series item with 360 variants), so `build_stratified_datasets` raises unless roughly `0.15 × n_train + n_eval ≤ 1080`. `launch_h100_stage1.sh` defaults to `N_TRAIN_SAMPLES=200000` / `N_EVAL_SAMPLES=2000`, so `EXP-002` as scripted stops before any training. Reproduce on CPU in seconds from `phase1_pilot/`: `uv run python -m phase1_pilot.train --mode supervised --preset micro-small --max-steps 1 --n-train-samples 200000 --n-eval-samples 2000` exits with `RuntimeError: Could only generate 1082/30000 unique natural_stream samples`. Separately, `HFTokenizerAdapter` and `load_h100_web_mixture` are not called by `train.py`, so at sizes that fit, the `160m`/`410m`/`1b` presets would train on the 512-ID synthetic data rather than real web text (this compounds item 3).

---

## 6. Copy-Paste Resume Commands

```bash
# 1. Verify environment & all 50 unit tests (CPU, ~7s)
cd phase1_pilot   # from the repo root
uv run pytest -v

# 2. Re-run the 7-arm CPU micro-pilot end-to-end (~93s on 12 cores)
uv run python scripts/run_micro_pilot.py

# 3. Inspect Pareto ledger & dry-run logging a metrics JSON into research/experiments.jsonl
uv run python scripts/log_experiment.py --render-only
uv run python scripts/log_experiment.py \
  --metrics results/micro_pilot_metrics.json \
  --exp-id EXP-001-CHECK \
  --title "Dry-run verification" \
  --dry-run

# 4. Publish artifacts to HuggingFace Hub (models + Parquet datasets)
uv run --with huggingface_hub --with pyarrow python scripts/publish_hf.py \
  --namespace queelius --tag micro-pilot-v1 --run-dir results/micro_runs

# 5. Launch full Stage 1 on an H100 node (35-40 H100 GPU-hrs)
#    BLOCKED at the default dataset sizes until BL-11 is fixed (see §5 item 9)
uv sync --extra h100
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124
NPROC_PER_NODE=4 OUT_ROOT=runs/h100_stage1 bash scripts/launch_h100_stage1.sh
```

---

## 7. First 3 Decisions on Resume

1. **Decision 1 — Compute Track (Local CPU Today vs. Rent H100 Now):**
   - *If staying CPU-only today ($0):* Pick `BL-01` (longer multi-seed CPU micro-run to test whether `ΔNLL_swap > 0` emerges once `micro-small` leaves uniform logits), `BL-02` (`T0`/`T1`/`T2-self`/`T2-scale` + `R2` vs. `R3` CPU ablation), or `BL-03`/`BL-04` (runnable CPU synthetic testbeds for `H2` stealth Goodharting and `H3` AIPW causal replay). See [`research/BACKLOG.md`](BACKLOG.md).
   - *If provisioning 1–4× H100 (~$80–$150 cloud budget):* First fix the Stage 1 data path (`BL-11`, CPU-testable), write `scripts/eval_stage1.py` (`BL-07`), tune adaptive/sparse inference FLOPs so `I2-Hybrid` can hit `≤ 1.0× B2 FLOPs`, preregister `EXP-002` using [`research/templates/EXP_PREREG.md`](templates/EXP_PREREG.md), and launch `scripts/launch_h100_stage1.sh`.
2. **Decision 2 — Fix Adaptive Inference FLOPs Budget Before `EXP-002`:**
   - Decide whether `EXP-002` evaluates `I2-Hybrid` and `RL-R3` with fixed short thoughts (`thought_len=2–4`) OR entropy/variance-gated adaptive inference (thinking only on top-20% tokens at eval time, cutting effective `extra_steps_per_target` by ~5× to achieve `≤ 1.0× B2` FLOPs parity).
3. **Decision 3 — Prioritize Breadth (`H2`/`H3` CPU Prototypes) vs. Depth (`EXP-002` H100):**
   - Check [`research/QUESTIONS.md`](QUESTIONS.md) to choose whether the highest Value-of-Information next step is falsifying `H1b` at 160M scale (`EXP-002`) or validating `H2`/`H3` mechanisms in fast synthetic environments before spending 500+ GPU-hrs in Phases 3–4.
