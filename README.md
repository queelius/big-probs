# Self-Supervised RL & Universal Verification (`big-probs`)

> **Core Thesis:** *Doing for reinforcement learning (RL) and verification what self-supervised learning (SSL) did for unsupervised learning.*

Self-supervised learning unlocked web-scale representation learning because next-token prediction is a **degenerate boundary case of automatic verification**—trivial masking inversion (zero pushforward distribution shift) intersected with lossless predictive compression at zero verification latency ($\tau = 0$). Generalizing SSL beyond standard pretraining and narrow math/code RLVR does not require inventing a mythical universal scalar oracle; it requires systematically leaving that boundary along three orthogonal axes:

1. **Latent Compute Depth (Track 1):** Reframing next-token prediction as RL over unobserved latent chain-of-thought (`SSL-as-RL`), trading temporal compute depth for parameter count on compute-elastic tokens.
2. **Structural Verification Asymmetry (Track 2):** Exploiting *Evaluative Asymmetry* ($C_{\text{verify}}(x,y,w) \ll C_{\text{solve}}(x)$ without gold labels) and *Generative Asymmetry* (grounded inversions $x = g(y)$ with controlled pushforward shift) across physics, biology, hardware, law, and medicine.
3. **Verification Latency Compression (Track 3):** Dissolving the myth of "unverifiable" domains by compressing slow, costly, noisy, and multi-objective ground-truth channels via chronological replay, recalibrated surrogates, multi-fidelity GFlowNets, progress critics, and epistemic oversight—while explicitly quarantining the irreducible normative/causal remainder.

---

## Cold-Restart Research Operations

> **Resuming from zero context (human or AI agent)?** Start here:
> 1. **[`research/STATE.md`](research/STATE.md)** — **60-Second Cold-Restart Briefing:** current thesis, verified Phase 1 snapshot (`EXP-001`), compute constraints, open blind spots, and copy-paste resume commands.
> 2. **[`research/PROCEDURE.md`](research/PROCEDURE.md)** — **Research Operating System (SOP):** Platt's strong-inference 6-stage experiment lifecycle, isoFLOPs Pareto frontier discipline, `<2`-minute cold-restart protocol, and anti-pattern checklist.
> 3. Living registries: [`research/QUESTIONS.md`](research/QUESTIONS.md) (`RQ-01..03`, `H1..H3`, `H1a..H1c`) · [`research/EXPERIMENTS.md`](research/EXPERIMENTS.md) & [`research/experiments.jsonl`](research/experiments.jsonl) (Pareto frontier ledger) · [`research/BACKLOG.md`](research/BACKLOG.md) (VoI-ranked queue & dead-ends graveyard) · [`research/templates/EXP_PREREG.md`](research/templates/EXP_PREREG.md).

---

## Repository Navigation

| Document / Module | Scope | Key Contents |
| :--- | :--- | :--- |
| **[research/STATE.md](research/STATE.md)** | **Cold-Restart Briefing** | 60-second operational snapshot, verified `EXP-001` status, hardware budgets, next decisions, and GitHub/HF provenance links. |
| **[research/PROCEDURE.md](research/PROCEDURE.md)** | **Research Procedure SOP** | 6-stage experiment lifecycle, isoFLOPs Pareto discipline, `<2`-minute cold-restart protocol, failure-mode forensics, and Git+HF artifact rules. |
| **[00-unified-framework.md](00-unified-framework.md)** | **Master Synthesis** | Unified 4-Tier Verification Stack (Tier 0A $\to$ Tier 3), 4 cross-cutting mathematical primitives, 3 open research frontiers & falsifiable hypotheses (`H1`–`H3`), and 4-phase experimental roadmap. |
| **[01-ssl-as-rl-latent-cot.md](01-ssl-as-rl-latent-cot.md)** | **Track 1: SSL-as-RL & Latent CoT** | Formal SSL $\leftrightarrow$ RL EM identity, shared-$\theta$ baseline sabotage proof, 7-reward taxonomy, Knowledge–Algorithm Decomposition, compute-elastic filtering (`T2-self`), and Stage 1/2 small-model protocols. |
| **[02-verification-asymmetry-and-inversion.md](02-verification-asymmetry-and-inversion.md)** | **Track 2: Verification Asymmetry & Inversion** | Evaluative vs. Generative Asymmetry, Pushforward Shift Theorem, 8-mechanism structural taxonomy, 6-transform Verification Playbook (lexicographic gating, Hadamard priors), and Medicine/Law/Engineering blueprints. |
| **[03-slow-and-unknown-verification.md](03-slow-and-unknown-verification.md)** | **Track 3: Slow & "Unknown" Verification** | Four pseudo-unverifiable regimes vs. 4 genuine epistemic remainders, latency wall, asymmetric FP/FN RLVR noise, 6 latency-compression mechanisms (replay, WCO/UWO, GFlowNets, PAVs/CCMV, Debate/Peer Prediction), and worked slow-domain stacks. |
| **[phase1_pilot/](phase1_pilot/)** | **Phase 1 Experimental Testbed** | Runnable PyTorch implementation of Track 1 §5: 3-regime data generators, Llama + pause/continuous/hybrid latent models, Dr. GRPO + DAPO + strictly proper Brier rewards, 5 mandatory §5.5 diagnostics, 50 unit tests, CPU micro-pilot, H100 Stage 1 launcher, and HF/ledger scripts. |

---

## Phase 1 Experimental Pilot — CPU Micro-Scale Results

> **Scope Notice:** The table below reports a live, reproducible **CPU micro-scale proof-of-life run** (`micro-small` ~0.98M params vs. `micro-large` ~2.85M params, 80 supervised steps + 20 RL steps, `seed=42`, completed in **93.1s on CPU**, **50/50 pytest passed**). It validates end-to-end training stability, anti-sabotage invariants, and all 5 Track 1 §5.5 diagnostics. It is **distinct from** the full `160M` / `410M` / `1.0B` H100 Stage 1 run (`phase1_pilot/scripts/launch_h100_stage1.sh`, ~30–45 H100 GPU-hours).

Full diagnostic report: [`phase1_pilot/results/micro_pilot_report.md`](phase1_pilot/results/micro_pilot_report.md) | Machine-readable metrics: [`phase1_pilot/results/micro_pilot_metrics.json`](phase1_pilot/results/micro_pilot_metrics.json)

| Arm | Params | FLOPs/tok (vs B2) | Exact Acc | Hard-BPB | OOD `pass@16` | $\Delta\text{NLL}_{\text{swap}}$ | $\Delta\text{NLL}_{\text{trunc}}$ | Mono Depth | Reg3 ECE | Reg3 Brier |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| `B1` (`micro-small` NTP) | 984,192 | 0.34× | 3.1% | 2.318 | 31.2% | — | — | — | 0.005 | -0.994 |
| `B2` (`micro-large` NTP) | 2,853,312 | 1.00× | 6.2% | 2.266 | 31.2% | — | — | — | 0.040 | -1.006 |
| `B3` (Pause control) | 984,704 | 1.71× | 0.0% | 2.317 | 31.2% | — | — | — | 0.032 | -1.003 |
| `B4` (Continuous latent) | 1,000,704 | 1.74× | 3.1% | 2.272 | 18.8% | -0.001 | +2.909 | No | 0.045 | -0.989 |
| `RL-R1` (Discrete + exact-match) | 984,192 | 3.10× | 3.1% | 2.326 | 31.2% | -0.002 | +0.001 | No | 0.026 | -0.997 |
| `RL-R3` (Discrete + Brier) | 984,192 | 3.10× | 1.6% | 2.309 | 25.0% | +0.000 | -0.001 | No | 0.040 | -1.005 |
| `I2-Hybrid` (Continuous $\to$ HRPO) | 1,033,600 | 2.51× | 3.1% | 2.271 | **37.5%** | -0.003 | +2.732 | No | 0.032 | -0.998 |

**Key Micro-Pilot Takeaways:**
- **Verdict (`FALSIFIED` at 80+20 CPU micro-steps, as expected for micro-scale):** No small-model arm simultaneously beat pause control `B3` (0.0%) and FLOPs-matched `B2` (6.2%), confirming the falsification harness rejects under-trained candidates cleanly.
- **Reward Variance Collapse Verified (`R1` vs. `R3`):** Binary exact-match (`RL-R1`) suffered `sigma=0` fraction `1.00` (100% zero-variance groups), whereas strictly proper Brier scoring (`RL-R3` and `I2-Hybrid`) maintained `sigma=0` fraction `0.00` throughout RL training.
- **Strongest Small-Model Candidate (`I2-Hybrid`):** Reached Hard-BPB `2.271` (nearly matching 2.9× larger `B2` at `2.266`), expanded OOD `pass@16` to `37.5%` (vs. `31.2%` for `B1`/`B2`/`B3`), and showed strong format-matched truncation sensitivity ($\Delta\text{NLL}_{\text{trunc}} = +2.732$ nats) with stable aleatoric calibration (`Reg3 ECE = 0.032`, `Reg3 Brier = -0.998`).

---

## Models & Datasets on HuggingFace Hub

Large model checkpoints, training logs, and Parquet-formatted stratified benchmark splits are hosted on HuggingFace Hub to keep this Git repository lean and reproducible:

- **Models & Checkpoints:** [huggingface.co/queelius/big-probs-phase1-models](https://huggingface.co/queelius/big-probs-phase1-models) *(public, 26 files, 33.88 MB)* — Model Card + `micro-pilot-v1/` with all 7 trained Phase 1 arms (`b1`, `b2`, `b3`, `b4`, `rl_r1`, `rl_r3`, `i2_hybrid`: `checkpoint.pt`, `config.json`, `metrics.jsonl`), `micro_pilot_report.md`, `micro_pilot_metrics.json`, and `manifest.json` (ready to host future H100 Stage 1 `160M`/`410M`/`1.0B` checkpoints under `--tag h100-stage1`).
- **Stratified Datasets:** [huggingface.co/datasets/queelius/big-probs-phase1-data](https://huggingface.co/datasets/queelius/big-probs-phase1-data) *(public, 23 files, 3.52 MB)* — Dataset Card with HF Dataset Viewer configs + Parquet & JSONL splits with verified zero train/OOD leakage:
  - `micro_pilot` *(default, 792 rows)*: `train` (600), `eval_id` (48), `eval_ood` (48), `eval_natural` (48), `eval_aleatoric` (48).
  - `stage1_benchmark` *(3,200 rows)*: `train` (2,000), `eval_id` (300), `eval_ood` (300), `eval_natural` (300), `eval_aleatoric` (300).

---

## Quickstart

### Local CPU (Unit Tests & Micro-Pilot)

```bash
cd phase1_pilot

# 1. Install dependencies (CPU PyTorch by default)
uv sync

# 2. Run the full 50-test verification suite
uv run pytest -v

# 3. Reproduce the 7-arm CPU micro-scale pilot (~93s on 12 CPU cores)
uv run python scripts/run_micro_pilot.py
```

### H100 Stage 1 Launch (~30–45 H100 GPU-Hours)

```bash
cd phase1_pilot
uv sync --extra h100
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124

NPROC_PER_NODE=4 OUT_ROOT=runs/h100_stage1 bash scripts/launch_h100_stage1.sh
```

See [`phase1_pilot/README.md`](phase1_pilot/README.md) for full architecture details, CLI options, and HuggingFace Hub publishing commands.
