# Experiment Preregistration & Post-Mortem — `EXP-XXX`

> **Operating Rule:** Complete **Sections 1–5 BEFORE launching compute** (Platt's *Strong Inference*, isoFLOPs parity, explicit kill criteria). Complete **Sections 6–8 AFTER execution** and roll results into `research/experiments.jsonl`, `research/EXPERIMENTS.md`, `research/QUESTIONS.md`, `research/STATE.md`, and `research/BACKLOG.md`.

---

## 1. Identity & Provenance (Pre-Run)

- **Experiment ID:** `EXP-XXX`
- **Date Preregistered:** `YYYY-MM-DD`
- **Owner / Session:** `<human or agent id>`
- **Linked RQs / Hypotheses:** `RQ-XX`, `H-X` (from [`research/QUESTIONS.md`](../QUESTIONS.md))
- **Code Commit SHA:** `<git rev-parse HEAD>`
- **Dataset / Seed Specification:** `<dataset version / HF URI / seeds>`
- **Target Hardware & Compute Budget:** `<e.g., 4x H100 80GB, max 40 GPU-hrs>`

---

## 2. Competing Hypotheses (Platt's Strong Inference)

State at least **two mutually exclusive hypotheses** this experiment decisively distinguishes:

- **Hypothesis A (Primary):** `<Specific quantitative prediction>`
- **Hypothesis B (Null / Alternative Explanation):** `<e.g., gains are pure extra forward-pass FLOPs explained by pause control B3, or require teacher bootstrapping B5>`
- **Decisive Discriminator:** `<Exact observable difference between arms that rules out A or B>`

---

## 3. Experimental Arms & IsoFLOPs / Iso-Parameter Parity

Every arm MUST declare exact parameter count, inference FLOPs multiplier vs. weight-class ceiling (`B2`), and training token/step budget:

| Arm | Role / Control | Params | Target Inference FLOPs (vs `B2`) | Train Budget (Tokens / Steps) | Key Hyperparameters |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `B1` | Same-size NTP lower bound | — | — | — | — |
| `B2` | Larger NTP weight-class ceiling | — | `1.00×` | — | — |
| `B3` | FLOPs-matched pause control | — | `≤ 1.00×` | — | — |
| `<Candidate>` | `<Method under test>` | — | `≤ 1.00×` | — | — |

---

## 4. Preregistered Go / Partial / Kill Criteria

> **Anti-Moving-Goalposts Rule:** Do NOT edit these thresholds after observing evaluation metrics.

- **Strong Confirmation (Go):**
  - [ ] Matches or beats `B2` at `≤ 1.0× B2` inference FLOPs on primary quality metric(s)
  - [ ] Beats FLOPs-matched control `B3` by `≥ 3.0 pp` (or domain-equivalent threshold)
  - [ ] Passes causal thought faithfulness (`ΔNLL_swap > 0.15` nats, format-matched `ΔNLL_trunc > 0`, monotonic OOD depth scaling)
  - [ ] Strictly improves OOD `pass@64` (support expansion, not sharpening-only)
  - [ ] Maintains Regime 3 aleatoric calibration & bounded `nothought_nll_drift`
- **Partial Confirmation (Conditional):**
  - [ ] `<State exact conditional outcome, e.g., works only under continuous pretraining I2 or warm-start>`
- **Falsification / Kill Criterion (Stop or Pivot):**
  - [ ] `<Exact condition triggering immediate halt, budget cap, or dead-end logging>`
- **Mid-Run Abort Triggers:**
  - [ ] `NaN/Inf` loss or unrecovered gradient explosion
  - [ ] `sigma=0` group fraction `> 0.90` for `> 100` consecutive RL steps
  - [ ] `nothought_nll_drift > +0.50` nats (baseline sabotage)

---

## 5. Pre-Flight Verification Checklist

- [ ] `uv run pytest -v` passes 100% (including train-vs-eval interface parity test `< 1e-4` and distributed sync test if `WORLD_SIZE > 1`).
- [ ] Overfit-a-single-batch / 2-step smoke test verified with zero NaNs and expected FLOPs accounting.
- [ ] Zero train/eval/OOD data leakage verified.
- [ ] Output directory and HF publishing target tag (`--tag <exp-tag>`) confirmed.

---

## 6. Post-Run Execution Record & Provenance (Fill After Completion)

- **Date Completed:** `YYYY-MM-DD`
- **Actual Hardware & Wall-Clock / Compute Hours:** `<exact hours>`
- **Exact Command Executed:**
  ```bash
  <copy-paste exact command>
  ```
- **Artifacts:**
  - Local Report: `<path>`
  - Local Metrics JSON: `<path>`
  - HuggingFace Models URI: `<https://huggingface.co/queelius/big-probs-phase1-models/...>`
  - HuggingFace Datasets URI: `<https://huggingface.co/datasets/queelius/big-probs-phase1-data/...>`
  - GitHub Commit SHA: `<sha>`

---

## 7. Quantitative Results & Verdict

- **Preregistered Verdict:** `[ CONFIRMED | PARTIALLY CONFIRMED | FALSIFIED | ABORTED ]`
- **Summary Table (paste output of `scripts/log_experiment.py`):**

| Arm | Params | FLOPs (vs B2) | Exact Acc | Hard-BPB | OOD pass@k | ΔNLL_swap | ΔNLL_trunc | Reg3 ECE | σ=0 Frac | Pareto? |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `...` | — | — | — | — | — | — | — | — | — | — |

---

## 8. Post-Mortem & State Roll-Up

- **What Worked / Confirmed:**
  - ...
- **What Failed / Disproved:**
  - ...
- **Bugs Caught During Audit:**
  - ...
- **Dead Ends Added to [`research/BACKLOG.md`](../BACKLOG.md) Graveyard:**
  - `DE-XXX`: ...
- **Mandatory End-of-Session Roll-Up Checklist:**
  - [ ] Appended structured record via `uv run python scripts/log_experiment.py --metrics <json> --exp-id EXP-XXX --update-markdown`
  - [ ] Updated [`research/EXPERIMENTS.md`](../EXPERIMENTS.md) with post-mortem
  - [ ] Updated statuses & evidence links in [`research/QUESTIONS.md`](../QUESTIONS.md)
  - [ ] Updated [`research/STATE.md`](../STATE.md) frontier snapshot & next decisions
  - [ ] Updated [`research/BACKLOG.md`](../BACKLOG.md) queue & graveyard
  - [ ] Uploaded binary checkpoints/splits to HuggingFace Hub (never committed large binaries to Git)
