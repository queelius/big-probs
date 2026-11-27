# Cold-Restart Briefing — `big-probs` Research Program

> **Read This First (<60 seconds):** This document is the authoritative entry point for any human researcher or fresh AI session resuming the `big-probs` program from zero context. Follow [`research/PROCEDURE.md`](PROCEDURE.md) for operating discipline.

- **Last Updated:** `2026-11-27`
- **Current Phase:** **Phase 1 (Track 1 Small-Model Latent-CoT RL Pilot)**: CPU micro-pilot (`EXP-001`) complete, audited, and published, but its Regime 1 results sit below the prompt-blind floor and are uninformative about reasoning (`EXP-006`, `DE-007`); no micro base model has yet cleared its task floor (`BL-12`); H100 Stage 1 (`EXP-002`) launcher now runs at its default dataset sizes, but on synthetic Regime 2 text only. The real-text path `EXP-002` specifies is wired as far as a local corpus file (`train.py --tokenizer` / `--corpus`, CPU-tested with stand-in documents). Building that file from the Hub, choosing the tokenizer, and passing both through the launcher are still open (§5 item 9, `BL-11`).
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
| **`EXP-001` CPU Micro-Pilot** | **Completed** | Live 7-arm run (`micro-small` 0.98M vs. `micro-large` 2.85M, 80 sup + 20 RL steps, `seed=42`, **93.1s**, zero NaNs). Report: [`phase1_pilot/results/micro_pilot_report.md`](../phase1_pilot/results/micro_pilot_report.md); raw JSON: [`phase1_pilot/results/micro_pilot_metrics.json`](../phase1_pilot/results/micro_pilot_metrics.json). Reproduced on `2026-11-24` at commit `f03ff41`, again on the `BL-11` generator change, and again on `2026-11-26` on the real-text plumbing (branch `stage1-real-text`): every metric identical, only `elapsed_seconds` and `config.output_dir` differ. Its datasets are pinned by `test_generated_datasets_match_committed_artifacts`. |
| **Audit Bugs Caught & Fixed** | **Resolved** | (1) Fixed critical `I2-Hybrid` train/eval interface mismatch in `diagnostics.py` (now `< 1e-4` parity with `recompute_hybrid_logprobs_and_continuation`, `extra_steps == thought_len`); (2) Added `torch.distributed` process-group init/broadcast/sharding/grad all-reduce for multi-GPU `torchrun` in `train.py` & `grpo.py`. |
| **H100 Stage 1 Launcher** | **Synthetic only** | [`phase1_pilot/scripts/launch_h100_stage1.sh`](../phase1_pilot/scripts/launch_h100_stage1.sh) passes static checks (`bash -n` clean; 8/8 CLI invocations match `phase1_pilot.train`). Its default dataset sizes failed in job 1 at commit `f03ff41`; after the `BL-11` generator change the same CLI path builds them (~12 s on CPU) and trains. Regime 2 is still synthetic, not the real web text `EXP-002` specifies: the launcher does not pass `--tokenizer` or `--corpus` yet (§5 item 9). Model vocabulary now follows the tokenizer, so with the default `PilotTokenizer` the `160m` preset builds 85.7M parameters (134.1M at the preset's own 32,000). Targets ~35–40 H100 GPU-hrs (`160M`/`410M`/`1.0B`). |
| **GitHub + HuggingFace Publishing** | **Wired** | [`phase1_pilot/scripts/publish_hf.py`](../phase1_pilot/scripts/publish_hf.py) publishes checkpoints + Parquet splits to `queelius/big-probs-phase1-models` and `queelius/big-probs-phase1-data`. |
| **Research Operating System (`research/`)** | **Active** | Cold-restart registries (`STATE.md`, `PROCEDURE.md`, `QUESTIONS.md`, `EXPERIMENTS.md`, `experiments.jsonl`, `BACKLOG.md`, `templates/EXP_PREREG.md`) + [`phase1_pilot/scripts/log_experiment.py`](../phase1_pilot/scripts/log_experiment.py). |

---

## 3. Current Frontier Snapshot (`EXP-001` CPU Micro-Scale)

> **Scope Warning:** `EXP-001` is a **0.98M vs. 2.85M CPU proof-of-life** (80 supervised + 20 RL steps, 600 train samples). Its `FALSIFIED` verdict validates the diagnostic pipeline and surfaces real small-model RL dynamics, but does **not** falsify the 160M / 410M / 1.0B H100 hypothesis (`EXP-002`).

- **Track 1 §5.5 Verdict (`EXP-001`):** `FALSIFIED` as computed: best small-model candidate `I2-Hybrid` reached `3.1%` exact acc at `2.51×` `B2` FLOPs vs. `B2` (`6.2%`, `1.00×`) and `B3` (`0.0%`, `1.71×`). **Reinterpreted `2026-11-26` (`EXP-006`, `DE-007`):** every arm scored worse on Regime 1 than a predictor that never reads the prompt (NLL `3.362`–`3.452` vs. `3.150`), so the verdict and every Regime 1 comparison are uninformative about reasoning in either direction.
- **What survives from `EXP-001`:** (1) a mechanism observation: binary `R1` reward had `sigma=0` fraction `1.00` in this run, from a policy at chance (`DE-001`), while `R3_brier` and `I2-Hybrid` give `0.00`. That shows the continuous reward varies within groups, not that its variation is useful. (2) Pipeline checks: anchor wiring (`DE-002`), train/eval parity (`DE-003`), and FLOPs accounting. The two "positive signals" this section used to list (`I2-Hybrid` matching `B2` Hard-BPB and leading OOD `pass@16`) are withdrawn: the Hard-BPB gap lies above the prompt-blind floor, and `37.5%` OOD `pass@16` is below the `39.8%` of uniform guessing over node IDs (and the `99.85%` of guessing among the three listed candidates).
- **`EXP-006` (Regime 1 learnability scoping, `7.29` CPU-hrs summed over overlapping runs):** no supervised NTP micro model left its task's structural floor in 13 probes (3k–20k steps), except `python_trace` `d=1` trained alone (`17.0%` vs. `6.0%`). A single 3-pair lookup plateaus at the `1/3` copy floor for 20k steps. The micro testbed cannot test thought arms until a base model is competent (`BL-12`). Summary: [`phase1_pilot/results/exp006_learnability_summary.json`](../phase1_pilot/results/exp006_learnability_summary.json).
- **`EXP-007` (trainer check on Zoology MQAR, preregistered, `3.95` CPU-hrs summed):** rule A. `micro-small` with default `train_supervised` settings learns MQAR (`1.000` test accuracy with 4 labelled queries per row), so a general trainer or backbone defect is unlikely. With one labelled query per row it fits its training batches and stays at `0.20` held-out (likely memorization; train-set accuracy not measured). The `EXP-006` pointer model copies a listed literal: its greedy answer changes with the queried key in only `3`–`5.5%` of prompts. Summary: [`phase1_pilot/results/exp007_trainer_check_summary.json`](../phase1_pilot/results/exp007_trainer_check_summary.json).

See [`research/EXPERIMENTS.md`](EXPERIMENTS.md) for the full 7-arm table, the `EXP-001` correction note, and the `EXP-006` and `EXP-007` post-mortems.

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

1. **No H100 Capacity-Regime Data Yet, and No Competent Micro Base Model:** At 0.98M / 80 steps, greedy accuracy is `0.0%–6.2%` and `ΔNLL_swap ≈ 0`; `EXP-006` found that 3k–20k steps do not move a micro model off its task floor either (`BL-12`). We cannot confirm or falsify `H1`/`H1b` until models reach non-trivial Regime 1 accuracy (`>30%` ID) on `160M` / `410M` / `1.0B` (`EXP-002`).
2. **Single-Seed Only (`seed=42`):** Zero multi-seed confidence intervals on `EXP-001`.
3. **No Real Web-Text Regime 2 Evaluation:** `EXP-001` uses synthetic Regime 2 streams; real OpenWebMath / Stack v2 / FineWeb-Edu (`40/40/20`) has not been evaluated yet.
4. **No `T0`/`T1`/`T2-self`/`T2-scale` Comparative Ablation Yet:** `EXP-001` only exercised `T2-self` (testing `R1` vs. `R3` and `I1` vs. `I2`).
5. **No `B5` Off-Policy Teacher Distillation Baseline or PCA Latent-Drift Probe:** Required to test Tension 1 (`H1`) transfer geometry.
6. **FLOPs-Budget Mismatch on `I2-Hybrid`:** In `EXP-001`, `I2-Hybrid` ran at `2.51×` `B2` FLOPs (`thought_len=6` on every target token); Stage 1 needs sparse triggering (`top-20%` adaptive inference) or shorter `thought_len` to test the strict `≤ 1.0× B2 FLOPs` Pareto gate.
7. **No One-Command Stage 1 Checkpoint Evaluator:** `launch_h100_stage1.sh` trains all 8 Stage 1 checkpoints, but a dedicated `scripts/eval_stage1.py` (emitting `stage1_metrics.json` for `log_experiment.py`) is not yet written.
8. **No Runnable Code Testbeds for `H2` or `H3`:** Tracks 2 & 3 hypotheses (`H2` stealth Goodharting, `H3` causal ordering-leakage / AIPW replay) are specified theoretically in `00-unified-framework.md` §4, with zero runnable code yet. Both can be prototyped on CPU!
9. **Stage 1 Trains on Synthetic Regime 2 Text Only (`BL-11`):** Until `2026-11-24`, `generate_natural_stream` capped datasets at ~1,080 natural samples (every third sample was an arithmetic-series item with 360 variants), so `launch_h100_stage1.sh` at its defaults (`N_TRAIN_SAMPLES=200000` / `N_EVAL_SAMPLES=2000`) failed in job 1. The generator now widens a template's parameter domain once its original domain is used up; the defaults build and train, and the data at every previously supported size is byte-identical. What remains: `HFTokenizerAdapter` and `load_h100_web_mixture` are not called by `train.py`, so the `160m`/`410m`/`1b` presets train on the 512-ID synthetic data, not real web text (this compounds item 3). Wiring real text needs decisions, not just code: how documents become prompt/target `Sample`s, which tokenizer, and a code source. `bigcode/the-stack-v2-dedup` is gated (the `queelius` HF account had not been granted access as of `2026-11-24`) and stores only blob IDs; the file contents sit in Software Heritage's S3 bucket and need AWS credentials. **Decided `2026-11-26`:** Stage 1 uses real Regime 2 text, with `codeparrot/codeparrot-clean` (open, Python only, file contents inline) in place of Stack v2. This drops the planned C++ share and must be recorded in the `EXP-002` preregistration. A measurement behind the decision: the launcher's default synthetic train set is about 11M tokens (200,000 samples, mean 55 tokens, longest 94), roughly 0.5% of the 2B tokens `EXP-002` plans. **Step 1 done `2026-11-26` (branch `stage1-real-text`):** `data/corpus.py` cuts documents into one-boundary token windows, `train.py` takes token IDs and vocabulary size from the tokenizer, and checkpoints record the tokenizer and held-out split they were trained with (design in `CLAUDE.md`, "Real-text Regime 2"). Verified on CPU with stand-in documents, a stand-in HuggingFace-style tokenizer, and a tiny BPE trained locally with the real `transformers` 5.24.0 / `tokenizers` 0.23.3 (the `uv.lock` versions): supervised and RL training both run. No real corpus and no tokenizer from the Hub has been run. **Still open:** a script that writes the corpus file from the Hub at pinned revisions; the tokenizer choice; launcher flags; a token cache and batching that scale to 2B tokens; and the regime mix, which is fixed by sample count (`TRAIN_REGIME_MIX`) while `EXP-002` budgets tokens. **Preset sizes depend on the vocabulary** (`estimate_parameter_count`): at 32,000 the presets are 134.1M / 373.9M / 1,050.6M (2.79× and 7.83×); at 49,152 they are 160.5M / 409.0M / 1,112.1M (2.55× and 6.93×). The names and the planned 2.5× ratio match a vocabulary near 49k–50k, not the presets' 32,000.

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
#    Regime 2 is synthetic until BL-11's real-text half is done (see §5 item 9)
uv sync --extra h100
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124
NPROC_PER_NODE=4 OUT_ROOT=runs/h100_stage1 bash scripts/launch_h100_stage1.sh
```

---

## 7. First 3 Decisions on Resume

1. **Decision 1 — Compute Track (Local CPU Today vs. Rent H100 Now):**
   - *If staying CPU-only today ($0):* Pick `BL-12` (find a Regime 1 task and budget where `B1` clears its structural floor; step 1, the trainer check, passed in `EXP-007`; step 2 tests label density on the pointer task and needs a preregistration) or `BL-03`/`BL-04` (runnable CPU synthetic testbeds for `H2` stealth Goodharting and `H3` AIPW causal replay). `BL-01` and `BL-02` are blocked on `BL-12` since `EXP-006`. See [`research/BACKLOG.md`](BACKLOG.md).
   - *If provisioning 1–4× H100 (~$80–$150 cloud budget):* First wire the real-text Regime 2 path decided on `2026-11-26` (`BL-11`, §5 item 9), write `scripts/eval_stage1.py` (`BL-07`), tune adaptive/sparse inference FLOPs so `I2-Hybrid` can hit `≤ 1.0× B2 FLOPs`, preregister `EXP-002` using [`research/templates/EXP_PREREG.md`](templates/EXP_PREREG.md), and launch `scripts/launch_h100_stage1.sh`.
2. **Decision 2 — Fix Adaptive Inference FLOPs Budget Before `EXP-002`:**
   - Decide whether `EXP-002` evaluates `I2-Hybrid` and `RL-R3` with fixed short thoughts (`thought_len=2–4`) OR entropy/variance-gated adaptive inference (thinking only on top-20% tokens at eval time, cutting effective `extra_steps_per_target` by ~5× to achieve `≤ 1.0× B2` FLOPs parity).
3. **Decision 3 — Prioritize Breadth (`H2`/`H3` CPU Prototypes) vs. Depth (`EXP-002` H100):**
   - Check [`research/QUESTIONS.md`](QUESTIONS.md) to choose whether the highest Value-of-Information next step is falsifying `H1b` at 160M scale (`EXP-002`) or validating `H2`/`H3` mechanisms in fast synthetic environments before spending 500+ GPU-hrs in Phases 3–4.
