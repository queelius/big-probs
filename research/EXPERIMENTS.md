# Experiment Ledger & Pareto Frontier Tracker

> **Machine-Readable Source of Truth:** [`research/experiments.jsonl`](experiments.jsonl) (append-only NDJSON).
> **Logging Tool:** `cd phase1_pilot && uv run python scripts/log_experiment.py --help`
> **Preregistration Template:** [`research/templates/EXP_PREREG.md`](templates/EXP_PREREG.md)

---

## 1. Master Experiment Index

| Exp ID | Date | Title | RQs / Hypotheses | Scale | Compute | Status | Verdict |
| :--- | :--- | :--- | :--- | :--- | ---: | :---: | :---: |
| **`EXP-001`** | `2026-11-23` | Phase 1 CPU Micro-Scale 7-Arm Proof-of-Life | `RQ-01`, `H1`, `H1a`, `H1b`, `H1c` | `0.98M` vs `2.85M` (80+20 steps) | `0.026` CPU-hrs (`93.1s`) | `completed` | `FALSIFIED` *(micro-scale)* |
| **`EXP-006`** | `2026-11-26` | Regime 1 Learnability Scoping (reinterprets `EXP-001`) | `RQ-01`, `H1`, `H1b` | `0.98M` (one `2.85M`), 13 probes × 3k–20k steps | `7.29` CPU-hrs (summed; runs overlapped) | `completed` | `NEGATIVE` *(not preregistered)* |
| **`EXP-007`** | `2026-11-27` | Trainer Check on Zoology MQAR (`BL-12` step 1) | `RQ-01`, `H1` | `0.92M`/`0.86M`, 3 arms × 10k steps | `3.95` CPU-hrs (summed; runs overlapped) | `completed` | `RULE A` *(stack learns MQAR)* |
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

#### Interpretation Correction (`2026-11-26`, `EXP-006`, `DE-007`)
Takeaways 2 and 4 above, and the Regime 1 accuracy, Hard-BPB, `pass@k`, and `ΔNLL_swap` comparisons in both tables, are uninformative about reasoning in either direction. This does not show that the arms ignore the prompt; it shows that no arm did better than a predictor that does. On the same 64 Regime 1 samples, each task's answer marginal from the 600-sample training split scores NLL `3.150` and accuracy `7.8%` without reading the prompt (a default-argument rerun of `run_micro_pilot.py` into `runs/exp001_rerun_ref`, which adds the `prompt_blind_reference` key and reproduced every committed metric exactly). Every arm did worse (NLL `3.362` to `3.452`, accuracy at most `6.2%`). The eval slice holds only `dag_planning` and `python_trace` items; `I2-Hybrid`'s `37.5%` OOD `pass@16` (6 of 16 DAG items) is below the `39.8%` that 16 uniform guesses over the 32 node IDs give, and far below the `99.85%` that 16 uniform guesses among each item's three listed candidates give. Takeaway 1 stands as an observation for this run (binary `R1` reward had zero within-group variance from a policy at chance, although such a policy can produce mixed rewards in general); `R3`'s `sigma=0 = 0.00` shows that rewards varied, not that they helped. Takeaway 3 (anchor wiring) and the FLOPs lesson (5) are unaffected. The ledger row is unchanged (append-only); see `EXP-006`.

#### Bugs Caught During Audit & Verified Fixed
1. **`I2-Hybrid` Train-vs-Eval Interface Skew (`CRITICAL`, caught by `verifier`):** `diagnostics.py` initially prepended `4` pure continuous latents AND wrapped the `6` HRPO-blended steps in `<think>`/`</think>` (`12` extra steps total), whereas `ContinuousLatentWrapper.rollout_hybrid` trains with `thought_len` blended steps directly after `prefix_ids` and no delimiters. That mismatch produced spurious `NLL=7.68`, `Hard-BPB=5.030`, `ECE=0.844`, and `4.71×` `B2` FLOPs! Fixed to exact `< 1e-4` parity with `recompute_hybrid_logprobs_and_continuation` (`extra_steps == thought_len`), locked with `test_hybrid_latent_eval_matches_recompute_hybrid_parity`, and regenerated `EXP-001` cleanly.
2. **Multi-GPU `torchrun` Distributed Sync (`MEDIUM`, caught by `verifier`):** `train.py` read `LOCAL_RANK`/`RANK` without initializing `torch.distributed` or all-reducing gradients, which would have run 4 uncoordinated duplicates on 4× H100. Added `init_distributed`/`cleanup_distributed`, rank-0 broadcast, rank-sharded sampling, and `dist.all_reduce(..., op=AVG)` in `train.py` & `grpo.py`, verified by a 2-rank Gloo unit test.

### `EXP-006`: Regime 1 Learnability Scoping (`2026-11-26`, not preregistered)
**Question.** `BL-01` (a longer, multi-seed micro-pilot) assumes that the micro arms can learn Regime 1 if trained longer. Can one supervised NTP micro model leave the prompt-blind floor on Regime 1 at all, within a CPU budget? This was a scoping check that makes no capability claim, so it was not preregistered. The status `NEGATIVE` answers this feasibility question only; it does not bear on `H1`.

**Setup.** 13 single-model probes (`scripts/probe_learnability.py`): `B1` (`micro-small`, `0.98M`) in all but one, `B2` (`micro-large`, `2.85M`) in one. Each used a 50k-sample train set, batch 32, the `run_micro_pilot.py` learning rates, and 3,000 steps (one 6,000, one 20,000). Train depths ranged from `{2,4}` (the `EXP-001` setting) down to `{1}`. The probes used the stratified Regime 1 tasks or a new pointer-chain task: three equal-depth copy chains per prompt, so copying one of the three literals scores `1/3`; depth 0 is a single key-value lookup. Each family ran with full-sequence loss (the default) and with answer-only loss. Each cell is scored against its **structural floor**: the prompt-blind reference, or `1/3` for `dag_planning` and pointer chains, whose answer is one of three options listed in the prompt. The test is one-sided exact binomial, Bonferroni-corrected over all 89 cells. Per-cell accuracy and NLL, val curves, answer label shares, and an equivalent command for every probe: [`phase1_pilot/results/exp006_learnability_summary.json`](../phase1_pilot/results/exp006_learnability_summary.json).

**Findings.**
1. **`EXP-001`'s Regime 1 results are below the prompt-blind floor** (see the correction note above and `DE-007`).
2. **Only 1 of 89 cells beats its structural floor:** `python_trace` `d=1` trained on `d=1` alone (`17.0%` vs. `6.0%`, n=200; its `d=2` OOD cell, `10.5%` vs. `5.0%`, is suggestive but does not survive correction). Trained at shallow depths, `dag_planning` beats the prompt-blind reference (`19`–`36%` vs. `1.5`–`7.6%`) but never significantly beats `1/3` (best `35.8%`), so it has learned to pick a listed candidate, not to plan. `B2` (`micro-large`) is no better than `B1` at `d∈{2,4}`.
3. **A single lookup is not learned at this budget.** With three key-value pairs, single-token names, and answer-only loss, final accuracy is `31`–`36%` across depths 0–2 after 20,000 steps (640k training draws, 34 CPU-min). Answer NLL is near `ln 3` (`1.12`–`1.14`), which is what spreading probability evenly over the three listed literals gives. The averages fit that reading, but they also fit a position heuristic or a mix of right lookups and confident errors; no candidate-mass or key-swap probe was run, so the mechanism is not identified. The final train-batch loss (`0.53`) and val loss (`0.57`, answer plus `<eos>`) are close, so there is no sign of overfitting, but that does not establish why training stalls. Accuracy was measured only at the final checkpoint. The result shows that the task is not learned at this budget, not that it is unlearnable.
4. **The backbone forward pass matches a reference implementation.** With copied weights, `LlamaLM` gives the same logits as `transformers`' `LlamaForCausalLM` (max difference `0.0`, with and without a padding mask, at init and for a trained probe checkpoint); a RoPE-theta mismatch is detected. This is now a test (`tests/test_llama_reference.py`). It rules out a forward-pass bug in these configurations; training behavior, the optimizer and schedule, and whether the architecture suits the task remain open.
5. **Answer-only loss helps on pointer chains but does not cross the floor.** Under full-sequence loss, answers are `3.57%` of supervised labels on single-token pointer chains (`1.40%` in the `EXP-001` stratified mix), and pointer accuracy sits below `1/3` (`22`–`26%`).

**Bug caught.** In the first pointer probes, names `v16`–`v31` tokenize as two `PilotTokenizer` tokens whose first token (`v1`, `v2`, `v3`) is itself a variable name, which confounds lookup with multi-token key matching. The default pool is now `v0`–`v15` (`--pool-size 32` reproduces the first probes exactly; test `test_default_pool_names_are_single_pilot_tokens`). `modular_varbind` uses the same `v<k>` naming but only `v0`–`v15` at the depths tested, so it is not affected.

**Second bug caught (Codex review).** The stratified split deduplicates rendered prompts, so a prompt can reappear in another split with its equations in a different order. `modular_varbind` `eval_id` `d=1` holds 7 of 200 such duplicates of training prompts (trained on `d=1`) and 1 of 67 (trained on `{1,2,3}`). No overlap was found at `d∈{2,4}` or in pointer chains, and the one cell above its floor (`python_trace`) is not affected. Fixing it changes generated data, so it is a data-version change (`PROCEDURE.md` §4.4, `BL-12`). The val loader also overlaps the reported ID cells, so a follow-up chosen by looking at these cells needs a fresh holdout.

**Consequences.**
- `BL-01` and `BL-02` cannot make capability comparisons on these tasks at this budget: the binding constraint is base-model competence. They can still test reward mechanics, or whether an intervention lifts a weak baseline above its floor; `EXP-006` tested only supervised NTP at one seed and does not rule that out.
- `EXP-002` gets a competence gate (see its plan below).
- The next decision is a testbed choice (`BL-12`): find a task and budget where `B1` clears the floor, or move the first real comparison to `L2`.

**Mechanism identified (`2026-11-27`, `EXP-007` Part A).** The candidate-mass and key-swap probe on the 20k-step single-lookup checkpoint shows that the model puts `99.3`–`99.5%` of its mass on the three listed literals, but its greedy answer changes with the queried key in only `3`–`5.5%` of prompts. It copies a listed literal without matching the key. Finding 3 above is left as written.

**Not tested** (as of `EXP-006`; the candidate-mass and key-swap probes have since run, see above): candidate-mass and key-swap probes, accuracy curves over training, larger batches, other learning rates or schedules, curricula, more than 20k steps, tied embeddings, and multiple seeds (every probe used seed 42).

### `EXP-007`: Trainer Check on Zoology MQAR (`BL-12` step 1; preregistered `2026-11-26`)
**Preregistration** (written and committed before the MQAR runs; not edited after them).
- **Question.** Is the single-lookup plateau of `EXP-006` caused by our trainer or backbone, or by the pointer task's setup? Test: train our model and trainer on a task that small attention models are known to learn.
- **Reference task.** Multi-query associative recall (MQAR) as generated by Zoology (Arora et al. 2023, `HazyResearch/zoology`, `zoology/data/multiquery_ar.py`). A context of `K` interleaved key/value tokens with disjoint key and value vocabularies and no separators. Then a query region where each key appears once at a power-law gap (`power_a=0.01`); the next position is labelled with the key's value, every other position is ignored, and filler is random tokens. Data and optimizer settings follow Zoology's minimal example (`zoology/experiments/basic_examples/basic.py`): 127 keys and 128 values, sequence length 64, `K=4`, 10k train and 1k test rows, AdamW lr `1e-3`, batch 32. Zoology reports that attention solves MQAR at these sizes (paper Figure 2); we did not read exact values from the figure.
- **What stays ours.** `micro-small` `LlamaLM` (4 layers, d=128, 4 heads, RoPE, SwiGLU, untied head; `0.92M` parameters at `M1`'s 264-ID vocabulary, `0.86M` at `M3`'s 44) and `train_supervised` with its defaults (5% warmup, cosine to 10%, clip 1.0, weight decay 0.01). Zoology's model uses 2 layers, one head, learned absolute positions, no MLP in the Figure 2 sweep, tied embeddings, dropout 0.1, weight decay 0.1, and up to 100 epochs with early stopping. So this checks whether our stack learns MQAR; it is not a replication.
- **Arms** (seed 42, 10,000 steps = 32 epochs, test accuracy on all queries of 1,000 held-out rows every 500 steps; `scripts/trainer_check_mqar.py`):
  - `M1` `mqar_basic`: the reference task, all 4 queries labelled.
  - `M2` `mqar_1q`: `M1`'s inputs, one random query labelled per row (the pointer task has one answer label per row).
  - `M3` `mqar_smallvocab`: 16 keys, 19 values, `K=3`, all queries labelled (the pointer task's key and value vocabulary and pair count).
- **Hypotheses and decision rules** (on final test accuracy of `M1`; single seed, so these are triage rules, not proofs):
  - *A, stack can learn associative recall:* `M1` ≥ 0.99. A general defect in the trainer or backbone is then unlikely to explain the `EXP-006` plateau; a problem specific to the pointer task's format or its interaction with the trainer stays possible. `BL-12` moves to step 2 (pointer-task sweep).
  - *B, stack fails on MQAR:* `M1` < 0.90. That can come from a trainer defect or from a sound trainer with an unsuitable model, schedule, or budget. One more run separates the first from the rest, with settings fixed now: Zoology's optimizer recipe on the same model, data, seed, and 10,000 steps: AdamW (lr `1e-3`, weight decay 0.1), no warmup, no gradient clipping, cosine annealing to 0 stepped once per epoch over 32 epochs (313 steps per epoch). If it reaches ≥ 0.99, our trainer settings are at fault. If not, the cause is in the model or the budget; this experiment does not say which, and the next check is a Zoology-shaped model.
  - *Partial:* 0.90 ≤ `M1` < 0.99. Our stack learns MQAR but does not finish within 10,000 steps; no defect claim (Zoology gives no step count to compare with); step 2 proceeds with a longer budget.
  - *Secondary, descriptive:* the copy floor is `1/K` (picking any context value). An arm *plateaus* if its final accuracy is at most `1/K + 0.05`. If `M1` learns and `M2` plateaus, label density is the first suspect for the pointer plateau, and step 2 sweeps batch size first (the pointer loss also supervises `<eos>`, which `M2` lacks). If `M3` plateaus, its changes are the suspect, but they are confounded: pair count, labels per row, vocabulary size, and the rate of filler that collides with a key all change together.
- **Abort:** NaN or Inf train loss (checked every step by the runner).
- **Budget:** 3 runs in parallel on 4 threads each, about 25 min wall clock (0.11 s per step measured), under 1.5 CPU-hours.
- **Part A, analysis of an existing checkpoint, run before this preregistration was written:** the candidate-mass and key-swap probe (`key_swap_probe` in `probe_learnability.py`) on the `EXP-006` 20k-step single-lookup checkpoint. It is reported with the results but was not a preregistered prediction.

**Results** (`2026-11-27`; per-arm curves, configs, commands, and the Part A report: [`phase1_pilot/results/exp007_trainer_check_summary.json`](../phase1_pilot/results/exp007_trainer_check_summary.json)).

| Arm | Labels per row | Params | Final test acc | First eval ≥ 0.99 | Final train-batch loss | Copy floor `1/K` | Reading |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M1` `mqar_basic` | 4 | `0.92M` | `1.000` | step 3,500 | `1.4e-5` | `0.25` | learns |
| `M2` `mqar_1q` | 1 | `0.92M` | `0.202` | never | `2.3e-4` | `0.25` | plateaus (≤ `0.30`) |
| `M3` `mqar_smallvocab` | 3 | `0.86M` | `1.000` | step 1,000 | `7.0e-6` | `0.33` | learns |

- **Decision: rule A.** `M1` reaches `1.000` (first ≥ 0.99 at step 3,500; `0.258`, about the copy floor, at step 500 and `0.914` at step 1,000). Our model and trainer, with default settings, learn MQAR, so a general defect in either is unlikely to explain the `EXP-006` plateau. The fallback run was not triggered.
- **`M2` fits its training rows and does not generalize.** Its train-batch loss falls from `3.88` to `2.3e-4` (a summed NLL below `ln 2` over the batch's 32 labels, so all 32 were predicted correctly), while held-out accuracy stays between `0.12` and `0.20` at every evaluation, below even the `0.25` that copying any context value gives. This reading is memorization of the 10,000 training rows (32 epochs) in place of the recall solution `M1` finds on the same inputs. Accuracy on the whole training set, and on the unlabelled queries of training rows, was not measured; those would separate row-specific fitting from a partial or position-based rule. By the secondary rule, label density is the first suspect for the pointer plateau. The run does not separate the two ways one label per row differs from four: four times fewer answers per step, and four times fewer distinct supervised answers in the training set (10k vs. 40k).
- **`M3` learns faster than `M1`** (`0.995` at step 1,000, first `1.000` at step 4,500). With every query labelled, the combined small setting (16 keys, 19 values, `K=3`, and the higher filler-collision rate that comes with it) is learnable.
- **Part A: the `EXP-006` model copies a listed literal without reading the query.** On 200 prompts per depth, `99.3`–`99.5%` of next-token probability sits on the three listed literals and mean `p(correct)` is `0.331`. When the bindings are held fixed and only the queried key changes, the greedy answer changes in `3.0%` (d=0), `5.0%` (d=1), and `5.5%` (d=2) of prompts, and no prompt gets all three keys right. (These are greedy answers; the probe does not show whether the distribution over the three literals shifts with the key.) Re-scoring the checkpoint reproduces the `EXP-006` cells exactly (accuracy `0.339`/`0.360`/`0.308`, NLL `1.128`/`1.116`/`1.137`). This identifies the `EXP-006` finding 3 mechanism. `M1`'s accuracy is also at its copy floor at step 500 (`0.258`; its candidate mass was not measured) and leaves it by step 1,000; the pointer model is still at it after 20,000 steps (its accuracy was measured only at the final checkpoint).
- **The `M2` failure does not look like the pointer plateau.** For the pointer model, `EXP-006` finding 3 reports a final train-batch loss of `0.53` (one batch) against a val loss of `0.57` (answer plus `<eos>` in both; 50k rows over 12.8 epochs). That is weak evidence against substantial overfitting, not a measurement of train-set accuracy. The pointer model also assigns its mass to the listed literals, while `M2` scores below the copy floor on held-out rows. What the two share is one supervised answer per row.
- **Budget overrun.** The arms took about 80 min wall clock each, against the preregistered estimate of 25 min, and `3.95` arm-hours of elapsed time summed over the three concurrent arms (the "CPU-hrs" in the index, by the `EXP-006` convention), against "under 1.5". The `0.11` s/step estimate was measured on a solo run; the host's load average was about 20 on 12 hardware threads during these runs. No abort rule covered the budget. Results are unaffected; the estimate method was wrong.
- **Single seed (42) for every arm**, as preregistered.

**Next (`BL-12` step 2, not yet preregistered).** The preregistered secondary rule says "sweep batch size first". `M2` adds a second suspect that batch size alone does not test, the number of distinct supervised answers. Draft design, revised after a Codex review of this write-up: (i) on MQAR with one labelled query per row, a 2×2 of batch {32, 128} × train rows {10k, 40k} on a common holdout, with matched-update and matched-exposure comparisons and the schedule stated, plus train-set accuracy on labelled and unlabelled queries; (ii) on pointer chains at `d=0` (pool 16, answer-only, seed 42), identical inputs that query all three chain heads, with one vs. three answer labels and the same `<eos>` treatment, a rerun one-query `d=0` control (the `EXP-006` checkpoint trained on depths 0 and 1), batch 128, and a fresh holdout split by binding set. Step 2 needs its own preregistration before it runs.

---

## 4. Planned Experiments (`EXP-002` – `EXP-005`)

### `EXP-002` — Phase 1 Stage 1 H100 Core Pilot (`~35–40` H100 GPU-hrs)
- **Hypotheses:** `RQ-01`, `H1`, `H1a`, `H1b`, `H1c`
- **Arms (8):** `B1` (`160M` NTP, ~1.5h), `B2a` (`410M` NTP, ~3.5h), `B2b` (`1.0B` NTP, ~9.0h), `B3` (`160M` Pause, ~3.0h), `B4` (`160M` Continuous Latent `L=4`, ~4.5h), `RL-R2` (`160M` `T2-self` + clipped `R2_log`, ~6.0h), `RL-R3` (`160M` `T2-self` + `R3_brier`, ~6.0h), `I2-Hybrid` (`B4` $\to$ HRPO + `R3_brier`, ~6.5h).
- **Data:** 2B tokens stratified across Regime 1 (DAGs $d\in\{2,4,6,8\}$, Python traces, modular var-bind), Regime 2 (OpenWebMath 40% / Stack v2 40% / FineWeb-Edu 20%), and Regime 3 (TriviaQA / aleatoric negative control).
- **Go / No-Go Gate:** `160M` + latent CoT matches/beats `410M` (`B2a`) at $\le 1.0\times$ inference FLOPs, beats `B3` by $\ge 3$ pp on Regime 1, passes $\Delta\text{NLL}_{\text{swap}} > 0.15$ nats + $\Delta\text{NLL}_{\text{trunc}} > 0$ + monotonic OOD depth scaling, maintains Regime 3 calibration, and strictly increases OOD `pass@64` ($d=6,8$).
- **Competence gate (added `2026-11-26`, `EXP-006`, `DE-007`):** before any arm comparison is read, `B1` must beat each Regime 1 task's structural floor on ID depths (the prompt-blind reference, or `1/3` for tasks whose answer is one of three listed options), by a margin fixed in the preregistration. If it does not, the verdict is `INCONCLUSIVE`, not `FALSIFIED`. Report ID and OOD accuracy separately. The preregistration must also check one Regime 1 task at `L2` scale on a cheap run first: no micro model in `EXP-006` cleared the floor (`EXP-006` did not evaluate this gate as written, since its margin is unset and its depth grid differs).
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
