# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research rules, the Git-vs-HuggingFace artifact boundary, and the end-of-session handoff checklist live in @AGENTS.md (canonical for every agent; not repeated here). On a fresh context also read [`research/STATE.md`](research/STATE.md) (current frontier, hardware, next decisions) and [`research/PROCEDURE.md`](research/PROCEDURE.md) (experiment lifecycle, isoFLOPs rules, anti-patterns), and check the Dead-Ends Graveyard in [`research/BACKLOG.md`](research/BACKLOG.md) before proposing an experiment.

## Layout

The root holds the theory (`00-unified-framework.md` synthesizes Tracks 1–3 in `01..03-*.md`) and `research/` (operating memory plus the `experiments.jsonl` ledger). The only code is `phase1_pilot/`, a self-contained `uv` project implementing Track 1 §5: latent chain-of-thought trained as RL ("SSL-as-RL"). Tracks 2 and 3 have no code. Development is on a CPU-only machine, so only the `micro-*` presets run locally; `160m`/`410m`/`1b` need an H100 node.

## Commands (run from `phase1_pilot/`)

```bash
uv sync                                    # CPU torch; pyproject pins the pytorch-cpu index
uv run pytest -q                           # full suite: ~50 tests, ~10 s on CPU
uv run pytest tests/test_rl.py::test_ssl_rl_trainer_anti_sabotage_and_smoke_micro_small -v   # one test
uv run pytest tests/test_eval.py -k parity -v                                                # by keyword
uv run --with pytest-cov pytest --cov=phase1_pilot --cov-report=term-missing                 # coverage (pytest-cov is not a project dependency)
uv run python scripts/run_micro_pilot.py --output-dir runs/<tag>   # 7-arm CPU micro-pilot, ~90 s
uv run python -m phase1_pilot.train --help                         # train one arm
uv run python scripts/log_experiment.py --render-only              # print the Pareto table from the ledger
uv run python scripts/log_experiment.py --metrics <json> --exp-id EXP-xxx --title "..." --dry-run
```

No linter, formatter, or type checker is configured.

- **`run_micro_pilot.py` writes to `results/` by default**, which overwrites the committed `EXP-001` artifacts (`results/micro_pilot_{report.md,metrics.json}`). A default-argument rerun on the dev machine reproduced every committed metric exactly (seed 42, CPU); only run metadata differed (`elapsed_seconds`, `config.output_dir`, the report timestamp), so a rerun into `results/` only dirties the tree. Send every other run to `--output-dir runs/<tag>` (gitignored).
- **Dataset sizes are capped.** `build_stratified_datasets` raises `RuntimeError: Could only generate … unique natural_stream samples` unless roughly `0.15 × n_train + n_eval ≤ 1080` (see Data). The micro-pilot and `train.py` defaults fit; the H100 launcher's defaults do not.
- **If pytest collection fails with `No module named 'torch'` after `uv sync`**, check `head -1 .venv/bin/pytest`. A stale absolute shebang (the repo directory was moved) makes `pytest` resolve to another interpreter on `PATH`. Fix with `uv sync --locked --reinstall`.
- **GPU nodes:** `uv sync --extra h100`, `uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124`, `source .venv/bin/activate`, then `NPROC_PER_NODE=4 OUT_ROOT=runs/h100_stage1 bash scripts/launch_h100_stage1.sh`. The launcher runs 8 training jobs in sequence through bare `python`/`torchrun`.
  - Its default `N_TRAIN_SAMPLES=200000 N_EVAL_SAMPLES=2000` exceed the dataset cap, so the first job exits during dataset construction, before any training. Until the generator is fixed, pass sizes that fit, such as `N_TRAIN_SAMPLES=5000 N_EVAL_SAMPLES=300`. `SUP_STEPS=200 RL_STEPS=50 NPROC_PER_NODE=1` shortens it to a smoke pass.
  - After the CUDA reinstall, do not use plain `uv run` on that node. It re-syncs the venv to `uv.lock`, which pins the CPU torch wheel. `--device auto` then falls back to CPU without an error, and `--bf16` is ignored off CUDA. Use `uv run --no-sync` or the activated venv.

## Architecture (`phase1_pilot/phase1_pilot/`)

**One thinking position per sample.** Every sample is `prompt … <ans>` followed by `target`. `tokenize_sample` records the boundary as `prompt_len` and `collate_batch` emits `prompt_lens`. All extra compute is spent at that boundary: pause embeddings, continuous latent steps, or a sampled thought go between prompt and answer, and RL scores the `n_ahead` tokens that follow it. The "tokens" a `T*` filter selects are therefore batch items (one prefix each), not positions in a text stream.

**Names live in `config.py`; behavior is dispatched by string at several sites.** The `VALID_VARIANTS` / `VALID_REWARDS` / `VALID_FILTERS` / `VALID_INTERFACES` tuples drive dataclass validation and the `train.py` argparse choices. Each name's behavior is implemented separately:

| Kind | Dispatch sites |
| :--- | :--- |
| variant | `models/latent.py::build_model`; `models/llama.py::estimate_inference_flops_per_token` (adapter FLOPs); `eval/diagnostics.py::_build_prefix_embeds_for_model` and the `mode` branches of `evaluate_thought_faithfulness` |
| reward | `rl/rewards.py::compute_reward` |
| filter | `rl/filters.py::select_tokens`, plus the `if/elif` in `SSLRLTrainer.train_step` that feeds each filter its inputs |
| interface | `SSLRLTrainer._is_hybrid_interface` (true only for `I2_hybrid` **and** a `ContinuousLatentWrapper` model) |

Adding an arm to the two fixed suites also needs wiring in `scripts/run_micro_pilot.py` (a training block, an `arm_specs` entry, the verdict inputs) and a block in `scripts/launch_h100_stage1.sh`. A new latent or thought interface also needs a parity test (see Invariants).

**Arms are compositions of one transformer.** `models/llama.py::LlamaLM` is the only backbone. `build_model` wraps it by `variant`: `ntp` → bare `LlamaLM` (`B1` small, `B2` large); `pause` → `PauseTokenWrapper` (`B3`); `continuous_latent` → `ContinuousLatentWrapper` (`B4`); `hybrid_latent` → the same wrapper with an HRPO-style gate (`I2-Hybrid`). Discrete-thought RL arms (`RL-R*`, interface `I1_discrete`) have no wrapper: `DiscreteThoughtHelper` samples `thought_len` tokens between fixed `<think>`/`</think>` delimiters from a plain `ntp` model. Consequences:
- Interface/variant combinations are not validated. `I2_hybrid` with `ntp` or `pause`, and `I1_discrete` with `hybrid_latent`, train discrete thoughts without an error. `I2_hybrid` with `continuous_latent` runs the hybrid rollout with no gate, so the sampled tokens never enter the latent state.
- RL arms start from supervised weights: `RL-R*` from `B1`, `I2-Hybrid` from `B4`. The launcher does this with `--init-checkpoint`; the micro-pilot copies the in-memory models. `train.py --mode rl` without `--init-checkpoint` starts RL from random weights.
- `--init-checkpoint` loads with `strict=False`. Wrapper keys are prefixed `base_lm.`, so keys that do not match are skipped without a warning: a `B1` checkpoint loaded into a wrapper loads nothing.
- `checkpoint.pt` stores `model_state_dict` and the train/RL configs but no `ModelConfig`. Rebuild the model with the same model flags (`--preset`, `--variant`, `--n-pause-tokens`, `--n-latent-steps`, `--max-seq-len`). `scripts/publish_hf.py` synthesizes a `config.json` per arm for the Hub.

**Token protocol.** `data/tokenizer.py::SPECIAL_TOKENS` fixes `<pad>=0 <bos>=1 <eos>=2 <think>=3 </think>=4 <pause>=5 <sep>=6 <ans>=7`. IDs 8–263 are a lossless UTF-8 byte fallback and 264–511 are multi-character tokens (vocab 512). The same IDs are literal defaults in `SSLRLTrainer`, `DiscreteThoughtHelper`, and `build_model(pause_token_id=5)`.

**Data.** `data/generators.py::build_stratified_datasets` returns five splits: `train` (by default 75% Regime 1 algorithmic at hop depths {2,4}, 15% Regime 2 synthetic natural text, 10% Regime 3 aleatoric cloze), `eval_id`, `eval_ood` (depths {6,8}), `eval_natural`, and `eval_aleatoric`. The splits are separated by prompt-signature dedup (applied before tokenization), disjoint depths, and disjoint entity namespaces. `train.py --regime` only picks which eval split is the validation loader; training always uses the mixed `train` split.

`generate_natural_stream` is the size bottleneck. Every third sample is an arithmetic-series item with only 360 variants, so the generator yields about 1,080 unique samples, shared by `train` and `eval_natural`. The algorithmic generators are not the limit (150k unique samples build in seconds).

**The scale path is not wired.** `HFTokenizerAdapter` and `load_h100_web_mixture` exist and are unit-tested with mocks, but `train.py` always builds `PilotTokenizer` plus the synthetic generators. With the `160m`/`410m`/`1b` presets the model gets a 32k-token vocabulary but still trains on the 512-ID synthetic data; real tokenizers and web text reach neither the trainer nor the launcher.

**RL step** (`rl/grpo.py::SSLRLTrainer.train_step`):
1. Sample `G` thoughts per prefix, without gradients.
2. Reward each thought from the continuation logits it produces (`rl/rewards.py`, under `no_grad`). `R1_exact` is absolute. `R2_log` and `R3_brier` are gains over the no-thought logits of `ref_model`, by default a deep-copied frozen snapshot of the policy.
3. Compute Dr. GRPO advantages: centered within each group, with no division by the standard deviation.
4. Mask the policy-gradient term: keep the batch items the filter selects, drop zero-variance groups (`filter_zero_var`), and inside each thought keep only the top-p highest-entropy tokens.
5. Loss = DAPO clip-higher policy gradient + `lambda_ntp` × no-thought NTP loss + `continuation_sft_weight` × cross-entropy of the continuation given the thought (over all groups).

Things to know when reading RL runs:
- Watch `zero_var_frac` (σ=0 reward collapse, `DE-001`) and `nothought_nll_drift` (baseline sabotage, `DE-002`).
- `val_loss` comes from the supervised forward path (`evaluate_nll`), not from sampled thoughts, so it does not measure thought quality.
- `--filter-type T2_scale` without `--teacher-checkpoint` ranks tokens with a randomly initialized teacher.
- Multi-GPU is hand-rolled rather than `DistributedDataParallel`: `init_distributed`, a rank-0 `_broadcast_model_state`, `DistributedSampler`, and `_all_reduce_model_grads` after each backward pass.

**Evaluation.** `train.py` never runs the diagnostics; it logs losses and RL telemetry. The five Track 1 §5.5 diagnostics in `eval/diagnostics.py` (`evaluate_pareto_point`, `evaluate_pass_at_k`, `evaluate_thought_faithfulness`, `evaluate_calibration`, then `compute_pilot_verdict`) are called only by `scripts/run_micro_pilot.py`, on models still in memory. Nothing evaluates a saved checkpoint yet (`BL-07`).
- The eval code assembles each variant's inference sequence itself, reusing wrapper primitives such as `generate_latents`, instead of calling the training path. That duplication is where train/eval skew comes from (`DE-003`).
- The path is chosen from the model's wrapper type plus caller flags. Pareto, pass@k, and calibration take `use_thought`/`thought_len`: a discrete-thought RL model is architecturally `ntp`, so omitting `use_thought=True` scores it as a no-thought baseline with no warning. Faithfulness takes `mode`, and `mode="auto"` treats every bare `ntp` model, baselines included, as discrete-thought.
- FLOPs come from `estimate_inference_flops_per_token(config, seq_len, extra_steps)`, where `extra_steps` is the inserted positions amortized per predicted target token.
- The verdict logic lives in `compute_pilot_verdict`. Four thresholds are keyword defaults (ΔNLL_swap > 0.15, ≥ 3 pp over `B3`, ECE within +0.05 of `B1`, drift ≤ 0.15) and the `≤ 1.0× B2` FLOPs cap is hardcoded. Changing any of them changes the verdict logic, which `PROCEDURE.md` Stage 2 requires to be frozen before compute.

**Ledger and publishing.** `scripts/log_experiment.py` reads the `run_micro_pilot.py` metrics schema and appends one row to `research/experiments.jsonl`.
- It does not validate the schema. A JSON without `arms` or `verdict` (a per-job `metrics.jsonl` line from the launcher, for example) is logged as a completed zero-arm row with verdict `UNKNOWN`.
- `--update-markdown` regenerates the table between the `PARETO_TABLE_START`/`END` markers in `research/EXPERIMENTS.md`, so do not hand-edit inside them.
- The ledger holds `"date": "planned"` preregistration rows for `EXP-002`–`EXP-005`. Logging one of those IDs is refused as a duplicate unless `--allow-update`, which replaces the row and rewrites the file. `PROCEDURE.md` §4.4 makes history append-only and does not spell out this exception, so never use the flag on a completed row.
- The script records neither `git_sha` nor `supersedes`, although `PROCEDURE.md` §6.2 lists both.
- `scripts/publish_hf.py` takes arm directories directly under `--run-dir`, while `--report` and `--metrics` default to the committed `results/` files. To publish a run from a custom output directory, pass all three (`--run-dir <dir>/micro_runs --report <dir>/micro_pilot_report.md --metrics <dir>/micro_pilot_metrics.json`), or the new checkpoints are paired with `EXP-001` metrics.

## Invariants enforced by tests

When you touch the listed code, keep the matching test green and extend it for any new interface:
- **Train/eval parity < 1e-4** (`DE-003`): a change to a latent or thought path in `models/latent.py` or `eval/diagnostics.py` must keep the eval path's continuation logits equal to the training recompute path. Model test: `test_hybrid_latent_eval_matches_recompute_hybrid_parity` in `tests/test_eval.py`. Only the hybrid path has such a test today; the pause, continuous, and discrete eval paths are covered only by wrapper-internal consistency checks in `tests/test_models.py`. A new interface needs its own parity test.
- **Frozen anchor** (`DE-002`), all in `tests/test_rl.py`: `test_ssl_rl_trainer_anti_sabotage_and_smoke_micro_small` checks that `ref_model` has no gradients and is unchanged after training steps (`ref_ema_decay=1.0`); `test_compute_reward_modes_and_stop_gradient` checks that rewards are detached; `test_filters_t0_t1_t2_self_t2_scale` checks that `T2_self` rejects teacher logits. The trainer's own rejection of a `teacher_model` under `T2_self` has no test.
- **Distributed sync** (`DE-004`): a change to `train.py` or `SSLRLTrainer` must keep the 2-rank Gloo test passing (`tests/test_rl.py::test_torch_distributed_sync_supervised_and_rl`; it uses `mp.spawn`, so it runs on CPU).
- **FLOPs accounting**: if a variant's step count or adapter changes, update `estimate_inference_flops_per_token` and `tests/test_models.py` together, because every Pareto claim depends on it.
