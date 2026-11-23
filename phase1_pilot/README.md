# Phase 1 Small-Model Latent-CoT RL Pilot

Implementation of the **Track 1 §5 Experimental Testbed** from [`01-ssl-as-rl-latent-cot.md`](../01-ssl-as-rl-latent-cot.md) testing whether framing self-supervised next-token prediction as reinforcement learning with unsupervised latent chain-of-thought (`SSL-as-RL`) enables small models to punch above their parameter weight class at matched inference FLOPs.

## Architecture Map

```text
phase1_pilot/
├── pyproject.toml                       # uv project config (CPU PyTorch default, [h100] extra)
├── uv.lock                              # Reproducible locked dependency graph
├── phase1_pilot/
│   ├── config.py                        # Shared ModelConfig (5 presets), RLConfig, DataConfig, TrainConfig
│   ├── data/
│   │   ├── tokenizer.py                 # Lossless vocab=512 PilotTokenizer + HFTokenizerAdapter
│   │   └── generators.py                # 3-regime generators (Regime 1 Algorithmic ID/OOD, Regime 2 Natural + H100 loader, Regime 3 Aleatoric)
│   ├── models/
│   │   ├── llama.py                     # LlamaLM (RMSNorm, RoPE, GQA, SwiGLU, KV-cache) + exact FLOPs/param accounting
│   │   └── latent.py                    # B3 PauseTokenWrapper, B4/I2 ContinuousLatentWrapper, DiscreteThoughtHelper, build_model
│   ├── rl/
│   │   ├── rewards.py                   # Detached R1_exact, R2_log, and exact O(|V|) strictly proper R3_brier
│   │   ├── filters.py                   # T0_all, T1_entropy, teacher-free T2_self, and T2_scale filters
│   │   └── grpo.py                      # Dr. GRPO + DAPO Clip-Higher + frozen reference anchor + aux lambda_ntp + torch.distributed sync
│   ├── eval/
│   │   └── diagnostics.py               # 5 mandatory Track 1 §5.5 diagnostic probes + falsification verdict engine
│   └── train.py                         # Unified supervised (B1–B4) & SSL-as-RL (I1/I2) CLI trainer (single & multi-GPU torchrun)
├── scripts/
│   ├── run_micro_pilot.py               # End-to-end 7-arm CPU micro-scale pilot (~93s on CPU)
│   ├── launch_h100_stage1.sh            # Full Stage 1 ~30–45 H100 GPU-hour launcher (160M / 410M / 1.0B)
│   ├── publish_hf.py                    # Reusable HuggingFace Hub publisher (models & Parquet datasets)
│   └── log_experiment.py                # Experiment ledger CLI -> research/experiments.jsonl & research/EXPERIMENTS.md
├── results/
│   ├── micro_pilot_report.md            # Verified CPU micro-scale diagnostic report
│   └── micro_pilot_metrics.json         # Full machine-readable JSON telemetry & diagnostics
└── tests/
    ├── test_data.py                     # Zero-leakage, exact BFS/exec/modular & aleatoric entropy tests
    ├── test_models.py                   # Param/FLOPs accounting, KV-cache parity, BPTT & rollout tests
    ├── test_rl.py                       # Strict Brier propriety, anti-sabotage, Dr. GRPO/DAPO, T2_self & distributed sync tests
    └── test_eval.py                     # Unbiased pass@k, ECE/Brier bounds, swap/trunc faithfulness, I2-Hybrid parity & verdict tests
```

## Five Mandatory Diagnostics (Track 1 §5.5)

1. **FLOPs-Matched Pareto Frontier (`evaluate_pareto_point`):** Exact answer accuracy, Hard-BPB, and exact autoregressive `inference_flops_per_token` (accounting for pause, continuous latent, and discrete thought steps).
2. **Support-Boundary Probe (`evaluate_pass_at_k`):** Unbiased $\text{pass@}k = 1 - \binom{n-c}{k}/\binom{n}{k}$ across $k \in \{1, 4, 16, 64\}$ broken down by hop depth $d \in \{2, 4, 6, 8\}$ to distinguish OOD capability expansion from `pass@1` sharpening.
3. **Causal Thought Faithfulness (`evaluate_thought_faithfulness`):** Swap-sensitivity ($\Delta\text{NLL}_{\text{swap}}$), format-matched intermediate truncation/orthogonal perturbation ($\Delta\text{NLL}_{\text{trunc}}$ preserving delimiters & length/norm), and monotonic OOD depth scaling.
4. **Distributional Calibration (`evaluate_calibration`):** 10-bin ECE, strictly proper Brier score $\text{BS}(q, v^*) = 2q(v^*) - \sum_v q(v)^2 - 1 \in [-2, 0]$, and overconfidence gap evaluated on both Regime 1/2 and **Regime 3 (Aleatoric Negative Control)**.
5. **Falsification Verdict Engine (`compute_pilot_verdict`):** Evaluates the exact Strong Confirmation / Partial Confirmation / Falsification criteria from Track 1 §5.5.

## Verified CPU Micro-Pilot Results

> **Scope Notice:** Proof-of-life & pipeline validation on CPU (`seed=42`, `600` train / `64` Regime 1 eval / `48` Regime 3 eval samples, `80` supervised + `20` RL steps, **93.1s** total runtime, **50/50 pytest passed**). Distinct from the full `160M` / `410M` / `1.0B` H100 Stage 1 run. Full report in [`results/micro_pilot_report.md`](results/micro_pilot_report.md).

**Track 1 §5.5 Verdict (CPU Micro-Scale):** `FALSIFIED` — no small-model arm simultaneously beat pause control `B3` (`0.0%`) and FLOPs-matched `B2` (`6.2%`); best candidate `I2-Hybrid` reached `3.1%` at `2.51×` `B2` FLOPs.

| Arm | Params | FLOPs/tok (vs B2) | Exact Acc | Hard-BPB | OOD `pass@16` | $\Delta\text{NLL}_{\text{swap}}$ | $\Delta\text{NLL}_{\text{trunc}}$ | Mono Depth | Reg3 ECE | Reg3 Brier | `sigma=0` Frac |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `B1` | 984,192 | 0.34× | 3.1% | 2.318 | 31.2% | — | — | — | 0.005 | -0.994 | — |
| `B2` | 2,853,312 | 1.00× | 6.2% | 2.266 | 31.2% | — | — | — | 0.040 | -1.006 | — |
| `B3` | 984,704 | 1.71× | 0.0% | 2.317 | 31.2% | — | — | — | 0.032 | -1.003 | — |
| `B4` | 1,000,704 | 1.74× | 3.1% | 2.272 | 18.8% | -0.001 | +2.909 | No | 0.045 | -0.989 | — |
| `RL-R1` | 984,192 | 3.10× | 3.1% | 2.326 | 31.2% | -0.002 | +0.001 | No | 0.026 | -0.997 | 1.00 |
| `RL-R3` | 984,192 | 3.10× | 1.6% | 2.309 | 25.0% | +0.000 | -0.001 | No | 0.040 | -1.005 | 0.00 |
| `I2-Hybrid` | 1,033,600 | 2.51× | 3.1% | 2.271 | 37.5% | -0.003 | +2.732 | No | 0.032 | -0.998 | 0.00 |

- **`I2-Hybrid`** is the strongest small-model arm (`Hard-BPB = 2.271`, `OOD pass@16 = 37.5%` vs. `31.2%` for `B1`/`B2`/`B3`, $\Delta\text{NLL}_{\text{trunc}} = +2.732$ nats, `Reg3 ECE = 0.0325` / `Reg3 Brier = -0.9982`).
- **`RL-R1` vs. `RL-R3` variance collapse:** Binary exact-match (`RL-R1`) hits `sigma=0` fraction `1.00`, whereas strictly proper Brier (`RL-R3` and `I2-Hybrid`) eliminates zero-variance collapse (`sigma=0` fraction `0.00`).

## HuggingFace Hub Artifacts & Publishing

Trained checkpoints and stratified benchmark splits are published on HuggingFace Hub (keeping large `.pt` and `.parquet` binaries out of Git):

- **Models:** [`queelius/big-probs-phase1-models`](https://huggingface.co/queelius/big-probs-phase1-models) *(26 files, 33.88 MB — all 7 `micro-pilot-v1/` arm checkpoints, configs, metrics, and Model Card)*
- **Datasets:** [`queelius/big-probs-phase1-data`](https://huggingface.co/datasets/queelius/big-probs-phase1-data) *(23 files, 3.52 MB — `micro_pilot` [792 rows] & `stage1_benchmark` [3,200 rows] Parquet/JSONL splits with zero train/OOD leakage)*

Publish micro-pilot artifacts or future H100 Stage 1 runs with `scripts/publish_hf.py`:

```bash
cd phase1_pilot

# Inspect all CLI flags
uv run --with huggingface_hub python scripts/publish_hf.py --help

# Publish CPU micro-pilot checkpoints + Parquet stratified datasets
uv run --with huggingface_hub --with pyarrow python scripts/publish_hf.py \
  --namespace queelius \
  --tag micro-pilot-v1 \
  --run-dir results/micro_runs

# Publish future H100 Stage 1 runs (160M / 410M / 1.0B)
uv run --with huggingface_hub --with pyarrow python scripts/publish_hf.py \
  --namespace queelius \
  --tag h100-stage1 \
  --run-dir runs/h100_stage1 \
  --skip-datasets
```

## Quickstart (Local CPU)

```bash
cd phase1_pilot

# 1. Create environment and install dependencies (CPU PyTorch by default)
uv sync

# 2. Run the full 50-test unit test suite across data, models, RL, and diagnostics
uv run pytest -v

# 3. Run the 7-arm CPU micro-scale end-to-end pilot (~93s on 12 cores)
uv run python scripts/run_micro_pilot.py
```

Outputs are written to `results/micro_pilot_report.md` and `results/micro_pilot_metrics.json`.

## Scaling to H100 Stage 1 (~30–45 H100 GPU-Hours)

On a CUDA / H100 node, install CUDA-enabled PyTorch and optional HuggingFace streaming extras:

```bash
cd phase1_pilot
uv sync --extra h100
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124

# Launch the full Stage 1 Track A suite (160M / 410M / 1.0B baselines + B3 + B4 + RL-R2 + RL-R3 + I2-Hybrid)
NPROC_PER_NODE=4 OUT_ROOT=runs/h100_stage1 bash scripts/launch_h100_stage1.sh
```

Individual runs can also be launched directly via `uv run python -m phase1_pilot.train --help`.
