#!/usr/bin/env bash
# =============================================================================
# Phase 1 Small-Model Latent-CoT RL Pilot — Full Stage 1 H100 Launcher
# Reference: 01-ssl-as-rl-latent-cot.md §5 (Track A Minimal Core Pilot)
#
# Compute Budget (BF16 @ ~40% MFU, ~1.44e18 FLOPs/H100-hour, 2B tokens):
#   - B1 (160M NTP baseline, 2B tokens):               ~1.5 H100 GPU-hrs
#   - B2a (410M ~2.5x NTP ceiling, 2B tokens):         ~3.5 H100 GPU-hrs
#   - B2b (1.0B ~6x NTP ceiling, 2B tokens):           ~9.0 H100 GPU-hrs
#   - B3 (160M + Pause tokens, FLOPs-matched):         ~3.0 H100 GPU-hrs
#   - B4 (160M + Continuous Latent BPTT, L=4):         ~4.5 H100 GPU-hrs
#   - RL-R2 (160M T2_self + R2_log frozen anchor):     ~6.0 H100 GPU-hrs
#   - RL-R3 (160M T2_self + R3_brier frozen anchor):   ~6.0 H100 GPU-hrs
#   - I2-Hybrid (B4 -> HRPO gated RL + R3_brier):      ~6.5 H100 GPU-hrs
#   Total Stage 1 Minimal Core Pilot:                  ~35–40 H100 GPU-hrs
#   (Fits comfortably in ~10 hours on 4x H100 or a single weekend on 1x H100.)
#
# Usage:
#   # Default 4x H100 run:
#   bash scripts/launch_h100_stage1.sh
#
#   # Override GPU count, output dir, or step budget (e.g. dry-run smoke test):
#   NPROC_PER_NODE=1 OUT_ROOT=runs/h100_smoke SUP_STEPS=200 RL_STEPS=50 \
#     bash scripts/launch_h100_stage1.sh
# =============================================================================
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
OUT_ROOT="${OUT_ROOT:-runs/h100_stage1}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

# Tokenizer (BL-11, decided 2026-11-27): SmolLM2's 49,152-token BPE, pinned to a Hub commit.
# With the 8 protocol tokens the vocabulary is 49,160, so the presets build 160.5M / 409.0M /
# 1,112.1M parameters. Every job gets the same tokenizer: RL arms load supervised checkpoints,
# and train.py refuses a checkpoint whose tokenizer fingerprint differs.
# TOKENIZER=pilot restores the 512-ID PilotTokenizer (no revision). Any other Hub tokenizer
# needs TOKENIZER_REVISION set to a commit of that repository.
DEFAULT_TOKENIZER="HuggingFaceTB/SmolLM2-135M"
DEFAULT_TOKENIZER_REVISION="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TOKENIZER="${TOKENIZER:-${DEFAULT_TOKENIZER}}"
if [[ "${TOKENIZER}" == "pilot" ]]; then
  TOKENIZER_ARGS=(--tokenizer pilot)
else
  if [[ "${TOKENIZER}" == "${DEFAULT_TOKENIZER}" ]]; then
    TOKENIZER_REVISION="${TOKENIZER_REVISION:-${DEFAULT_TOKENIZER_REVISION}}"
  elif [[ -z "${TOKENIZER_REVISION:-}" ]]; then
    echo "error: TOKENIZER=${TOKENIZER} needs TOKENIZER_REVISION (a commit of that repository)" >&2
    exit 2
  fi
  TOKENIZER_ARGS=(--tokenizer "${TOKENIZER}" --tokenizer-revision "${TOKENIZER_REVISION}")
fi

# Training budgets (override for scaling to full 2B tokens across multi-node/grad-accum)
BATCH_SIZE="${BATCH_SIZE:-32}"
SUP_STEPS="${SUP_STEPS:-15000}"
RL_STEPS="${RL_STEPS:-2000}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-200000}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-2000}"
LOG_EVERY="${LOG_EVERY:-100}"

# Shared SSL-as-RL hyperparameters (Track 1 §5.4)
GROUP_SIZE="${GROUP_SIZE:-8}"
N_AHEAD="${N_AHEAD:-4}"
THOUGHT_LEN="${THOUGHT_LEN:-8}"
CLIP_LOW="${CLIP_LOW:-0.2}"
CLIP_HIGH="${CLIP_HIGH:-0.28}"
REWARD_CLIP_TAU="${REWARD_CLIP_TAU:-5.0}"
LAMBDA_NTP="${LAMBDA_NTP:-0.2}"
TOP_P_TOKENS="${TOP_P_TOKENS:-0.2}"
TOP_P_THOUGHT_ENTROPY="${TOP_P_THOUGHT_ENTROPY:-0.2}"

mkdir -p "${OUT_ROOT}"

# Select torchrun for multi-GPU DDP or python directly for single-GPU
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCHER=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m phase1_pilot.train)
else
  LAUNCHER=(python -m phase1_pilot.train)
fi

echo "==================================================================="
echo " Phase 1 Stage 1 H100 Pilot Launcher"
echo " GPUs: ${NPROC_PER_NODE} | Output: ${OUT_ROOT} | Seed: ${SEED}"
echo " Sup steps: ${SUP_STEPS} | RL steps: ${RL_STEPS} | Seq len: ${MAX_SEQ_LEN}"
echo " Tokenizer: ${TOKENIZER_ARGS[*]}"
echo "==================================================================="

# -----------------------------------------------------------------------------
# 1. B1 — 160M Standard NTP Baseline (~1.5 H100 GPU-hrs @ 2B tokens)
# -----------------------------------------------------------------------------
echo "[1/8] Training B1: 160M Standard NTP Baseline..."
"${LAUNCHER[@]}" \
  --mode supervised \
  --preset 160m \
  --variant ntp \
  --regime all \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${SUP_STEPS}" \
  --lr 6e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --bf16 \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --n-train-samples "${N_TRAIN_SAMPLES}" \
  --n-eval-samples "${N_EVAL_SAMPLES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${TOKENIZER_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/b1_160m_ntp"

# -----------------------------------------------------------------------------
# 2. B2a — 410M (~2.5x) Weight-Class Ceiling (~3.5 H100 GPU-hrs @ 2B tokens)
# -----------------------------------------------------------------------------
echo "[2/8] Training B2a: 410M (~2.5x) Standard NTP Ceiling..."
"${LAUNCHER[@]}" \
  --mode supervised \
  --preset 410m \
  --variant ntp \
  --regime all \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${SUP_STEPS}" \
  --lr 4e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --bf16 \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --n-train-samples "${N_TRAIN_SAMPLES}" \
  --n-eval-samples "${N_EVAL_SAMPLES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${TOKENIZER_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/b2a_410m_ntp"

# -----------------------------------------------------------------------------
# 3. B2b — 1.0B (~6x) Weight-Class Ceiling (~9.0 H100 GPU-hrs @ 2B tokens)
# -----------------------------------------------------------------------------
echo "[3/8] Training B2b: 1.0B (~6x) Standard NTP Ceiling..."
"${LAUNCHER[@]}" \
  --mode supervised \
  --preset 1b \
  --variant ntp \
  --regime all \
  --batch-size "$(( BATCH_SIZE / 2 ))" \
  --max-steps "${SUP_STEPS}" \
  --lr 3e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --bf16 \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --n-train-samples "${N_TRAIN_SAMPLES}" \
  --n-eval-samples "${N_EVAL_SAMPLES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${TOKENIZER_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/b2b_1b_ntp"

# -----------------------------------------------------------------------------
# 4. B3 — 160M Pause-Token Control (Goyal et al. 2023/2024; ~3.0 H100 GPU-hrs)
# -----------------------------------------------------------------------------
echo "[4/8] Training B3: 160M Pause-Token Control (n_pause_tokens=4)..."
"${LAUNCHER[@]}" \
  --mode supervised \
  --preset 160m \
  --variant pause \
  --n-pause-tokens 4 \
  --regime all \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${SUP_STEPS}" \
  --lr 6e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --bf16 \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --n-train-samples "${N_TRAIN_SAMPLES}" \
  --n-eval-samples "${N_EVAL_SAMPLES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${TOKENIZER_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/b3_160m_pause"

# -----------------------------------------------------------------------------
# 5. B4 — 160M Continuous Latent BPTT (PonderLM-2 / Coconut; ~4.5 H100 GPU-hrs)
# -----------------------------------------------------------------------------
echo "[5/8] Training B4: 160M Continuous Latent BPTT (n_latent_steps=4)..."
"${LAUNCHER[@]}" \
  --mode supervised \
  --preset 160m \
  --variant continuous_latent \
  --n-latent-steps 4 \
  --regime all \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${SUP_STEPS}" \
  --lr 6e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --bf16 \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --n-train-samples "${N_TRAIN_SAMPLES}" \
  --n-eval-samples "${N_EVAL_SAMPLES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${TOKENIZER_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/b4_160m_continuous_latent"

# -----------------------------------------------------------------------------
# 6. RL-R2 — 160M Discrete SSL-as-RL (T2_self + R2_log frozen anchor; ~6 hrs)
# -----------------------------------------------------------------------------
echo "[6/8] Training RL-R2: 160M Teacher-Free T2_self + R2_log (Dr. GRPO + DAPO)..."
"${LAUNCHER[@]}" \
  --mode rl \
  --preset 160m \
  --variant ntp \
  --init-checkpoint "${OUT_ROOT}/b1_160m_ntp/checkpoint.pt" \
  --reward-type R2_log \
  --filter-type T2_self \
  --interface I1_discrete \
  --group-size "${GROUP_SIZE}" \
  --n-ahead "${N_AHEAD}" \
  --thought-len "${THOUGHT_LEN}" \
  --clip-low "${CLIP_LOW}" \
  --clip-high "${CLIP_HIGH}" \
  --reward-clip-tau "${REWARD_CLIP_TAU}" \
  --lambda-ntp "${LAMBDA_NTP}" \
  --top-p-tokens "${TOP_P_TOKENS}" \
  --top-p-thought-entropy "${TOP_P_THOUGHT_ENTROPY}" \
  --ref-ema-decay 1.0 \
  --regime all \
  --batch-size "$(( BATCH_SIZE / 2 ))" \
  --max-steps "${RL_STEPS}" \
  --lr 1e-4 \
  --grad-clip 1.0 \
  --bf16 \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --n-train-samples "${N_TRAIN_SAMPLES}" \
  --n-eval-samples "${N_EVAL_SAMPLES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${TOKENIZER_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/rl_160m_t2self_r2log"

# -----------------------------------------------------------------------------
# 7. RL-R3 — 160M Discrete SSL-as-RL (T2_self + R3_brier frozen anchor; ~6 hrs)
# -----------------------------------------------------------------------------
echo "[7/8] Training RL-R3: 160M Teacher-Free T2_self + R3_brier (Dr. GRPO + DAPO)..."
"${LAUNCHER[@]}" \
  --mode rl \
  --preset 160m \
  --variant ntp \
  --init-checkpoint "${OUT_ROOT}/b1_160m_ntp/checkpoint.pt" \
  --reward-type R3_brier \
  --filter-type T2_self \
  --interface I1_discrete \
  --group-size "${GROUP_SIZE}" \
  --n-ahead "${N_AHEAD}" \
  --thought-len "${THOUGHT_LEN}" \
  --clip-low "${CLIP_LOW}" \
  --clip-high "${CLIP_HIGH}" \
  --lambda-ntp "${LAMBDA_NTP}" \
  --top-p-tokens "${TOP_P_TOKENS}" \
  --top-p-thought-entropy "${TOP_P_THOUGHT_ENTROPY}" \
  --ref-ema-decay 1.0 \
  --regime all \
  --batch-size "$(( BATCH_SIZE / 2 ))" \
  --max-steps "${RL_STEPS}" \
  --lr 1e-4 \
  --grad-clip 1.0 \
  --bf16 \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --n-train-samples "${N_TRAIN_SAMPLES}" \
  --n-eval-samples "${N_EVAL_SAMPLES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${TOKENIZER_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/rl_160m_t2self_r3brier"

# -----------------------------------------------------------------------------
# 8. I2-Hybrid — B4 Continuous Latent -> HRPO Gated RL + R3_brier (~6.5 hrs)
# -----------------------------------------------------------------------------
echo "[8/8] Training I2-Hybrid: 160M B4 -> HRPO Hybrid RL + R3_brier..."
"${LAUNCHER[@]}" \
  --mode rl \
  --preset 160m \
  --variant hybrid_latent \
  --n-latent-steps 4 \
  --init-checkpoint "${OUT_ROOT}/b4_160m_continuous_latent/checkpoint.pt" \
  --reward-type R3_brier \
  --filter-type T2_self \
  --interface I2_hybrid \
  --group-size "${GROUP_SIZE}" \
  --n-ahead "${N_AHEAD}" \
  --thought-len "${THOUGHT_LEN}" \
  --clip-low "${CLIP_LOW}" \
  --clip-high "${CLIP_HIGH}" \
  --lambda-ntp "${LAMBDA_NTP}" \
  --top-p-tokens "${TOP_P_TOKENS}" \
  --top-p-thought-entropy "${TOP_P_THOUGHT_ENTROPY}" \
  --ref-ema-decay 1.0 \
  --regime all \
  --batch-size "$(( BATCH_SIZE / 2 ))" \
  --max-steps "${RL_STEPS}" \
  --lr 1e-4 \
  --grad-clip 1.0 \
  --bf16 \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --n-train-samples "${N_TRAIN_SAMPLES}" \
  --n-eval-samples "${N_EVAL_SAMPLES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${TOKENIZER_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/i2_160m_hybrid_r3brier"

echo "==================================================================="
echo " Stage 1 H100 training complete. Checkpoints saved under ${OUT_ROOT}"
echo " Run full 5-diagnostic evaluation suite against saved checkpoints"
echo " using phase1_pilot.eval.diagnostics (n_samples=64 for pass@64)."
echo "==================================================================="
