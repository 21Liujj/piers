#!/usr/bin/env bash
# Auto iterative pseudo-label refinement + training (shell wrapper).
# - Runs tools/auto_iterative_pl_train.py with sensible defaults
# - Supports background running and log file output
# - Policy: first PL round uses TEACHER_CKPT_INIT (default: best_sam2_s_iou60.36.pth), then
#   train CHUNK_EPOCHS epochs on that PL before regen / next round (default CHUNK_EPOCHS=2).
#
# Usage:
#   bash scripts/auto_iterative_pl_train_overnight.sh
#   RUN_IN_BACKGROUND=1 TOTAL_EPOCHS=15 CHUNK_EPOCHS=2 bash scripts/auto_iterative_pl_train_overnight.sh
#
# Optional env vars:
#   DATA_ROOT, LOG_DIR
#   PD_TRAIN
#   TEACHER_CKPT_INIT
#   TOTAL_EPOCHS, CHUNK_EPOCHS
#   WARMUP_PL, ITERS_PL
#   PL_BATCH_SIZE, PL_NUM_WORKERS
#   BATCH_SIZE, CONF_THR
#   AMP (1/0), TF32 (1/0)
#   PLATEAU_EPS, LOOKBACK
#   MAX_ROUNDS
#   WANDB_MODE (default offline)
#   RUN_IN_BACKGROUND (default 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-data}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"

PD_TRAIN="${PD_TRAIN:-${ROOT}/pd_auto_iter}"
# First-round pseudo-labels: teacher = 60.36 mIoU checkpoint (override with TEACHER_CKPT_INIT).
TEACHER_CKPT_INIT="${TEACHER_CKPT_INIT:-${ROOT}/pd_uda_densepass/checkpoints/Cityscapes/sam2_s/best_sam2_s_iou60.36.pth}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
# Train this many epochs per fixed PL before the driver may regen / advance (1 PL → 2 epochs).
CHUNK_EPOCHS="${CHUNK_EPOCHS:-2}"
WARMUP_PL="${WARMUP_PL:-2000}"
ITERS_PL="${ITERS_PL:-0}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-8}"

BATCH_SIZE="${BATCH_SIZE:-4}"
CONF_THR="${CONF_THR:-0.85}"
AMP="${AMP:-1}"
TF32="${TF32:-1}"

PLATEAU_EPS="${PLATEAU_EPS:-0.10}"
LOOKBACK="${LOOKBACK:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${TEACHER_CKPT_INIT}" ]]; then
  echo "Error: TEACHER_CKPT_INIT not found: ${TEACHER_CKPT_INIT}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/auto_iterative_pl_train_${STAMP}.log"

CMD=(
  python tools/auto_iterative_pl_train.py
  --pd_train "${PD_TRAIN}"
  --teacher_ckpt_init "${TEACHER_CKPT_INIT}"
  --data_root "${DATA_ROOT}"
  --dataset Cityscapes
  --class_set 13
  --backbone sam2_s
  --total_epochs "${TOTAL_EPOCHS}"
  --chunk_epochs "${CHUNK_EPOCHS}"
  --warmup_pl "${WARMUP_PL}"
  --iters_pl "${ITERS_PL}"
  --pl_batch_size "${PL_BATCH_SIZE}"
  --pl_num_workers "${PL_NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --conf_thr "${CONF_THR}"
  --plateau_eps "${PLATEAU_EPS}"
  --lookback "${LOOKBACK}"
  --max_rounds "${MAX_ROUNDS}"
  --log_dir "${LOG_DIR}"
)

if [[ "${AMP}" == "1" ]]; then
  CMD+=(--amp)
else
  CMD+=(--no-amp)
fi

if [[ "${TF32}" == "1" ]]; then
  CMD+=(--tf32)
else
  CMD+=(--no-tf32)
fi

{
  echo "=========================================="
  echo "AUTO ITERATIVE PL TRAIN (shell wrapper)"
  echo "Time: $(date -Iseconds)"
  echo "PD_TRAIN: ${PD_TRAIN}"
  echo "TEACHER_CKPT_INIT: ${TEACHER_CKPT_INIT}"
  echo "TOTAL_EPOCHS: ${TOTAL_EPOCHS}"
  echo "CHUNK_EPOCHS: ${CHUNK_EPOCHS}"
  echo "WARMUP_PL: ${WARMUP_PL}"
  echo "ITERS_PL: ${ITERS_PL}"
  echo "PL_BATCH_SIZE: ${PL_BATCH_SIZE}"
  echo "PL_NUM_WORKERS: ${PL_NUM_WORKERS}"
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  echo "CONF_THR: ${CONF_THR}"
  echo "AMP: ${AMP}"
  echo "TF32: ${TF32}"
  echo "PLATEAU_EPS: ${PLATEAU_EPS}"
  echo "LOOKBACK: ${LOOKBACK}"
  echo "MAX_ROUNDS: ${MAX_ROUNDS}"
  echo "RUN_IN_BACKGROUND: ${RUN_IN_BACKGROUND}"
  echo "LOG_FILE: ${LOG_FILE}"
  echo "=========================================="
} | tee "${LOG_FILE}"

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
  nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "Started in background. PID=${PID}" | tee -a "${LOG_FILE}"
  echo "Follow log: tail -f ${LOG_FILE}"
else
  "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  echo "Done: $(date -Iseconds)" | tee -a "${LOG_FILE}"
fi

