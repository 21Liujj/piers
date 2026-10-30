#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
INTERNAL_BG="${INTERNAL_BG:-0}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/text_crossattn_diag_matrix50}"
RUN_NAME="${RUN_NAME:-crossattn_diag_matrix50_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
MASTER_LOG="${RUN_DIR}/00_master.log"
PID_FILE="${RUN_DIR}/run.pid"

EPOCHS="${EPOCHS:-1}"
WARMUP_PL="${WARMUP_PL:-50}"
ITERS_PL="${ITERS_PL:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-4}"
TEXT_CROSS_BETA_INIT="${TEXT_CROSS_BETA_INIT:-0.05}"
TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL:-10}"
MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY:-1}"
BACKBONE="${BACKBONE:-sam2_s}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
CASE_SET="${CASE_SET:-all}"
START_FROM="${START_FROM:-linear}"

mkdir -p "${RUN_DIR}"

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 \
    LOG_ROOT="${LOG_ROOT}" RUN_NAME="${RUN_NAME}" START_FROM="${START_FROM}" \
    EPOCHS="${EPOCHS}" WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
    TEXT_CROSS_BETA_INIT="${TEXT_CROSS_BETA_INIT}" TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL}" \
    MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY}" \
    BACKBONE="${BACKBONE}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" DATA_ROOT="${DATA_ROOT}" \
    SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" CASE_SET="${CASE_SET}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Started in background."
  echo "PID: ${pid} (saved to ${PID_FILE})"
  echo "Master log: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Text CrossAttn DiagMatrix50 =========="
echo "run_dir=${RUN_DIR}"
echo "epochs=${EPOCHS}, warmup_pl=${WARMUP_PL}, iters_pl=${ITERS_PL}"
echo "batch_size=${BATCH_SIZE}, pl_batch_size=${PL_BATCH_SIZE}"
echo "text_cross_beta_init=${TEXT_CROSS_BETA_INIT}, text_log_interval=${TEXT_LOG_INTERVAL}"
echo "mem_compress_train_only=${MEM_COMPRESS_TRAIN_ONLY}"
echo "case_set=${CASE_SET}, start_from=${START_FROM}"
echo "================================================="

run_variant() {
  local variant="$1"
  local projector_mode="$2"
  local freeze_projector="$3"
  local attn_temp="$4"

  echo "[variant] ${variant}"
  RUN_IN_BACKGROUND=0 \
  LOG_ROOT="${RUN_DIR}" \
  RUN_NAME="${variant}" \
  EPOCHS="${EPOCHS}" \
  WARMUP_PL="${WARMUP_PL}" \
  ITERS_PL="${ITERS_PL}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
  TEXT_CROSS_BETA_INIT="${TEXT_CROSS_BETA_INIT}" \
  TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL}" \
  MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY}" \
  BACKBONE="${BACKBONE}" \
  DATASET="${DATASET}" \
  CLASS_SET="${CLASS_SET}" \
  DATA_ROOT="${DATA_ROOT}" \
  SRC_CKPT="${SRC_CKPT}" \
  TGT_CKPT="${TGT_CKPT}" \
  CASE_SET="${CASE_SET}" \
  PROJECTOR_MODE="${projector_mode}" \
  FREEZE_PROJECTOR="${freeze_projector}" \
  ATTN_TEMP="${attn_temp}" \
  bash scripts/run_text_crossattn_diagnose50.sh
}

variants=(
  "linear|linear|0|1.0"
  "freezeproj|mlp|1|1.0"
  "temp05|mlp|0|0.5"
)

start_found=0
for item in "${variants[@]}"; do
  IFS='|' read -r variant projector_mode freeze_projector attn_temp <<< "${item}"
  if [[ "${start_found}" == "0" ]]; then
    if [[ "${variant}" == "${START_FROM}" ]]; then
      start_found=1
    else
      continue
    fi
  fi
  run_variant "${variant}" "${projector_mode}" "${freeze_projector}" "${attn_temp}"
done

if [[ "${start_found}" == "0" ]]; then
  echo "[error] START_FROM=${START_FROM} not found in variant list."
  exit 1
fi

echo "All diag-matrix50 runs finished."
echo "Logs: ${RUN_DIR}"
