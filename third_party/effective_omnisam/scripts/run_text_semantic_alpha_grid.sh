#!/usr/bin/env bash
set -euo pipefail

# Run semantic text ablation for multiple alpha_init values in sequence.
# Each alpha run executes A1/A4/A5 via run_text_semantic_fixed200.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
INTERNAL_BG="${INTERNAL_BG:-0}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/text_fusion_quick}"
GRID_RUN_NAME="${GRID_RUN_NAME:-alpha_grid_$(date +%Y%m%d_%H%M%S)}"
GRID_DIR="${LOG_ROOT}/${GRID_RUN_NAME}"
MASTER_LOG="${GRID_DIR}/00_alpha_grid_master.log"
PID_FILE="${GRID_DIR}/grid.pid"

# Grid candidates; can be overridden as space-separated values, e.g. "0.10 0.20".
ALPHA_LIST="${ALPHA_LIST:-0.10 0.20}"

# Shared defaults (can be overridden from env)
START_FROM="${START_FROM:-A1_text_branch}"
EPOCHS="${EPOCHS:-1}"
SUBSET_NUM="${SUBSET_NUM:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-4}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-2}"
TEXT_PL_BATCH_SIZE="${TEXT_PL_BATCH_SIZE:-2}"
TAU_T="${TAU_T:-0.10}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.02}"
TEXT_ADAPTER_LR_MULT="${TEXT_ADAPTER_LR_MULT:-10}"
TEXT_FUSION_ALPHA_LR_MULT="${TEXT_FUSION_ALPHA_LR_MULT:-10}"
TEXT_SCALE_LR_MULT="${TEXT_SCALE_LR_MULT:-5}"
TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL:-10}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

mkdir -p "${GRID_DIR}"

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 \
    LOG_ROOT="${LOG_ROOT}" GRID_RUN_NAME="${GRID_RUN_NAME}" ALPHA_LIST="${ALPHA_LIST}" \
    START_FROM="${START_FROM}" EPOCHS="${EPOCHS}" SUBSET_NUM="${SUBSET_NUM}" \
    BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
    TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE}" TEXT_PL_BATCH_SIZE="${TEXT_PL_BATCH_SIZE}" \
    TAU_T="${TAU_T}" WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" \
    TEXT_ADAPTER_LR_MULT="${TEXT_ADAPTER_LR_MULT}" \
    TEXT_FUSION_ALPHA_LR_MULT="${TEXT_FUSION_ALPHA_LR_MULT}" \
    TEXT_SCALE_LR_MULT="${TEXT_SCALE_LR_MULT}" \
    TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Started alpha grid in background."
  echo "PID: ${pid} (saved to ${PID_FILE})"
  echo "Master log: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Text Semantic Alpha Grid =========="
echo "grid_run_name=${GRID_RUN_NAME}"
echo "grid_dir=${GRID_DIR}"
echo "alpha_list=${ALPHA_LIST}"
echo "start_from=${START_FROM}, epochs=${EPOCHS}, subset_num=${SUBSET_NUM}"
echo "batch(base)=${BATCH_SIZE}, pl_batch(base)=${PL_BATCH_SIZE}"
echo "batch(text)=${TEXT_BATCH_SIZE}, pl_batch(text)=${TEXT_PL_BATCH_SIZE}"
echo "tau_t=${TAU_T}, weight_txt_align=${WEIGHT_TXT_ALIGN}"
echo "text_lr_mult: adapter=${TEXT_ADAPTER_LR_MULT}, alpha=${TEXT_FUSION_ALPHA_LR_MULT}, scale=${TEXT_SCALE_LR_MULT}"
echo "text_log_interval=${TEXT_LOG_INTERVAL}"
echo "=============================================="

for alpha in ${ALPHA_LIST}; do
  alpha_tag="$(echo "${alpha}" | tr '.' 'p')"
  run_name="${GRID_RUN_NAME}_alpha${alpha_tag}"
  echo
  echo "[grid] launch alpha=${alpha}, run_name=${run_name}"
  env \
    RUN_IN_BACKGROUND=0 \
    RUN_NAME="${run_name}" \
    START_FROM="${START_FROM}" \
    EPOCHS="${EPOCHS}" \
    SUBSET_NUM="${SUBSET_NUM}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
    TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE}" \
    TEXT_PL_BATCH_SIZE="${TEXT_PL_BATCH_SIZE}" \
    TAU_T="${TAU_T}" \
    WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" \
    TEXT_ALPHA_INIT="${alpha}" \
    TEXT_ADAPTER_LR_MULT="${TEXT_ADAPTER_LR_MULT}" \
    TEXT_FUSION_ALPHA_LR_MULT="${TEXT_FUSION_ALPHA_LR_MULT}" \
    TEXT_SCALE_LR_MULT="${TEXT_SCALE_LR_MULT}" \
    TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL}" \
    LOG_ROOT="${LOG_ROOT}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    bash "${SCRIPT_DIR}/run_text_semantic_fixed200.sh"
done

echo
echo "[grid] All alpha runs finished."
echo "[grid] Log root: ${LOG_ROOT}"
echo "[grid] Grid prefix: ${GRID_RUN_NAME}_alpha*"
