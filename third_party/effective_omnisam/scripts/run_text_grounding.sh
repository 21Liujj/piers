#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# Unified controller for V3 grounding experiments.
# Modes:
#   c1_fixed20   -> C1 real only, fixed20
#   all_fixed20  -> C1/C4/C5, fixed20
#   all_fixed50  -> C1/C4/C5, fixed50
#   watch        -> tail latest grounding master log

MODE="${MODE:-all_fixed20}"

if [[ "${MODE}" == "watch" ]]; then
  LOG_BASE="logs/text_grounding_diagnose"
  if [[ ! -d "${LOG_BASE}" ]]; then
    echo "No log directory: ${LOG_BASE}"
    exit 1
  fi
  LATEST_LOG="$(ls -1t "${LOG_BASE}"/*/00_master.log 2>/dev/null | head -n 1 || true)"
  if [[ -z "${LATEST_LOG}" ]]; then
    echo "No 00_master.log found under ${LOG_BASE}"
    exit 1
  fi
  echo "${LATEST_LOG}"
  tail -f "${LATEST_LOG}"
  exit 0
fi

# defaults
RUN_NAME="${RUN_NAME:-grounding_${MODE}_$(date +%m%d_%H%M%S)}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
GROUNDING_GAMMA_INIT="${GROUNDING_GAMMA_INIT:-0.05}"
GROUNDING_NUM_LAYERS="${GROUNDING_NUM_LAYERS:-2}"
GROUNDING_NUM_HEADS="${GROUNDING_NUM_HEADS:-4}"
GROUNDING_LOSS_WEIGHT="${GROUNDING_LOSS_WEIGHT:-0.05}"
TEXT_ADAPTER_PROJECTOR_TYPE="${TEXT_ADAPTER_PROJECTOR_TYPE:-linear}"
TEXT_ADAPTER_FREEZE="${TEXT_ADAPTER_FREEZE:-0}"
SAVE_DEBUG_VIS="${SAVE_DEBUG_VIS:-1}"
MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY:-1}"

case "${MODE}" in
  c1_fixed20)
    CASE_SET="${CASE_SET:-c1}"
    WARMUP_PL="${WARMUP_PL:-20}"
    ITERS_PL="${ITERS_PL:-20}"
    RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
    ;;
  all_fixed20)
    CASE_SET="${CASE_SET:-all}"
    WARMUP_PL="${WARMUP_PL:-20}"
    ITERS_PL="${ITERS_PL:-20}"
    ;;
  all_fixed50)
    CASE_SET="${CASE_SET:-all}"
    WARMUP_PL="${WARMUP_PL:-50}"
    ITERS_PL="${ITERS_PL:-50}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. expected c1_fixed20|all_fixed20|all_fixed50|watch"
    exit 2
    ;;
esac

echo "========== Unified Text Grounding Runner =========="
echo "mode=${MODE}"
echo "run_name=${RUN_NAME}"
echo "run_in_background=${RUN_IN_BACKGROUND}"
echo "case_set=${CASE_SET}"
echo "epochs=${EPOCHS}, warmup_pl=${WARMUP_PL}, iters_pl=${ITERS_PL}"
echo "batch_size=${BATCH_SIZE}, pl_batch_size=${PL_BATCH_SIZE}"
echo "gamma_init=${GROUNDING_GAMMA_INIT}, layers=${GROUNDING_NUM_LAYERS}, heads=${GROUNDING_NUM_HEADS}"
echo "loss_weight=${GROUNDING_LOSS_WEIGHT}, projector=${TEXT_ADAPTER_PROJECTOR_TYPE}, freeze=${TEXT_ADAPTER_FREEZE}"
echo "save_debug_vis=${SAVE_DEBUG_VIS}, mem_compress_train_only=${MEM_COMPRESS_TRAIN_ONLY}"
echo "==================================================="

RUN_NAME="${RUN_NAME}" \
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND}" \
CASE_SET="${CASE_SET}" \
EPOCHS="${EPOCHS}" \
WARMUP_PL="${WARMUP_PL}" \
ITERS_PL="${ITERS_PL}" \
BATCH_SIZE="${BATCH_SIZE}" \
PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
GROUNDING_GAMMA_INIT="${GROUNDING_GAMMA_INIT}" \
GROUNDING_NUM_LAYERS="${GROUNDING_NUM_LAYERS}" \
GROUNDING_NUM_HEADS="${GROUNDING_NUM_HEADS}" \
GROUNDING_LOSS_WEIGHT="${GROUNDING_LOSS_WEIGHT}" \
TEXT_ADAPTER_PROJECTOR_TYPE="${TEXT_ADAPTER_PROJECTOR_TYPE}" \
TEXT_ADAPTER_FREEZE="${TEXT_ADAPTER_FREEZE}" \
SAVE_DEBUG_VIS="${SAVE_DEBUG_VIS}" \
MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY}" \
bash scripts/run_text_grounding_diagnose.sh
