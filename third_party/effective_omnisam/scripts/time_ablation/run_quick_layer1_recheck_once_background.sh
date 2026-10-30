#!/usr/bin/env bash
set -euo pipefail

# Minimal recheck run:
# - same fixed subset (100 images by default)
# - forced rerun (SKIP_COMPLETED=0)
# - shuffled method order to reduce first-run cold-start bias

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

INTERNAL_BG="${INTERNAL_BG:-0}"

GPU="${GPU:-0}"
SEED="${SEED:-0}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-9}"

QUICK_IMAGES="${QUICK_IMAGES:-100}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-200}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
WARMUP_PL="${WARMUP_PL:-${QUICK_IMAGES}}"
ITERS_PL="${ITERS_PL:-${QUICK_IMAGES}}"
REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG:---no-regenerate_pl_each_epoch}"

USE_MULTIMODAL="${USE_MULTIMODAL:-1}"
TEXT_PROTO_PATH="${TEXT_PROTO_PATH:-/root/autodl-tmp/OmniSAM-main/assets/text_prototypes_openclip_text_latest.pt}"
SSL_GATE_MODE="${SSL_GATE_MODE:-hard}"
TAU_T="${TAU_T:-0.10}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.02}"

PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/quick_recheck_once/main4}"
SUBSET_DIR="${OUT_ROOT}/subsets"
RUN_TAG="${RUN_TAG:-layer1_quick_recheck_once}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_DIR="${OUT_ROOT}/_launcher_logs"
MASTER_LOG="${MASTER_DIR}/${RUN_TAG}_${STAMP}.log"
PID_FILE="${MASTER_DIR}/${RUN_TAG}.pid"

mkdir -p "${MASTER_DIR}" "${SUBSET_DIR}"

if [[ "${INTERNAL_BG}" != "1" ]]; then
  nohup env \
    INTERNAL_BG=1 \
    GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
    MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE}" QUICK_IMAGES="${QUICK_IMAGES}" TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" NUM_EPOCHS="${NUM_EPOCHS}" \
    WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG}" \
    USE_MULTIMODAL="${USE_MULTIMODAL}" TEXT_PROTO_PATH="${TEXT_PROTO_PATH}" SSL_GATE_MODE="${SSL_GATE_MODE}" TAU_T="${TAU_T}" WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" \
    PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" BATCH_SIZE="${BATCH_SIZE}" INPUT_LIST="${INPUT_LIST}" \
    OUT_ROOT="${OUT_ROOT}" RUN_TAG="${RUN_TAG}" MASTER_LOG="${MASTER_LOG}" PID_FILE="${PID_FILE}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "已后台启动 quick 复核（1次，顺序打乱）"
  echo "PID: ${pid} (保存于 ${PID_FILE})"
  echo "总日志: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Quick Recheck Once =========="
echo "start: $(date -Iseconds)"
echo "OUT_ROOT=${OUT_ROOT}"
echo "METHOD_ORDER=mem_only both baseline aps_only"
echo "SKIP_COMPLETED=0"
echo "WARMUP_PL=${WARMUP_PL}, ITERS_PL=${ITERS_PL}"
echo "======================================="

SUBSET_FILE="${SUBSET_DIR}/${DATASET}_seed${SEED}_quick${QUICK_IMAGES}.txt"
python "${SCRIPT_DIR}/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${SUBSET_FILE}" \
  --num_images "${QUICK_IMAGES}" \
  --offset 0

env \
  GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
  MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE}" NUM_EPOCHS="${NUM_EPOCHS}" TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
  BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
  WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG}" \
  USE_MULTIMODAL="${USE_MULTIMODAL}" TEXT_PROTO_PATH="${TEXT_PROTO_PATH}" SSL_GATE_MODE="${SSL_GATE_MODE}" TAU_T="${TAU_T}" WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" \
  PL_SUBSET_FILE="${SUBSET_FILE}" \
  SKIP_COMPLETED=0 \
  METHOD_ORDER="mem_only both baseline aps_only" \
  OUT_ROOT="${OUT_ROOT}" \
  bash "${SCRIPT_DIR}/run_time_ablation_main4.sh"

echo "[done] quick recheck once finished at: $(date -Iseconds)"
echo "summary tip:"
echo "python tools/summarize_time_ablation.py --root ${OUT_ROOT} --out_dir ${OUT_ROOT}/summaries"

