#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned Layer-2 (formal) launcher:
# - Multimodal disabled (align to OmniSAM paper baseline by default)
# - PL update count aligns to 400 images/epoch (warmup_pl/iters_pl)
# - Training runs full epoch by default (TRAIN_MAX_ITERS=0), and multiple epochs (NUM_EPOCHS default 10)
# - Optional: PAPER_SUBSET_MODE=fixed to force the same 400-image subset every epoch for stable timing.

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

NUM_EPOCHS="${NUM_EPOCHS:-10}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-0}"  # 0 = full epoch

PAPER_PL_IMAGES="${PAPER_PL_IMAGES:-400}"
WARMUP_PL="${WARMUP_PL:-${PAPER_PL_IMAGES}}"
ITERS_PL="${ITERS_PL:-${PAPER_PL_IMAGES}}"
REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG:---regenerate_pl_each_epoch}"

# Paper baseline: no multimodal
USE_MULTIMODAL="${USE_MULTIMODAL:-0}"

# Subset mode: random (paper-like) vs fixed (stable timing)
PAPER_SUBSET_MODE="${PAPER_SUBSET_MODE:-random}" # random|fixed
INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/paper_align/formal/main4}"
SUBSET_DIR="${OUT_ROOT}/subsets"
RUN_TAG="${RUN_TAG:-paper_align_formal}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_DIR="${OUT_ROOT}/_launcher_logs"
MASTER_LOG="${MASTER_DIR}/${RUN_TAG}_${STAMP}.log"
PID_FILE="${MASTER_DIR}/${RUN_TAG}.pid"

PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"

mkdir -p "${MASTER_DIR}" "${SUBSET_DIR}"

if [[ "${INTERNAL_BG}" != "1" ]]; then
  nohup env \
    INTERNAL_BG=1 \
    GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
    MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE}" \
    NUM_EPOCHS="${NUM_EPOCHS}" TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
    PAPER_PL_IMAGES="${PAPER_PL_IMAGES}" WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG}" \
    USE_MULTIMODAL="${USE_MULTIMODAL}" PAPER_SUBSET_MODE="${PAPER_SUBSET_MODE}" \
    PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" BATCH_SIZE="${BATCH_SIZE}" \
    INPUT_LIST="${INPUT_LIST}" OUT_ROOT="${OUT_ROOT}" RUN_TAG="${RUN_TAG}" MASTER_LOG="${MASTER_LOG}" PID_FILE="${PID_FILE}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "已后台启动 paper-align formal（4组）"
  echo "PID: ${pid} (保存于 ${PID_FILE})"
  echo "总日志: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Paper-align Formal (Layer-2) =========="
echo "start: $(date -Iseconds)"
echo "DATASET=${DATASET}, CLASS_SET=${CLASS_SET}, BACKBONE=${BACKBONE}"
echo "SEED=${SEED}, GPU=${GPU}, MEMORY_BANK_SIZE=${MEMORY_BANK_SIZE}"
echo "NUM_EPOCHS=${NUM_EPOCHS}, TRAIN_MAX_ITERS=${TRAIN_MAX_ITERS}"
echo "PAPER_PL_IMAGES=${PAPER_PL_IMAGES} (warmup_pl=${WARMUP_PL}, iters_pl=${ITERS_PL})"
echo "PAPER_SUBSET_MODE=${PAPER_SUBSET_MODE}"
echo "USE_MULTIMODAL=${USE_MULTIMODAL} (paper baseline default=0)"
echo "OUT_ROOT=${OUT_ROOT}"
echo "MASTER_LOG=${MASTER_LOG}"
echo "================================================="

PL_SHUFFLE_FLAG="--pl_shuffle"
PL_SUBSET_FILE=""
if [[ "${PAPER_SUBSET_MODE}" == "fixed" ]]; then
  PL_SHUFFLE_FLAG="--no-pl_shuffle"
  PL_SUBSET_FILE="${SUBSET_DIR}/${DATASET}_seed${SEED}_paperfixed${PAPER_PL_IMAGES}.txt"
  python "${SCRIPT_DIR}/build_fixed_subset.py" \
    --input_list "${INPUT_LIST}" \
    --output_list "${PL_SUBSET_FILE}" \
    --num_images "${PAPER_PL_IMAGES}" \
    --offset 0
  echo "[step] fixed subset ready: ${PL_SUBSET_FILE}"
else
  echo "[step] random subset mode: dataloader shuffle enabled"
fi

env \
  GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
  MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE}" \
  NUM_EPOCHS="${NUM_EPOCHS}" TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
  PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" BATCH_SIZE="${BATCH_SIZE}" \
  USE_MULTIMODAL="${USE_MULTIMODAL}" \
  WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
  REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG}" \
  PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG}" \
  PL_SUBSET_FILE="${PL_SUBSET_FILE}" \
  OUT_ROOT="${OUT_ROOT}" \
  bash "${SCRIPT_DIR}/run_time_ablation_main4.sh"

echo "[done] paper-align formal finished at: $(date -Iseconds)"
echo "summary tip:"
echo "python tools/summarize_time_ablation.py --root ${OUT_ROOT} --out_dir ${OUT_ROOT}/summaries"

