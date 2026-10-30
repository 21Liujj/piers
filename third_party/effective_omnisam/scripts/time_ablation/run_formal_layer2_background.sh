#!/usr/bin/env bash
set -euo pipefail

# One-shot launcher for Layer-2 formal time ablation (4 methods):
# baseline / mem_only / aps_only / both
# Background execution with full stdout/stderr logging.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

INTERNAL_BG="${INTERNAL_BG:-0}"

# -------- user-facing knobs --------
GPU="${GPU:-0}"
SEED="${SEED:-0}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-9}"

# Formal-layer defaults
FORMAL_IMAGES="${FORMAL_IMAGES:-400}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-500}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
WARMUP_PL="${WARMUP_PL:-${FORMAL_IMAGES}}"
ITERS_PL="${ITERS_PL:-${FORMAL_IMAGES}}"
REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG:---no-regenerate_pl_each_epoch}"

# r9 + multimodal defaults
USE_MULTIMODAL="${USE_MULTIMODAL:-1}"
TEXT_PROTO_PATH="${TEXT_PROTO_PATH:-/root/autodl-tmp/OmniSAM-main/assets/text_prototypes_openclip_text_latest.pt}"
SSL_GATE_MODE="${SSL_GATE_MODE:-hard}"
TAU_T="${TAU_T:-0.10}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.02}"

PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"

INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/formal/main4}"
SUBSET_DIR="${OUT_ROOT}/subsets"
RUN_TAG="${RUN_TAG:-layer2_formal}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_DIR="${OUT_ROOT}/_launcher_logs"
MASTER_LOG="${MASTER_DIR}/${RUN_TAG}_${STAMP}.log"
PID_FILE="${MASTER_DIR}/${RUN_TAG}.pid"

mkdir -p "${MASTER_DIR}" "${SUBSET_DIR}"

if [[ "${INTERNAL_BG}" != "1" ]]; then
  nohup env \
    INTERNAL_BG=1 \
    GPU="${GPU}" \
    SEED="${SEED}" \
    DATASET="${DATASET}" \
    CLASS_SET="${CLASS_SET}" \
    BACKBONE="${BACKBONE}" \
    MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE}" \
    FORMAL_IMAGES="${FORMAL_IMAGES}" \
    TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    USE_MULTIMODAL="${USE_MULTIMODAL}" \
    TEXT_PROTO_PATH="${TEXT_PROTO_PATH}" \
    SSL_GATE_MODE="${SSL_GATE_MODE}" \
    TAU_T="${TAU_T}" \
    WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" \
    PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
    PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    INPUT_LIST="${INPUT_LIST}" \
    OUT_ROOT="${OUT_ROOT}" \
    RUN_TAG="${RUN_TAG}" \
    MASTER_LOG="${MASTER_LOG}" \
    PID_FILE="${PID_FILE}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "已后台启动第二层 formal（4组）"
  echo "PID: ${pid} (保存于 ${PID_FILE})"
  echo "总日志: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Layer-2 Formal Time Ablation =========="
echo "start: $(date -Iseconds)"
echo "ROOT=${ROOT}"
echo "DATASET=${DATASET}, CLASS_SET=${CLASS_SET}, BACKBONE=${BACKBONE}"
echo "SEED=${SEED}, GPU=${GPU}, MEMORY_BANK_SIZE=${MEMORY_BANK_SIZE}"
echo "FORMAL_IMAGES=${FORMAL_IMAGES}, TRAIN_MAX_ITERS=${TRAIN_MAX_ITERS}, NUM_EPOCHS=${NUM_EPOCHS}"
echo "USE_MULTIMODAL=${USE_MULTIMODAL}, r9: gate=${SSL_GATE_MODE}, tau_t=${TAU_T}, w_txt_align=${WEIGHT_TXT_ALIGN}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "MASTER_LOG=${MASTER_LOG}"
echo "================================================="

SUBSET_FILE="${SUBSET_DIR}/${DATASET}_seed${SEED}_formal${FORMAL_IMAGES}.txt"
python "${SCRIPT_DIR}/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${SUBSET_FILE}" \
  --num_images "${FORMAL_IMAGES}" \
  --offset 0

echo "[step] subset ready: ${SUBSET_FILE}"
echo "[step] launching main4 formal ablation..."

env \
  GPU="${GPU}" \
  SEED="${SEED}" \
  DATASET="${DATASET}" \
  CLASS_SET="${CLASS_SET}" \
  BACKBONE="${BACKBONE}" \
  MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE}" \
  NUM_EPOCHS="${NUM_EPOCHS}" \
  TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
  PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
  USE_MULTIMODAL="${USE_MULTIMODAL}" \
  TEXT_PROTO_PATH="${TEXT_PROTO_PATH}" \
  SSL_GATE_MODE="${SSL_GATE_MODE}" \
  TAU_T="${TAU_T}" \
  WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" \
  WARMUP_PL="${WARMUP_PL}" \
  ITERS_PL="${ITERS_PL}" \
  REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG}" \
  PL_SUBSET_FILE="${SUBSET_FILE}" \
  OUT_ROOT="${OUT_ROOT}" \
  bash "${SCRIPT_DIR}/run_time_ablation_main4.sh"

echo "[done] layer-2 formal finished at: $(date -Iseconds)"
echo "summary tip:"
echo "python tools/summarize_time_ablation.py --root ${OUT_ROOT} --out_dir ${OUT_ROOT}/summaries"
