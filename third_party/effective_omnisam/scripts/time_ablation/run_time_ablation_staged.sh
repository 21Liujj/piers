#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

DATASET="${DATASET:-Cityscapes}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
BANK="${BANK:-9}"
INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/staged}"

# Stage-1 quick trend
QUICK_IMAGES="${QUICK_IMAGES:-100}"
QUICK_ITERS="${QUICK_ITERS:-200}"
# Stage-2 formal-ish
FORMAL_IMAGES="${FORMAL_IMAGES:-400}"
FORMAL_ITERS="${FORMAL_ITERS:-500}"

mkdir -p "${OUT_ROOT}/subsets"

quick_subset="${OUT_ROOT}/subsets/${DATASET}_seed${SEED}_quick_${QUICK_IMAGES}.txt"
formal_subset="${OUT_ROOT}/subsets/${DATASET}_seed${SEED}_formal_${FORMAL_IMAGES}.txt"

python "${SCRIPT_DIR}/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${quick_subset}" \
  --num_images "${QUICK_IMAGES}" \
  --offset 0

python "${SCRIPT_DIR}/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${formal_subset}" \
  --num_images "${FORMAL_IMAGES}" \
  --offset 0

echo "[Stage-1 Quick] subset=${quick_subset}, train_max_iters=${QUICK_ITERS}"
GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" MEMORY_BANK_SIZE="${BANK}" \
OUT_ROOT="${OUT_ROOT}/stage1_quick" \
PL_SUBSET_FILE="${quick_subset}" \
NUM_EPOCHS=1 TRAIN_MAX_ITERS="${QUICK_ITERS}" \
bash "${SCRIPT_DIR}/run_time_ablation_main4.sh"

echo "[Stage-2 Formal] subset=${formal_subset}, train_max_iters=${FORMAL_ITERS}"
GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" MEMORY_BANK_SIZE="${BANK}" \
OUT_ROOT="${OUT_ROOT}/stage2_formal" \
PL_SUBSET_FILE="${formal_subset}" \
NUM_EPOCHS=1 TRAIN_MAX_ITERS="${FORMAL_ITERS}" \
bash "${SCRIPT_DIR}/run_time_ablation_main4.sh"

echo "Staged time ablation launch complete: ${OUT_ROOT}"
