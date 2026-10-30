#!/usr/bin/env bash
set -euo pipefail

# Depth prior ablation (50-image fixed subset) for Route-A:
# three-way comparison:
#   (1) base
#   (2) base + accelerate (your existing speed/stability config)
#   (3) base + accelerate + depth prior (soft loss weighting, no model input change)
#
# Usage:
#   bash scripts/run_ablation_depth_prior_50.sh
#
# Override via env vars:
#   BACKBONE=sam2_s DATASET=Cityscapes EPOCHS=1 BATCH_SIZE=2 bash scripts/run_ablation_depth_prior_50.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# Most training knobs are unified in YAML configs under configs/experiments/depth_prior/.
# Use env vars to override only when necessary.
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
DATA_ROOT="${DATA_ROOT:-data}"
LOG_DIR="${LOG_DIR:-./logs/depth_prior}"
RUN_TAG="${RUN_TAG:-depth_prior_50}"

DEFAULT_CKPT="$(cd "${ROOT}/.." && pwd)/omnisam_s.pth"
SRC_CKPT="${SRC_CKPT:-${DEFAULT_CKPT}}"
TGT_CKPT="${TGT_CKPT:-${SRC_CKPT}}"

# Offline depth prior root (generated earlier)
DEPTH_PRIOR_ROOT="${DEPTH_PRIOR_ROOT:-${ROOT}/aux_outputs/depth_anything_v2/WildPASS}"

# Build a fixed subset list (50 images) from the same list used by WildPASSforPL in this repo.
SUBSET_DIR="${SUBSET_DIR:-${ROOT}/scripts/time_ablation/subsets}"
SUBSET_FILE="${SUBSET_FILE:-${SUBSET_DIR}/wildpass_fixed50_seed0.txt}"
INPUT_LIST="${INPUT_LIST:-${ROOT}/dataset/densepass_list/train.txt}"
SEED="${SEED:-0}"

mkdir -p "${LOG_DIR}" "${SUBSET_DIR}"
python scripts/time_ablation/build_fixed_subset.py \
  --input_list "${INPUT_LIST}" \
  --output_list "${SUBSET_FILE}" \
  --num_images 50 \
  --shuffle \
  --seed "${SEED}"

COMMON_ARGS=(
  --src_ckpt "${SRC_CKPT}"
  --tgt_ckpt "${TGT_CKPT}"
  --pl_subset_file "${SUBSET_FILE}"
  --data_root "${DATA_ROOT}"
  --dataset "${DATASET}"
  --class_set "${CLASS_SET}"
  --backbone "${BACKBONE}"
)

run_case() {
  local tag="$1"; shift
  local pd_root="$1"; shift
  local cfg="$1"; shift
  local log_file="${LOG_DIR}/ablation_${RUN_TAG}_${tag}.log"
  mkdir -p "${LOG_DIR}"
  {
    echo "=========================================="
    echo "DepthPrior Ablation (fixed50)"
    echo "Time: $(date -Iseconds)"
    echo "RUN_TAG: ${RUN_TAG}"
    echo "CASE: ${tag}"
    echo "CONFIG: ${cfg}"
    echo "DATASET: ${DATASET}  CLASS_SET: ${CLASS_SET}  BACKBONE: ${BACKBONE}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "PD_ROOT: ${pd_root}"
    echo "SRC_CKPT: ${SRC_CKPT}"
    echo "TGT_CKPT: ${TGT_CKPT}"
    echo "SUBSET_FILE: ${SUBSET_FILE}"
    echo "DEPTH_PRIOR_ROOT: ${DEPTH_PRIOR_ROOT}"
    echo "CONFIG/EXTRA_ARGS: $*"
    echo "=========================================="
    python run_adaptation.py \
      "${COMMON_ARGS[@]}" \
      --pd_root "${pd_root}" \
      --config "${cfg}" \
      "$@"
    echo "Done: $(date -Iseconds)"
  } 2>&1 | tee "${log_file}"
}

# A1: base + accelerate/stability (plus4train+plus5train)
run_case "A1_base_accel" "./pd_ablation_${RUN_TAG}_A1" \
  "configs/experiments/depth_prior/ablation50_A1_base_accel.yaml"

# A2: base + accelerate/stability + depth prior
run_case "A2_base_accel_depth" "./pd_ablation_${RUN_TAG}_A2" \
  "configs/experiments/depth_prior/ablation50_A2_base_accel_depth.yaml" \
  --depth_prior_root "${DEPTH_PRIOR_ROOT}" \
  --opts "plugins.depth_boundary_prior.depth_prior_root=${DEPTH_PRIOR_ROOT}"

echo "All done. Logs: ${LOG_DIR}/ablation_${RUN_TAG}_*.log"

