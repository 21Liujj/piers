#!/usr/bin/env bash
# scripts/run_pl_threshold_scan.sh
#
# Runs pl_threshold_scan.py on fixed50 subset.
# Does NOT start training.
#
# Usage:
#   SRC_CKPT=/path/to/omnisam_s.pth bash scripts/run_pl_threshold_scan.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/root/miniconda3/envs/omnisam/bin/python}"
SRC_CKPT="${SRC_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"
DATA_ROOT="${DATA_ROOT:-data}"
NUM_IMAGES="${NUM_IMAGES:-50}"
SEED="${SEED:-1}"
OUT_DIR="${OUT_DIR:-logs/pl_threshold_scan}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-}"       # pass --amp to enable mixed precision

mkdir -p "${OUT_DIR}"

CMD=(
    "${PYTHON}" scripts/pl_threshold_scan.py
    --checkpoint "${SRC_CKPT}"
    --dataset Cityscapes
    --class_set 13
    --data_root "${DATA_ROOT}"
    --num_images "${NUM_IMAGES}"
    --seed "${SEED}"
    --out_dir "${OUT_DIR}"
    --device "${DEVICE}"
)
if [ -n "${AMP}" ]; then
    CMD+=(--amp)
fi

echo "[run_pl_threshold_scan] ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${OUT_DIR}/scan_stdout.log"
echo "[run_pl_threshold_scan] done. Results in ${OUT_DIR}/"
