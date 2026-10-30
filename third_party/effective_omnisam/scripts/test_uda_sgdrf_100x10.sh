#!/usr/bin/env bash
# UDA + SG-DRF: 100样本 × 10 epochs 快速验证
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
cd "${ROOT}"

DEFAULT_CKPT="/root/autodl-tmp/omnisam_s.pth"
DATA_ROOT="data"
PD_ROOT="pd_uda_sgdrf_100x10"
LOG_DIR="logs"
NUM_EPOCHS=10
BATCH_SIZE=2
ITERS_PL=100

mkdir -p "${LOG_DIR}" "${PD_ROOT}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/uda_sgdrf_100x10_${STAMP}.log"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "========================================"
echo "UDA + SG-DRF: 100样本 × 10 epochs"
echo "Time: $(date)"
echo "========================================"

/root/miniconda3/bin/python run_adaptation.py \
  --backbone sam2_s \
  --dataset Cityscapes \
  --data_root "${DATA_ROOT}" \
  --pd_root "${PD_ROOT}" \
  --src_ckpt "${DEFAULT_CKPT}" \
  --tgt_ckpt "${DEFAULT_CKPT}" \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --warmup_pl 100 \
  --iters_pl ${ITERS_PL} \
  --enable_sgdrf \
  --enable_dynamic_fsmc \
  --enable_dynamic_scm \
  --sgdrf_cold_start_epochs 2 \
  --sgdrf_density_weight 0.1 \
  --sgdrf_density_lr 5e-5 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Done. Log: ${LOG_FILE}"
