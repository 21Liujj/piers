#!/usr/bin/env bash
# 复用 epoch1 全量伪标签进行优化训练（默认从 epoch2 续训到 epoch6），支持一键后台运行。
#
# 用法：
#   bash scripts/run_adaptation_reuse_epoch1_optimized.sh
#   RUN_IN_BACKGROUND=1 bash scripts/run_adaptation_reuse_epoch1_optimized.sh
#   BATCH_SIZE=4 CONF_THR=0.90 NUM_EPOCHS=6 bash scripts/run_adaptation_reuse_epoch1_optimized.sh
#
# 常用环境变量：
#   SRC_CKPT, TGT_CKPT
#   DATA_ROOT, PD_ROOT, LOG_DIR
#   REUSE_PL_DIR
#   RESUME_CKPT (默认使用 pd_uda_densepass 下 resume_sam2_s.pth)
#   NUM_EPOCHS (默认 6；若 resume 在 epoch1 后，则会跑 epoch2..epoch6 共 5 轮)
#   BATCH_SIZE (默认 4；更稳，若显存充足可手动设为 6)
#   CONF_THR (默认 0.85)
#   AMP (1/0, 默认 1)
#   TF32 (1/0, 默认 1)
#   RUN_IN_BACKGROUND (1/0, 默认 1)
#   WANDB_MODE (默认 offline)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DEFAULT_CKPT="$(cd "${ROOT}/.." && pwd)/omnisam_s.pth"
SRC_CKPT="${SRC_CKPT:-${DEFAULT_CKPT}}"
TGT_CKPT="${TGT_CKPT:-${SRC_CKPT}}"
DATA_ROOT="${DATA_ROOT:-data}"
PD_ROOT="${PD_ROOT:-${ROOT}/pd_uda_densepass}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
REUSE_PL_DIR="${REUSE_PL_DIR:-${PD_ROOT}/DensePASS/sam2_s/Cityscapes/epoch1}"
RESUME_CKPT="${RESUME_CKPT:-${PD_ROOT}/checkpoints/Cityscapes/sam2_s/resume_sam2_s.pth}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONF_THR="${CONF_THR:-0.85}"
AMP="${AMP:-1}"
TF32="${TF32:-1}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

mkdir -p "${LOG_DIR}" "${PD_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/uda_densepass_reuse_epoch1_opt_${STAMP}.log"

if [[ ! -d "${REUSE_PL_DIR}" ]]; then
  echo "Error: REUSE_PL_DIR not found: ${REUSE_PL_DIR}" >&2
  exit 1
fi
if [[ ! -f "$(dirname "${REUSE_PL_DIR}")/used_path.txt" ]]; then
  echo "Error: used_path.txt not found next to REUSE_PL_DIR: $(dirname "${REUSE_PL_DIR}")" >&2
  exit 1
fi
if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "Error: RESUME_CKPT not found: ${RESUME_CKPT}" >&2
  echo "Hint: if you want to continue after epoch1, please set RESUME_CKPT explicitly." >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CMD=(
  python run_adaptation.py
  --backbone sam2_s
  --dataset Cityscapes
  --data_root "${DATA_ROOT}"
  --pd_root "${PD_ROOT}"
  --src_ckpt "${SRC_CKPT}"
  --tgt_ckpt "${TGT_CKPT}"
  --resume_ckpt "${RESUME_CKPT}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --reuse_pl_dir "${REUSE_PL_DIR}"
  --conf_thr "${CONF_THR}"
)

if [[ "${AMP}" == "1" ]]; then
  CMD+=(--amp)
fi
if [[ "${TF32}" == "1" ]]; then
  CMD+=(--tf32)
else
  CMD+=(--no-tf32)
fi

{
  echo "=========================================="
  echo "UDA REUSE EPOCH1 OPTIMIZED"
  echo "Time: $(date -Iseconds)"
  echo "PD_ROOT: ${PD_ROOT}"
  echo "REUSE_PL_DIR: ${REUSE_PL_DIR}"
  echo "SRC_CKPT: ${SRC_CKPT}"
  echo "TGT_CKPT: ${TGT_CKPT}"
  echo "DATA_ROOT: ${DATA_ROOT}"
  echo "RESUME_CKPT: ${RESUME_CKPT}"
  echo "NUM_EPOCHS: ${NUM_EPOCHS}"
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  echo "CONF_THR: ${CONF_THR}"
  echo "AMP: ${AMP}"
  echo "TF32: ${TF32}"
  echo "RUN_IN_BACKGROUND: ${RUN_IN_BACKGROUND}"
  echo "LOG_FILE: ${LOG_FILE}"
  echo "=========================================="
} | tee "${LOG_FILE}"

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
  nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "Started in background. PID=${PID}" | tee -a "${LOG_FILE}"
  echo "Follow log: tail -f ${LOG_FILE}"
else
  "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  echo "Done: $(date -Iseconds)" | tee -a "${LOG_FILE}"
fi

