#!/usr/bin/env bash
# UDA 恢复/自修复脚本：
# 1) 自动读取默认 resume 状态；
# 2) 自动校验伪标签完整性；
# 3) 若 epoch 伪标签损坏则自动重建；
# 4) 日志单独写入 logs/uda_densepass_resume_<时间戳>.log
#
# 用法:
#   bash scripts/resume_adaptation_uda_densepass.sh
#   RESUME_CKPT=/path/to/resume_sam2_s.pth bash scripts/resume_adaptation_uda_densepass.sh
#   SRC_CKPT=/path/to.pth TGT_CKPT=/path/to.pth bash scripts/resume_adaptation_uda_densepass.sh
#
# 环境变量:
#   SRC_CKPT, TGT_CKPT
#   DATA_ROOT
#   PD_ROOT
#   LOG_DIR
#   RESUME_CKPT
#   NUM_EPOCHS
#   BATCH_SIZE
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
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
RESUME_CKPT="${RESUME_CKPT:-${PD_ROOT}/checkpoints/Cityscapes/sam2_s/resume_sam2_s.pth}"

mkdir -p "${LOG_DIR}" "${PD_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/uda_densepass_resume_${STAMP}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/uda_densepass_resume.pid}"

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG:-0}" != "1" ]]; then
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 LOG_FILE="${LOG_FILE}" \
    bash "$0" "$@" > "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "${PID}" > "${PID_FILE}"
  echo "Started in background."
  echo "PID: ${PID} (saved to ${PID_FILE})"
  echo "Log: ${LOG_FILE}"
  exit 0
fi

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "Error: resume checkpoint not found: ${RESUME_CKPT}" >&2
  echo "Hint: run the normal adaptation script once first, or set RESUME_CKPT manually." >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

{
  echo "=========================================="
  echo "UDA RESUME — tag=densepass"
  echo "Time: $(date -Iseconds)"
  echo "PD_ROOT: ${PD_ROOT}"
  echo "RESUME_CKPT: ${RESUME_CKPT}"
  echo "SRC_CKPT: ${SRC_CKPT}"
  echo "TGT_CKPT: ${TGT_CKPT}"
  echo "DATA_ROOT: ${DATA_ROOT}"
  echo "NUM_EPOCHS: ${NUM_EPOCHS}"
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  echo "=========================================="
  python run_adaptation.py \
    --backbone sam2_s \
    --dataset Cityscapes \
    --data_root "${DATA_ROOT}" \
    --pd_root "${PD_ROOT}" \
    --src_ckpt "${SRC_CKPT}" \
    --tgt_ckpt "${TGT_CKPT}" \
    --resume_ckpt "${RESUME_CKPT}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    "$@"
  echo "Done: $(date -Iseconds)"
} 2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
echo "Resume state: ${RESUME_CKPT}"
echo "Checkpoints under: ${PD_ROOT}/checkpoints/"
