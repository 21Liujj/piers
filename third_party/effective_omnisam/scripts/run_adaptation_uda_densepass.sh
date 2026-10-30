#!/usr/bin/env bash
# UDA 智能续跑脚本：
# - 若存在 resume 状态，则自动恢复训练；
# - 若不存在 resume 但已有完整 epoch1 伪标签，则直接跳过生成进入 train；
# - 若伪标签只生成了一部分，则从已有进度继续补齐，不再整轮重来；
# - 仅当确实没有可复用资产时，才从头生成伪标签。
#
# 用法:
#   bash scripts/run_adaptation_uda_densepass.sh
#   NUM_EPOCHS=3 BATCH_SIZE=2 bash scripts/run_adaptation_uda_densepass.sh
#   SRC_CKPT=/path/to.pth TGT_CKPT=/path/to.pth bash scripts/run_adaptation_uda_densepass.sh
#
# 环境变量: SRC_CKPT, TGT_CKPT, DATA_ROOT, PD_ROOT, LOG_DIR, NUM_EPOCHS, BATCH_SIZE
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
RESUME_CKPT="${PD_ROOT}/checkpoints/Cityscapes/sam2_s/resume_sam2_s.pth"
mkdir -p "${LOG_DIR}" "${PD_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/uda_densepass_${STAMP}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/uda_densepass.pid}"

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

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXTRA_ARGS=()
if [[ -f "${RESUME_CKPT}" ]]; then
  EXTRA_ARGS+=(--resume_ckpt "${RESUME_CKPT}")
fi

{
  echo "=========================================="
  echo "UDA — tag=densepass (smart resume/reuse)"
  echo "Time: $(date -Iseconds)"
  echo "PD_ROOT: ${PD_ROOT}"
  echo "SRC_CKPT: ${SRC_CKPT}"
  echo "TGT_CKPT: ${TGT_CKPT}"
  echo "DATA_ROOT: ${DATA_ROOT}"
  echo "NUM_EPOCHS: ${NUM_EPOCHS}"
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  if [[ -f "${RESUME_CKPT}" ]]; then
    echo "RESUME_CKPT: ${RESUME_CKPT}"
  else
    echo "RESUME_CKPT: <none>"
  fi
  echo "=========================================="
  python run_adaptation.py \
    --backbone sam2_s \
    --dataset Cityscapes \
    --data_root "${DATA_ROOT}" \
    --pd_root "${PD_ROOT}" \
    --src_ckpt "${SRC_CKPT}" \
    --tgt_ckpt "${TGT_CKPT}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
  echo "Done: $(date -Iseconds)"
} 2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
echo "Checkpoints under: ${PD_ROOT}/checkpoints/"
