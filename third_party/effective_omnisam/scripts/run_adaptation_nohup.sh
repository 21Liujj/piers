#!/usr/bin/env bash
# 后台运行 run_adaptation.py，stdout/stderr 写入 logs/run_adaptation_<时间戳>.log
#
# 用法（在仓库根目录，由脚本自行 cd）:
#   bash scripts/run_adaptation_nohup.sh --backbone sam2_s --dataset Cityscapes ...
#
# 环境变量:
#   RUN_IN_BACKGROUND=1 (默认) 用 nohup 后台；=0 则前台并 tee
#   LOG_DIR           默认 <repo>/logs
#   WANDB_MODE        默认 offline
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_adaptation_${STAMP}.log"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  echo "=========================================="
  echo "run_adaptation.py (nohup wrapper)"
  echo "Time: $(date -Iseconds)"
  echo "CWD: ${ROOT}"
  echo "CMD: python run_adaptation.py $*"
  echo "LOG_FILE: ${LOG_FILE}"
  echo "RUN_IN_BACKGROUND: ${RUN_IN_BACKGROUND}"
  echo "=========================================="
} | tee "${LOG_FILE}"

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
  nohup python run_adaptation.py "$@" >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "Started in background. PID=${PID}" | tee -a "${LOG_FILE}"
  echo "${PID}" > "${LOG_DIR}/run_adaptation_${STAMP}.pid"
  echo "PID file: ${LOG_DIR}/run_adaptation_${STAMP}.pid"
  echo "Follow log: tail -f ${LOG_FILE}"
else
  python run_adaptation.py "$@" 2>&1 | tee -a "${LOG_FILE}"
  echo "Finished: $(date -Iseconds)" | tee -a "${LOG_FILE}"
fi
