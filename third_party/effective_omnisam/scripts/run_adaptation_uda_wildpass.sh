#!/usr/bin/env bash
# UDA（run_adaptation.py）单独实验：伪标签与 checkpoint 写入独立目录，日志单独文件。
# 说明：OmniSAM 的 adaptation 在代码里会同时用到 WildPASS（伪标签）与 DensePASS（验证），
# 本脚本仅将「输出根目录」与「日志」与 densepass 版本区分开，便于并行/对照实验。
#
# 用法:
#   bash scripts/run_adaptation_uda_wildpass.sh
#   SRC_CKPT=/path/to.pth PD_ROOT=/path/to/pd_root bash scripts/run_adaptation_uda_wildpass.sh
#
# 环境变量:
#   SRC_CKPT, TGT_CKPT  默认指向 omnisam_s.pth
#   DATA_ROOT           默认 <仓库>/data
#   PD_ROOT             伪标签与 ckpt 根目录，默认 <仓库>/pd_uda_wildpass
#   LOG_DIR             默认 <仓库>/logs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DEFAULT_CKPT="$(cd "${ROOT}/.." && pwd)/omnisam_s.pth"
SRC_CKPT="${SRC_CKPT:-${DEFAULT_CKPT}}"
TGT_CKPT="${TGT_CKPT:-${SRC_CKPT}}"
DATA_ROOT="${DATA_ROOT:-data}"
PD_ROOT="${PD_ROOT:-${ROOT}/pd_uda_wildpass}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
mkdir -p "${LOG_DIR}" "${PD_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/uda_wildpass_${STAMP}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/uda_wildpass.pid}"

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

{
  echo "=========================================="
  echo "UDA — tag=wildpass (pd_root isolated)"
  echo "Time: $(date -Iseconds)"
  echo "PD_ROOT: ${PD_ROOT}"
  echo "SRC_CKPT: ${SRC_CKPT}"
  echo "TGT_CKPT: ${TGT_CKPT}"
  echo "DATA_ROOT: ${DATA_ROOT}"
  echo "=========================================="
  python run_adaptation.py \
    --backbone sam2_s \
    --dataset Cityscapes \
    --data_root "${DATA_ROOT}" \
    --pd_root "${PD_ROOT}" \
    --src_ckpt "${SRC_CKPT}" \
    --tgt_ckpt "${TGT_CKPT}" \
    --num_epochs 1 \
    --batch_size 2 \
    "$@"
  echo "Done: $(date -Iseconds)"
} 2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
echo "Checkpoints under: ${PD_ROOT}/checkpoints/"
