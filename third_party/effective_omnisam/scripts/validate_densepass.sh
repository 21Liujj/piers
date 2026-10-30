#!/usr/bin/env bash
# 仅在 DensePASS 验证集上评测，单独写日志。
#
# 用法:
#   bash scripts/validate_densepass.sh
#   CKPT=/path/to.pth STAGE=baseline bash scripts/validate_densepass.sh
#   CKPT=/path/to_19cls.pth CLASS_SET=19 STAGE=baseline19 bash scripts/validate_densepass.sh
#
# 环境变量: CKPT, STAGE, DATA_ROOT, LOG_DIR, CLASS_SET（同仓库内其他 validate 脚本）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DEFAULT_CKPT="$(cd "${ROOT}/.." && pwd)/omnisam_s.pth"
CKPT="${CKPT:-${DEFAULT_CKPT}}"
STAGE="${STAGE:-benchmark}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
CLASS_SET="${CLASS_SET:-13}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/validate_densepass_${STAGE}_${STAMP}.log"

if [[ ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

{
  echo "=========================================="
  echo "OmniSAM validation — DensePASS — ${STAGE}"
  echo "Time: $(date -Iseconds)"
  echo "Repo: ${ROOT}"
  echo "CKPT: ${CKPT}"
  echo "DATA_ROOT: ${DATA_ROOT}"
  echo "=========================================="
  python validation.py \
    --dataset DensePASS \
    --class_set "${CLASS_SET}" \
    --data_root "${DATA_ROOT}" \
    --backbone sam2_s \
    --use_mem_bank \
    --batch_size 1 \
    --ckpt "${CKPT}"
  echo "Done: $(date -Iseconds)"
} 2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
