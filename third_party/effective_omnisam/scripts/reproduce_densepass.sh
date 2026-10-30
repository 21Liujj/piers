#!/usr/bin/env bash
# DensePASS 验证：在项目根目录运行 validation.py，日志写入 logs/validate_densepass_<时间戳>.log
#
# 用法:
#   bash scripts/reproduce_densepass.sh
#   CKPT=/path/to/weights.pth bash scripts/reproduce_densepass.sh
#
# 默认 CKPT 为与 OmniSAM-main 同目录下的 omnisam_s.pth（即 <仓库父目录>/omnisam_s.pth）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DEFAULT_CKPT="$(cd "${ROOT}/.." && pwd)/omnisam_s.pth"
CKPT="${CKPT:-${DEFAULT_CKPT}}"

LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/validate_densepass_${STAMP}.log"

if [[ ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found: ${CKPT}" >&2
  echo "Set CKPT to your .pth file, e.g. CKPT=/root/autodl-tmp/omnisam_s.pth bash scripts/reproduce_densepass.sh" >&2
  exit 1
fi

{
  echo "Project root: ${ROOT}"
  echo "Checkpoint:   ${CKPT}"
  echo "Started:      $(date -Iseconds)"
  echo "----------"
  python validation.py \
    --backbone sam2_s \
    --dataset DensePASS \
    --use_mem_bank \
    --batch_size 1 \
    --ckpt "${CKPT}"
  echo "Finished:     $(date -Iseconds)"
} 2>&1 | tee "${LOG_FILE}"

echo "Log saved: ${LOG_FILE}"
