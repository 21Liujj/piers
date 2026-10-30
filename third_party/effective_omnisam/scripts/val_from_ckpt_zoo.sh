#!/usr/bin/env bash
# 用 ckpt_zoo 统一选择 ckpt 并跑 validation.py
#
# 用法：
#   bash scripts/val_from_ckpt_zoo.sh list
#   bash scripts/val_from_ckpt_zoo.sh collect
#   bash scripts/val_from_ckpt_zoo.sh val --id 3 --dataset DensePASS
#   bash scripts/val_from_ckpt_zoo.sh val --match iou60.36 --dataset DensePASS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

python tools/ckpt_zoo.py "$@"

