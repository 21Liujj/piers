#!/usr/bin/env bash
# DensePASS val：① 原论文 omnisam_s  ② UDA best_iou60.36（单次调用 validation 多 --ckpt）
set -euo pipefail
REPO="/root/autodl-tmp/OmniSAM-main"
cd "$REPO"
PYTHON="${PYTHON:-/root/miniconda3/envs/omnisam/bin/python}"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$LOG_DIR/val_compare_omnisam_vs_6036_${STAMP}.log"
JSON_OUT="$LOG_DIR/val_compare_omnisam_vs_6036_${STAMP}.json"

{
  echo "=========================================="
  echo "compare_val_two_ckpts.sh (one validation.py, two --ckpt)"
  echo "Time: $(date -Iseconds)"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "JSON: $JSON_OUT"
  echo "=========================================="
} | tee "$SUMMARY_LOG"

"$PYTHON" validation.py \
  --backbone sam2_s \
  --dataset DensePASS \
  --class_set 13 \
  --use_mem_bank \
  --data_root "$DATA_ROOT" \
  --ckpt "/root/autodl-tmp/omnisam_s.pth" \
  --ckpt "$REPO/pd_uda_densepass/checkpoints/Cityscapes/sam2_s/best_sam2_s_iou60.36.pth" \
  --ckpt_names "official_omnisam_s,uda_best_60_36" \
  --metrics_json "$JSON_OUT" \
  2>&1 | tee -a "$SUMMARY_LOG"

echo "Done. Log: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
