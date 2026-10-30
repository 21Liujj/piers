#!/usr/bin/env bash
# Run inference for baseline and compressed models, save per-image predictions for spatial diagnostic

set -e

BASELINE_CKPT="experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth"
COMPRESSED_CKPT="experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.02.pth"

PRED_OUT_DIR="experiments/tecma/spatial_diagnostic_preds"
mkdir -p "$PRED_OUT_DIR"

echo "=== Running baseline inference (save predictions) ==="
/root/miniconda3/bin/python validation.py \
  --dataset DensePASS \
  --class_set 13 \
  --backbone sam2_s \
  --use_mem_bank \
  --num_maskmem 9 \
  --ckpt "$BASELINE_CKPT" \
  --batch_size 1 \
  --save_preds_to "$PRED_OUT_DIR/baseline" \
  --metrics_json "$PRED_OUT_DIR/baseline_metrics.json"

echo "=== Running compressed inference (save predictions) ==="
/root/miniconda3/bin/python validation.py \
  --dataset DensePASS \
  --class_set 13 \
  --backbone sam2_s \
  --use_mem_bank \
  --num_maskmem 9 \
  --ckpt "$COMPRESSED_CKPT" \
  --batch_size 1 \
  --save_preds_to "$PRED_OUT_DIR/compressed" \
  --metrics_json "$PRED_OUT_DIR/compressed_metrics.json"

echo "Inference done. Predictions saved to $PRED_OUT_DIR"
