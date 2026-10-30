#!/bin/bash
# Quick test: Baseline vs +SCM on small subset (first 10 samples)

REPO_ROOT="/root/autodl-tmp/OmniSAM-main"
cd $REPO_ROOT

# Use existing checkpoint
CKPT="experiments/route1/runs/boundary_0625_n500_seed1_20260826_101721/checkpoints/Cityscapes_13cls/sam2_s/best_iou59.99.pth"

echo "========================================"
echo "SCM Quick Comparison Test"
echo "========================================"

# Test 1: Baseline (no SCM)
echo ""
echo "[Test 1/2] Running Baseline (no circular padding)..."
python validation.py \
    --dataset DensePASS \
    --class_set 13 \
    --resume $CKPT \
    --save_dir experiments/scm_test/baseline \
    2>&1 | tee experiments/scm_test/baseline.log

# Test 2: +SCM
echo ""
echo "[Test 2/2] Running +SCM (circular padding enabled)..."
python validation.py \
    --dataset DensePASS \
    --class_set 13 \
    --resume $CKPT \
    --save_dir experiments/scm_test/scm_enabled \
    --enable_scm \
    --scm_pad_width 512 \
    2>&1 | tee experiments/scm_test/scm_enabled.log

echo ""
echo "========================================"
echo "Tests completed!"
echo "========================================"

# Extract mIoU from logs
echo ""
echo "Results Summary:"
echo "----------------"
grep -i 'miou\|mean iou' experiments/scm_test/baseline.log | tail -3
echo ""
grep -i 'miou\|mean iou' experiments/scm_test/scm_enabled.log | tail -3

echo ""
echo "Detailed logs:"
echo "  Baseline: experiments/scm_test/baseline.log"
echo "  +SCM: experiments/scm_test/scm_enabled.log"
