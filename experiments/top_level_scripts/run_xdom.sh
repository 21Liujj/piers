#!/bin/bash
E=/root/autodl-tmp/envs/piers/bin/python
P=/root/autodl-tmp/piers-main
LOG=$P/xdom.log
echo "[XDOM] start $(date)" >> $LOG
run_one(){ DS=$1; MODE=$2; KR=$3; N=$4; TAG=$5;
  echo "[XDOM] $TAG start $(date)" | tee -a $LOG
  $E $P/m_eval_xdom.py --dataset $DS --mode $MODE --kr $KR --n_img $N --tag $TAG >> $P/xdom.log 2>&1
  echo "[XDOM] $TAG done $(date)" | tee -a $LOG
}
# WildPASS 500: OmniSAM baseline(dense) + I2 (ca/bg/topk/pool @N=2048, kr=0.0625)
run_one WildPASS dense 0.0625 500 XW500-dense
run_one WildPASS class_aware 0.0625 500 XW500-ca2048
run_one WildPASS boundary_guarded 0.0625 500 XW500-bg2048
run_one WildPASS feature_norm 0.0625 500 XW500-topk2048
run_one WildPASS pool 0.0625 500 XW500-pool2048
# SPAN8 260: same matrix
run_one Stanford2D3D dense 0.0625 260 XS260-dense
run_one Stanford2D3D class_aware 0.0625 260 XS260-ca2048
run_one Stanford2D3D boundary_guarded 0.0625 260 XS260-bg2048
run_one Stanford2D3D feature_norm 0.0625 260 XS260-topk2048
run_one Stanford2D3D pool 0.0625 260 XS260-pool2048
echo "[XDOM] all done $(date)" | tee -a $LOG
