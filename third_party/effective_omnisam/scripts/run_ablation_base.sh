#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATA_ROOT=data PD_ROOT=data_pd TGT_CKPT=omnisam_s.pth SRC_CKPT=omnisam_s.pth \
#   bash scripts/run_ablation_base.sh

: "${DATA_ROOT:=data}"
: "${PD_ROOT:=data_pd}"
: "${TGT_CKPT:=exp2/Cityscapes/best_sam2_s_10.pth}"
: "${SRC_CKPT:=exp2/Cityscapes/best_sam2_s_10.pth}"
: "${DATASET:=Cityscapes}"
: "${BACKBONE:=sam2_s}"
: "${CLASS_SET:=13}"

python run_adaptation.py \
  --config configs/ablations/omnisam_base.yaml \
  --data_root "${DATA_ROOT}" \
  --pd_root "${PD_ROOT}" \
  --dataset "${DATASET}" \
  --backbone "${BACKBONE}" \
  --class_set "${CLASS_SET}" \
  --tgt_ckpt "${TGT_CKPT}" \
  --src_ckpt "${SRC_CKPT}"

