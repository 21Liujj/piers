#!/bin/bash
set -u
cd /root/autodl-tmp/piers-main
LOG=/root/autodl-tmp/piers-main/runs/night_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "[NIGHT] start $(date) gpu:"; nvidia-smi --query-gpu=index,memory.used --format=csv | head -n 5
run_one(){ echo "[NIGHT] $@ $(date)"; timeout 900 /root/autodl-tmp/envs/piers/bin/python m_eval2.py "$@" || echo "[NIGHT-FAIL] $@ code=$?"; sleep 5; }
# 20-img dev set: control + variants (each ~1min)
run_one feature_norm 0.0625 20 M-night-topk20
run_one boundary_guarded 0.0625 20 M-night-bg20
run_one class_aware 0.0625 20 M-night-ca20
run_one random 0.0625 20 M-night-random20
# budget curve 20-img for topk/bg (1024/4096) -> 4 runs
run_one feature_norm 0.03125 20 M-night-topk1024-20
run_one feature_norm 0.125 20 M-night-topk4096-20
run_one boundary_guarded 0.03125 20 M-night-bg1024-20
run_one boundary_guarded 0.125 20 M-night-bg4096-20
# S-fixed-scm 20-img via validation CLI
echo "[NIGHT] S-scm20 $(date)"
timeout 900 /root/autodl-tmp/envs/piers/bin/python third_party/effective_omnisam/validation.py --backbone sam2_s --dataset DensePASS --class_set 13 --data_root /root/autodl-tmp/data --ckpt /root/autodl-tmp/weights/omnisam_s.pth --enable_scm --scm_pad_width 512 --metrics_json /root/autodl-tmp/piers-main/runs/S-night-scm20/metrics.json || echo "[NIGHT-FAIL] scm20 $?"
echo "[NIGHT] done $(date)"
