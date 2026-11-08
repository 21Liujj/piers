#!/bin/bash
cd /root/autodl-tmp/piers-main
P=/root/autodl-tmp/envs/piers/bin/python
export PIERS_WEIGHTS_DIR=/root/autodl-tmp/weights
export PIERS_GROUNDING_DIR=/root/autodl-tmp/weights/grounding-dino-tiny-hf
while pgrep -f "formal100_run.sh" >/dev/null 2>&1 || pgrep -f "s_adaptive_eval.py S100" >/dev/null 2>&1; do echo "[I1] waiting formal100/S100 $(date)" >> /root/autodl-tmp/piers-main/i1.log 2>&1; sleep 30; done
echo "[I1] probe start $(date)" >> /root/autodl-tmp/piers-main/i1.log 2>&1
cd /root/autodl-tmp/piers-main/third_party/grounded_sam2
timeout 1800 $P /root/autodl-tmp/piers-main/t_probe.py >> /root/autodl-tmp/piers-main/i1.log 2>&1
echo "[I1] probe done $(date)" >> /root/autodl-tmp/piers-main/i1.log 2>&1
timeout 3600 $P -m piers_ljj.workers.entry grounded --config /root/autodl-tmp/piers-main/configs/piers/T0_grounded_direct.yaml --manifest /root/autodl-tmp/piers-main/configs/piers/manifest.dev-calib-10.json --out /root/autodl-tmp/piers-main/runs/T0-10 --device cuda >> /root/autodl-tmp/piers-main/i1.log 2>&1
echo "[I1] T0 done $(date)" >> /root/autodl-tmp/piers-main/i1.log 2>&1
timeout 3600 $P -m piers_ljj.workers.entry grounded --config /root/autodl-tmp/piers-main/configs/piers/T1_grounded_fov.yaml --manifest /root/autodl-tmp/piers-main/configs/piers/manifest.dev-calib-10.json --out /root/autodl-tmp/piers-main/runs/T1-10 --device cuda >> /root/autodl-tmp/piers-main/i1.log 2>&1
echo "[I1] T1 done $(date)" >> /root/autodl-tmp/piers-main/i1.log 2>&1
echo "[I1] all done $(date)" >> /root/autodl-tmp/piers-main/i1.log 2>&1
