#!/bin/bash
export PYTHONPATH=/root/autodl-tmp/piers-main/src:$PYTHONPATH
E=/root/autodl-tmp/envs/piers/bin/python
P=/root/autodl-tmp/piers-main
LOG=$P/i1_100.log
echo "[I1-100] start $(date)" | tee -a $LOG
cd $P/third_party/grounded_sam2
echo "[I1-100] T0 start $(date)" | tee -a $LOG
timeout 5400 $E -m piers_ljj.workers.entry grounded --config $P/configs/piers/T0_grounded_direct.yaml --manifest $P/configs/piers/manifest.dev-calib-100.json --out $P/runs/T0-100 --device cuda >> $LOG 2>&1
echo "[I1-100] T0 done $(date) exit=$?" | tee -a $LOG
echo "[I1-100] T1 start $(date)" | tee -a $LOG
timeout 5400 $E -m piers_ljj.workers.entry grounded --config $P/configs/piers/T1_grounded_fov.yaml --manifest $P/configs/piers/manifest.dev-calib-100.json --out $P/runs/T1-100 --device cuda >> $LOG 2>&1
echo "[I1-100] T1 done $(date) exit=$?" | tee -a $LOG
echo "[I1-100] all done $(date)" | tee -a $LOG
