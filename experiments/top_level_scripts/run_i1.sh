#!/bin/bash
export PYTHONPATH=/root/autodl-tmp/piers-main/src:$PYTHONPATH
E=/root/autodl-tmp/envs/piers/bin/python
P=/root/autodl-tmp/piers-main
LOG=$P/i1_10.log
echo "[I1-10] start $(date)" | tee -a $LOG
cd $P/third_party/grounded_sam2
echo "[I1-10] T0 start $(date)" | tee -a $LOG
timeout 3600 $E -m piers_ljj.workers.entry grounded --config $P/configs/piers/T0_grounded_direct.yaml --manifest $P/configs/piers/manifest.dev-calib-10.json --out $P/runs/T0-10 --device cuda >> $LOG 2>&1
echo "[I1-10] T0 done $(date) exit=$?" | tee -a $LOG
echo "[I1-10] T1 start $(date)" | tee -a $LOG
timeout 3600 $E -m piers_ljj.workers.entry grounded --config $P/configs/piers/T1_grounded_fov.yaml --manifest $P/configs/piers/manifest.dev-calib-10.json --out $P/runs/T1-10 --device cuda >> $LOG 2>&1
echo "[I1-10] T1 done $(date) exit=$?" | tee -a $LOG
echo "[I1-10] all done $(date)" | tee -a $LOG
