#!/bin/bash
E=/root/autodl-tmp/envs/piers/bin/python; P=/root/autodl-tmp/piers-main
# GT gen (fixed script already uploaded)
setsid $E $P/_gt_gen.py > $P/runs/_gtgen2.log 2>&1 < /dev/null &
echo GT2-$!
for i in 0 1 2 3; do
  LOG=$P/runs/T1-100m-s$i/worker.log
  if [ -s "$LOG" ] && grep -q "executable" "$LOG" 2>/dev/null; then echo "S$i-SKIP(running)"; continue; fi
  mkdir -p $P/runs/T1-100m-s$i
  cd $P/third_party/grounded_sam2
  setsid $E -m piers_ljj.workers.entry grounded --config ../../configs/piers/T1_grounded_fov.yaml --manifest ../../configs/piers/manifest.100-s$i.json --out ../../runs/T1-100m-s$i --device cuda > $LOG 2>&1 < /dev/null &
  echo "S$i-$!"
  cd $P
done
exit 0
