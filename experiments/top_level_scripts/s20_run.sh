#!/bin/bash
cd /root/autodl-tmp/piers-main
P=/root/autodl-tmp/envs/piers/bin/python
# wait for budget20 to finish (avoid GPU contention)
while pgrep -f "m_eval2.py" >/dev/null 2>&1; do echo "[S20] waiting budget $(date)" >> /root/autodl-tmp/piers-main/s20.log 2>&1; sleep 30; done
echo "[S20] start $(date)" >> /root/autodl-tmp/piers-main/s20.log 2>&1
timeout 1200 $P s_adaptive_eval.py S20-b0-adapt 0 adaptive 20 >> /root/autodl-tmp/piers-main/s20.log 2>&1
echo "[S20] b0 done $(date)" >> /root/autodl-tmp/piers-main/s20.log 2>&1
timeout 1200 $P s_adaptive_eval.py S20-b1-adapt 1 adaptive 20 >> /root/autodl-tmp/piers-main/s20.log 2>&1
echo "[S20] b1-adapt done $(date)" >> /root/autodl-tmp/piers-main/s20.log 2>&1
timeout 1200 $P s_adaptive_eval.py S20-b1-uniform 1 uniform 20 >> /root/autodl-tmp/piers-main/s20.log 2>&1
echo "[S20] b1-uniform done $(date)" >> /root/autodl-tmp/piers-main/s20.log 2>&1
timeout 1200 $P s_adaptive_eval.py S20-b2-adapt 2 adaptive 20 >> /root/autodl-tmp/piers-main/s20.log 2>&1
echo "[S20] b2-adapt done $(date)" >> /root/autodl-tmp/piers-main/s20.log 2>&1
echo "[S20] all done $(date)" >> /root/autodl-tmp/piers-main/s20.log 2>&1
