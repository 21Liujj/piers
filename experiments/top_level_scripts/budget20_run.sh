#!/bin/bash
cd /root/autodl-tmp/piers-main
P=/root/autodl-tmp/envs/piers/bin/python
echo "[BUDGET20] start $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
timeout 900 $P m_eval2.py feature_norm 0.03125 20 M-b20-topk1024 >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] topk1024 done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
timeout 900 $P m_eval2.py feature_norm 0.125 20 M-b20-topk4096 >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] topk4096 done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
timeout 900 $P m_eval2.py boundary_guarded 0.03125 20 M-b20-bg1024 >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] bg1024 done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
timeout 900 $P m_eval2.py boundary_guarded 0.125 20 M-b20-bg4096 >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] bg4096 done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
timeout 900 $P m_eval2.py cams 0.03125 20 M-b20-cams1024 >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] cams1024 done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
timeout 900 $P m_eval2.py cams 0.125 20 M-b20-cams4096 >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] cams4096 done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
timeout 900 $P m_eval2.py uniform 0.0625 20 M-b20-uniform2048 >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] uniform done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
timeout 900 $P m_eval2.py pool 0.0625 20 M-b20-pool2048 >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] pool done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
echo "[BUDGET20] all done $(date)" >> /root/autodl-tmp/piers-main/budget20.log 2>&1
