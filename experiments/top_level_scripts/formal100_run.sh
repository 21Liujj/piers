#!/bin/bash
cd /root/autodl-tmp/piers-main
P=/root/autodl-tmp/envs/piers/bin/python
while pgrep -f "s_adaptive_eval.py" >/dev/null 2>&1; do echo "[F100] waiting s20 $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1; sleep 30; done
echo "[F100] start $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1200 $P m_eval2.py dense 0.5 100 M100-dense >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] dense done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1200 $P m_eval2.py feature_norm 0.0625 100 M100-topk2048 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] topk done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1200 $P m_eval2.py uniform 0.0625 100 M100-uniform2048 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] uniform done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1200 $P m_eval2.py pool 0.0625 100 M100-pool2048 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] pool done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1200 $P m_eval2.py boundary_guarded 0.0625 100 M100-bg2048 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] bg done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1200 $P m_eval2.py class_aware 0.0625 100 M100-ca2048 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] ca done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1200 $P m_eval2.py cams 0.0625 100 M100-cams2048 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] cams done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1800 $P s_adaptive_eval.py S100-b0 0 adaptive 100 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] s-b0 done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1800 $P s_adaptive_eval.py S100-b1-adapt 1 adaptive 100 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] s-b1a done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1800 $P s_adaptive_eval.py S100-b1-uniform 1 uniform 100 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] s-b1u done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
timeout 1800 $P s_adaptive_eval.py S100-b2-adapt 2 adaptive 100 >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] s-b2 done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
echo "[F100] all done $(date)" >> /root/autodl-tmp/piers-main/formal100.log 2>&1
