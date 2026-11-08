#!/bin/bash
cd /root/autodl-tmp/piers-main
P=/root/autodl-tmp/envs/piers/bin/python
echo "[DEV20] start $(date)" >> /root/autodl-tmp/piers-main/dev20.log 2>&1
timeout 900 $P m_eval2.py dense 0.5 20 M-dev20-dense >> /root/autodl-tmp/piers-main/dev20.log 2>&1
echo "[DEV20] dense done $(date)" >> /root/autodl-tmp/piers-main/dev20.log 2>&1
timeout 900 $P m_eval2.py feature_norm 0.0625 20 M-dev20-topk2048 >> /root/autodl-tmp/piers-main/dev20.log 2>&1
echo "[DEV20] topk done $(date)" >> /root/autodl-tmp/piers-main/dev20.log 2>&1
timeout 900 $P m_eval2.py random 0.0625 20 M-dev20-random >> /root/autodl-tmp/piers-main/dev20.log 2>&1
echo "[DEV20] random done $(date)" >> /root/autodl-tmp/piers-main/dev20.log 2>&1
timeout 900 $P m_eval2.py boundary_guarded 0.0625 20 M-dev20-bg2048 >> /root/autodl-tmp/piers-main/dev20.log 2>&1
echo "[DEV20] bg done $(date)" >> /root/autodl-tmp/piers-main/dev20.log 2>&1
timeout 900 $P m_eval2.py class_aware 0.0625 20 M-dev20-ca2048 >> /root/autodl-tmp/piers-main/dev20.log 2>&1
echo "[DEV20] ca done $(date)" >> /root/autodl-tmp/piers-main/dev20.log 2>&1
timeout 900 $P m_eval2.py cams 0.0625 20 M-dev20-cams2048 >> /root/autodl-tmp/piers-main/dev20.log 2>&1
echo "[DEV20] cams done $(date)" >> /root/autodl-tmp/piers-main/dev20.log 2>&1
