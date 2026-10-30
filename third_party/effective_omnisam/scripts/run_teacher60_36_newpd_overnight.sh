#!/usr/bin/env bash
# Overnight run: generate new pseudo-labels using teacher=60.36, then train multiple epochs on them.
# - Uses a fresh PD_ROOT (separate from pd_uda_densepass)
# - Does NOT reuse any old pseudo-labels
# - Writes logs under logs/
#
# Usage:
#   bash scripts/run_teacher60_36_newpd_overnight.sh
#   RUN_IN_BACKGROUND=1 bash scripts/run_teacher60_36_newpd_overnight.sh
#
# Env vars (optional):
#   DATA_ROOT, LOG_DIR
#   PD_ROOT_NEW
#   TEACHER_CKPT
#   NUM_EPOCHS (default 6)
#   BATCH_SIZE (default 4)
#   CONF_THR (default 0.85)
#   AMP (default 1), TF32 (default 1)
#   WANDB_MODE (default offline)
#   RUN_IN_BACKGROUND (default 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-data}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"

TEACHER_CKPT="${TEACHER_CKPT:-/root/autodl-tmp/OmniSAM-main/pd_uda_densepass/checkpoints/Cityscapes/sam2_s/best_sam2_s_iou60.36.pth}"
PD_ROOT_NEW="${PD_ROOT_NEW:-${ROOT}/pd_uda_densepass_teacher60.36}"

NUM_EPOCHS="${NUM_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONF_THR="${CONF_THR:-0.85}"
AMP="${AMP:-1}"
TF32="${TF32:-1}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "Error: teacher ckpt not found: ${TEACHER_CKPT}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/uda_teacher60.36_newpd_${STAMP}.log"

echo "==========================================" | tee "${LOG_FILE}"
echo "UDA OVERNIGHT (new PD_ROOT + teacher60.36)" | tee -a "${LOG_FILE}"
echo "Time: $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "DATA_ROOT: ${DATA_ROOT}" | tee -a "${LOG_FILE}"
echo "PD_ROOT_NEW: ${PD_ROOT_NEW}" | tee -a "${LOG_FILE}"
echo "TEACHER_CKPT: ${TEACHER_CKPT}" | tee -a "${LOG_FILE}"
echo "NUM_EPOCHS: ${NUM_EPOCHS}" | tee -a "${LOG_FILE}"
echo "BATCH_SIZE: ${BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "CONF_THR: ${CONF_THR}" | tee -a "${LOG_FILE}"
echo "AMP: ${AMP}" | tee -a "${LOG_FILE}"
echo "TF32: ${TF32}" | tee -a "${LOG_FILE}"
echo "RUN_IN_BACKGROUND: ${RUN_IN_BACKGROUND}" | tee -a "${LOG_FILE}"
echo "LOG_FILE: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

PL_DIR="${PD_ROOT_NEW}/DensePASS/sam2_s/Cityscapes/epoch1"

CMD_GEN=(
  python run_adaptation.py
  --backbone sam2_s
  --dataset Cityscapes
  --data_root "${DATA_ROOT}"
  --pd_root "${PD_ROOT_NEW}"
  --src_ckpt "/root/autodl-tmp/omnisam_s.pth"
  --tgt_ckpt "${TEACHER_CKPT}"
  --warmup_pl 2000
  --iters_pl 0
  --batch_size 1
  --num_epochs 1
  --generate_only
)

CMD_TRAIN=(
  python run_adaptation.py
  --backbone sam2_s
  --dataset Cityscapes
  --data_root "${DATA_ROOT}"
  --pd_root "${PD_ROOT_NEW}"
  --src_ckpt "/root/autodl-tmp/omnisam_s.pth"
  --tgt_ckpt "${TEACHER_CKPT}"
  --reuse_pl_dir "${PL_DIR}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --conf_thr "${CONF_THR}"
)

if [[ "${AMP}" == "1" ]]; then
  CMD_TRAIN+=(--amp)
fi
if [[ "${TF32}" == "1" ]]; then
  CMD_TRAIN+=(--tf32)
else
  CMD_TRAIN+=(--no-tf32)
fi

RUN_ALL=(
  bash -lc
  "$(printf '%q ' "${CMD_GEN[@]}") && $(printf '%q ' "${CMD_TRAIN[@]}")"
)

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
  nohup "${RUN_ALL[@]}" >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "Started in background. PID=${PID}" | tee -a "${LOG_FILE}"
  echo "Follow log: tail -f ${LOG_FILE}"
else
  "${RUN_ALL[@]}" 2>&1 | tee -a "${LOG_FILE}"
  echo "Done: $(date -Iseconds)" | tee -a "${LOG_FILE}"
fi

