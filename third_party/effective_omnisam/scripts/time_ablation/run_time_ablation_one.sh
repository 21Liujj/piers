#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

METHOD="${METHOD:-baseline}" # baseline|mem_only|aps_only|both
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"

DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-4}"
PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG:---no-pl_shuffle}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/one}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-9}"

# r9 multimodal defaults
USE_MULTIMODAL="${USE_MULTIMODAL:-1}"
TEXT_PROTO_PATH="${TEXT_PROTO_PATH:-/root/autodl-tmp/OmniSAM-main/assets/text_prototypes_openclip_text_latest.pt}"
SSL_GATE_MODE="${SSL_GATE_MODE:-hard}"
TAU_T="${TAU_T:-0.10}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.02}"

COMPRESS_MODE_DEFAULT="${COMPRESS_MODE_DEFAULT:-hybrid}"
COMPRESS_LOCATION_DEFAULT="${COMPRESS_LOCATION_DEFAULT:-before_attention}"
APS_MODE_DEFAULT="${APS_MODE_DEFAULT:-adaptive_stride}"
APS_SCORE_DEFAULT="${APS_SCORE_DEFAULT:-hybrid}"
APS_HIGH_STRIDE="${APS_HIGH_STRIDE:-384}"
APS_LOW_STRIDE="${APS_LOW_STRIDE:-768}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${METHOD}/${DATASET}_seed${SEED}_bank${MEMORY_BANK_SIZE}_${STAMP}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/ckpts" "${RUN_DIR}/metrics" "${RUN_DIR}/summaries"
MASTER_LOG="${RUN_DIR}/logs/${METHOD}.master.log"
PID_FILE="${RUN_DIR}/logs/${METHOD}.pid"
TRAIN_LOG_BACKUP="${OUT_ROOT}/_trainlogs/${METHOD}_${STAMP}.log"
mkdir -p "${OUT_ROOT}/_trainlogs"

USE_MEM="--no-use_memory_compression"
USE_APS="--no-use_patch_scheduler"
COMP_MODE="none"
SCHED_MODE="none"
case "${METHOD}" in
  baseline) ;;
  mem_only) USE_MEM="--use_memory_compression"; COMP_MODE="${COMPRESS_MODE_DEFAULT}" ;;
  aps_only) USE_APS="--use_patch_scheduler"; SCHED_MODE="${APS_MODE_DEFAULT}" ;;
  both) USE_MEM="--use_memory_compression"; USE_APS="--use_patch_scheduler"; COMP_MODE="${COMPRESS_MODE_DEFAULT}"; SCHED_MODE="${APS_MODE_DEFAULT}" ;;
  *) echo "[error] METHOD must be baseline|mem_only|aps_only|both"; exit 1 ;;
esac

SUBSET_ARG=""
if [[ -n "${PL_SUBSET_FILE}" ]]; then
  SUBSET_ARG="--pl_subset_file ${PL_SUBSET_FILE}"
fi
MM_ARGS=""
if [[ "${USE_MULTIMODAL}" == "1" ]]; then
  MM_ARGS="--text_proto_path ${TEXT_PROTO_PATH} --use_text_gate_ssl --ssl_gate_mode ${SSL_GATE_MODE} --tau_t ${TAU_T} --weight_txt_align ${WEIGHT_TXT_ALIGN}"
fi

CMD="CUDA_VISIBLE_DEVICES=${GPU} python run_adaptation.py \
  --dataset ${DATASET} \
  --class_set ${CLASS_SET} \
  --backbone ${BACKBONE} \
  --seed ${SEED} \
  --num_epochs ${NUM_EPOCHS} \
  --train_max_iters ${TRAIN_MAX_ITERS} \
  --batch_size ${BATCH_SIZE} \
  --pl_batch_size ${PL_BATCH_SIZE} \
  --pl_num_workers ${PL_NUM_WORKERS} \
  --num_maskmem ${MEMORY_BANK_SIZE} \
  --memory_bank_size ${MEMORY_BANK_SIZE} \
  --pd_root ${RUN_DIR} \
  --tb_log_dir ${RUN_DIR}/logs/tb \
  --tgt_ckpt ${TGT_CKPT} \
  --src_ckpt ${SRC_CKPT} \
  --amp --tf32 \
  ${PL_SHUFFLE_FLAG} \
  ${USE_MEM} \
  --memory_select_mode hybrid \
  --memory_recent_keep 4 \
  --memory_score_keep 4 \
  --memory_compress_mode ${COMP_MODE} \
  --memory_compress_location ${COMPRESS_LOCATION_DEFAULT} \
  --memory_spatial_window_size 2 \
  --memory_global_summary_tokens 4 \
  ${USE_APS} \
  --patch_scheduler_mode ${SCHED_MODE} \
  --patch_score_source ${APS_SCORE_DEFAULT} \
  --adaptive_stride_high_complexity ${APS_HIGH_STRIDE} \
  --adaptive_stride_low_complexity ${APS_LOW_STRIDE} \
  --adaptive_apply_stage pseudo_label \
  ${MM_ARGS} \
  ${SUBSET_ARG}"

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
  nohup bash -lc "${CMD}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Started ${METHOD} in background."
  echo "PID: ${pid} (saved to ${PID_FILE})"
  echo "Log: ${MASTER_LOG}"
  echo "Run dir: ${RUN_DIR}"
  exit 0
fi

echo "${CMD}" | tee "${MASTER_LOG}"
bash -lc "${CMD}" 2>&1 | tee -a "${MASTER_LOG}" | tee -a "${TRAIN_LOG_BACKUP}"
