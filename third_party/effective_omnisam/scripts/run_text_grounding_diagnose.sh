#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
INTERNAL_BG="${INTERNAL_BG:-0}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/text_grounding_diagnose}"
RUN_NAME="${RUN_NAME:-grounding_diag_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
MASTER_LOG="${RUN_DIR}/00_master.log"
PID_FILE="${RUN_DIR}/run.pid"

EPOCHS="${EPOCHS:-1}"
WARMUP_PL="${WARMUP_PL:-20}"
ITERS_PL="${ITERS_PL:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
BACKBONE="${BACKBONE:-sam2_s}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
CASE_SET="${CASE_SET:-all}"
TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL:-10}"
GROUNDING_GAMMA_INIT="${GROUNDING_GAMMA_INIT:-0.05}"
GROUNDING_NUM_LAYERS="${GROUNDING_NUM_LAYERS:-2}"
GROUNDING_NUM_HEADS="${GROUNDING_NUM_HEADS:-4}"
GROUNDING_LOSS_WEIGHT="${GROUNDING_LOSS_WEIGHT:-0.05}"
GROUNDING_KV_STRIDE="${GROUNDING_KV_STRIDE:-1}"
TEXT_ADAPTER_PROJECTOR_TYPE="${TEXT_ADAPTER_PROJECTOR_TYPE:-linear}"
TEXT_ADAPTER_FREEZE="${TEXT_ADAPTER_FREEZE:-0}"
SAVE_DEBUG_VIS="${SAVE_DEBUG_VIS:-0}"
MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY:-1}"

mkdir -p "${RUN_DIR}"

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 \
    LOG_ROOT="${LOG_ROOT}" RUN_NAME="${RUN_NAME}" \
    EPOCHS="${EPOCHS}" WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
    BACKBONE="${BACKBONE}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" DATA_ROOT="${DATA_ROOT}" \
    SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" CASE_SET="${CASE_SET}" \
    TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL}" \
    GROUNDING_GAMMA_INIT="${GROUNDING_GAMMA_INIT}" GROUNDING_NUM_LAYERS="${GROUNDING_NUM_LAYERS}" \
    GROUNDING_NUM_HEADS="${GROUNDING_NUM_HEADS}" GROUNDING_LOSS_WEIGHT="${GROUNDING_LOSS_WEIGHT}" \
    GROUNDING_KV_STRIDE="${GROUNDING_KV_STRIDE}" \
    TEXT_ADAPTER_PROJECTOR_TYPE="${TEXT_ADAPTER_PROJECTOR_TYPE}" TEXT_ADAPTER_FREEZE="${TEXT_ADAPTER_FREEZE}" \
    SAVE_DEBUG_VIS="${SAVE_DEBUG_VIS}" MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Started in background."
  echo "PID: ${pid} (saved to ${PID_FILE})"
  echo "Master log: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Text Grounding Diagnose =========="
echo "run_dir=${RUN_DIR}"
echo "case_set=${CASE_SET}"
echo "epochs=${EPOCHS}, warmup_pl=${WARMUP_PL}, iters_pl=${ITERS_PL}"
echo "batch_size=${BATCH_SIZE}, pl_batch_size=${PL_BATCH_SIZE}"
echo "gamma_init=${GROUNDING_GAMMA_INIT}, layers=${GROUNDING_NUM_LAYERS}, heads=${GROUNDING_NUM_HEADS}"
echo "loss_weight=${GROUNDING_LOSS_WEIGHT}, kv_stride=${GROUNDING_KV_STRIDE}"
echo "projector=${TEXT_ADAPTER_PROJECTOR_TYPE}, freeze=${TEXT_ADAPTER_FREEZE}, save_debug_vis=${SAVE_DEBUG_VIS}"
echo "mem_compress_train_only=${MEM_COMPRESS_TRAIN_ONLY}"
echo "============================================"

case "${CASE_SET}" in
  all|c1c4c5|C1C4C5) ;;
  c1|C1) ;;
  c4|C4) ;;
  c5|C5) ;;
  *)
    echo "Unsupported CASE_SET=${CASE_SET}; expected all|c1|c4|c5"
    exit 2
    ;;
esac

COMMON_OPTS="num_epochs=${EPOCHS},warmup_pl=${WARMUP_PL},iters_pl=${ITERS_PL},text_log_interval=${TEXT_LOG_INTERVAL},text_ground_gamma_init=${GROUNDING_GAMMA_INIT},text_ground_num_layers=${GROUNDING_NUM_LAYERS},text_ground_num_heads=${GROUNDING_NUM_HEADS},loss_text_grounding_weight=${GROUNDING_LOSS_WEIGHT},text_ground_kv_stride=${GROUNDING_KV_STRIDE},text_adapter_projector_type=${TEXT_ADAPTER_PROJECTOR_TYPE},text_adapter_freeze=$( [[ "${TEXT_ADAPTER_FREEZE}" == "1" ]] && echo true || echo false ),text_fusion_save_debug_vis=$( [[ "${SAVE_DEBUG_VIS}" == "1" ]] && echo true || echo false )"
if [[ "${MEM_COMPRESS_TRAIN_ONLY}" == "1" ]]; then
  COMMON_OPTS="${COMMON_OPTS},plugins.efficient_tam_memory.enabled=true,plugins.efficient_tam_memory.mode=pool,plugins.efficient_tam_memory.pool_stride=2,plugins.efficient_tam_memory.apply_in_train=true,plugins.efficient_tam_memory.apply_in_pl=false"
  COMMON_OPTS="${COMMON_OPTS},plugins.efficient_sam2_sparse.enabled=true,plugins.efficient_sam2_sparse.mode=attn_topk,plugins.efficient_sam2_sparse.keep_ratio=0.5,plugins.efficient_sam2_sparse.min_keep_tokens=64,plugins.efficient_sam2_sparse.saliency_source=feature_norm,plugins.efficient_sam2_sparse.cache_saliency=true,plugins.efficient_sam2_sparse.fallback_full_when_uncertain=true,plugins.efficient_sam2_sparse.apply_in_train=true,plugins.efficient_sam2_sparse.apply_in_pl=false,mem_compression_enable_in_pseudo_label=false"
fi

run_case() {
  local case_name="$1"
  local cfg="$2"
  local pd_root="${RUN_DIR}/${case_name}"
  local log_file="${RUN_DIR}/${case_name}.log"
  local status_file="${RUN_DIR}/${case_name}.status"
  echo "RUN ${case_name}"
  echo "cfg=${cfg}"
  echo "pd_root=${pd_root}"

  set +e
  python run_adaptation.py \
    --config "${cfg}" \
    --amp --tf32 \
    --backbone "${BACKBONE}" \
    --dataset "${DATASET}" \
    --class_set "${CLASS_SET}" \
    --batch_size "${BATCH_SIZE}" \
    --pl_batch_size "${PL_BATCH_SIZE}" \
    --data_root "${DATA_ROOT}" \
    --src_ckpt "${SRC_CKPT}" \
    --tgt_ckpt "${TGT_CKPT}" \
    --opts "${COMMON_OPTS}" \
    --pd_root "${pd_root}" \
    2>&1 | tee "${log_file}"
  rc=${PIPESTATUS[0]}
  set -e

  {
    echo "case=${case_name}"
    echo "rc=${rc}"
    echo "cfg=${cfg}"
    echo "log=${log_file}"
    echo "pd_root=${pd_root}"
    echo "time=$(date -Iseconds)"
  } > "${status_file}"

  if [[ ${rc} -ne 0 ]]; then
    echo "[error] case ${case_name} failed with rc=${rc}"
    exit ${rc}
  fi
  echo "[ok] case ${case_name} rc=0"
}

if [[ "${CASE_SET}" == "c1" || "${CASE_SET}" == "C1" ]]; then
  run_case "C1_real" "configs/experiments/text_modality_grounding/grounding_C1_real_proto.yaml"
elif [[ "${CASE_SET}" == "c4" || "${CASE_SET}" == "C4" ]]; then
  run_case "C4_shuffle" "configs/experiments/text_modality_grounding/grounding_C4_shuffle_proto.yaml"
elif [[ "${CASE_SET}" == "c5" || "${CASE_SET}" == "C5" ]]; then
  run_case "C5_random" "configs/experiments/text_modality_grounding/grounding_C5_random_proto.yaml"
else
  run_case "C1_real" "configs/experiments/text_modality_grounding/grounding_C1_real_proto.yaml"
  run_case "C4_shuffle" "configs/experiments/text_modality_grounding/grounding_C4_shuffle_proto.yaml"
  run_case "C5_random" "configs/experiments/text_modality_grounding/grounding_C5_random_proto.yaml"
fi

echo "All text grounding diagnose runs finished."
echo "Logs: ${RUN_DIR}"
