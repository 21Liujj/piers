#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
INTERNAL_BG="${INTERNAL_BG:-0}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/text_crossattn_diagnose50}"
RUN_NAME="${RUN_NAME:-crossattn_diag50_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
MASTER_LOG="${RUN_DIR}/00_master.log"
PID_FILE="${RUN_DIR}/run.pid"

EPOCHS="${EPOCHS:-1}"
WARMUP_PL="${WARMUP_PL:-50}"
ITERS_PL="${ITERS_PL:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-4}"
TEXT_CROSS_BETA_INIT="${TEXT_CROSS_BETA_INIT:-0.05}"
TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL:-10}"
MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY:-1}"
BACKBONE="${BACKBONE:-sam2_s}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"

PROJECTOR_MODE="${PROJECTOR_MODE:-mlp}"
FREEZE_PROJECTOR="${FREEZE_PROJECTOR:-0}"
ATTN_TEMP="${ATTN_TEMP:-1.0}"
CASE_SET="${CASE_SET:-all}"

mkdir -p "${RUN_DIR}"

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 \
    LOG_ROOT="${LOG_ROOT}" RUN_NAME="${RUN_NAME}" \
    EPOCHS="${EPOCHS}" WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
    TEXT_CROSS_BETA_INIT="${TEXT_CROSS_BETA_INIT}" TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL}" \
    MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY}" \
    BACKBONE="${BACKBONE}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" DATA_ROOT="${DATA_ROOT}" \
    SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" \
    PROJECTOR_MODE="${PROJECTOR_MODE}" FREEZE_PROJECTOR="${FREEZE_PROJECTOR}" \
    ATTN_TEMP="${ATTN_TEMP}" CASE_SET="${CASE_SET}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Started in background."
  echo "PID: ${pid} (saved to ${PID_FILE})"
  echo "Master log: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

if [[ "${PROJECTOR_MODE}" != "mlp" && "${PROJECTOR_MODE}" != "linear" ]]; then
  echo "Unsupported PROJECTOR_MODE=${PROJECTOR_MODE}; expected mlp|linear"
  exit 2
fi

case "${CASE_SET}" in
  all|b1b4b5|B1B4B5) ;;
  b1|B1) ;;
  *)
    echo "Unsupported CASE_SET=${CASE_SET}; expected all|b1"
    exit 2
    ;;
esac

echo "========== Text CrossAttn Diagnose50 =========="
echo "run_dir=${RUN_DIR}"
echo "epochs=${EPOCHS}, warmup_pl=${WARMUP_PL}, iters_pl=${ITERS_PL}"
echo "batch_size=${BATCH_SIZE}, pl_batch_size=${PL_BATCH_SIZE}"
echo "projector_mode=${PROJECTOR_MODE}, freeze_projector=${FREEZE_PROJECTOR}, attn_temp=${ATTN_TEMP}"
echo "text_cross_beta_init=${TEXT_CROSS_BETA_INIT}, text_log_interval=${TEXT_LOG_INTERVAL}"
echo "mem_compress_train_only=${MEM_COMPRESS_TRAIN_ONLY}"
echo "case_set=${CASE_SET}"
echo "==============================================="

COMMON_OPTS="num_epochs=${EPOCHS},warmup_pl=${WARMUP_PL},iters_pl=${ITERS_PL},text_cross_beta_init=${TEXT_CROSS_BETA_INIT},text_log_interval=${TEXT_LOG_INTERVAL},text_adapter_projector_type=${PROJECTOR_MODE},text_adapter_freeze=$( [[ "${FREEZE_PROJECTOR}" == "1" ]] && echo true || echo false ),text_cross_attn_temperature=${ATTN_TEMP}"
if [[ "${MEM_COMPRESS_TRAIN_ONLY}" == "1" ]]; then
  COMMON_OPTS="${COMMON_OPTS},plugins.efficient_tam_memory.enabled=true,plugins.efficient_tam_memory.mode=pool,plugins.efficient_tam_memory.pool_stride=2,plugins.efficient_tam_memory.apply_in_train=true,plugins.efficient_tam_memory.apply_in_pl=false"
  COMMON_OPTS="${COMMON_OPTS},plugins.efficient_sam2_sparse.enabled=true,plugins.efficient_sam2_sparse.mode=attn_topk,plugins.efficient_sam2_sparse.keep_ratio=0.5,plugins.efficient_sam2_sparse.min_keep_tokens=64,plugins.efficient_sam2_sparse.saliency_source=feature_norm,plugins.efficient_sam2_sparse.cache_saliency=true,plugins.efficient_sam2_sparse.fallback_full_when_uncertain=true,plugins.efficient_sam2_sparse.apply_in_train=true,plugins.efficient_sam2_sparse.apply_in_pl=false"
fi

run_case() {
  local case_name="$1"
  local cfg="$2"
  local pd_root="${RUN_DIR}/${case_name}"
  local log_file="${RUN_DIR}/${case_name}.log"
  local status_file="${RUN_DIR}/${case_name}.status"

  echo "[run] ${case_name}"
  echo "[run] cfg=${cfg}"
  echo "[run] pd_root=${pd_root}"
  echo "[run] opts=${COMMON_OPTS}"

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
    echo "log=${log_file}"
    echo "pd_root=${pd_root}"
    echo "projector_mode=${PROJECTOR_MODE}"
    echo "freeze_projector=${FREEZE_PROJECTOR}"
    echo "attn_temp=${ATTN_TEMP}"
    echo "time=$(date -Iseconds)"
  } > "${status_file}"

  if [[ ${rc} -ne 0 ]]; then
    echo "[error] case ${case_name} failed with rc=${rc}"
    exit ${rc}
  fi
}

if [[ "${CASE_SET}" == "b1" || "${CASE_SET}" == "B1" ]]; then
  run_case "B1_real" "configs/experiments/text_modality_crossattn/crossattn_B1_real_proto.yaml"
else
  run_case "B1_real" "configs/experiments/text_modality_crossattn/crossattn_B1_real_proto.yaml"
  run_case "B4_shuffle" "configs/experiments/text_modality_crossattn/crossattn_B4_shuffle_proto.yaml"
  run_case "B5_random" "configs/experiments/text_modality_crossattn/crossattn_B5_random_proto.yaml"
fi

echo "All cross-attn diagnose50 runs finished."
echo "Logs: ${RUN_DIR}"
