#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

# ----------------------------
# User-tunable shared controls
# ----------------------------
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-4}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
AMP="${AMP:---amp}"
TF32="${TF32:---tf32}"
PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG:---no-pl_shuffle}"
WARMUP_PL="${WARMUP_PL:-0}"
ITERS_PL="${ITERS_PL:-0}"
REGENERATE_PL_EACH_EPOCH_FLAG="${REGENERATE_PL_EACH_EPOCH_FLAG:---no-regenerate_pl_each_epoch}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/main4}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
USE_MULTIMODAL="${USE_MULTIMODAL:-1}"
TEXT_PROTO_PATH="${TEXT_PROTO_PATH:-/root/autodl-tmp/OmniSAM-main/assets/text_prototypes_openclip_text_latest.pt}"
SSL_GATE_MODE="${SSL_GATE_MODE:-hard}"
TAU_T="${TAU_T:-0.10}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.02}"

# Fixed controls for fair timing comparison
MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-9}"
COMPRESS_MODE_DEFAULT="${COMPRESS_MODE_DEFAULT:-hybrid}"
COMPRESS_LOCATION_DEFAULT="${COMPRESS_LOCATION_DEFAULT:-before_attention}"
APS_MODE_DEFAULT="${APS_MODE_DEFAULT:-adaptive_stride}"
APS_SCORE_DEFAULT="${APS_SCORE_DEFAULT:-hybrid}"
APS_HIGH_STRIDE="${APS_HIGH_STRIDE:-384}"
APS_LOW_STRIDE="${APS_LOW_STRIDE:-768}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"  # 1: skip methods with existing metrics.jsonl
METHOD_ORDER="${METHOD_ORDER:-baseline mem_only aps_only both}" # space-separated method order
AUTO_CLEAN_UNUSED_PSEUDO="${AUTO_CLEAN_UNUSED_PSEUDO:-0}"       # 1: run cleanup helper before main loop
CLEANUP_DRY_RUN="${CLEANUP_DRY_RUN:-0}"                         # used when AUTO_CLEAN_UNUSED_PSEUDO=1
CLEANUP_MIN_FREE_GB="${CLEANUP_MIN_FREE_GB:-10}"

mkdir -p "${OUT_ROOT}"
MAIN_LOG="${OUT_ROOT}/main4_driver.log"
STATUS_CSV="${OUT_ROOT}/main4_status.csv"
TRAINLOG_BACKUP_DIR="${OUT_ROOT}/_trainlogs"
echo "method,exit_code,run_dir,train_log" > "${STATUS_CSV}"
mkdir -p "${TRAINLOG_BACKUP_DIR}"

log() {
  local msg="$1"
  echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] ${msg}" | tee -a "${MAIN_LOG}"
}

if [[ "${AUTO_CLEAN_UNUSED_PSEUDO}" == "1" ]]; then
  log "AUTO_CLEAN_UNUSED_PSEUDO enabled, start cleanup helper"
  DRY_RUN="${CLEANUP_DRY_RUN}" MIN_FREE_GB="${CLEANUP_MIN_FREE_GB}" \
    bash "${SCRIPT_DIR}/cleanup_unused_pseudo_labels.sh" | tee -a "${MAIN_LOG}"
fi

run_one() {
  local method="$1"            # baseline | mem_only | aps_only | both
  local use_mem="$2"           # --use_memory_compression / --no-use_memory_compression
  local use_aps="$3"           # --use_patch_scheduler / --no-use_patch_scheduler
  local comp_mode="$4"
  local sched_mode="$5"

  local run_name="${DATASET}_seed${SEED}_bank${MEMORY_BANK_SIZE}_comp-${comp_mode}_loc-${COMPRESS_LOCATION_DEFAULT}_sched-${sched_mode}"
  local run_dir="${OUT_ROOT}/${method}/${run_name}"
  mkdir -p "${run_dir}/logs" "${run_dir}/ckpts" "${run_dir}/metrics" "${run_dir}/summaries"
  local metrics_jsonl="${run_dir}/metrics.jsonl"
  local train_log="${run_dir}/logs/train.log"
  local train_log_backup="${TRAINLOG_BACKUP_DIR}/${method}_${run_name}.log"

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${metrics_jsonl}" ]]; then
    log "SKIP method=${method} existing metrics found: ${metrics_jsonl}"
    echo "${method},0,${run_dir},${train_log}" >> "${STATUS_CSV}"
    return 0
  fi

  local subset_arg=""
  if [[ -n "${PL_SUBSET_FILE}" ]]; then
    subset_arg="--pl_subset_file ${PL_SUBSET_FILE}"
  fi
  local mm_args=""
  if [[ "${USE_MULTIMODAL}" == "1" ]]; then
    mm_args="--text_proto_path ${TEXT_PROTO_PATH} --use_text_gate_ssl --ssl_gate_mode ${SSL_GATE_MODE} --tau_t ${TAU_T} --weight_txt_align ${WEIGHT_TXT_ALIGN}"
  fi

  log "START method=${method} run_dir=${run_dir}"
  set +e
  mkdir -p "${run_dir}/logs" "${run_dir}/metrics" "${run_dir}/summaries" "${run_dir}/ckpts"
  CUDA_VISIBLE_DEVICES="${GPU}" python run_adaptation.py \
    --dataset "${DATASET}" \
    --class_set "${CLASS_SET}" \
    --backbone "${BACKBONE}" \
    --seed "${SEED}" \
    --num_epochs "${NUM_EPOCHS}" \
    --train_max_iters "${TRAIN_MAX_ITERS}" \
    --batch_size "${BATCH_SIZE}" \
    --pl_batch_size "${PL_BATCH_SIZE}" \
    --pl_num_workers "${PL_NUM_WORKERS}" \
    --warmup_pl "${WARMUP_PL}" \
    --iters_pl "${ITERS_PL}" \
    ${REGENERATE_PL_EACH_EPOCH_FLAG} \
    --num_maskmem "${MEMORY_BANK_SIZE}" \
    --memory_bank_size "${MEMORY_BANK_SIZE}" \
    --pd_root "${run_dir}" \
    --tb_log_dir "${run_dir}/logs/tb" \
    --tgt_ckpt "${TGT_CKPT}" \
    --src_ckpt "${SRC_CKPT}" \
    ${AMP} ${TF32} ${PL_SHUFFLE_FLAG} \
    ${use_mem} \
    --memory_select_mode hybrid \
    --memory_recent_keep 4 \
    --memory_score_keep 4 \
    --memory_compress_mode "${comp_mode}" \
    --memory_compress_location "${COMPRESS_LOCATION_DEFAULT}" \
    --memory_spatial_window_size 2 \
    --memory_global_summary_tokens 4 \
    ${use_aps} \
    --patch_scheduler_mode "${sched_mode}" \
    --patch_score_source "${APS_SCORE_DEFAULT}" \
    --adaptive_stride_high_complexity "${APS_HIGH_STRIDE}" \
    --adaptive_stride_low_complexity "${APS_LOW_STRIDE}" \
    --adaptive_apply_stage pseudo_label \
    ${mm_args} \
    ${subset_arg} \
    2>&1 | tee "${train_log}" | tee -a "${train_log_backup}"
  local rc=${PIPESTATUS[0]}
  set -e
  log "END method=${method} rc=${rc} train_log=${train_log}"
  echo "${method},${rc},${run_dir},${train_log}" >> "${STATUS_CSV}"
  return 0
}

for method in ${METHOD_ORDER}; do
  case "${method}" in
    baseline)
      run_one "baseline" "--no-use_memory_compression" "--no-use_patch_scheduler" "none" "none"
      ;;
    mem_only)
      run_one "mem_only" "--use_memory_compression" "--no-use_patch_scheduler" "${COMPRESS_MODE_DEFAULT}" "none"
      ;;
    aps_only)
      run_one "aps_only" "--no-use_memory_compression" "--use_patch_scheduler" "none" "${APS_MODE_DEFAULT}"
      ;;
    both)
      run_one "both" "--use_memory_compression" "--use_patch_scheduler" "${COMPRESS_MODE_DEFAULT}" "${APS_MODE_DEFAULT}"
      ;;
    *)
      log "WARN unknown method in METHOD_ORDER: ${method} (skip)"
      ;;
  esac
done

fail_cnt="$(awk -F',' 'NR>1 && $2 != 0 {c++} END{print c+0}' "${STATUS_CSV}")"
if [[ "${fail_cnt}" -gt 0 ]]; then
  log "Main-4 finished with ${fail_cnt} failed runs. See ${STATUS_CSV}"
else
  log "Main-4 finished successfully. Outputs: ${OUT_ROOT}"
fi
echo "Main-4 status csv: ${STATUS_CSV}"
