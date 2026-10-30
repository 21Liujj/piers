#!/usr/bin/env bash
set -euo pipefail

# Fixed200 semantic-causality ablation (no gate):
# A0 baseline / A1 real / A4 shuffle / A5 random
#
# Logging style:
# - master log: logs/text_fusion_fixed200/<run_name>/00_master.log
# - per-case log: <run_dir>/<case>.log
# - per-case status: <run_dir>/<case>.status
# - pid file: <run_dir>/run.pid

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
INTERNAL_BG="${INTERNAL_BG:-0}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/text_fusion_fixed200}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
MASTER_LOG="${RUN_DIR}/00_master.log"
PID_FILE="${RUN_DIR}/run.pid"

DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-1}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-${BATCH_SIZE}}"
TEXT_PL_BATCH_SIZE="${TEXT_PL_BATCH_SIZE:-${PL_BATCH_SIZE}}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"

INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
SUBSET_NUM="${SUBSET_NUM:-200}"
SUBSET_FILE="${SUBSET_FILE:-${RUN_DIR}/subsets/train_fixed${SUBSET_NUM}.txt}"
WARMUP_PL="${WARMUP_PL:-${SUBSET_NUM}}"
ITERS_PL="${ITERS_PL:-${SUBSET_NUM}}"

TAU_T="${TAU_T:-0.10}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.02}"
TXT_TAU="${TXT_TAU:-0.07}"
LAMBDA_TXT="${LAMBDA_TXT:-0.1}"
TXT_ALIGN_TAU="${TXT_ALIGN_TAU:-1.0}"
TEXT_ALPHA_INIT="${TEXT_ALPHA_INIT:-}"
TEXT_ADAPTER_LR_MULT="${TEXT_ADAPTER_LR_MULT:-1.0}"
TEXT_FUSION_ALPHA_LR_MULT="${TEXT_FUSION_ALPHA_LR_MULT:-1.0}"
TEXT_SCALE_LR_MULT="${TEXT_SCALE_LR_MULT:-1.0}"
TEXT_LOG_INTERVAL="${TEXT_LOG_INTERVAL:-50}"

MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY:-1}"
START_FROM="${START_FROM:-A0_baseline}" # A0_baseline | A1_text_branch | A4_shuffle_proto | A5_random_proto
RESET_CASE_DIR_ON_RERUN="${RESET_CASE_DIR_ON_RERUN:-1}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --class_set "${CLASS_SET}"
  --backbone "${BACKBONE}"
  --num_epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --pl_batch_size "${PL_BATCH_SIZE}"
  --pl_num_workers "${PL_NUM_WORKERS}"
  --data_root "${DATA_ROOT}"
  --src_ckpt "${SRC_CKPT}"
  --tgt_ckpt "${TGT_CKPT}"
  --amp
  --tf32
)

mkdir -p "${RUN_DIR}"

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 \
    LOG_ROOT="${LOG_ROOT}" RUN_NAME="${RUN_NAME}" \
    DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
    EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
    DATA_ROOT="${DATA_ROOT}" SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" \
    INPUT_LIST="${INPUT_LIST}" SUBSET_NUM="${SUBSET_NUM}" SUBSET_FILE="${SUBSET_FILE}" \
    WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    TAU_T="${TAU_T}" WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" TXT_TAU="${TXT_TAU}" LAMBDA_TXT="${LAMBDA_TXT}" TXT_ALIGN_TAU="${TXT_ALIGN_TAU}" \
    MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Started in background."
  echo "PID: ${pid} (saved to ${PID_FILE})"
  echo "Master log: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Text Semantic Fixed200 =========="
echo "run_dir=${RUN_DIR}"
echo "subset_file=${SUBSET_FILE}"
echo "subset_num=${SUBSET_NUM}, epochs=${EPOCHS}"
echo "batch(base)=${BATCH_SIZE}, pl_batch(base)=${PL_BATCH_SIZE}"
echo "batch(text)=${TEXT_BATCH_SIZE}, pl_batch(text)=${TEXT_PL_BATCH_SIZE}, pl_workers=${PL_NUM_WORKERS}"
echo "tau_t=${TAU_T}, weight_txt_align=${WEIGHT_TXT_ALIGN}"
echo "text_alpha_init=${TEXT_ALPHA_INIT:-default}, text_log_interval=${TEXT_LOG_INTERVAL}"
echo "text_lr_mult: adapter=${TEXT_ADAPTER_LR_MULT}, alpha=${TEXT_FUSION_ALPHA_LR_MULT}, scale=${TEXT_SCALE_LR_MULT}"
echo "mem_compress_train_only=${MEM_COMPRESS_TRAIN_ONLY}"
echo "start_from=${START_FROM}, reset_case_dir_on_rerun=${RESET_CASE_DIR_ON_RERUN}"
echo "============================================"

mkdir -p "$(dirname "${SUBSET_FILE}")"
if [[ ! -f "${SUBSET_FILE}" ]]; then
  python scripts/time_ablation/build_fixed_subset.py \
    --input_list "${INPUT_LIST}" \
    --output_list "${SUBSET_FILE}" \
    --num_images "${SUBSET_NUM}" \
    --offset 0
fi
echo "[subset] ready: ${SUBSET_FILE}"

run_case() {
  local case_name="$1"
  local cfg="$2"
  local extra_opts="$3"
  local pd_root="${RUN_DIR}/pd_${case_name}"
  local log_file="${RUN_DIR}/${case_name}.log"
  local status_file="${RUN_DIR}/${case_name}.status"

  local merged_opts
  merged_opts="pl_subset_file=${SUBSET_FILE},warmup_pl=${WARMUP_PL},iters_pl=${ITERS_PL},tau_t=${TAU_T},weight_txt_align=${WEIGHT_TXT_ALIGN},txt_tau=${TXT_TAU},lambda_txt=${LAMBDA_TXT},txt_align_tau=${TXT_ALIGN_TAU}"
  merged_opts="${merged_opts},text_adapter_lr_mult=${TEXT_ADAPTER_LR_MULT},text_fusion_alpha_lr_mult=${TEXT_FUSION_ALPHA_LR_MULT},text_scale_lr_mult=${TEXT_SCALE_LR_MULT},text_log_interval=${TEXT_LOG_INTERVAL}"
  if [[ "${MEM_COMPRESS_TRAIN_ONLY}" == "1" ]]; then
    merged_opts="${merged_opts},mem_compression_enable_in_train=true,mem_compression_enable_in_pseudo_label=false"
  fi
  if [[ "${case_name}" != "A0_baseline" && -n "${TEXT_ALPHA_INIT}" ]]; then
    merged_opts="${merged_opts},text_fusion_alpha_init=${TEXT_ALPHA_INIT}"
  fi
  if [[ -n "${extra_opts}" ]]; then
    merged_opts="${merged_opts},${extra_opts}"
  fi

  local case_batch_size="${TEXT_BATCH_SIZE}"
  local case_pl_batch_size="${TEXT_PL_BATCH_SIZE}"
  if [[ "${case_name}" == "A0_baseline" ]]; then
    case_batch_size="${BATCH_SIZE}"
    case_pl_batch_size="${PL_BATCH_SIZE}"
  fi

  if [[ "${RESET_CASE_DIR_ON_RERUN}" == "1" && -d "${pd_root}" ]]; then
    echo "[run] reset existing pd_root: ${pd_root}"
    rm -rf "${pd_root}"
  fi

  echo "[run] ${case_name}"
  echo "[run] cfg=${cfg}"
  echo "[run] pd_root=${pd_root}"
  echo "[run] batch_size=${case_batch_size}, pl_batch_size=${case_pl_batch_size}"
  echo "[run] opts=${merged_opts}"

  set +e
  python run_adaptation.py \
    "${COMMON_ARGS[@]}" \
    --batch_size "${case_batch_size}" \
    --pl_batch_size "${case_pl_batch_size}" \
    --config "${cfg}" \
    --pd_root "${pd_root}" \
    --opts "${merged_opts}" \
    2>&1 | tee "${log_file}"
  rc=${PIPESTATUS[0]}
  set -e

  {
    echo "case=${case_name}"
    echo "rc=${rc}"
    echo "log=${log_file}"
    echo "pd_root=${pd_root}"
    echo "time=$(date -Iseconds)"
  } > "${status_file}"

  if [[ ${rc} -ne 0 ]]; then
    echo "[error] case ${case_name} failed with rc=${rc}"
    exit ${rc}
  fi
}

cases=(
  "A0_baseline|configs/experiments/text_modality/text_ablation_A0_baseline.yaml|"
  "A1_text_branch|configs/experiments/text_modality/text_ablation_A1_text_branch.yaml|"
  "A4_shuffle_proto|configs/experiments/text_modality/text_ablation_A4_shuffle_proto.yaml|"
  "A5_random_proto|configs/experiments/text_modality/text_ablation_A5_random_proto.yaml|"
)

start_found=0
for item in "${cases[@]}"; do
  IFS='|' read -r tag cfg extra_opts <<< "${item}"
  if [[ "${start_found}" == "0" ]]; then
    if [[ "${tag}" == "${START_FROM}" ]]; then
      start_found=1
    else
      continue
    fi
  fi
  run_case "${tag}" "${cfg}" "${extra_opts}"
done

if [[ "${start_found}" == "0" ]]; then
  echo "[error] START_FROM=${START_FROM} not found in case list."
  exit 1
fi

echo "All fixed200 semantic runs finished."
echo "Logs: ${RUN_DIR}"
