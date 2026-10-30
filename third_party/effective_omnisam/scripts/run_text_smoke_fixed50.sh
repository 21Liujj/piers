#!/usr/bin/env bash
set -euo pipefail

# One-click text modality smoke + fixed50 ablation runner.
# Stage-0: ensure fixed50 subset file
# Stage-1: A1 smoke
# Stage-2: semantic causality (A0/A1/A4/A5)
# Stage-3: gate efficacy (A2/A3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
INTERNAL_BG="${INTERNAL_BG:-0}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/text_fusion}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/${STAMP}"
MASTER_LOG="${RUN_DIR}/00_master.log"
PID_FILE="${RUN_DIR}/run.pid"

DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-3}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"

INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
SUBSET_FILE="${SUBSET_FILE:-${RUN_DIR}/subsets/train_fixed50.txt}"
SUBSET_NUM="${SUBSET_NUM:-50}"
WARMUP_PL="${WARMUP_PL:-${SUBSET_NUM}}"
ITERS_PL="${ITERS_PL:-${SUBSET_NUM}}"

TAU_T="${TAU_T:-0.10}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.02}"
TXT_TAU="${TXT_TAU:-0.07}"
LAMBDA_TXT="${LAMBDA_TXT:-0.1}"
TXT_ALIGN_TAU="${TXT_ALIGN_TAU:-1.0}"
START_FROM="${START_FROM:-A1_smoke}" # A1_smoke | A0_baseline | A1_text_branch | A4_shuffle_proto | A5_random_proto | A2_text_branch_gate_soft | A3_text_branch_gate_hard
MEM_COMPRESS_TRAIN_ONLY="${MEM_COMPRESS_TRAIN_ONLY:-1}"

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
    LOG_ROOT="${LOG_ROOT}" \
    DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
    EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
    DATA_ROOT="${DATA_ROOT}" SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" \
    INPUT_LIST="${INPUT_LIST}" SUBSET_FILE="${SUBSET_FILE}" SUBSET_NUM="${SUBSET_NUM}" \
    TAU_T="${TAU_T}" WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" TXT_TAU="${TXT_TAU}" LAMBDA_TXT="${LAMBDA_TXT}" TXT_ALIGN_TAU="${TXT_ALIGN_TAU}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Started in background."
  echo "PID: ${pid} (saved to ${PID_FILE})"
  echo "Master log: ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Text Ablation Fixed50 =========="
echo "run_dir=${RUN_DIR}"
echo "log_root=${LOG_ROOT}"
echo "subset_file=${SUBSET_FILE}"
echo "batch_size=${BATCH_SIZE}, pl_batch_size=${PL_BATCH_SIZE}, amp/tf32=on"
echo "warmup_pl=${WARMUP_PL}, iters_pl=${ITERS_PL}"
echo "tau_t=${TAU_T}, weight_txt_align=${WEIGHT_TXT_ALIGN}"
echo "start_from=${START_FROM}"
echo "mem_compress_train_only=${MEM_COMPRESS_TRAIN_ONLY}"
echo "==========================================="

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
  local tag="$1"
  local cfg="$2"
  local opts="$3"
  local pd_root="${RUN_DIR}/pd_${tag}"
  local log_file="${RUN_DIR}/${tag}.log"
  echo "----- RUN ${tag} -----"
  local merged_opts
  merged_opts="pl_subset_file=${SUBSET_FILE},warmup_pl=${WARMUP_PL},iters_pl=${ITERS_PL},tau_t=${TAU_T},weight_txt_align=${WEIGHT_TXT_ALIGN},txt_tau=${TXT_TAU},lambda_txt=${LAMBDA_TXT},txt_align_tau=${TXT_ALIGN_TAU}"
  if [[ "${MEM_COMPRESS_TRAIN_ONLY}" == "1" ]]; then
    merged_opts="${merged_opts},mem_compression_enable_in_train=true,mem_compression_enable_in_pseudo_label=false"
  fi
  if [[ -n "${opts}" ]]; then
    merged_opts="${merged_opts},${opts}"
  fi
  python run_adaptation.py \
    "${COMMON_ARGS[@]}" \
    --config "${cfg}" \
    --pd_root "${pd_root}" \
    --opts "${merged_opts}" \
    2>&1 | tee "${log_file}"
}

cases=(
  "A1_smoke|configs/experiments/text_modality/text_ablation_A1_text_branch.yaml|"
  "A0_baseline|configs/experiments/text_modality/text_ablation_A0_baseline.yaml|"
  "A1_text_branch|configs/experiments/text_modality/text_ablation_A1_text_branch.yaml|"
  "A4_shuffle_proto|configs/experiments/text_modality/text_ablation_A4_shuffle_proto.yaml|"
  "A5_random_proto|configs/experiments/text_modality/text_ablation_A5_random_proto.yaml|"
  "A2_text_branch_gate_soft|configs/experiments/text_modality/text_ablation_A2_text_branch_gate_soft.yaml|"
  "A3_text_branch_gate_hard|configs/experiments/text_modality/text_ablation_A3_text_branch_gate_hard.yaml|"
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

echo "All runs finished."
echo "Logs: ${RUN_DIR}"
