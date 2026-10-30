#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

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
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/patch_internal}"
MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-9}"
MEM_OFF_FLAG="--no-use_memory_compression"
PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG:---no-pl_shuffle}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"

mkdir -p "${OUT_ROOT}"

run_one() {
  local tag="$1"
  local use_sched="$2"
  local mode="$3"
  local hi="$4"
  local lo="$5"
  local topk="$6"

  local run_name="${DATASET}_seed${SEED}_bank${MEMORY_BANK_SIZE}_comp-none_loc-attn_sched-${tag}"
  local run_dir="${OUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}/logs" "${run_dir}/ckpts" "${run_dir}/metrics" "${run_dir}/summaries"

  local subset_arg=""
  if [[ -n "${PL_SUBSET_FILE}" ]]; then
    subset_arg="--pl_subset_file ${PL_SUBSET_FILE}"
  fi

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
    --num_maskmem "${MEMORY_BANK_SIZE}" \
    --memory_bank_size "${MEMORY_BANK_SIZE}" \
    --pd_root "${run_dir}" \
    --tb_log_dir "${run_dir}/logs/tb" \
    --tgt_ckpt "${TGT_CKPT}" \
    --src_ckpt "${SRC_CKPT}" \
    ${PL_SHUFFLE_FLAG} ${MEM_OFF_FLAG} ${use_sched} \
    --patch_scheduler_mode "${mode}" \
    --patch_score_source hybrid \
    --adaptive_stride_high_complexity "${hi}" \
    --adaptive_stride_low_complexity "${lo}" \
    --patch_topk "${topk}" \
    --adaptive_apply_stage pseudo_label \
    ${subset_arg} \
    > "${run_dir}/logs/train.log" 2>&1
}

# off
run_one "off" "--no-use_patch_scheduler" "none" 384 768 0
# default
run_one "default" "--use_patch_scheduler" "adaptive_stride" 384 768 0
# budget_small / medium / large
run_one "budget-small" "--use_patch_scheduler" "adaptive_stride" 384 1024 0
run_one "budget-medium" "--use_patch_scheduler" "adaptive_stride" 384 768 0
run_one "budget-large" "--use_patch_scheduler" "adaptive_stride" 256 512 0

echo "Patch internal ablation launch finished. Outputs: ${OUT_ROOT}"
