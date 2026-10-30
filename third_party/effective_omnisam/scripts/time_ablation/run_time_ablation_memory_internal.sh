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
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/memory_internal}"
MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-9}"
APS_OFF_FLAG="--no-use_patch_scheduler"
PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG:---no-pl_shuffle}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"

mkdir -p "${OUT_ROOT}"

run_one() {
  local mode="$1"      # none/pool/spatial/hybrid
  local loc="$2"       # before_attention/before_bank
  local use_mem="$3"   # --use_memory_compression/--no-use_memory_compression
  local run_name="${DATASET}_seed${SEED}_bank${MEMORY_BANK_SIZE}_comp-${mode}_loc-${loc}_sched-off"
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
    ${PL_SHUFFLE_FLAG} ${use_mem} ${APS_OFF_FLAG} \
    --memory_select_mode hybrid \
    --memory_recent_keep 4 \
    --memory_score_keep 4 \
    --memory_compress_mode "${mode}" \
    --memory_compress_location "${loc}" \
    --memory_spatial_window_size 2 \
    --memory_global_summary_tokens 4 \
    ${subset_arg} \
    > "${run_dir}/logs/train.log" 2>&1
}

# none (single no-compression baseline)
run_one "none" "before_attention" "--no-use_memory_compression"
# pool/spatial/hybrid on both insertion locations
for mode in pool spatial hybrid; do
  for loc in before_attention before_bank; do
    run_one "${mode}" "${loc}" "--use_memory_compression"
  done
done

echo "Memory internal ablation launch finished. Outputs: ${OUT_ROOT}"
