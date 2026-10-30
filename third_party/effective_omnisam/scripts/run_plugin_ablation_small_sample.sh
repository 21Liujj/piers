#!/usr/bin/env bash
# 小样本插件消融（base / plus1 / plus4 / plus5 / all5）
# - 支持后台运行：RUN_IN_BACKGROUND=1（默认）
# - 日志目录：LOG_DIR（默认 ./logs）
# - 主日志：LOG_FILE（默认 logs/plugin_ablation_small_<timestamp>.master.log）
# - PID 文件：PID_FILE（默认 logs/plugin_ablation_small.pid）
#
# 推荐：
#   cd /root/autodl-tmp/OmniSAM-main
#   RUN_IN_BACKGROUND=1 \
#   SRC_CKPT=/root/autodl-tmp/omnisam_s.pth TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
#   bash scripts/run_plugin_ablation_small_sample.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

INTERNAL_BG="${INTERNAL_BG:-0}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

LOG_DIR="${LOG_DIR:-${ROOT}/logs/omnisam_time_ablation}"
mkdir -p "${LOG_DIR}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/plugin_ablation_small_${STAMP}.master.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/plugin_ablation_small.pid}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- user knobs ----
GPU="${GPU:-0}"
SEED="${SEED:-0}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
DATA_ROOT="${DATA_ROOT:-data}"
PD_ROOT="${PD_ROOT:-data_pd_exp_plugin_small}"
SRC_CKPT="${SRC_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"
TGT_CKPT="${TGT_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"

# Speed knobs (recommend enabling on 5090)
AMP="${AMP:-true}"
TF32="${TF32:-true}"

# Multimodal knobs (paper-aligned default: disable)
USE_MULTIMODAL="${USE_MULTIMODAL:-0}"
TEXT_PROTO_PATH="${TEXT_PROTO_PATH:-}"
LAMBDA_TXT="${LAMBDA_TXT:-0.0}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.0}"
USE_TEXT_GATE_SSL="${USE_TEXT_GATE_SSL:-false}"

# small-sample settings
QUICK_IMAGES="${QUICK_IMAGES:-20}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-5}"
WARMUP_PL="${WARMUP_PL:-${QUICK_IMAGES}}"
ITERS_PL="${ITERS_PL:-${QUICK_IMAGES}}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-4}"
PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG:---no-pl_shuffle}"

SUBSET_DIR="${LOG_DIR}/plugin_ablation_subsets"
mkdir -p "${SUBSET_DIR}"

# WildPASS-for-PL uses densepass_list/train.txt in this repo.
INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-${SUBSET_DIR}/${DATASET}_${CLASS_SET}cls_seed${SEED}_small${QUICK_IMAGES}.txt}"

_run_one () {
  local tag="$1"
  local cfg="$2"
  local run_log="${LOG_DIR}/plugin_ablation_${tag}_${STAMP}.log"

  echo "----------------------------------------------"
  echo "[run] ${tag}"
  echo "cfg=${cfg}"
  echo "log=${run_log}"
  echo "----------------------------------------------"

  # BooleanOptionalAction flags: must be passed as --flag or --no-flag.
  local amp_flag="--no-amp"
  local tf32_flag="--no-tf32"
  local use_text_proto_flag="--no-use_text_proto"
  local use_text_gate_ssl_flag="--no-use_text_gate_ssl"
  local use_wandb_flag="--no-use_wandb"
  local use_tb_flag="--no-use_tensorboard"
  if [[ "${AMP}" == "true" || "${AMP}" == "1" ]]; then amp_flag="--amp"; fi
  if [[ "${TF32}" == "true" || "${TF32}" == "1" ]]; then tf32_flag="--tf32"; fi
  if [[ "${USE_MULTIMODAL}" == "1" || "${USE_MULTIMODAL}" == "true" ]]; then use_text_proto_flag="--use_text_proto"; fi
  if [[ "${USE_TEXT_GATE_SSL}" == "true" || "${USE_TEXT_GATE_SSL}" == "1" ]]; then use_text_gate_ssl_flag="--use_text_gate_ssl"; fi
  # Logging (keep off by default for clean time-ablation logs)
  if [[ "${USE_WANDB:-0}" == "1" || "${USE_WANDB:-0}" == "true" ]]; then use_wandb_flag="--use_wandb"; fi
  if [[ "${USE_TENSORBOARD:-0}" == "1" || "${USE_TENSORBOARD:-0}" == "true" ]]; then use_tb_flag="--use_tensorboard"; fi

  # Note: run_adaptation.py uses CUDA if available; GPU env is provided for compatibility with other scripts.
  env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    python run_adaptation.py \
      --config "${cfg}" \
      --seed "${SEED}" \
      --dataset "${DATASET}" \
      --class_set "${CLASS_SET}" \
      --backbone "${BACKBONE}" \
      --data_root "${DATA_ROOT}" \
      --pd_root "${PD_ROOT}/${tag}" \
      --src_ckpt "${SRC_CKPT}" \
      --tgt_ckpt "${TGT_CKPT}" \
      ${amp_flag} \
      ${tf32_flag} \
      --num_epochs "${NUM_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --pl_batch_size "${PL_BATCH_SIZE}" \
      --pl_num_workers "${PL_NUM_WORKERS}" \
      ${PL_SHUFFLE_FLAG} \
      --pl_subset_file "${PL_SUBSET_FILE}" \
      --warmup_pl "${WARMUP_PL}" \
      --iters_pl "${ITERS_PL}" \
      --train_max_iters "${TRAIN_MAX_ITERS}" \
      ${use_text_proto_flag} \
      --text_proto_path "${TEXT_PROTO_PATH}" \
      --lambda_txt "${LAMBDA_TXT}" \
      --weight_txt_align "${WEIGHT_TXT_ALIGN}" \
      ${use_text_gate_ssl_flag} \
      ${use_wandb_flag} \
      ${use_tb_flag} \
      >> "${run_log}" 2>&1
  local rc=$?
  if [[ "${rc}" != "0" ]]; then
    echo "[ERROR] run failed: tag=${tag} rc=${rc}" | tee -a "${run_log}"
    echo "[ERROR] stop whole ablation due to failure (see ${run_log})"
    exit "${rc}"
  fi

  echo "[done] ${tag} finished at $(date -Iseconds)"
}

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env \
    INTERNAL_BG=1 \
    RUN_IN_BACKGROUND=1 \
    LOG_DIR="${LOG_DIR}" \
    LOG_FILE="${LOG_FILE}" \
    PID_FILE="${PID_FILE}" \
    STAMP="${STAMP}" \
    GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
    DATA_ROOT="${DATA_ROOT}" PD_ROOT="${PD_ROOT}" SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" \
    QUICK_IMAGES="${QUICK_IMAGES}" NUM_EPOCHS="${NUM_EPOCHS}" TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
    WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
    PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG}" INPUT_LIST="${INPUT_LIST}" PL_SUBSET_FILE="${PL_SUBSET_FILE}" \
    WANDB_MODE="${WANDB_MODE}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    bash "${BASH_SOURCE[0]}" > "${LOG_FILE}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "已后台启动小样本插件消融"
  echo "PID: ${pid} (保存于 ${PID_FILE})"
  echo "主日志: ${LOG_FILE}"
  echo "查看: tail -f ${LOG_FILE}"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========== Plugin Ablation Small Sample =========="
echo "start: $(date -Iseconds)"
echo "ROOT=${ROOT}"
echo "GPU=${GPU} (CUDA_VISIBLE_DEVICES)"
echo "DATASET=${DATASET}, CLASS_SET=${CLASS_SET}, BACKBONE=${BACKBONE}"
echo "SEED=${SEED}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PD_ROOT=${PD_ROOT}"
echo "SRC_CKPT=${SRC_CKPT}"
echo "TGT_CKPT=${TGT_CKPT}"
echo "AMP=${AMP}, TF32=${TF32}"
echo "USE_MULTIMODAL=${USE_MULTIMODAL}"
echo "TEXT_PROTO_PATH=${TEXT_PROTO_PATH}"
echo "LAMBDA_TXT=${LAMBDA_TXT}, WEIGHT_TXT_ALIGN=${WEIGHT_TXT_ALIGN}, USE_TEXT_GATE_SSL=${USE_TEXT_GATE_SSL}"
echo "QUICK_IMAGES=${QUICK_IMAGES}, NUM_EPOCHS=${NUM_EPOCHS}, TRAIN_MAX_ITERS=${TRAIN_MAX_ITERS}"
echo "WARMUP_PL=${WARMUP_PL}, ITERS_PL=${ITERS_PL}"
echo "BATCH_SIZE=${BATCH_SIZE}, PL_BATCH_SIZE=${PL_BATCH_SIZE}, PL_NUM_WORKERS=${PL_NUM_WORKERS}"
echo "PL_SUBSET_FILE=${PL_SUBSET_FILE}"
echo "LOG_DIR=${LOG_DIR}"
echo "MASTER_LOG=${LOG_FILE}"
echo "PID_FILE=${PID_FILE}"
echo "==============================================="

python "${ROOT}/scripts/time_ablation/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${PL_SUBSET_FILE}" \
  --num_images "${QUICK_IMAGES}" \
  --offset 0 \
  --shuffle \
  --seed "${SEED}"

_run_one "all5_first" "configs/ablations/omnisam_all5_full.yaml"
_run_one "base" "configs/ablations/omnisam_base.yaml"
_run_one "base_plus1_confmask" "configs/ablations/omnisam_base_plus1_confmask.yaml"
_run_one "base_plus4_efficienttam" "configs/ablations/omnisam_base_plus4_efficienttam.yaml"
_run_one "base_plus5_efficientsam2_sparse" "configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml"
_run_one "all5_last" "configs/ablations/omnisam_all5_full.yaml"

echo "[all done] finished at $(date -Iseconds)"
echo "tips:"
echo "  tail -f ${LOG_FILE}"
echo "  ls -lh ${LOG_DIR}/plugin_ablation_*_${STAMP}.log"

