#!/usr/bin/env bash
# 小样本消融：仅运行 base+5（efficient_sam2_sparse），且仅训练阶段开启（PL 阶段关闭）
# 与 run_plugin_ablation_base_only.sh 同条件，便于公平对比。
#
# 用法示例：
#   cd /root/autodl-tmp/OmniSAM-main
#   RUN_IN_BACKGROUND=1 \
#   SRC_CKPT=/root/autodl-tmp/omnisam_s.pth TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
#   bash scripts/run_plugin_ablation_base_plus5_train_only.sh
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
MASTER_LOG="${MASTER_LOG:-${LOG_DIR}/plugin_ablation_base_plus5_train_only_${STAMP}.master.log}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/plugin_ablation_base_plus5_train_only_${STAMP}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/plugin_ablation_base_plus5_train_only.pid}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- user knobs (默认与主消融脚本保持一致) ----
GPU="${GPU:-0}"
SEED="${SEED:-1}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
DATA_ROOT="${DATA_ROOT:-data}"
PD_ROOT="${PD_ROOT:-data_pd_exp_plugin_small}"
SRC_CKPT="${SRC_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"
TGT_CKPT="${TGT_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"

AMP="${AMP:-true}"
TF32="${TF32:-true}"

USE_MULTIMODAL="${USE_MULTIMODAL:-0}"
TEXT_PROTO_PATH="${TEXT_PROTO_PATH:-}"
LAMBDA_TXT="${LAMBDA_TXT:-0.0}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.0}"
USE_TEXT_GATE_SSL="${USE_TEXT_GATE_SSL:-false}"

QUICK_IMAGES="${QUICK_IMAGES:-50}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-100}"
WARMUP_PL="${WARMUP_PL:-${QUICK_IMAGES}}"
ITERS_PL="${ITERS_PL:-${QUICK_IMAGES}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-4}"
PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG:---no-pl_shuffle}"

SUBSET_DIR="${LOG_DIR}/plugin_ablation_subsets"
mkdir -p "${SUBSET_DIR}"
INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-${SUBSET_DIR}/${DATASET}_${CLASS_SET}cls_seed${SEED}_small${QUICK_IMAGES}.txt}"

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env \
    INTERNAL_BG=1 \
    RUN_IN_BACKGROUND=1 \
    LOG_DIR="${LOG_DIR}" \
    STAMP="${STAMP}" \
    MASTER_LOG="${MASTER_LOG}" \
    LOG_FILE="${LOG_FILE}" \
    PID_FILE="${PID_FILE}" \
    GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
    DATA_ROOT="${DATA_ROOT}" PD_ROOT="${PD_ROOT}" SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" \
    AMP="${AMP}" TF32="${TF32}" \
    USE_MULTIMODAL="${USE_MULTIMODAL}" TEXT_PROTO_PATH="${TEXT_PROTO_PATH}" \
    LAMBDA_TXT="${LAMBDA_TXT}" WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" USE_TEXT_GATE_SSL="${USE_TEXT_GATE_SSL}" \
    QUICK_IMAGES="${QUICK_IMAGES}" NUM_EPOCHS="${NUM_EPOCHS}" TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
    WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
    PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG}" INPUT_LIST="${INPUT_LIST}" PL_SUBSET_FILE="${PL_SUBSET_FILE}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "已后台启动 base+5(train_only) 消融"
  echo "PID: ${pid} (保存于 ${PID_FILE})"
  echo "主日志: ${MASTER_LOG}"
  echo "实验日志: ${LOG_FILE}"
  echo "查看: tail -f ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Plugin Ablation Base+5 Train Only =========="
echo "start: $(date -Iseconds)"
echo "ROOT=${ROOT}"
echo "GPU=${GPU}"
echo "SEED=${SEED}"
echo "DATASET=${DATASET}, CLASS_SET=${CLASS_SET}, BACKBONE=${BACKBONE}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PD_ROOT=${PD_ROOT}"
echo "SRC_CKPT=${SRC_CKPT}"
echo "TGT_CKPT=${TGT_CKPT}"
echo "AMP=${AMP}, TF32=${TF32}"
echo "USE_MULTIMODAL=${USE_MULTIMODAL}"
echo "QUICK_IMAGES=${QUICK_IMAGES}, NUM_EPOCHS=${NUM_EPOCHS}, TRAIN_MAX_ITERS=${TRAIN_MAX_ITERS}"
echo "WARMUP_PL=${WARMUP_PL}, ITERS_PL=${ITERS_PL}"
echo "BATCH_SIZE=${BATCH_SIZE}, PL_BATCH_SIZE=${PL_BATCH_SIZE}, PL_NUM_WORKERS=${PL_NUM_WORKERS}"
echo "PL_SUBSET_FILE=${PL_SUBSET_FILE}"
echo "MASTER_LOG=${MASTER_LOG}"
echo "LOG_FILE=${LOG_FILE}"
echo "========================================================"

python "${ROOT}/scripts/time_ablation/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${PL_SUBSET_FILE}" \
  --num_images "${QUICK_IMAGES}" \
  --offset 0 \
  --shuffle \
  --seed "${SEED}"

amp_flag="--no-amp"
tf32_flag="--no-tf32"
use_text_proto_flag="--no-use_text_proto"
use_text_gate_ssl_flag="--no-use_text_gate_ssl"
use_wandb_flag="--no-use_wandb"
use_tb_flag="--no-use_tensorboard"

if [[ "${AMP}" == "true" || "${AMP}" == "1" ]]; then amp_flag="--amp"; fi
if [[ "${TF32}" == "true" || "${TF32}" == "1" ]]; then tf32_flag="--tf32"; fi
if [[ "${USE_MULTIMODAL}" == "1" || "${USE_MULTIMODAL}" == "true" ]]; then use_text_proto_flag="--use_text_proto"; fi
if [[ "${USE_TEXT_GATE_SSL}" == "true" || "${USE_TEXT_GATE_SSL}" == "1" ]]; then use_text_gate_ssl_flag="--use_text_gate_ssl"; fi
if [[ "${USE_WANDB:-0}" == "1" || "${USE_WANDB:-0}" == "true" ]]; then use_wandb_flag="--use_wandb"; fi
if [[ "${USE_TENSORBOARD:-0}" == "1" || "${USE_TENSORBOARD:-0}" == "true" ]]; then use_tb_flag="--use_tensorboard"; fi

echo "[run] base_plus5_train_only"
echo "cfg=configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml + opts(apply_in_train=true, apply_in_pl=false)"
echo "log=${LOG_FILE}"

env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python run_adaptation.py \
    --config "configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml" \
    --seed "${SEED}" \
    --dataset "${DATASET}" \
    --class_set "${CLASS_SET}" \
    --backbone "${BACKBONE}" \
    --data_root "${DATA_ROOT}" \
    --pd_root "${PD_ROOT}/base_plus5_train_only_${STAMP}" \
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
    --opts plugins.efficient_sam2_sparse.apply_in_train=true,plugins.efficient_sam2_sparse.apply_in_pl=false \
    > "${LOG_FILE}" 2>&1

echo "[done] base_plus5_train_only finished at $(date -Iseconds)"
echo "实验日志: ${LOG_FILE}"
