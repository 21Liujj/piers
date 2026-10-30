zhe1l#!/usr/bin/env bash
# 小样本三组消融（seed=1），并在最后再跑一次 base 用于抵消冷启动：
# 1) base_first
# 2) base + 1 + 4
# 3) base + 1 + 4 + 5 (train_only: E仅训练开，PL关)
# 4) base_last
#
# 设计目标：
# - 与既有消融脚本风格一致（支持后台、主日志+子日志、PID、固定subset）
# - 默认另起日志目录与输出目录，避免污染历史实验
#
# 用法示例：
#   cd /root/autodl-tmp/OmniSAM-main
#   RUN_IN_BACKGROUND=1 \
#   SRC_CKPT=/root/autodl-tmp/omnisam_s.pth TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
#   bash scripts/run_plugin_ablation_seed1_triplet.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

INTERNAL_BG="${INTERNAL_BG:-0}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

# 另起文件夹（默认）
LOG_DIR="${LOG_DIR:-${ROOT}/logs/omnisam_time_ablation_seed1_triplet}"
mkdir -p "${LOG_DIR}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="${MASTER_LOG:-${LOG_DIR}/plugin_ablation_seed1_triplet_${STAMP}.master.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/plugin_ablation_seed1_triplet.pid}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- user knobs ----
GPU="${GPU:-0}"
SEED="${SEED:-1}"   # 默认固定 seed=1
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
DATA_ROOT="${DATA_ROOT:-data}"
PD_ROOT="${PD_ROOT:-data_pd_exp_plugin_seed1_triplet}"
SRC_CKPT="${SRC_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"
TGT_CKPT="${TGT_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"

AMP="${AMP:-true}"
TF32="${TF32:-true}"

# 纯视觉（与原论文主干一致）
USE_MULTIMODAL="${USE_MULTIMODAL:-0}"
TEXT_PROTO_PATH="${TEXT_PROTO_PATH:-}"
LAMBDA_TXT="${LAMBDA_TXT:-0.0}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.0}"
USE_TEXT_GATE_SSL="${USE_TEXT_GATE_SSL:-false}"

# 小样本建议默认值（你当前建议使用200）
QUICK_IMAGES="${QUICK_IMAGES:-200}"
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

run_one() {
  local tag="$1"
  local cfg="$2"
  local opts="$3"
  local run_log="${LOG_DIR}/plugin_ablation_${tag}_${STAMP}.log"

  echo "----------------------------------------------"
  echo "[run] ${tag}"
  echo "cfg=${cfg}"
  echo "opts=${opts}"
  echo "log=${run_log}"
  echo "----------------------------------------------"

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
  if [[ "${USE_WANDB:-0}" == "1" || "${USE_WANDB:-0}" == "true" ]]; then use_wandb_flag="--use_wandb"; fi
  if [[ "${USE_TENSORBOARD:-0}" == "1" || "${USE_TENSORBOARD:-0}" == "true" ]]; then use_tb_flag="--use_tensorboard"; fi

  env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    python run_adaptation.py \
      --config "${cfg}" \
      --seed "${SEED}" \
      --dataset "${DATASET}" \
      --class_set "${CLASS_SET}" \
      --backbone "${BACKBONE}" \
      --data_root "${DATA_ROOT}" \
      --pd_root "${PD_ROOT}/${tag}_${STAMP}" \
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
      --opts "${opts}" \
      > "${run_log}" 2>&1

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
    MASTER_LOG="${MASTER_LOG}" \
    PID_FILE="${PID_FILE}" \
    STAMP="${STAMP}" \
    GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
    DATA_ROOT="${DATA_ROOT}" PD_ROOT="${PD_ROOT}" SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" \
    QUICK_IMAGES="${QUICK_IMAGES}" NUM_EPOCHS="${NUM_EPOCHS}" TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
    WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
    PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG}" INPUT_LIST="${INPUT_LIST}" PL_SUBSET_FILE="${PL_SUBSET_FILE}" \
    AMP="${AMP}" TF32="${TF32}" USE_MULTIMODAL="${USE_MULTIMODAL}" TEXT_PROTO_PATH="${TEXT_PROTO_PATH}" \
    LAMBDA_TXT="${LAMBDA_TXT}" WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" USE_TEXT_GATE_SSL="${USE_TEXT_GATE_SSL}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "已后台启动 seed1 三组消融"
  echo "PID: ${pid} (保存于 ${PID_FILE})"
  echo "主日志: ${MASTER_LOG}"
  echo "查看: tail -f ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Plugin Ablation Seed1 Triplet =========="
echo "start: $(date -Iseconds)"
echo "ROOT=${ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "PD_ROOT=${PD_ROOT}"
echo "SEED=${SEED}"
echo "DATASET=${DATASET}, CLASS_SET=${CLASS_SET}, BACKBONE=${BACKBONE}"
echo "QUICK_IMAGES=${QUICK_IMAGES}, NUM_EPOCHS=${NUM_EPOCHS}, TRAIN_MAX_ITERS=${TRAIN_MAX_ITERS}"
echo "BATCH_SIZE=${BATCH_SIZE}, PL_BATCH_SIZE=${PL_BATCH_SIZE}, PL_NUM_WORKERS=${PL_NUM_WORKERS}"
echo "PL_SUBSET_FILE=${PL_SUBSET_FILE}"
echo "MASTER_LOG=${MASTER_LOG}"
echo "PID_FILE=${PID_FILE}"
echo "==================================================="

python "${ROOT}/scripts/time_ablation/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${PL_SUBSET_FILE}" \
  --num_images "${QUICK_IMAGES}" \
  --offset 0 \
  --shuffle \
  --seed "${SEED}"

# 1) base_first
run_one \
  "base_first" \
  "configs/ablations/omnisam_base.yaml" \
  "plugins.pseudo_conf_mask.enabled=false,plugins.efficient_tam_memory.enabled=false,plugins.efficient_sam2_sparse.enabled=false"

# 2) base + 1 + 4
run_one \
  "base_plus1_plus4" \
  "configs/ablations/omnisam_base_plus4_efficienttam.yaml" \
  "plugins.pseudo_conf_mask.enabled=true,plugins.efficient_tam_memory.enabled=true,plugins.efficient_sam2_sparse.enabled=false"

# 3) base + 1 + 4 + 5 (train_only)
run_one \
  "base_plus1_plus4_plus5_trainonly" \
  "configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml" \
  "plugins.pseudo_conf_mask.enabled=true,plugins.efficient_tam_memory.enabled=true,plugins.efficient_sam2_sparse.enabled=true,plugins.efficient_sam2_sparse.apply_in_train=true,plugins.efficient_sam2_sparse.apply_in_pl=false"

# 4) base_last (mitigate cold-start / caching effects)
run_one \
  "base_last" \
  "configs/ablations/omnisam_base.yaml" \
  "plugins.pseudo_conf_mask.enabled=false,plugins.efficient_tam_memory.enabled=false,plugins.efficient_sam2_sparse.enabled=false"

echo "[all done] finished at $(date -Iseconds)"
echo "tips:"
echo "  tail -f ${MASTER_LOG}"
echo "  ls -lh ${LOG_DIR}/plugin_ablation_*_${STAMP}.log"

