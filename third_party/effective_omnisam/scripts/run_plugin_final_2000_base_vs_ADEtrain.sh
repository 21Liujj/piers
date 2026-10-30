#!/usr/bin/env bash
# 最终对比（2000 张 PL 子集，短热身版）：
# warmup(base_first, 不参与最终对比) -> A + D_train + E_train(full) -> base_last(full)
#
# 方案定义（更严谨表述）：
# - A = pseudo_conf_mask：本实现主要作用于训练阶段的 unsup loss（loss masking），并非 PL 生成模块
# - PL 阶段：不启用 D/E（避免 token 压缩/稀疏调度引入额外开销），保持原始 PL 生成与置信/uncertain 机制以保证稳定性
# - 训练阶段：启用 A + D + E（D/E 仅训练开）
#
# 与既有消融脚本风格一致：
# - RUN_IN_BACKGROUND=1 默认后台 nohup
# - 主日志 + 每组子日志
# - PID 文件
# - 固定 subset（warmup + full）保证可复现
#
# 用法示例：
#   cd /root/autodl-tmp/OmniSAM-main
#   RUN_IN_BACKGROUND=1 \
#   SRC_CKPT=/root/autodl-tmp/omnisam_s.pth TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
#   bash scripts/run_plugin_final_2000_base_vs_ADEtrain.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

INTERNAL_BG="${INTERNAL_BG:-0}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

# 另起文件夹（默认）
LOG_DIR="${LOG_DIR:-${ROOT}/logs/omnisam_final_2000}"
mkdir -p "${LOG_DIR}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="${MASTER_LOG:-${LOG_DIR}/final_2000_base_vs_ADEtrain_${STAMP}.master.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/final_2000_base_vs_ADEtrain.pid}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- user knobs ----
GPU="${GPU:-0}"
SEED="${SEED:-1}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
DATA_ROOT="${DATA_ROOT:-data}"
PD_ROOT="${PD_ROOT:-data_pd_exp_final_2000}"
SRC_CKPT="${SRC_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"
TGT_CKPT="${TGT_CKPT:-${ROOT}/exp2/Cityscapes/best_sam2_s_10.pth}"

AMP="${AMP:-true}"
TF32="${TF32:-true}"

# 纯视觉（原论文主干）
USE_MULTIMODAL="${USE_MULTIMODAL:-0}"
TEXT_PROTO_PATH="${TEXT_PROTO_PATH:-}"
LAMBDA_TXT="${LAMBDA_TXT:-0.0}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.0}"
USE_TEXT_GATE_SSL="${USE_TEXT_GATE_SSL:-false}"

# 全量 PL 子集
QUICK_IMAGES="${QUICK_IMAGES:-2000}"
# 热身阶段（base_first）子集与迭代
WARMUP_IMAGES="${WARMUP_IMAGES:-100}"
WARMUP_TRAIN_MAX_ITERS="${WARMUP_TRAIN_MAX_ITERS:-20}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# 训练是否跑满一个 epoch：默认 0 表示不截断（不传 --train_max_iters）
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-0}"

# 训练 batch 固定 4；PL 建议用更小 batch 降低峰值显存
BATCH_SIZE="${BATCH_SIZE:-4}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-1}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-2}"
PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG:---no-pl_shuffle}"

WARMUP_PL="${WARMUP_PL:-${QUICK_IMAGES}}"
ITERS_PL="${ITERS_PL:-${QUICK_IMAGES}}"

SUBSET_DIR="${LOG_DIR}/subsets"
mkdir -p "${SUBSET_DIR}"
INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-${SUBSET_DIR}/${DATASET}_${CLASS_SET}cls_seed${SEED}_n${QUICK_IMAGES}.txt}"
PL_WARMUP_SUBSET_FILE="${PL_WARMUP_SUBSET_FILE:-${SUBSET_DIR}/${DATASET}_${CLASS_SET}cls_seed${SEED}_warmup${WARMUP_IMAGES}.txt}"

run_one() {
  local tag="$1"
  local cfg="$2"
  local opts="$3"
  local subset_file="$4"
  local warmup_pl="$5"
  local iters_pl="$6"
  local train_max_iters="$7"
  local run_log="${LOG_DIR}/${tag}_${STAMP}.log"

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

  local train_iters_arg=""
  if [[ "${train_max_iters}" != "0" && -n "${train_max_iters}" ]]; then
    train_iters_arg="--train_max_iters ${train_max_iters}"
  fi

  # shellcheck disable=SC2086
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
      --pl_subset_file "${subset_file}" \
      --warmup_pl "${warmup_pl}" \
      --iters_pl "${iters_pl}" \
      ${train_iters_arg} \
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
    echo "[ERROR] stop whole final run due to failure (see ${run_log})"
    exit "${rc}"
  fi
  echo "[done] ${tag} finished at $(date -Iseconds)"
}

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env \
    INTERNAL_BG=1 \
    RUN_IN_BACKGROUND=1 \
    LOG_DIR="${LOG_DIR}" \
    STAMP="${STAMP}" \
    MASTER_LOG="${MASTER_LOG}" \
    PID_FILE="${PID_FILE}" \
    GPU="${GPU}" SEED="${SEED}" DATASET="${DATASET}" CLASS_SET="${CLASS_SET}" BACKBONE="${BACKBONE}" \
    DATA_ROOT="${DATA_ROOT}" PD_ROOT="${PD_ROOT}" SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" \
    QUICK_IMAGES="${QUICK_IMAGES}" WARMUP_IMAGES="${WARMUP_IMAGES}" NUM_EPOCHS="${NUM_EPOCHS}" TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS}" \
    WARMUP_TRAIN_MAX_ITERS="${WARMUP_TRAIN_MAX_ITERS}" \
    WARMUP_PL="${WARMUP_PL}" ITERS_PL="${ITERS_PL}" \
    BATCH_SIZE="${BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
    PL_SHUFFLE_FLAG="${PL_SHUFFLE_FLAG}" INPUT_LIST="${INPUT_LIST}" PL_SUBSET_FILE="${PL_SUBSET_FILE}" \
    PL_WARMUP_SUBSET_FILE="${PL_WARMUP_SUBSET_FILE}" \
    AMP="${AMP}" TF32="${TF32}" USE_MULTIMODAL="${USE_MULTIMODAL}" TEXT_PROTO_PATH="${TEXT_PROTO_PATH}" \
    LAMBDA_TXT="${LAMBDA_TXT}" WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN}" USE_TEXT_GATE_SSL="${USE_TEXT_GATE_SSL}" \
    bash "${BASH_SOURCE[0]}" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "已后台启动 final_2000：base vs A+D_train+E_train"
  echo "PID: ${pid} (保存于 ${PID_FILE})"
  echo "主日志: ${MASTER_LOG}"
  echo "查看: tail -f ${MASTER_LOG}"
  exit 0
fi

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Final 2000: base vs A+D_train+E_train =========="
echo "start: $(date -Iseconds)"
echo "ROOT=${ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "PD_ROOT=${PD_ROOT}"
echo "SEED=${SEED}"
echo "DATASET=${DATASET}, CLASS_SET=${CLASS_SET}, BACKBONE=${BACKBONE}"
echo "WARMUP_IMAGES=${WARMUP_IMAGES}, WARMUP_TRAIN_MAX_ITERS=${WARMUP_TRAIN_MAX_ITERS}"
echo "QUICK_IMAGES=${QUICK_IMAGES}, NUM_EPOCHS=${NUM_EPOCHS}, TRAIN_MAX_ITERS=${TRAIN_MAX_ITERS} (0 means full epoch)"
echo "BATCH_SIZE=${BATCH_SIZE}, PL_BATCH_SIZE=${PL_BATCH_SIZE}, PL_NUM_WORKERS=${PL_NUM_WORKERS}"
echo "PL_WARMUP_SUBSET_FILE=${PL_WARMUP_SUBSET_FILE}"
echo "PL_SUBSET_FILE=${PL_SUBSET_FILE}"
echo "MASTER_LOG=${MASTER_LOG}"
echo "PID_FILE=${PID_FILE}"
echo "==========================================================="

python "${ROOT}/scripts/time_ablation/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${PL_WARMUP_SUBSET_FILE}" \
  --num_images "${WARMUP_IMAGES}" \
  --offset 0 \
  --shuffle \
  --seed "${SEED}"

python "${ROOT}/scripts/time_ablation/build_fixed_subset.py" \
  --input_list "${INPUT_LIST}" \
  --output_list "${PL_SUBSET_FILE}" \
  --num_images "${QUICK_IMAGES}" \
  --offset 0 \
  --shuffle \
  --seed "${SEED}"

# base_first（短热身，不参与最终对比）
run_one \
  "base_first" \
  "configs/ablations/omnisam_base.yaml" \
  "plugins.pseudo_conf_mask.enabled=false,plugins.efficient_tam_memory.enabled=false,plugins.efficient_sam2_sparse.enabled=false" \
  "${PL_WARMUP_SUBSET_FILE}" \
  "${WARMUP_IMAGES}" \
  "${WARMUP_IMAGES}" \
  "${WARMUP_TRAIN_MAX_ITERS}"

# A + D_train + E_train（正式全量）
run_one \
  "A_Dtrain_Etrain" \
  "configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml" \
  "plugins.pseudo_conf_mask.enabled=true,plugins.efficient_tam_memory.enabled=true,plugins.efficient_tam_memory.apply_in_train=true,plugins.efficient_tam_memory.apply_in_pl=false,plugins.efficient_sam2_sparse.enabled=true,plugins.efficient_sam2_sparse.apply_in_train=true,plugins.efficient_sam2_sparse.apply_in_pl=false" \
  "${PL_SUBSET_FILE}" \
  "${WARMUP_PL}" \
  "${ITERS_PL}" \
  "${TRAIN_MAX_ITERS}"

# base_last（正式全量基线）
run_one \
  "base_last" \
  "configs/ablations/omnisam_base.yaml" \
  "plugins.pseudo_conf_mask.enabled=false,plugins.efficient_tam_memory.enabled=false,plugins.efficient_sam2_sparse.enabled=false" \
  "${PL_SUBSET_FILE}" \
  "${WARMUP_PL}" \
  "${ITERS_PL}" \
  "${TRAIN_MAX_ITERS}"

echo "[all done] finished at $(date -Iseconds)"
echo "tips:"
echo "  tail -f ${MASTER_LOG}"
echo "  ls -lh ${LOG_DIR}/*_${STAMP}.log"

