#!/usr/bin/env bash
# UDA（DensePASS/Cityscapes 13cls）——对齐「早期复现」风格，从 60.36 teacher 出发继续训：
# - batch_size=2（与 uda_densepass_20260408 一类日志一致）
# - 不开 AMP（与当时默认一致）
# - conf_thr=-1：关闭像素级置信度掩码（等价于原版无 IMPROVEMENTS 里 conf-mask 的行为）
# - 默认 num_epochs=3，且 --regenerate_pl_each_epoch：每轮先按当前 tgt 模型重打伪标签再训 1 个 epoch
#
# 用法（在项目根目录）:
#   bash scripts/uda_densepass_originalish_from_60_36.sh
#   NUM_EPOCHS=3 PD_ROOT=/path/to/fresh_pd bash scripts/uda_densepass_originalish_from_60_36.sh
#
# 日志：
#   前台跑也会自动写入 logs/uda_densepass_originalish_<时间戳>.log（同时打印到终端）
# 后台跑（nohup，同上路径）:
#   RUN_IN_BACKGROUND=1 bash scripts/uda_densepass_originalish_from_60_36.sh
# 若不要写文件（仅终端）:
#   SKIP_FILE_LOG=1 bash scripts/uda_densepass_originalish_from_60_36.sh
#
# Python：请用装有 torch + wandb 的环境（如 conda env omnisam）。未设置 PYTHON 时，若存在
#   /root/miniconda3/envs/omnisam/bin/python 则默认用它；否则用 PATH 里的 python。
#   不要用 base conda 的 python 3.12 直接跑，易出现 `wandb` 无 init（与本仓库 ./wandb 目录冲突或包装损坏）。
#
# 建议：换新的 PD_ROOT 或清空旧伪标签目录，避免与加速实验混用。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_PREFIX}" == *omnisam* && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif [[ -x "/root/miniconda3/envs/omnisam/bin/python" ]]; then
    PYTHON="/root/miniconda3/envs/omnisam/bin/python"
  else
    PYTHON="python"
  fi
fi

DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-${ROOT}/pd_uda_densepass/checkpoints/Cityscapes/sam2_s/best_sam2_s_iou60.36.pth}"
PD_ROOT="${PD_ROOT:-${ROOT}/pd_uda_from60_36_originalish}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WARMUP_PL="${WARMUP_PL:-2000}"
# epoch>=1 每轮伪标签数量；须 >0，默认与 WARMUP_PL 相同（原 iters_pl=0 时多 epoch 无法生成标签）
ITERS_PL="${ITERS_PL:-${WARMUP_PL}}"
# -1 => 不做 max-softmax 过滤（strategies 里 conf_t > -1 恒成立）
CONF_THR="${CONF_THR:--1}"

# 续训/复用伪标签（默认关闭，保持“从头完整跑 3 epoch”）
#
# - 若 RESUME_CKPT 为空，则自动优先用 13cls 的 resume_ckpt（若存在）
# - 若 REUSE_PL_DIR 为空，则自动优先复用已有的 Cityscapes epoch1 伪标签（若存在且完整）
#
# 默认行为：从头跑（不续训、不复用）
# 需要开启“自动续训/复用”时：
#   FORCE_FROM_SCRATCH=0 bash scripts/uda_densepass_originalish_from_60_36.sh
FORCE_FROM_SCRATCH="${FORCE_FROM_SCRATCH:-1}"
RESUME_CKPT="${RESUME_CKPT:-}"
REUSE_PL_DIR="${REUSE_PL_DIR:-}"

# 磁盘空间保护：伪标签会持续落盘 PNG，空间不足会直接挂
MIN_FREE_GB="${MIN_FREE_GB:-10}"

# 与「未刻意开 AMP」的复现一致
AMP="${AMP:-0}"
# 若需与部分旧环境完全一致可设 TF32=0
TF32="${TF32:-1}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
SKIP_FILE_LOG="${SKIP_FILE_LOG:-0}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${TGT_CKPT}" ]]; then
  echo "Error: TGT_CKPT not found: ${TGT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${SRC_CKPT}" ]]; then
  echo "Error: SRC_CKPT not found: ${SRC_CKPT}" >&2
  exit 1
fi

mkdir -p "${PD_ROOT}"

check_free_gb() {
  local path="${1}"
  # df -BG 输出形如：Avail 29G
  local avail_g
  avail_g="$(df -BG "${path}" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
  if [[ -z "${avail_g}" ]]; then
    echo "Warning: cannot determine free space for ${path}" >&2
    return 0
  fi
  if (( avail_g < MIN_FREE_GB )); then
    echo "Error: low disk space under ${path}: ${avail_g}G free < MIN_FREE_GB=${MIN_FREE_GB}G" >&2
    echo "Tip: delete large zips/logs or move PD_ROOT to a larger disk." >&2
    exit 1
  fi
}

check_free_gb "${ROOT}"

auto_pick_resume_ckpt() {
  if [[ -n "${RESUME_CKPT}" ]]; then
    return 0
  fi
  local cand
  cand="${ROOT}/pd_uda_densepass/checkpoints/Cityscapes_13cls/sam2_s/resume_sam2_s.pth"
  if [[ -f "${cand}" ]]; then
    RESUME_CKPT="${cand}"
  fi
}

auto_pick_reuse_pl_dir() {
  if [[ -n "${REUSE_PL_DIR}" ]]; then
    return 0
  fi
  # 优先尝试复用已生成完成的伪标签（Cityscapes epoch1，2000 张）
  local cand
  cand="${ROOT}/pd_uda_densepass/DensePASS/sam2_s/Cityscapes/epoch1/gtFine"
  if [[ -d "${cand}" ]]; then
    # 快速检查数量（避免半成品）
    local n
    n="$(python - <<PY
import os
root='${cand}'
cnt=0
for r,ds,fs in os.walk(root):
    for f in fs:
        if f.endswith('_labelTrainIds.png'):
            cnt+=1
print(cnt)
PY
)"
    if [[ "${n}" == "2000" ]]; then
      REUSE_PL_DIR="${cand}"
    fi
  fi
}

if [[ "${FORCE_FROM_SCRATCH}" != "1" ]]; then
  auto_pick_resume_ckpt
  auto_pick_reuse_pl_dir
fi

CMD=(
  "${PYTHON}" run_adaptation.py
  --backbone sam2_s
  --dataset Cityscapes
  --class_set 13
  --data_root "${DATA_ROOT}"
  --pd_root "${PD_ROOT}"
  --src_ckpt "${SRC_CKPT}"
  --tgt_ckpt "${TGT_CKPT}"
  --warmup_pl "${WARMUP_PL}"
  --iters_pl "${ITERS_PL}"
  --batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --conf_thr "${CONF_THR}"
)

if [[ "${FORCE_FROM_SCRATCH}" == "1" ]]; then
  CMD+=(--regenerate_pl_each_epoch)
else
  if [[ -n "${REUSE_PL_DIR}" ]]; then
    CMD+=(--reuse_pl_dir "${REUSE_PL_DIR}" --no-regenerate_pl_each_epoch)
  else
    CMD+=(--regenerate_pl_each_epoch)
  fi
  if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ ! -f "${RESUME_CKPT}" ]]; then
      echo "Error: RESUME_CKPT not found: ${RESUME_CKPT}" >&2
      exit 1
    fi
    CMD+=(--resume_ckpt "${RESUME_CKPT}")
  fi
fi

if [[ "${AMP}" == "1" ]]; then
  CMD+=(--amp)
else
  CMD+=(--no-amp)
fi
if [[ "${TF32}" == "1" ]]; then
  CMD+=(--tf32)
else
  CMD+=(--no-tf32)
fi

echo "PYTHON=${PYTHON}"
echo "PD_ROOT=${PD_ROOT}"
echo "TGT_CKPT=${TGT_CKPT}"
echo "RESUME_CKPT=${RESUME_CKPT:-<none>}"
echo "REUSE_PL_DIR=${REUSE_PL_DIR:-<none>}"
echo "MIN_FREE_GB=${MIN_FREE_GB}"
echo "Running: ${CMD[*]}"

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/uda_densepass_originalish_${STAMP}.log"
  {
    echo "=========================================="
    echo "uda_densepass_originalish_from_60_36.sh"
    echo "Time: $(date -Iseconds)"
    echo "PYTHON=${PYTHON}"
    echo "PD_ROOT=${PD_ROOT}"
    echo "TGT_CKPT=${TGT_CKPT}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "ITERS_PL=${ITERS_PL} WARMUP_PL=${WARMUP_PL}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "=========================================="
  } | tee "${LOG_FILE}"
  nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "Started in background. PID=${PID}" | tee -a "${LOG_FILE}"
  echo "${PID}" > "${LOG_DIR}/uda_densepass_originalish_${STAMP}.pid"
  echo "PID file: ${LOG_DIR}/uda_densepass_originalish_${STAMP}.pid"
  echo "Follow log: tail -f ${LOG_FILE}"
else
  if [[ "${SKIP_FILE_LOG}" == "1" ]]; then
    exec "${CMD[@]}"
  fi
  mkdir -p "${LOG_DIR}"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/uda_densepass_originalish_${STAMP}.log"
  {
    echo "=========================================="
    echo "uda_densepass_originalish_from_60_36.sh (foreground + tee)"
    echo "Time: $(date -Iseconds)"
    echo "PYTHON=${PYTHON}"
    echo "PD_ROOT=${PD_ROOT}"
    echo "TGT_CKPT=${TGT_CKPT}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "ITERS_PL=${ITERS_PL} WARMUP_PL=${WARMUP_PL}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "=========================================="
  } | tee "${LOG_FILE}"
  "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  echo "Done: $(date -Iseconds)" | tee -a "${LOG_FILE}"
fi
