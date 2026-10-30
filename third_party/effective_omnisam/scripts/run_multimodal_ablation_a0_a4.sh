#!/usr/bin/env bash
set -euo pipefail

# A0-A5 text ablation launcher.
# Default behavior:
# - Skip A0 baseline (assumed already completed)
# - Write each run log into ./logs/
# Usage:
#   bash scripts/run_multimodal_ablation_a0_a4.sh
# Override defaults via env vars, e.g.:
#   DATASET=Cityscapes BACKBONE=sam2_s EPOCHS=1 bash scripts/run_multimodal_ablation_a0_a4.sh

DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-data}"
PD_ROOT_BASE="${PD_ROOT_BASE:-data_pd_exp}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
PROTO_PATH="${PROTO_PATH:-./assets/text_prototypes_openclip_text_latest.pt}"
LOG_DIR="${LOG_DIR:-./logs}"
RUN_TAG="${RUN_TAG:-a0_a5}"
SKIP_A0="${SKIP_A0:-1}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
PID_FILE="${PID_FILE:-${LOG_DIR}/multimodal_${RUN_TAG}.pid}"
ABLT_PHASE="${ABLT_PHASE:-all}" # all | semantic | gate

# Memory compression / acceleration defaults (on by default for this ablation script)
USE_MEMORY_COMPRESSION="${USE_MEMORY_COMPRESSION:-1}"
MEMORY_SELECT_MODE="${MEMORY_SELECT_MODE:-hybrid}"
MEMORY_RECENT_KEEP="${MEMORY_RECENT_KEEP:-4}"
MEMORY_SCORE_KEEP="${MEMORY_SCORE_KEEP:-4}"
MEMORY_COMPRESS_MODE="${MEMORY_COMPRESS_MODE:-hybrid}"
MEMORY_COMPRESS_LOCATION="${MEMORY_COMPRESS_LOCATION:-before_attention}"
MEMORY_SPATIAL_WINDOW_SIZE="${MEMORY_SPATIAL_WINDOW_SIZE:-2}"
MEMORY_GLOBAL_SUMMARY_TOKENS="${MEMORY_GLOBAL_SUMMARY_TOKENS:-4}"

mkdir -p "${LOG_DIR}"
if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG:-0}" != "1" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  MASTER_LOG="${LOG_DIR}/multimodal_${RUN_TAG}_${STAMP}.master.log"
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 LOG_DIR="${LOG_DIR}" RUN_TAG="${RUN_TAG}" \
    bash "$0" "$@" > "${MASTER_LOG}" 2>&1 &
  PID=$!
  echo "${PID}" > "${PID_FILE}"
  echo "Started in background."
  echo "PID: ${PID} (saved to ${PID_FILE})"
  echo "Master log: ${MASTER_LOG}"
  echo "Per-case logs: ${LOG_DIR}/multimodal_${RUN_TAG}_*.log"
  exit 0
fi

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
  --memory_select_mode "${MEMORY_SELECT_MODE}"
  --memory_recent_keep "${MEMORY_RECENT_KEEP}"
  --memory_score_keep "${MEMORY_SCORE_KEEP}"
  --memory_compress_mode "${MEMORY_COMPRESS_MODE}"
  --memory_compress_location "${MEMORY_COMPRESS_LOCATION}"
  --memory_spatial_window_size "${MEMORY_SPATIAL_WINDOW_SIZE}"
  --memory_global_summary_tokens "${MEMORY_GLOBAL_SUMMARY_TOKENS}"
)
if [[ "${USE_MEMORY_COMPRESSION}" == "1" ]]; then
  COMMON_ARGS+=(--use_memory_compression)
else
  COMMON_ARGS+=(--no-use_memory_compression)
fi

run_case() {
  local tag="$1"; shift
  mkdir -p "${LOG_DIR}"
  local log_file="${LOG_DIR}/multimodal_${RUN_TAG}_${tag}.log"
  echo "=============================="
  echo "Running ${tag}"
  echo "Log file: ${log_file}"
  echo "=============================="
  python run_adaptation.py \
    "${COMMON_ARGS[@]}" \
    --pd_root "${PD_ROOT_BASE}_${tag}" \
    "$@" 2>&1 | tee "${log_file}"
}

print_text_proto_meta() {
  if [[ -f "${PROTO_PATH}" ]]; then
    PROTO_PATH="${PROTO_PATH}" TEXT_PROTO_MODE="${TEXT_PROTO_MODE:-real}" python - <<'PY'
import json, os, torch
p = os.environ["PROTO_PATH"]
payload = torch.load(p, map_location="cpu")
meta = {}
if isinstance(payload, dict):
    for k in ("source_mode", "text_model", "text_pretrained"):
        if k in payload:
            meta[k] = payload[k]
print("[text_meta] text_proto_path=", p, sep="")
print("[text_meta] text_proto_mode=", os.environ.get("TEXT_PROTO_MODE","real"), sep="")
if meta:
    print("[text_meta] payload=", json.dumps(meta, ensure_ascii=False), sep="")
PY
  fi
}

echo "[ablation] phase=${ABLT_PHASE}"
echo "[ablation] default gate params: tau_t=0.10, weight_txt_align=0.02"

if [[ "${SKIP_A0}" != "1" ]]; then
  # A0: visual-only baseline
  run_case "A0_baseline" \
    --no-use_text_proto \
    --no-use_text_gate_ssl \
    --text_proto_path "" \
    --lambda_txt 0.0 \
    --weight_txt_align 0.0 \
    --text_proto_mode real
fi
print_text_proto_meta

# A1: text branch with real prototypes (no text gate)
if [[ "${ABLT_PHASE}" == "all" || "${ABLT_PHASE}" == "semantic" || "${ABLT_PHASE}" == "gate" ]]; then
  run_case "A1_text_branch" \
    --text_proto_path "${PROTO_PATH}" \
    --use_text_proto \
    --text_proto_mode real \
    --no-use_text_gate_ssl \
    --lambda_txt 0.1 \
    --txt_tau 0.07 \
    --weight_txt_align 0.02 \
    --txt_align_tau 1.0
fi

# A2: text branch + soft text gate (real prototypes)
if [[ "${ABLT_PHASE}" == "all" || "${ABLT_PHASE}" == "gate" ]]; then
  run_case "A2_text_branch_gate_soft" \
    --text_proto_path "${PROTO_PATH}" \
    --use_text_proto \
    --text_proto_mode real \
    --use_text_gate_ssl \
    --ssl_gate_mode soft \
    --lambda_txt 0.1 \
    --txt_tau 0.07 \
    --weight_txt_align 0.02 \
    --txt_align_tau 1.0 \
    --tau_t 0.10
fi

# A3: text branch + hard text gate (real prototypes)
if [[ "${ABLT_PHASE}" == "all" || "${ABLT_PHASE}" == "gate" ]]; then
  run_case "A3_text_branch_gate_hard" \
    --text_proto_path "${PROTO_PATH}" \
    --use_text_proto \
    --text_proto_mode real \
    --use_text_gate_ssl \
    --ssl_gate_mode hard \
    --lambda_txt 0.1 \
    --txt_tau 0.07 \
    --weight_txt_align 0.02 \
    --txt_align_tau 1.0 \
    --tau_t 0.10
fi

# A4: shuffled prototypes (negative control), gate disabled
if [[ "${ABLT_PHASE}" == "all" || "${ABLT_PHASE}" == "semantic" ]]; then
  run_case "A4_shuffle_proto" \
    --text_proto_path "${PROTO_PATH}" \
    --use_text_proto \
    --text_proto_mode shuffle \
    --no-use_text_gate_ssl \
    --lambda_txt 0.1 \
    --txt_tau 0.07 \
    --weight_txt_align 0.02 \
    --txt_align_tau 1.0
fi

# A5: random prototypes (negative control)
if [[ "${ABLT_PHASE}" == "all" || "${ABLT_PHASE}" == "semantic" ]]; then
  run_case "A5_random_proto" \
    --text_proto_path "${PROTO_PATH}" \
    --use_text_proto \
    --text_proto_mode random \
    --no-use_text_gate_ssl \
    --lambda_txt 0.1 \
    --txt_tau 0.07 \
    --weight_txt_align 0.02 \
    --txt_align_tau 1.0
fi

echo "All requested experiments finished. Logs saved to: ${LOG_DIR}"
