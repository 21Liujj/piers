#!/usr/bin/env bash
set -euo pipefail

# Minimal stage-1/2 experiment sequence (time-saving mode):
# 1) Generate pseudo-labels ONCE with A (baseline) and cache them
# 2) Reuse the same pseudo-label directory for A/B/C/D training
# 3) Train each setting for 1 epoch first, compare trend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
PROTO_PATH="${PROTO_PATH:-./assets/text_prototypes_openclip_text_latest.pt}"
TEXT_MODEL="${TEXT_MODEL:-ViT-B-32}"
TEXT_PRETRAINED="${TEXT_PRETRAINED:-openai}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
RUN_TAG="${RUN_TAG:-abcd}"
# IMPORTANT:
# Use per-run pd_root by default to avoid auto-resuming stale checkpoints
# from previous experiments with the same fixed directory name.
PD_ROOT_BASE="${PD_ROOT_BASE:-data_pd_${RUN_TAG}}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
PID_FILE="${PID_FILE:-${LOG_DIR}/multimodal_${RUN_TAG}.pid}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-8}"
WARMUP_PL="${WARMUP_PL:-2000}"
ITERS_PL="${ITERS_PL:-0}"
CONF_THR="${CONF_THR:-0.85}"
SKIP_GENERATE_ONLY="${SKIP_GENERATE_ONLY:-auto}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-2}"
SSL_GATE_MODE="${SSL_GATE_MODE:-hard}"
TAU_T="${TAU_T:-0.10}"
WEIGHT_TXT_ALIGN="${WEIGHT_TXT_ALIGN:-0.05}"
LAMBDA_TXT="${LAMBDA_TXT:-0.1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ONLY="${ONLY:-ABCD}"  # ABCD | CD | D | C | AB (simple filters)
REUSE_PL_DIR_OVERRIDE="${REUSE_PL_DIR_OVERRIDE:-}"

export PYTORCH_CUDA_ALLOC_CONF

print_text_proto_meta() {
  if [[ -f "${PROTO_PATH}" ]]; then
    PROTO_PATH="${PROTO_PATH}" TEXT_PROTO_MODE="${TEXT_PROTO_MODE:-real}" SSL_GATE_MODE="${SSL_GATE_MODE}" TAU_T="${TAU_T}" TEXT_FUSION_ALPHA_INIT="${TEXT_FUSION_ALPHA_INIT:-0.0}" python - <<'PY'
import json, os, torch
p = os.environ.get("PROTO_PATH", "")
payload = torch.load(p, map_location="cpu")
meta = {}
if isinstance(payload, dict):
    for k in ("source_mode", "text_model", "text_pretrained"):
        if k in payload:
            meta[k] = payload[k]
print("[text_meta] text_proto_path=", p, sep="")
print("[text_meta] text_proto_mode=", os.environ.get("TEXT_PROTO_MODE","real"), sep="")
print("[text_meta] ssl_gate_mode=", os.environ.get("SSL_GATE_MODE","hard"), " tau_t=", os.environ.get("TAU_T","0.10"), sep="")
print("[text_meta] fusion_alpha_init=", os.environ.get("TEXT_FUSION_ALPHA_INIT","0.0"), sep="")
if meta:
    print("[text_meta] payload=", json.dumps(meta, ensure_ascii=False), sep="")
PY
  fi
}

mkdir -p "${LOG_DIR}"
if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG:-0}" != "1" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  MASTER_LOG="${LOG_DIR}/multimodal_${RUN_TAG}_${STAMP}.master.log"
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 LOG_DIR="${LOG_DIR}" RUN_TAG="${RUN_TAG}" \
    bash "${SCRIPT_DIR}/run_multimodal_stage1_stage2_abcd.sh" "$@" > "${MASTER_LOG}" 2>&1 &
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
  --data_root "${DATA_ROOT}"
  --src_ckpt "${SRC_CKPT}"
  --tgt_ckpt "${TGT_CKPT}"
  --amp
  --tf32
  --no-use_wandb
  --use_tensorboard
  --text_proto_refresh cache
  --ssl_gate_mode "${SSL_GATE_MODE}"
  --tau_t "${TAU_T}"
)

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

run_generate_only_A() {
  local tag="A_generate_only"
  mkdir -p "${LOG_DIR}"
  local log_file="${LOG_DIR}/multimodal_${RUN_TAG}_${tag}.log"
  echo "=============================="
  echo "Running ${tag}"
  echo "Log file: ${log_file}"
  echo "=============================="
  python run_adaptation.py \
    "${COMMON_ARGS[@]}" \
    --pd_root "${PD_ROOT_BASE}_A_baseline" \
    --num_epochs 1 \
    --batch_size 1 \
    --pl_batch_size "${PL_BATCH_SIZE}" \
    --pl_num_workers "${PL_NUM_WORKERS}" \
    --warmup_pl "${WARMUP_PL}" \
    --iters_pl "${ITERS_PL}" \
    --generate_only \
    --no-use_text_proto \
    --no-use_text_gate_ssl \
    --no-use_language_adapter \
    --text_proto_path "" \
    --lambda_txt 0.0 \
    --weight_txt_align 0.0 2>&1 | tee "${log_file}"
}

A_REUSE_PL_DIR_DEFAULT="${PD_ROOT_BASE}_A_baseline/DensePASS/${BACKBONE}/${DATASET}_${CLASS_SET}cls/epoch1"
A_REUSE_PL_DIR="${A_REUSE_PL_DIR_DEFAULT}"
if [[ -n "${REUSE_PL_DIR_OVERRIDE}" ]]; then
  A_REUSE_PL_DIR="${REUSE_PL_DIR_OVERRIDE}"
fi

if [[ -n "${REUSE_PL_DIR_OVERRIDE}" && ! -d "${A_REUSE_PL_DIR}" ]]; then
  echo "[error] REUSE_PL_DIR_OVERRIDE not found: ${A_REUSE_PL_DIR}"
  exit 1
fi
ensure_text_prototypes() {
  if [[ -f "${PROTO_PATH}" ]]; then
    echo "[proto] found: ${PROTO_PATH}"
    return 0
  fi
  echo "[proto] missing: ${PROTO_PATH}"
  echo "[proto] auto building text prototypes..."
  echo "[proto] HF_ENDPOINT=${HF_ENDPOINT}"
  HF_ENDPOINT="${HF_ENDPOINT}" python tools/build_text_prototypes.py \
    --dataset "${DATASET}" \
    --class_set "${CLASS_SET}" \
    --text_model "${TEXT_MODEL}" \
    --text_pretrained "${TEXT_PRETRAINED}" \
    --out "${PROTO_PATH}"
  echo "[proto] built: ${PROTO_PATH}"
}

need_generate_only=true
if [[ "${SKIP_GENERATE_ONLY}" == "1" ]]; then
  need_generate_only=false
elif [[ "${SKIP_GENERATE_ONLY}" == "auto" && -d "${A_REUSE_PL_DIR}" ]]; then
  if compgen -G "${A_REUSE_PL_DIR}/*" > /dev/null; then
    need_generate_only=false
  fi
fi

# 0) Generate pseudo-labels once and cache
if [[ "${need_generate_only}" == "true" ]]; then
  run_generate_only_A
else
  echo "Skip A_generate_only (reuse existing pseudo-labels): ${A_REUSE_PL_DIR}"
fi

# Ensure prototypes exist before runs that need text branch.
ensure_text_prototypes
print_text_proto_meta

should_run() {
  local tag="$1"
  case "${ONLY}" in
    ABCD) return 0 ;;
    AB) [[ "${tag}" == A_* || "${tag}" == B_* ]] ;;
    CD) [[ "${tag}" == C_* || "${tag}" == D_* ]] ;;
    C) [[ "${tag}" == C_* ]] ;;
    D) [[ "${tag}" == D_* ]] ;;
    *) return 0 ;;
  esac
}

# A. baseline (reuse cached pseudo-labels)
if should_run "A_baseline"; then
  run_case "A_baseline" \
    --reuse_pl_dir "${A_REUSE_PL_DIR}" \
    --conf_thr "${CONF_THR}" \
    --no-use_text_proto \
    --no-use_text_gate_ssl \
    --no-use_language_adapter \
    --text_proto_path "" \
    --lambda_txt 0.0 \
    --weight_txt_align 0.0
else
  echo "Skip A_baseline (ONLY=${ONLY})"
fi

# B. visual-only gate (text branch on, text gate off)
if should_run "B_visual_gate_only"; then
  run_case "B_visual_gate_only" \
    --reuse_pl_dir "${A_REUSE_PL_DIR}" \
    --conf_thr "${CONF_THR}" \
    --batch_size "${TEXT_BATCH_SIZE}" \
    --use_text_proto \
    --no-use_text_gate_ssl \
    --no-use_language_adapter \
    --text_proto_path "${PROTO_PATH}" \
    --lambda_txt "${LAMBDA_TXT}" \
    --weight_txt_align "${WEIGHT_TXT_ALIGN}"
else
  echo "Skip B_visual_gate_only (ONLY=${ONLY})"
fi

# C. text-aware gate (hard, no adapter)
if should_run "C_text_gate_hard_no_adapter"; then
  run_case "C_text_gate_hard_no_adapter" \
    --reuse_pl_dir "${A_REUSE_PL_DIR}" \
    --conf_thr "${CONF_THR}" \
    --batch_size "${TEXT_BATCH_SIZE}" \
    --use_text_proto \
    --use_text_gate_ssl \
    --no-use_language_adapter \
    --text_proto_path "${PROTO_PATH}" \
    --lambda_txt "${LAMBDA_TXT}" \
    --weight_txt_align "${WEIGHT_TXT_ALIGN}"
else
  echo "Skip C_text_gate_hard_no_adapter (ONLY=${ONLY})"
fi

# D. text-aware gate (hard, + adapter)
if should_run "D_text_gate_hard_with_adapter"; then
  run_case "D_text_gate_hard_with_adapter" \
    --reuse_pl_dir "${A_REUSE_PL_DIR}" \
    --conf_thr "${CONF_THR}" \
    --batch_size "${TEXT_BATCH_SIZE}" \
    --use_text_proto \
    --use_text_gate_ssl \
    --use_language_adapter \
    --adapter_num_heads 4 \
    --adapter_scale 0.1 \
    --text_proto_path "${PROTO_PATH}" \
    --lambda_txt "${LAMBDA_TXT}" \
    --weight_txt_align "${WEIGHT_TXT_ALIGN}"
else
  echo "Skip D_text_gate_hard_with_adapter (ONLY=${ONLY})"
fi

echo "A/B/C/D experiments finished. Logs in: ${LOG_DIR}"

echo "Generating ABCD summary (CSV + Markdown)..."
python tools/summarize_abcd_results.py \
  --log_dirs "${LOG_DIR}" \
  --glob "multimodal_*_*.log" \
  --out_csv "${LOG_DIR}/abcd_summary.csv" \
  --out_md "${LOG_DIR}/abcd_summary.md"
echo "Summary generated:"
echo "  - ${LOG_DIR}/abcd_summary.csv"
echo "  - ${LOG_DIR}/abcd_summary.md"
