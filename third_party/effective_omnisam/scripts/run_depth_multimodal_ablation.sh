#!/usr/bin/env bash
set -euo pipefail

# Unified depth ablation runner on optimized base:
#   1) Optimized Base
#   2) D-Conf
#   3) D-Adapter
#   4) D-Conf + D-Adapter
#
# Usage:
#   bash scripts/run_depth_multimodal_ablation.sh
#
# Optional env overrides:
#   MODE=smoke|formal                 (default: smoke)
#   BASE_CFG=...                      (default: configs/ablations/omnisam_A_plus4train_plus5train.yaml)
#   DEPTH_DIR=...                     (default: /root/autodl-tmp/OmniSAM-main/aux_outputs/depth_anything_v2/WildPASS)
#   DATASET=Cityscapes
#   BACKBONE=sam2_s
#   BATCH_SIZE=4
#   PL_BATCH_SIZE=2
#   PL_NUM_WORKERS=4
#   NUM_EPOCHS=1                      (smoke default)
#   WARMUP_PL=50                      (smoke default)
#   ITERS_PL=50                       (smoke default)
#   TRAIN_MAX_ITERS=20                (smoke default)
#   RUN_TAG=depth_mm_ablation
#   LOG_ROOT=logs/pseudo-depth
#   PL_MODE=separate|shared              (default: separate)
#   FORCE_REGEN_PL=true|false            (default: true)
#   SEED=1234
#   PL_SUBSET_FILE=...                   (default: auto-generate fixed subset file)
#   PL_SUBSET_SIZE=300                   (default: 300)
#   PL_SUBSET_SEED=1234                  (default: 1234)
#   SKIP_BASE=true|false                 (default: false)
#   ONLY_BASE=true|false                 (default: false)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OMP_NUM_THREADS_RAW="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS_RAW="${MKL_NUM_THREADS:-8}"
if [[ ! "${OMP_NUM_THREADS_RAW}" =~ ^[0-9]+$ ]]; then
  echo "[warn] Invalid OMP_NUM_THREADS='${OMP_NUM_THREADS_RAW}', fallback to 8"
  OMP_NUM_THREADS_RAW=8
fi
if [[ ! "${MKL_NUM_THREADS_RAW}" =~ ^[0-9]+$ ]]; then
  echo "[warn] Invalid MKL_NUM_THREADS='${MKL_NUM_THREADS_RAW}', fallback to 8"
  MKL_NUM_THREADS_RAW=8
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS_RAW}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_RAW}"

MODE="${MODE:-smoke}"
BASE_CFG="${BASE_CFG:-configs/ablations/omnisam_A_plus4train_plus5train.yaml}"
DEPTH_DIR="${DEPTH_DIR:-/root/autodl-tmp/OmniSAM-main/aux_outputs/depth_anything_v2/WildPASS}"
DATASET="${DATASET:-Cityscapes}"
BACKBONE="${BACKBONE:-sam2_s}"
BATCH_SIZE_DEFAULT=4
PL_BATCH_SIZE_DEFAULT=2
BATCH_SIZE="${BATCH_SIZE:-${BATCH_SIZE_DEFAULT}}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-${PL_BATCH_SIZE_DEFAULT}}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-4}"
RUN_TAG="${RUN_TAG:-depth_mm_ablation}"
LOG_ROOT="${LOG_ROOT:-logs/pseudo-depth}"
PL_MODE="${PL_MODE:-separate}"
FORCE_REGEN_PL="${FORCE_REGEN_PL:-true}"
SEED="${SEED:-1234}"
PL_SUBSET_SIZE="${PL_SUBSET_SIZE:-300}"
PL_SUBSET_SEED="${PL_SUBSET_SEED:-1234}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-}"
SKIP_BASE="${SKIP_BASE:-false}"
ONLY_BASE="${ONLY_BASE:-false}"

if [[ "${MODE}" == "formal" ]]; then
  NUM_EPOCHS="${NUM_EPOCHS:-10}"
  WARMUP_PL="${WARMUP_PL:-2000}"
  ITERS_PL="${ITERS_PL:-2000}"
  TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-0}"
else
  NUM_EPOCHS="${NUM_EPOCHS:-1}"
  WARMUP_PL="${WARMUP_PL:-50}"
  ITERS_PL="${ITERS_PL:-50}"
  TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-20}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_TAG}_${TS}}"
mkdir -p "${LOG_DIR}"
CONSOLE_LOG="${LOG_DIR}/00_console.log"
# Save full console output into logs directory while still printing to terminal.
exec > >(tee -a "${CONSOLE_LOG}") 2>&1

if [[ -z "${PL_SUBSET_FILE}" ]]; then
  PL_SUBSET_FILE="${LOG_ROOT}/subsets/wildpass_${PL_SUBSET_SIZE}_seed${PL_SUBSET_SEED}.txt"
fi
mkdir -p "$(dirname "${PL_SUBSET_FILE}")"
if [[ ! -f "${PL_SUBSET_FILE}" ]]; then
  echo "[step] Build fixed PL subset file: ${PL_SUBSET_FILE}"
  python - "dataset/densepass_list/train.txt" "${PL_SUBSET_FILE}" "${PL_SUBSET_SIZE}" "${PL_SUBSET_SEED}" <<'PY'
import random
import sys

src, dst, k, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, "r", encoding="utf-8") as f:
    items = [ln.strip() for ln in f if ln.strip()]
if not items:
    raise RuntimeError(f"Empty list file: {src}")
rng = random.Random(seed)
if k >= len(items):
    subset = items
else:
    subset = rng.sample(items, k)
with open(dst, "w", encoding="utf-8") as f:
    for x in subset:
        f.write(x + "\n")
print(f"[subset] source={src}")
print(f"[subset] total={len(items)} selected={len(subset)} seed={seed}")
print(f"[subset] saved={dst}")
PY
else
  echo "[step] Reuse fixed PL subset file: ${PL_SUBSET_FILE}"
fi

COMMON_ARGS=(
  --config "${BASE_CFG}"
  --dataset "${DATASET}"
  --backbone "${BACKBONE}"
  --resume_ckpt "${LOG_DIR}/__no_resume__.pth"
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --pl_batch_size "${PL_BATCH_SIZE}"
  --pl_num_workers "${PL_NUM_WORKERS}"
  --num_epochs "${NUM_EPOCHS}"
  --warmup_pl "${WARMUP_PL}"
  --iters_pl "${ITERS_PL}"
  --train_max_iters "${TRAIN_MAX_ITERS}"
  --pl_subset_file "${PL_SUBSET_FILE}"
  --amp
  --tf32
)

echo "[info] Root: ${ROOT_DIR}"
echo "[info] Mode: ${MODE}"
echo "[info] Base cfg: ${BASE_CFG}"
echo "[info] Depth dir: ${DEPTH_DIR}"
echo "[info] OMP_NUM_THREADS: ${OMP_NUM_THREADS}, MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "[info] Batch size: ${BATCH_SIZE}, PL batch size: ${PL_BATCH_SIZE}"
echo "[info] Logs: ${LOG_DIR}"
echo "[info] PL mode: ${PL_MODE}"
echo "[info] FORCE_REGEN_PL: ${FORCE_REGEN_PL}"
echo "[info] PL subset file: ${PL_SUBSET_FILE}"
echo "[info] SKIP_BASE: ${SKIP_BASE}"
echo "[info] ONLY_BASE: ${ONLY_BASE}"
{
  echo "[meta] start_time=$(date '+%F %T')"
  echo "[meta] mode=${MODE}"
  echo "[meta] base_cfg=${BASE_CFG}"
  echo "[meta] depth_dir=${DEPTH_DIR}"
  echo "[meta] dataset=${DATASET}"
  echo "[meta] backbone=${BACKBONE}"
  echo "[meta] batch_size=${BATCH_SIZE}"
  echo "[meta] pl_batch_size=${PL_BATCH_SIZE}"
  echo "[meta] num_epochs=${NUM_EPOCHS}"
  echo "[meta] warmup_pl=${WARMUP_PL}"
  echo "[meta] iters_pl=${ITERS_PL}"
  echo "[meta] train_max_iters=${TRAIN_MAX_ITERS}"
  echo "[meta] pl_mode=${PL_MODE}"
  echo "[meta] force_regen_pl=${FORCE_REGEN_PL}"
  echo "[meta] seed=${SEED}"
  echo "[meta] pl_subset_file=${PL_SUBSET_FILE}"
  echo "[meta] pl_subset_size=${PL_SUBSET_SIZE}"
  echo "[meta] pl_subset_seed=${PL_SUBSET_SEED}"
} | tee "${LOG_DIR}/00_run_meta.log"

echo "[step] Depth coverage check..."
python tools/check_depth_prior.py \
  --data_root data/WildPASS \
  --list_path dataset/densepass_list/train.txt \
  --depth_dir "${DEPTH_DIR}" | tee "${LOG_DIR}/00_depth_check.log"

run_case() {
  local case_name="$1"
  local exp_name="$2"
  local opts="$3"
  local log_file="${LOG_DIR}/${case_name}.log"
  local status_file="${LOG_DIR}/${case_name}.status"
  local pd_root=""
  local pl_meta_file=""
  local existed_epoch1=0
  if [[ "${PL_MODE}" == "shared" ]]; then
    pd_root="${LOG_DIR}/pl_cache/fixed_base_pl"
  else
    pd_root="${LOG_DIR}/pl_cache/${exp_name}"
  fi
  pl_meta_file="${pd_root}/pl_meta.json"
  if [[ -d "${pd_root}/DensePASS/${BACKBONE}/${DATASET}_13cls/epoch1" ]]; then
    existed_epoch1=1
  fi
  mkdir -p "${pd_root}"

  if [[ -f "${pl_meta_file}" ]]; then
    set +e
    python - "${pl_meta_file}" "${PL_MODE}" "${exp_name}" "${BASE_CFG}" "${DEPTH_DIR}" "${SEED}" <<'PY'
import json, sys
meta_path, pl_mode, exp_name, base_cfg, depth_dir, seed = sys.argv[1:]
ok = True
msg = []
with open(meta_path, "r", encoding="utf-8") as f:
    m = json.load(f)
for k, v in {
    "pl_mode": pl_mode,
    "exp_name": exp_name if pl_mode == "separate" else "fixed_base_pl",
    "base_cfg": base_cfg,
    "depth_dir": depth_dir,
    "seed": int(seed),
}.items():
    if m.get(k) != v:
        ok = False
        msg.append(f"{k}: expected={v} got={m.get(k)}")
if not ok:
    print("[warn] existing pl_meta mismatch: " + "; ".join(msg))
    sys.exit(2)
print("[info] existing pl_meta matches current setup")
PY
    meta_check_code=$?
    set -e
    if [[ ${meta_check_code} -ne 0 && "${FORCE_REGEN_PL}" != "true" ]]; then
      echo "[error] pl_meta mismatch and FORCE_REGEN_PL=false for ${case_name}"
      exit 2
    fi
  fi

  if [[ "${FORCE_REGEN_PL}" == "true" ]]; then
    echo "[info] FORCE_REGEN_PL=true, clearing pd_root: ${pd_root}"
    rm -rf "${pd_root}"
    mkdir -p "${pd_root}"
    existed_epoch1=0
  fi

  python - "${pl_meta_file}" "${PL_MODE}" "${exp_name}" "${BASE_CFG}" "${DEPTH_DIR}" "${SEED}" "${case_name}" "${opts}" <<'PY'
import json, os, sys, datetime
path, pl_mode, exp_name, base_cfg, depth_dir, seed, case_name, opts = sys.argv[1:]
os.makedirs(os.path.dirname(path), exist_ok=True)
meta = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "pl_mode": pl_mode,
    "exp_name": exp_name if pl_mode == "separate" else "fixed_base_pl",
    "case_name": case_name,
    "base_cfg": base_cfg,
    "depth_dir": depth_dir,
    "seed": int(seed),
    "opts": opts,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
PY

  echo "[run] ${case_name}"
  echo "[run] opts=${opts}"
  echo "[run] pd_root=${pd_root}"
  echo "[run] pl_mode=${PL_MODE}"
  echo "[run] force_regen_pl=${FORCE_REGEN_PL}"
  echo "[run] existed_epoch1_before_run=${existed_epoch1}"
  {
    echo "case=${case_name}"
    echo "start_time=$(date '+%F %T')"
    echo "opts=${opts}"
    echo "pd_root=${pd_root}"
    echo "pl_mode=${PL_MODE}"
    echo "force_regen_pl=${FORCE_REGEN_PL}"
    echo "existed_epoch1_before_run=${existed_epoch1}"
  } > "${status_file}"
  set +e
  python run_adaptation.py \
    "${COMMON_ARGS[@]}" \
    --pd_root "${pd_root}" \
    --opts "${opts}" 2>&1 | tee "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  set -e
  {
    echo "end_time=$(date '+%F %T')"
    echo "exit_code=${exit_code}"
  } >> "${status_file}"
  if [[ ${exit_code} -ne 0 ]]; then
    echo "[error] ${case_name} failed with exit_code=${exit_code}. See ${log_file}"
    exit ${exit_code}
  fi
}

# IMPORTANT:
# - plugin path: efficient_tam_memory.apply_in_pl=false
# - main mem-compression path: mem_compression_enable_in_pseudo_label=false
# This enforces "compress in train only, never in pseudo-label stage".
ACCEL_OPTS="plugins.pseudo_conf_mask.enabled=true,plugins.efficient_tam_memory.enabled=true,plugins.efficient_tam_memory.apply_in_train=true,plugins.efficient_tam_memory.apply_in_pl=false,plugins.efficient_sam2_sparse.enabled=true,plugins.efficient_sam2_sparse.apply_in_train=true,plugins.efficient_sam2_sparse.apply_in_pl=false,mem_compression_enable_in_train=true,mem_compression_enable_in_pseudo_label=false"

if [[ "${SKIP_BASE}" != "true" ]]; then
  run_case "01_optimized_base" \
    "base" \
  "depth_prior.enable=false,depth_confidence.enable=false,depth_adapter.enable=false,${ACCEL_OPTS}"
else
  echo "[skip] 01_optimized_base (SKIP_BASE=true)"
fi

if [[ "${ONLY_BASE}" == "true" ]]; then
  echo "[skip] ONLY_BASE=true, skip 02_d_conf/03_d_adapter/04_d_conf_d_adapter"
else
  run_case "02_d_conf" \
    "d_conf" \
  "depth_prior.enable=true,depth_prior.depth_dir=${DEPTH_DIR},depth_prior.normalize=minmax,depth_confidence.enable=true,depth_confidence.edge_consistency=true,depth_confidence.smooth_region_consistency=true,depth_confidence.weight_min=0.3,depth_confidence.weight_max=1.0,depth_confidence.edge_beta=0.5,depth_confidence.smooth_beta=0.5,depth_adapter.enable=false,${ACCEL_OPTS}"

  run_case "03_d_adapter" \
    "d_adapter" \
  "depth_prior.enable=true,depth_prior.depth_dir=${DEPTH_DIR},depth_prior.normalize=minmax,depth_confidence.enable=false,depth_adapter.enable=true,depth_adapter.inject_position=decoder_feature,depth_adapter.encoder_type=light_cnn,depth_adapter.mid_channels=32,depth_adapter.norm=gn,depth_adapter.gate_init=0.0,depth_adapter.depth_loss_weight=0.0,${ACCEL_OPTS}"

  run_case "04_d_conf_d_adapter" \
    "d_conf_adapter" \
  "depth_prior.enable=true,depth_prior.depth_dir=${DEPTH_DIR},depth_prior.normalize=minmax,depth_confidence.enable=true,depth_confidence.edge_consistency=true,depth_confidence.smooth_region_consistency=true,depth_confidence.weight_min=0.3,depth_confidence.weight_max=1.0,depth_confidence.edge_beta=0.5,depth_confidence.smooth_beta=0.5,depth_adapter.enable=true,depth_adapter.inject_position=decoder_feature,depth_adapter.encoder_type=light_cnn,depth_adapter.mid_channels=32,depth_adapter.norm=gn,depth_adapter.gate_init=0.0,depth_adapter.depth_loss_weight=0.0,${ACCEL_OPTS}"
fi

echo "[done] All 4 ablation runs finished."
echo "[done] Check logs under: ${LOG_DIR}"
echo "[meta] end_time=$(date '+%F %T')" | tee -a "${LOG_DIR}/00_run_meta.log"

