#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PAPER_CKPT=""
FINAL_CKPT=""
BASE_CKPT=""
PSEUDO_LABEL_ROOT=""
SOURCE_ROOT=""
MODE="quick"
DRY_RUN=false
ONLY="all"
LOG_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${REPO_ROOT}/logs/Standford2d3d_results/run_logs/run_standford2d3d_${LOG_TS}.log"

mkdir -p "${REPO_ROOT}/logs/Standford2d3d_results/"{reports,vis_check,val_ckpts,train_time_reuse_pl,summaries,nohup_logs,run_logs}

exec > >(tee -a "${RUN_LOG}") 2>&1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper_ckpt) PAPER_CKPT="$2"; shift 2 ;;
    --final_ckpt) FINAL_CKPT="$2"; shift 2 ;;
    --base_ckpt) BASE_CKPT="$2"; shift 2 ;;
    --pseudo_label_root) PSEUDO_LABEL_ROOT="$2"; shift 2 ;;
    --source_root) SOURCE_ROOT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

run_prepare() {
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[dry-run] 00_prepare_standford2d3d.sh"
    return 0
  fi
  bash scripts/extra_exp/standford2d3d/00_prepare_standford2d3d.sh
}

run_convert() {
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[dry-run] 00_convert_standford2d3d_labels.sh"
    return 0
  fi
  bash scripts/extra_exp/standford2d3d/00_convert_standford2d3d_labels.sh
}

run_check() {
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[dry-run] 00_check_standford2d3d.sh"
    return 0
  fi
  bash scripts/extra_exp/standford2d3d/00_check_standford2d3d.sh
}

run_val() {
  if [[ -z "${PAPER_CKPT}" || -z "${FINAL_CKPT}" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    D="${REPO_ROOT}/logs/Standford2d3d_results/val_ckpts/${TS}"
    mkdir -p "${D}"
    cat >"${D}/MISSING_CKPT.md" <<'EOF'
# MISSING_CKPT

val 步骤需要 --paper_ckpt 与 --final_ckpt（均为 indoor 8 类 OmniSAM checkpoint）。
EOF
    echo "[warn] val skipped -> ${D}/MISSING_CKPT.md"
    return 0
  fi
  EXTRA=()
  if [[ "${DRY_RUN}" == true ]]; then EXTRA+=(--dry-run); fi
  bash scripts/extra_exp/standford2d3d/01_val_ckpts_standford2d3d.sh \
    --paper_ckpt "${PAPER_CKPT}" \
    --final_ckpt "${FINAL_CKPT}" \
    "${EXTRA[@]}"
}

run_train_time() {
  if [[ -z "${BASE_CKPT}" || -z "${FINAL_CKPT}" || -z "${PSEUDO_LABEL_ROOT}" || -z "${SOURCE_ROOT}" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    D="${REPO_ROOT}/logs/Standford2d3d_results/train_time_reuse_pl/${TS}"
    mkdir -p "${D}"
    cat >"${D}/MISSING_PATHS.md" <<'EOF'
# MISSING_PATHS

train_time 需要: --base_ckpt --final_ckpt --pseudo_label_root --source_root
EOF
    echo "[warn] train_time skipped -> ${D}/MISSING_PATHS.md"
    return 0
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[dry-run] 02_train_time_standford2d3d_reuse_pl.sh (reuse_pl + spin8 source)"
    return 0
  fi
  bash scripts/extra_exp/standford2d3d/02_train_time_standford2d3d_reuse_pl.sh \
    --base_ckpt "${BASE_CKPT}" \
    --final_ckpt "${FINAL_CKPT}" \
    --pseudo_label_root "${PSEUDO_LABEL_ROOT}" \
    --source_root "${SOURCE_ROOT}" \
    --mode "${MODE}"
}

run_collect() {
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[dry-run] 03_collect_standford2d3d_results.sh"
    return 0
  fi
  bash scripts/extra_exp/standford2d3d/03_collect_standford2d3d_results.sh
}

case "${ONLY}" in
  prepare) run_prepare ;;
  convert) run_convert ;;
  check) run_check ;;
  val) run_val ;;
  train_time) run_train_time ;;
  collect) run_collect ;;
  all)
    run_prepare
    run_convert
    run_check
    run_val
    run_train_time
    run_collect
    ;;
  *)
    echo "unknown --only ${ONLY} (use prepare|convert|check|val|train_time|collect|all)"
    exit 2
    ;;
esac

echo "[done] log: ${RUN_LOG}"
