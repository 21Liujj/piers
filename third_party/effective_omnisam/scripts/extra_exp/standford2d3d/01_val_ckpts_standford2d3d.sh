#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PAPER_CKPT=""
FINAL_CKPT=""
DRY_RUN=false
ONLY="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper_ckpt) PAPER_CKPT="$2"; shift 2 ;;
    --final_ckpt) FINAL_CKPT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
VAL_ROOT="${REPO_ROOT}/logs/Standford2d3d_results/val_ckpts/${TS}"
mkdir -p "${VAL_ROOT}"

write_missing_ckpt() {
  cat >"$1" <<EOF
# MISSING_CKPT

需要提供有效的 indoor **8 类** OmniSAM checkpoint 路径。

- paper_ckpt: ${PAPER_CKPT:-empty}
- final_ckpt: ${FINAL_CKPT:-empty}
EOF
}

run_eval() {
  local ROLE="$1"
  local CKPT="$2"
  local SUB="$3"
  if [[ -z "${CKPT}" ]]; then
    write_missing_ckpt "${VAL_ROOT}/MISSING_CKPT.md"
    echo "[skip ${ROLE}] empty ckpt -> ${VAL_ROOT}/MISSING_CKPT.md"
    return 2
  fi
  if [[ ! -f "${CKPT}" ]]; then
    write_missing_ckpt "${VAL_ROOT}/MISSING_CKPT.md"
    echo "[skip ${ROLE}] not a file: ${CKPT}"
    return 2
  fi
  local OUT="${VAL_ROOT}/${SUB}"
  mkdir -p "${OUT}"
  EXTRA=()
  if [[ "${DRY_RUN}" == true ]]; then EXTRA+=(--dry-run); fi
  python3 "${REPO_ROOT}/tools/eval_standford2d3d_span8.py" \
    --ckpt "${CKPT}" \
    --out-dir "${OUT}" \
    --repo-root "${REPO_ROOT}" \
    --method-name "${ROLE}" \
    "${EXTRA[@]}"
}

if [[ "${ONLY}" == "all" || "${ONLY}" == "paper" ]]; then
  run_eval paper "${PAPER_CKPT}" paper_ckpt || true
fi
if [[ "${ONLY}" == "all" || "${ONLY}" == "final" ]]; then
  run_eval final "${FINAL_CKPT}" final_ckpt || true
fi

echo "[ok] val outputs under ${VAL_ROOT}"
