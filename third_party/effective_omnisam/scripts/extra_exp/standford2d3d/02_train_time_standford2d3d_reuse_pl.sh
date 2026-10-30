#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

BASE_CKPT=""
FINAL_CKPT=""
PSEUDO_ROOT=""
SOURCE_ROOT=""
MODE="quick"
ITERS=500
EPOCHS=1
BATCH_SIZE=4
DRY_RUN=false
ONLY="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_ckpt) BASE_CKPT="$2"; shift 2 ;;
    --final_ckpt) FINAL_CKPT="$2"; shift 2 ;;
    --pseudo_label_root) PSEUDO_ROOT="$2"; shift 2 ;;
    --source_root) SOURCE_ROOT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${REPO_ROOT}/logs/Standford2d3d_results/train_time_reuse_pl/${TS}"
mkdir -p "${OUT_ROOT}/base" "${OUT_ROOT}/final_optimized"

write_missing_pl() {
  mkdir -p "$(dirname "$1")"
  cat >"$1" <<'EOF'
# MISSING_PL_ROOT

需要 **Stanford360 / SPan8** 伪标签目录（例如某次实验的 `.../epoch1`，包含与 Stanford 全景样本对齐的预测标签）。

这不是 DensePASS / WildPASS 的 **13 类** pseudo labels。

请手动提供 `--pseudo_label_root` 后重跑。
EOF
}

write_missing_paths() {
  cat >"$1" <<EOF
# MISSING_PATHS

缺少训练所需的 checkpoint 或路径参数。

- base_ckpt: ${BASE_CKPT:-"(empty)"}
- final_ckpt: ${FINAL_CKPT:-"(empty)"}
- pseudo_label_root: ${PSEUDO_ROOT:-"(empty)"}
EOF
}

write_missing_spin8() {
  cat >"$1" <<'EOF'
# MISSING_SPIN8_SOURCE

训练阶段需要从 **SPin8（Stanford Pin 视角）** 源域读取数据（`StanfordPin8DataSet` 默认根目录 `data_root/Stanford2d3d_Seg`）。

请提供 `--source_root` 指向 SPin8 数据根，或通过 `--stanford_spin8_root` 传入相同路径。
EOF
}

if [[ ! -d "${PSEUDO_ROOT}" ]]; then
  write_missing_pl "${OUT_ROOT}/MISSING_PL_ROOT.md"
  echo "[error] pseudo_label_root missing -> ${OUT_ROOT}/MISSING_PL_ROOT.md"
  exit 2
fi

if [[ -z "${BASE_CKPT}" || -z "${FINAL_CKPT}" ]]; then
  write_missing_paths "${OUT_ROOT}/MISSING_PATHS.md"
  echo "[error] missing ckpt args -> ${OUT_ROOT}/MISSING_PATHS.md"
  exit 2
fi

if [[ -z "${SOURCE_ROOT}" ]]; then
  write_missing_spin8 "${OUT_ROOT}/MISSING_SPIN8_SOURCE.md"
  echo "[error] source_root required -> ${OUT_ROOT}/MISSING_SPIN8_SOURCE.md"
  exit 2
fi

run_one() {
  local ROLE="$1"
  local CKPT="$2"
  local SUB="$3"
  local CFG="$4"
  local DDIR="${OUT_ROOT}/${SUB}"
  mkdir -p "${DDIR}"
  echo "${CKPT}" >"${DDIR}/checkpoint.txt"
  echo "${PSEUDO_ROOT}" >"${DDIR}/pseudo_label_root.txt"
  cat >"${DDIR}/config_used.yaml" <"${REPO_ROOT}/${CFG}"

  CMD=(
    python3
    run_adaptation.py
    --config "${REPO_ROOT}/${CFG}"
    --dataset Stanford2D3D
    --data_root data
    --stanford_processed_root data_extra/processed/Stanford360_SPAN8
    --stanford_flat_layout true
    --stanford_spin8_root "${SOURCE_ROOT}"
    --reuse_pl_dir "${PSEUDO_ROOT}"
    --regenerate_pl_each_epoch false
    --validate_pseudo_labels false
    --auto_rebuild_pseudo_labels false
    --tgt_ckpt "${CKPT}"
    --src_ckpt "${CKPT}"
    --pd_root "$(dirname "${PSEUDO_ROOT}")"
    --num_epochs "${EPOCHS}"
    --train_max_iters "${ITERS}"
    --batch_size "${BATCH_SIZE}"
    --amp true
    --tf32 true
  )

  if [[ "${DRY_RUN}" == true ]]; then
    echo "[dry-run ${ROLE}] ${CMD[*]}"
    return 0
  fi

  printf "%s\n" "${CMD[*]}" >"${DDIR}/command.txt"
  git -C "${REPO_ROOT}" diff -- . >"${DDIR}/git_diff.patch" 2>/dev/null || true

  START=$(date +%s)
  set +e
  "${CMD[@]}" 2>&1 | tee "${DDIR}/train.log"
  RC=${PIPESTATUS[0]}
  set -e
  END=$(date +%s)
  TRAIN_T=$((END - START))

  python3 <<PY
import json, math, os
sub = "${SUB}"
role = "${ROLE}"
ddir = "${DDIR}"
train_t = float(${TRAIN_T})
iters = float(${ITERS})
bs = float(${BATCH_SIZE})
avg_iter = train_t / max(iters, 1.0)
ips = (iters * bs) / max(train_t, 1e-9)
payload = {
    "dataset": "Stanford360_SPAN8",
    "method": role,
    "pseudo_label_reuse": True,
    "pseudo_label_root": r"""${PSEUDO_ROOT}""",
    "num_epochs": int(${EPOCHS}),
    "num_iters": int(${ITERS}),
    "batch_size": int(${BATCH_SIZE}),
    "train_time_sec": train_t,
    "avg_iter_time_sec": avg_iter,
    "images_per_sec": ips,
    "pl_time_sec": 0,
    "eval_time_sec": 0,
    "total_time_sec": train_t,
    "mIoU": None,
    "mAcc": None,
    "PixAcc": None,
}
open(os.path.join(ddir, "timing.json"), "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n")
PY

  echo "# ${ROLE} summary" >"${DDIR}/summary.md"
  echo "exit_code=${RC} train_time_sec=${TRAIN_T}" >>"${DDIR}/summary.md"
}

if [[ "${ONLY}" == "all" || "${ONLY}" == "base" ]]; then
  run_one base "${BASE_CKPT}" base configs/extra_exp/standford2d3d/stanford360_span8_train_base.yaml
fi
if [[ "${ONLY}" == "all" || "${ONLY}" == "final" || "${ONLY}" == "final_optimized" ]]; then
  run_one final_optimized "${FINAL_CKPT}" final_optimized configs/extra_exp/standford2d3d/stanford360_span8_train_final_optimized.yaml
fi

echo "[ok] outputs under ${OUT_ROOT}"
