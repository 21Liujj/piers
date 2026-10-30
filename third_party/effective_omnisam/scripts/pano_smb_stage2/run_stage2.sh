#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

EXP_ROOT="${ROOT}/experiments/pano_smb_stage2"
SUMMARY_DIR="${EXP_ROOT}/summary"
MARKER_DIR="${EXP_ROOT}/markers"
B0_CONFIG="${EXP_ROOT}/B0_legacy/config.yaml"
B1_CONFIG="${EXP_ROOT}/B1_pano_smb/config.yaml"
B0_RUN_DIR="${EXP_ROOT}/B0_legacy/run"
B1_RUN_DIR="${EXP_ROOT}/B1_pano_smb/run"
MASTER_LOG="${EXP_ROOT}/stage2_master.log"
PID_FILE="${EXP_ROOT}/stage2.pid"

PYTHON="${PYTHON:-/root/miniconda3/envs/omnisam/bin/python}"
GPU_ID="${GPU_ID:-0}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
FIXED_PL_DIR="${FIXED_PL_DIR:-${ROOT}/data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-${ROOT}/experiments/tecma/runs/_subsets/Cityscapes_13cls_seed1_n2000.txt}"
SOURCE_LIST="${ROOT}/dataset/cityscapes_list/train.txt"
TARGET_LIST="${ROOT}/dataset/densepass_list/train.txt"
VAL_LIST="${ROOT}/dataset/densepass_list/val.txt"
MIOU_ANCHOR="${MIOU_ANCHOR:-60.02}"
MIOU_TOLERANCE="${MIOU_TOLERANCE:-0.30}"
BFSCORE_ANCHOR="${BFSCORE_ANCHOR:-39.23}"
BFSCORE_TOLERANCE="${BFSCORE_TOLERANCE:-0.50}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
if [[ $# -gt "${DRY_RUN}" ]]; then
  printf 'usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

mkdir -p "${SUMMARY_DIR}" "${MARKER_DIR}"
if [[ "${DRY_RUN}" -eq 0 && "${STAGE2_INTERNAL_RUN:-0}" != "1" ]]; then
  [[ ! -e "${PID_FILE}" ]] || { printf 'stage2 pid file already exists: %s\n' "${PID_FILE}" >&2; exit 4; }
  [[ ! -e "${MASTER_LOG}" ]] || { printf 'refusing to overwrite master log: %s\n' "${MASTER_LOG}" >&2; exit 5; }
  STAGE2_INTERNAL_RUN=1 exec bash "${BASH_SOURCE[0]}" "$@" >> "${MASTER_LOG}" 2>&1
fi
if [[ "${DRY_RUN}" -eq 0 ]]; then
  printf '%s\n' "$$" > "${PID_FILE}"
fi

CURRENT_STAGE="STARTUP"
on_error() {
  local rc=$?
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    printf 'stage=%s exit_code=%s timestamp=%s\n' \
      "${CURRENT_STAGE}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MARKER_DIR}/.FAILED"
  fi
  exit "${rc}"
}
trap on_error ERR
trap '[[ "${DRY_RUN}" -eq 1 ]] || rm -f "${PID_FILE}"' EXIT

record_stage() {
  local stage="$1" start="$2" end="$3" elapsed="$4" rc="$5" command="$6"
  "${PYTHON}" - "${SUMMARY_DIR}/${stage}.json" "${stage}" "${start}" "${end}" "${elapsed}" "${rc}" "${command}" <<'PY'
import json, sys
out, stage, start, end, elapsed, rc, command = sys.argv[1:]
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"stage": stage, "start": start, "end": end, "elapsed_seconds": float(elapsed), "exit_code": int(rc), "command": command}, handle, indent=2)
    handle.write("\n")
PY
}

run_stage() {
  local stage="$1"
  shift
  local command_text start end start_s end_s rc elapsed
  CURRENT_STAGE="${stage}"
  printf -v command_text '%q ' "$@"
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_s="$(date +%s)"
  printf '[%s] START %s\n' "${start}" "${stage}"
  printf '%s\n' "${command_text}" > "${SUMMARY_DIR}/${stage}.command.txt"
  set +e
  "$@" > "${SUMMARY_DIR}/${stage}.log" 2>&1
  rc=$?
  set -e
  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_s="$(date +%s)"
  elapsed="$((end_s - start_s))"
  printf '%s\n' "${rc}" > "${SUMMARY_DIR}/${stage}.exit_code"
  record_stage "${stage}" "${start}" "${end}" "${elapsed}" "${rc}" "${command_text}"
  printf '[%s] END %s rc=%s elapsed=%ss\n' "${end}" "${stage}" "${rc}" "${elapsed}"
  return "${rc}"
}

json_field_equals() {
  "${PYTHON}" - "$1" "$2" "$3" <<'PY'
import json, sys
path, key, expected = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
for part in key.split("."):
    value = value[part]
if str(value).lower() != expected.lower():
    raise SystemExit(f"{path}: {key}={value!r}, expected {expected!r}")
PY
}

validate_run_artifacts() {
  local run_dir="$1"
  local save_dir="${run_dir}/checkpoints/Cityscapes_13cls/sam2_s"
  [[ -s "${run_dir}/launch_command.txt" ]]
  [[ -s "${run_dir}/run.log" && -s "${run_dir}/training.log" && -s "${run_dir}/evaluation.log" ]]
  [[ -s "${run_dir}/train_stage.json" && -s "${run_dir}/eval_stage.json" ]]
  json_field_equals "${run_dir}/train_stage.json" process_exit_code 0
  json_field_equals "${run_dir}/train_stage.json" boundary_observed true
  json_field_equals "${run_dir}/eval_stage.json" process_exit_code 0
  [[ -s "${save_dir}/metrics.jsonl" && -s "${save_dir}/metrics/train_timing.jsonl" ]]
  grep -q '\[ValSummary\] BFScore=' "${run_dir}/run.log"
  find "${save_dir}" -maxdepth 1 -type f -name 'best_iou*.pth' -print -quit | grep -q .
}

precheck() {
  [[ -x "${PYTHON}" ]]
  [[ -f "${SRC_CKPT}" && -f "${TGT_CKPT}" ]]
  [[ -d "${FIXED_PL_DIR}/gtFine" && -f "${FIXED_PL_DIR}/used_path.txt" ]]
  [[ "$(find "${FIXED_PL_DIR}/gtFine" -type f | wc -l)" -ge 2000 ]]
  [[ -f "${PL_SUBSET_FILE}" && "$(wc -l < "${PL_SUBSET_FILE}")" -eq 2000 ]]
  [[ -f "${SOURCE_LIST}" && -f "${TARGET_LIST}" && -f "${VAL_LIST}" ]]
  [[ -f "${B0_CONFIG}" && -f "${B1_CONFIG}" && -f "${ROOT}/run_adaptation.py" ]]
  "${PYTHON}" "${SCRIPT_DIR}/check_config_diff.py" \
    --b0 "${B0_CONFIG}" --b1 "${B1_CONFIG}" --output "${SUMMARY_DIR}/config_diff.txt"
  "${PYTHON}" - "${B0_CONFIG}" "${B1_CONFIG}" <<'PY'
import sys, yaml
for path, expected in zip(sys.argv[1:], (False, True)):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    plugins = cfg["plugins"]
    assert plugins["pseudo_conf_mask"]["enabled"] is False
    assert plugins["efficient_tam_memory"]["enabled"] is True
    assert plugins["efficient_sam2_sparse"]["enabled"] is True
    assert plugins["efficient_sam2_sparse"]["saliency_source"] == "feature_norm"
    assert plugins["pano_smb"]["enabled"] is expected
    assert plugins["pano_smb"]["min_tokens"] == 256
PY
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible")
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("PyTorch:", torch.__version__)
PY
  fi
  {
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'fixed_pl_dir=%s\n' "${FIXED_PL_DIR}"
    printf 'subset=%s\n' "${PL_SUBSET_FILE}"
    printf 'dry_run=%s\n' "${DRY_RUN}"
    printf 'git_status_begin\n'
    git status --short
    printf 'git_status_end\n'
  } > "${SUMMARY_DIR}/precheck_environment.txt"
}

build_command() {
  local config="$1" run_dir="$2"
  printf '%s\0' \
    env WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" run_adaptation.py \
    --config "${config}" --seed 1 --dataset Cityscapes --class_set 13 --backbone sam2_s \
    --data_root data --pd_root "${run_dir}" --src_ckpt "${SRC_CKPT}" --tgt_ckpt "${TGT_CKPT}" \
    --amp --tf32 --num_epochs 1 --batch_size 4 --pl_batch_size 1 --pl_num_workers 2 \
    --no-pl_shuffle --pl_subset_file "${PL_SUBSET_FILE}" --warmup_pl 2000 --iters_pl 2000 \
    --no-use_text_proto --text_proto_path '' --lambda_txt 0.0 --weight_txt_align 0.0 \
    --no-use_text_gate_ssl --no-use_wandb --no-use_tensorboard --reuse_pl_dir "${FIXED_PL_DIR}"
}

print_command() {
  local label="$1" config="$2" run_dir="$3"
  local -a command
  mapfile -d '' -t command < <(build_command "${config}" "${run_dir}")
  printf '%s: ' "${label}"
  printf '%q ' "${PYTHON}" "${SCRIPT_DIR}/run_integrated.py" --label "${label}" --run-dir "${run_dir}" --marker-dir "${MARKER_DIR}" -- "${command[@]}"
  printf '\n'
}

run_experiment() {
  local label="$1" config="$2" run_dir="$3"
  local -a command wrapper_args
  mapfile -d '' -t command < <(build_command "${config}" "${run_dir}")
  wrapper_args=(--label "${label}" --run-dir "${run_dir}" --marker-dir "${MARKER_DIR}")
  if [[ -e "${run_dir}" ]]; then
    if [[ -f "${run_dir}/train_stage.json" && -f "${run_dir}/eval_stage.json" ]] && \
       json_field_equals "${run_dir}/train_stage.json" process_exit_code 0 && \
       json_field_equals "${run_dir}/eval_stage.json" process_exit_code 0; then
      validate_run_artifacts "${run_dir}"
      return 0
    fi
    [[ -s "${run_dir}/checkpoints/Cityscapes_13cls/sam2_s/resume_sam2_s.pth" ]] || {
      printf 'incomplete run has no valid resume checkpoint: %s\n' "${run_dir}" >&2
      return 20
    }
    wrapper_args+=(--resume)
  fi
  "${PYTHON}" "${SCRIPT_DIR}/run_integrated.py" \
    "${wrapper_args[@]}" -- "${command[@]}"
  validate_run_artifacts "${run_dir}"
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
  CURRENT_STAGE="DRY_RUN_PRECHECK"
  precheck
  print_command B0 "${B0_CONFIG}" "${B0_RUN_DIR}"
  print_command B1 "${B1_CONFIG}" "${B1_RUN_DIR}"
  printf 'DRY-RUN PASS: no training or evaluation command executed.\n'
  exit 0
fi

if [[ ! -f "${MARKER_DIR}/.PRECHECK_DONE" ]]; then
  run_stage PRECHECK precheck
  touch "${MARKER_DIR}/.PRECHECK_DONE"
else
  [[ -s "${SUMMARY_DIR}/config_diff.txt" && -s "${SUMMARY_DIR}/precheck_environment.txt" ]]
fi

if [[ -f "${MARKER_DIR}/.B0_EVAL_DONE" ]]; then
  validate_run_artifacts "${B0_RUN_DIR}"
else
  run_stage B0_INTEGRATED run_experiment B0 "${B0_CONFIG}" "${B0_RUN_DIR}"
  validate_run_artifacts "${B0_RUN_DIR}"
  touch "${MARKER_DIR}/.B0_TRAIN_DONE" "${MARKER_DIR}/.B0_EVAL_DONE"
fi
run_stage B0_METRICS "${PYTHON}" "${SCRIPT_DIR}/extract_metrics.py" \
  --run-dir "${B0_RUN_DIR}" --output "${SUMMARY_DIR}/B0_metrics.json" --label B0
run_stage B0_SANITY "${PYTHON}" "${SCRIPT_DIR}/check_b0_sanity.py" \
  --metrics "${SUMMARY_DIR}/B0_metrics.json" --output "${SUMMARY_DIR}/B0_sanity.json" \
  --miou-anchor "${MIOU_ANCHOR}" --miou-tolerance "${MIOU_TOLERANCE}" \
  --bf-anchor "${BFSCORE_ANCHOR}" --bf-tolerance "${BFSCORE_TOLERANCE}"
json_field_equals "${SUMMARY_DIR}/B0_sanity.json" status PASS
touch "${MARKER_DIR}/.B0_METRICS_DONE" "${MARKER_DIR}/.B0_SANITY_DONE"

if [[ -f "${MARKER_DIR}/.B1_EVAL_DONE" ]]; then
  validate_run_artifacts "${B1_RUN_DIR}"
else
  run_stage B1_INTEGRATED run_experiment B1 "${B1_CONFIG}" "${B1_RUN_DIR}"
  validate_run_artifacts "${B1_RUN_DIR}"
  touch "${MARKER_DIR}/.B1_TRAIN_DONE" "${MARKER_DIR}/.B1_EVAL_DONE"
fi
run_stage B1_METRICS "${PYTHON}" "${SCRIPT_DIR}/extract_metrics.py" \
  --run-dir "${B1_RUN_DIR}" --output "${SUMMARY_DIR}/B1_metrics.json" --label B1
touch "${MARKER_DIR}/.B1_METRICS_DONE"

QUOTA_JSON="${B1_RUN_DIR}/checkpoints/Cityscapes_13cls/sam2_s/metrics/pano_smb_quota_summary.json"
run_stage QUOTA_AGGREGATION "${PYTHON}" "${SCRIPT_DIR}/validate_quota.py" \
  --input "${QUOTA_JSON}" --output "${SUMMARY_DIR}/B1_quota_summary.json"
touch "${MARKER_DIR}/.QUOTA_DONE"

run_stage SUMMARY "${PYTHON}" "${SCRIPT_DIR}/build_summary.py" \
  --b0 "${SUMMARY_DIR}/B0_metrics.json" --b1 "${SUMMARY_DIR}/B1_metrics.json" \
  --quota "${SUMMARY_DIR}/B1_quota_summary.json" --sanity "${SUMMARY_DIR}/B0_sanity.json" \
  --output-json "${SUMMARY_DIR}/stage2_summary.json" --output-md "${SUMMARY_DIR}/stage2_summary.md" \
  --per-class-csv "${SUMMARY_DIR}/per_class_delta.csv"
[[ -s "${SUMMARY_DIR}/stage2_summary.json" && -s "${SUMMARY_DIR}/stage2_summary.md" && -s "${SUMMARY_DIR}/per_class_delta.csv" ]]
touch "${MARKER_DIR}/.SUMMARY_DONE"
rm -f "${MARKER_DIR}/.FAILED"
printf 'Stage 2 runtime complete. Human scientific analysis is still required.\n'
