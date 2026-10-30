#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_ROOT="${ROOT}/experiments/pano_smb_stage2"
MARKER_DIR="${EXP_ROOT}/markers"
MASTER_LOG="${EXP_ROOT}/stage2_master.log"
PID_FILE="${EXP_ROOT}/stage2.pid"

state="NOT_STARTED"
stage="PRECHECK"
if [[ -f "${PID_FILE}" ]]; then
  pid="$(tr -cd '0-9' < "${PID_FILE}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    state="RUNNING"
  fi
fi
if [[ -f "${MARKER_DIR}/.FAILED" ]]; then
  state="FAILED"
elif [[ -f "${MARKER_DIR}/.SUMMARY_DONE" ]]; then
  state="DONE"
elif [[ -f "${MARKER_DIR}/.B1_TRAIN_DONE" ]]; then
  stage="B1_EVAL"
elif [[ -f "${MARKER_DIR}/.B0_EVAL_DONE" ]]; then
  stage="B1_TRAIN"
elif [[ -f "${MARKER_DIR}/.B0_TRAIN_DONE" ]]; then
  stage="B0_EVAL"
elif [[ -f "${MARKER_DIR}/.PRECHECK_DONE" ]]; then
  stage="B0_TRAIN"
fi

printf '%s\n\n' "${state}"
printf 'Current stage: %s\n' "${stage}"
printf 'PID: %s\n' "${pid:-N/A}"
printf 'Master log: %s\n' "${MASTER_LOG}"
printf '\nMarkers:\n'
if [[ -d "${MARKER_DIR}" ]]; then
  find "${MARKER_DIR}" -maxdepth 1 -type f -printf '  %f\n' | sort
else
  printf '  none\n'
fi
printf '\nLast log lines:\n'
stage_log="${EXP_ROOT}/summary/${stage}.log"
if [[ -f "${stage_log}" ]]; then
  tail -n 30 "${stage_log}"
elif [[ -f "${MASTER_LOG}" ]]; then
  tail -n 30 "${MASTER_LOG}"
else
  printf '  no log yet\n'
fi
