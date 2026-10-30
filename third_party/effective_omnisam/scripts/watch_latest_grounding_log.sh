#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

LOG_BASE="logs/text_grounding_diagnose"
if [[ ! -d "${LOG_BASE}" ]]; then
  echo "No log directory: ${LOG_BASE}"
  exit 1
fi

LATEST_LOG="$(ls -1t "${LOG_BASE}"/*/00_master.log 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_LOG}" ]]; then
  echo "No 00_master.log found under ${LOG_BASE}"
  exit 1
fi

echo "${LATEST_LOG}"
tail -f "${LATEST_LOG}"
