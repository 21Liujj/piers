#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

# Quick smoke: ensure 4 methods can start and write timing logs.
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export PL_BATCH_SIZE="${PL_BATCH_SIZE:-1}"
export PL_NUM_WORKERS="${PL_NUM_WORKERS:-0}"
export OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/time_ablation/smoke}"

bash "${SCRIPT_DIR}/run_time_ablation_main4.sh"
echo "Smoke launch completed. Inspect ${OUT_ROOT}/*/*/metrics/{pl_timing.jsonl,train_timing.jsonl}"
