#!/usr/bin/env bash
set -euo pipefail

# Two-stage text ablation runner:
# Stage-1 semantic causality: A0/A1/A4/A5
# Stage-2 gate efficacy: A1/A2/A3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-3}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-8}"
SKIP_A0="${SKIP_A0:-0}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"

run_phase() {
  local phase="$1"
  local tag="$2"
  echo "========== Run phase=${phase}, tag=${tag} =========="
  ABLT_PHASE="${phase}" \
  RUN_TAG="${tag}" \
  SKIP_A0="${SKIP_A0}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
  PL_NUM_WORKERS="${PL_NUM_WORKERS}" \
  RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND}" \
  LOG_DIR="${LOG_DIR}" \
  bash "${SCRIPT_DIR}/run_multimodal_ablation_a0_a4.sh"
}

run_phase "semantic" "text_semantic_a0_a1_a4_a5"
run_phase "gate" "text_gate_a1_a2_a3"

echo "All two-stage text ablations finished."
