#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# Two-stage sweep:
# Stage 1: run ONLY=D for all configs, rank by mIoU, pick TopK.
# Stage 2: deep-run TopK with D (3-5 epochs), optional C review.
TAU_T_LIST="${TAU_T_LIST:-0.05 0.08 0.10}"
SSL_GATE_MODE_LIST="${SSL_GATE_MODE_LIST:-hard soft}"
WEIGHT_TXT_ALIGN_LIST="${WEIGHT_TXT_ALIGN_LIST:-0.02 0.05}"

LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
BASE_RUN_TAG="${BASE_RUN_TAG:-cdsweep}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
INTERNAL_BG="${INTERNAL_BG:-0}"
SKIP_GENERATE_ONLY="${SKIP_GENERATE_ONLY:-auto}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-2}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-2}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
SHARED_REUSE_PL_DIR="${SHARED_REUSE_PL_DIR:-data_pd_abcd_A_baseline/DensePASS/sam2_s/Cityscapes_13cls/epoch1}"

# Stage control
PHASE1_ONLY="${PHASE1_ONLY:-D}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-1}"
TOPK="${TOPK:-2}"
ENABLE_STAGE2="${ENABLE_STAGE2:-1}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-3}"
PHASE2_MAIN_ONLY="${PHASE2_MAIN_ONLY:-D}"
PHASE2_RUN_C_REVIEW="${PHASE2_RUN_C_REVIEW:-1}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/${BASE_RUN_TAG}_${STAMP}.master.log"
PHASE1_RESULTS_CSV="${LOG_DIR}/${BASE_RUN_TAG}_${STAMP}.phase1_results.csv"
TOPK_CSV="${LOG_DIR}/${BASE_RUN_TAG}_${STAMP}.topk.csv"
PID_FILE="${PID_FILE:-${LOG_DIR}/${BASE_RUN_TAG}.pid}"

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${INTERNAL_BG}" != "1" ]]; then
  nohup env INTERNAL_BG=1 RUN_IN_BACKGROUND=0 \
    LOG_DIR="${LOG_DIR}" BASE_RUN_TAG="${BASE_RUN_TAG}" \
    TAU_T_LIST="${TAU_T_LIST}" SSL_GATE_MODE_LIST="${SSL_GATE_MODE_LIST}" WEIGHT_TXT_ALIGN_LIST="${WEIGHT_TXT_ALIGN_LIST}" \
    SKIP_GENERATE_ONLY="${SKIP_GENERATE_ONLY}" HF_ENDPOINT="${HF_ENDPOINT}" \
    BATCH_SIZE="${BATCH_SIZE}" TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE}" PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
    SRC_CKPT="${SRC_CKPT}" TGT_CKPT="${TGT_CKPT}" SHARED_REUSE_PL_DIR="${SHARED_REUSE_PL_DIR}" \
    PHASE1_ONLY="${PHASE1_ONLY}" PHASE1_EPOCHS="${PHASE1_EPOCHS}" TOPK="${TOPK}" \
    ENABLE_STAGE2="${ENABLE_STAGE2}" PHASE2_EPOCHS="${PHASE2_EPOCHS}" PHASE2_MAIN_ONLY="${PHASE2_MAIN_ONLY}" \
    PHASE2_RUN_C_REVIEW="${PHASE2_RUN_C_REVIEW}" PID_FILE="${PID_FILE}" \
    bash "${SCRIPT_DIR}/run_multimodal_cd_sweep.sh" > "${MASTER_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Started in background."
  echo "PID: ${pid} (saved to ${PID_FILE})"
  echo "Master log: ${MASTER_LOG}"
  exit 0
fi

{
  echo "=========================================="
  echo "Two-stage CD sweep started: $(date -Iseconds)"
  echo "ROOT=${ROOT}"
  echo "TAU_T_LIST=${TAU_T_LIST}"
  echo "SSL_GATE_MODE_LIST=${SSL_GATE_MODE_LIST}"
  echo "WEIGHT_TXT_ALIGN_LIST=${WEIGHT_TXT_ALIGN_LIST}"
  echo "PHASE1_ONLY=${PHASE1_ONLY}, PHASE1_EPOCHS=${PHASE1_EPOCHS}"
  echo "TOPK=${TOPK}, ENABLE_STAGE2=${ENABLE_STAGE2}, PHASE2_EPOCHS=${PHASE2_EPOCHS}"
  echo "PHASE2_MAIN_ONLY=${PHASE2_MAIN_ONLY}, PHASE2_RUN_C_REVIEW=${PHASE2_RUN_C_REVIEW}"
  echo "BATCH_SIZE=${BATCH_SIZE}, TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE}, PL_BATCH_SIZE=${PL_BATCH_SIZE}"
  echo "SKIP_GENERATE_ONLY=${SKIP_GENERATE_ONLY}, HF_ENDPOINT=${HF_ENDPOINT}"
  echo "PHASE1_RESULTS_CSV=${PHASE1_RESULTS_CSV}"
  echo "TOPK_CSV=${TOPK_CSV}"
  echo "=========================================="
} | tee "${MASTER_LOG}"

if [[ ! -d "${SHARED_REUSE_PL_DIR}" ]]; then
  echo "[error] shared pseudo-label dir not found: ${SHARED_REUSE_PL_DIR}" | tee -a "${MASTER_LOG}"
  echo "[hint] run ABCD generate once or set SHARED_REUSE_PL_DIR explicitly." | tee -a "${MASTER_LOG}"
  exit 1
fi

echo "run_id,run_tag,tau_t,ssl_gate_mode,weight_txt_align,val_mIoU,status,run_log" > "${PHASE1_RESULTS_CSV}"

run_one() {
  local run_tag="$1"
  local only="$2"
  local epochs="$3"
  local tau_t="$4"
  local gate_mode="$5"
  local w_align="$6"
  local run_log="$7"

  RUN_IN_BACKGROUND=0 \
  RUN_TAG="${run_tag}" \
  ONLY="${only}" \
  EPOCHS="${epochs}" \
  SKIP_GENERATE_ONLY="1" \
  REUSE_PL_DIR_OVERRIDE="${SHARED_REUSE_PL_DIR}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE}" \
  PL_BATCH_SIZE="${PL_BATCH_SIZE}" \
  SRC_CKPT="${SRC_CKPT}" \
  TGT_CKPT="${TGT_CKPT}" \
  TAU_T="${tau_t}" \
  SSL_GATE_MODE="${gate_mode}" \
  WEIGHT_TXT_ALIGN="${w_align}" \
  bash "${SCRIPT_DIR}/run_multimodal_stage1_stage2_abcd.sh" 2>&1 | tee "${run_log}" | tee -a "${MASTER_LOG}"
}

extract_miou() {
  local log_file="$1"
  awk -F'val mIoU=' '
    /Epoch [0-9]+: val mIoU=/ {
      split($2, a, ",")
      val = a[1]
    }
    END {
      if (val == "") {
        print "NA"
      } else {
        print val
      }
    }
  ' "${log_file}"
}

run_id=0
for tau_t in ${TAU_T_LIST}; do
  for gate_mode in ${SSL_GATE_MODE_LIST}; do
    for w_align in ${WEIGHT_TXT_ALIGN_LIST}; do
      run_id=$((run_id + 1))
      run_tag="${BASE_RUN_TAG}_p1_r${run_id}_tt${tau_t}_gm${gate_mode}_wa${w_align}"
      run_log="${LOG_DIR}/${run_tag}.log"
      {
        echo ""
        echo "------------------------------------------"
        echo "[Stage1] RUN ${run_id}: RUN_TAG=${run_tag}"
        echo "tau_t=${tau_t}, ssl_gate_mode=${gate_mode}, weight_txt_align=${w_align}"
        echo "start: $(date -Iseconds)"
      } | tee -a "${MASTER_LOG}"

      status="ok"
      if ! run_one "${run_tag}" "${PHASE1_ONLY}" "${PHASE1_EPOCHS}" "${tau_t}" "${gate_mode}" "${w_align}" "${run_log}"; then
        status="failed"
      fi
      miou="$(extract_miou "${run_log}")"
      echo "${run_id},${run_tag},${tau_t},${gate_mode},${w_align},${miou},${status},${run_log}" >> "${PHASE1_RESULTS_CSV}"

      {
        echo "done: $(date -Iseconds)"
        echo "[Stage1] status=${status}, val_mIoU=${miou}, run_log=${run_log}"
      } | tee -a "${MASTER_LOG}"
    done
  done
done

{
  echo ""
  echo "=========================================="
  echo "[Stage1] ranking Top${TOPK}"
} | tee -a "${MASTER_LOG}"

awk -F',' '
  NR == 1 { next }                # skip csv header
  $7 == "ok" && $6 != "NA" { print }
' "${PHASE1_RESULTS_CSV}" | sort -t',' -k6,6nr | awk -F',' -v k="${TOPK}" '
  BEGIN {
    print "rank,run_id,run_tag,tau_t,ssl_gate_mode,weight_txt_align,val_mIoU,status,run_log"
  }
  NR <= k {
    print NR "," $0
  }
' > "${TOPK_CSV}"

cat "${TOPK_CSV}" | tee -a "${MASTER_LOG}"

if [[ "${ENABLE_STAGE2}" == "1" ]]; then
  {
    echo ""
    echo "=========================================="
    echo "[Stage2] deep-run Top${TOPK} started: $(date -Iseconds)"
  } | tee -a "${MASTER_LOG}"

  while IFS=',' read -r rank run_id_csv run_tag_csv tau_t gate_mode w_align miou status run_log_path; do
    if [[ "${rank}" == "rank" ]]; then
      continue
    fi

    stage2_d_tag="${BASE_RUN_TAG}_p2_top${rank}_D_tt${tau_t}_gm${gate_mode}_wa${w_align}"
    stage2_d_log="${LOG_DIR}/${stage2_d_tag}.log"

    {
      echo ""
      echo "------------------------------------------"
      echo "[Stage2][D] rank=${rank}, src_run=${run_tag_csv}, stage2_tag=${stage2_d_tag}"
      echo "tau_t=${tau_t}, ssl_gate_mode=${gate_mode}, weight_txt_align=${w_align}"
      echo "start: $(date -Iseconds)"
    } | tee -a "${MASTER_LOG}"

    run_one "${stage2_d_tag}" "${PHASE2_MAIN_ONLY}" "${PHASE2_EPOCHS}" "${tau_t}" "${gate_mode}" "${w_align}" "${stage2_d_log}" || true
    d_miou="$(extract_miou "${stage2_d_log}")"
    echo "[Stage2][D] done: $(date -Iseconds), val_mIoU=${d_miou}, run_log=${stage2_d_log}" | tee -a "${MASTER_LOG}"

    if [[ "${PHASE2_RUN_C_REVIEW}" == "1" ]]; then
      stage2_c_tag="${BASE_RUN_TAG}_p2_top${rank}_C_tt${tau_t}_gm${gate_mode}_wa${w_align}"
      stage2_c_log="${LOG_DIR}/${stage2_c_tag}.log"
      {
        echo "[Stage2][C] review start: tag=${stage2_c_tag}, epochs=${PHASE2_EPOCHS}"
      } | tee -a "${MASTER_LOG}"
      run_one "${stage2_c_tag}" "C" "${PHASE2_EPOCHS}" "${tau_t}" "${gate_mode}" "${w_align}" "${stage2_c_log}" || true
      c_miou="$(extract_miou "${stage2_c_log}")"
      echo "[Stage2][C] done: $(date -Iseconds), val_mIoU=${c_miou}, run_log=${stage2_c_log}" | tee -a "${MASTER_LOG}"
    fi
  done < "${TOPK_CSV}"
fi

{
  echo ""
  echo "Sweep finished: $(date -Iseconds)"
  echo "Master log: ${MASTER_LOG}"
  echo "Phase1 csv: ${PHASE1_RESULTS_CSV}"
  echo "TopK csv: ${TOPK_CSV}"
} | tee -a "${MASTER_LOG}"
