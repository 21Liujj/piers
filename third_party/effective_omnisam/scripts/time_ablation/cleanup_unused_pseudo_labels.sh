#!/usr/bin/env bash
set -euo pipefail

# Cleanup helper for historical pseudo-label directories.
# Default is dry-run for safety.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

DRY_RUN="${DRY_RUN:-1}"                  # 1: preview only, 0: actually delete
MIN_FREE_GB="${MIN_FREE_GB:-10}"         # stop cleanup once free space reaches this threshold
TARGET_ROOT="${TARGET_ROOT:-${ROOT}}"

# comma-separated directory basenames to keep (exact match)
KEEP_DIRS="${KEEP_DIRS:-data_pd_cdsweep_p2_top1_D_tt0.10_gmhard_wa0.02_D_text_gate_hard_with_adapter,data_pd_exp_A0_baseline,data_pd_cdsweep_clean_A_baseline}"

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] $*"; }

free_gb() {
  df -BG "${TARGET_ROOT}" | awk 'NR==2 {gsub(/G/, "", $4); print $4+0}'
}

is_kept() {
  local base="$1"
  IFS=',' read -r -a keep_arr <<< "${KEEP_DIRS}"
  for k in "${keep_arr[@]}"; do
    [[ "${base}" == "${k}" ]] && return 0
  done
  return 1
}

collect_candidates() {
  # Historical pseudo-label/cache directories that are typically safe to clean.
  # Keep list can be overridden via KEEP_DIRS.
  printf '%s\n' \
    data_pd_abcd_A_baseline \
    data_pd_abcd_B_visual_gate_only \
    data_pd_abcd_C_text_gate_hard_no_adapter \
    data_pd_abcd_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_only_clean_p1_r1_tt0.05_gmhard_wa0.02_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r1_tt0.05_gmhard_wa0.02_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r2_tt0.05_gmhard_wa0.05_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r3_tt0.05_gmsoft_wa0.02_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r4_tt0.05_gmsoft_wa0.05_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r5_tt0.08_gmhard_wa0.02_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r6_tt0.08_gmhard_wa0.05_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r7_tt0.08_gmsoft_wa0.02_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r8_tt0.08_gmsoft_wa0.05_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r9_tt0.10_gmhard_wa0.02_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r10_tt0.10_gmhard_wa0.05_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r11_tt0.10_gmsoft_wa0.02_D_text_gate_hard_with_adapter \
    data_pd_cdsweep_p1_r12_tt0.10_gmsoft_wa0.05_D_text_gate_hard_with_adapter \
    pd_auto_iter \
    pd_uda_densepass \
    pd_uda_from60_36_originalish
}

log "cleanup start dry_run=${DRY_RUN} min_free_gb=${MIN_FREE_GB} target_root=${TARGET_ROOT}"
log "free space before: $(free_gb)G"
log "keep dirs: ${KEEP_DIRS}"

while IFS= read -r name; do
  [[ -z "${name}" ]] && continue
  path="${ROOT}/${name}"
  [[ -d "${path}" ]] || continue
  if is_kept "${name}"; then
    log "KEEP ${name}"
    continue
  fi
  sz="$(du -sh "${path}" 2>/dev/null | awk '{print $1}')"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY-RUN delete ${name} (${sz})"
  else
    log "DELETE ${name} (${sz})"
    rm -rf "${path}"
    log "free space now: $(free_gb)G"
    if [[ "$(free_gb)" -ge "${MIN_FREE_GB}" ]]; then
      log "free space reached threshold ${MIN_FREE_GB}G, stop cleanup"
      break
    fi
  fi
done < <(collect_candidates)

log "cleanup done. free space after: $(free_gb)G"

# Remove empty directories left after cleanup (bottom-up).
# This keeps workspace tidy and avoids confusion in later manual checks.
log "remove empty directories (bottom-up)"
python - <<'PY'
import os

root = "/root/autodl-tmp/OmniSAM-main"
skip_prefixes = {
    ".git",
    "__pycache__",
}

removed = 0
for dirpath, dirnames, filenames in os.walk(root, topdown=False):
    base = os.path.basename(dirpath)
    if base in skip_prefixes:
        continue
    try:
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            removed += 1
    except OSError:
        pass
print(f"[cleanup-empty] removed_dirs={removed}")
PY

