#!/usr/bin/env bash
# cleanup_safe_repository_assets.sh
# 读取 docs/claude/repository-assets-safe-delete.tsv，逐条删除 SAFE_DELETE 文件。
# 默认 dry-run（DRY_RUN=1），设 DRY_RUN=0 才真正删除。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TSV="${REPO_ROOT}/docs/claude/repository-assets-safe-delete.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs/cleanup"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/repository_assets_cleanup_${STAMP}.log"

DRY_RUN="${DRY_RUN:-1}"

# 禁止删除的路径前缀/名称
FORBIDDEN_PATTERNS=(
  "${REPO_ROOT}/.git"
  "${REPO_ROOT}/configs"
  "${REPO_ROOT}/docs/claude"
  "${REPO_ROOT}/CLAUDE.md"
  "${REPO_ROOT}/adaptation"
  "${REPO_ROOT}/model"
  "${REPO_ROOT}/dataset"
  "${REPO_ROOT}/utils"
  "${REPO_ROOT}/tools"
  "${REPO_ROOT}/sam"
  "${REPO_ROOT}/data"
  "${REPO_ROOT}/data_pd_exp_final_2000"
  "${REPO_ROOT}/assets"
  "${REPO_ROOT}/markdown_archive"
)

log() { echo "$*" | tee -a "${LOG_FILE}"; }

is_forbidden() {
  local target="$1"
  [[ -z "${target}" ]] && return 0
  [[ "${target}" == "/" ]] && return 0
  # 仓库外路径
  if [[ "${target}" != "${REPO_ROOT}"* ]]; then
    return 0
  fi
  for pat in "${FORBIDDEN_PATTERNS[@]}"; do
    if [[ "${target}" == "${pat}"* ]]; then
      return 0
    fi
  done
  return 1
}

is_git_tracked() {
  local relpath="${1#${REPO_ROOT}/}"
  cd "${REPO_ROOT}"
  git ls-files --error-unmatch "${relpath}" &>/dev/null && return 0
  return 1
}

is_symlink_target() {
  find "${REPO_ROOT}" -type l -ls 2>/dev/null | grep -q "$(basename "$1")" && return 0
  return 1
}

log "============================================"
log "Repository Assets Cleanup"
log "REPO_ROOT: ${REPO_ROOT}"
log "TSV: ${TSV}"
log "DRY_RUN: ${DRY_RUN}"
log "START: $(date)"
log "============================================"

log ""
log "=== BEFORE: disk and repo size ==="
df -h "${REPO_ROOT}" | tee -a "${LOG_FILE}"
du -sh "${REPO_ROOT}" 2>/dev/null | tee -a "${LOG_FILE}"

DELETED=()
SKIPPED=()
FAILED=()

while IFS=$'\t' read -r id path type size_bytes size_human reason replacement; do
  [[ "${id}" == "id" ]] && continue  # header
  [[ -z "${path}" ]] && { log "SKIP: empty path (id=${id})"; SKIPPED+=("(empty)"); continue; }

  full_path="${REPO_ROOT}/${path}"

  # Forbidden check
  if is_forbidden "${full_path}"; then
    log "SKIP (forbidden): ${full_path}"
    SKIPPED+=("${full_path}")
    continue
  fi

  # Git tracked check
  if is_git_tracked "${full_path}"; then
    log "SKIP (git tracked): ${full_path}"
    SKIPPED+=("${full_path}")
    continue
  fi

  # Existence check
  if [[ ! -e "${full_path}" ]]; then
    log "SKIP (not found): ${full_path}"
    SKIPPED+=("${full_path}")
    continue
  fi

  FSIZE=$(du -sh "${full_path}" 2>/dev/null | cut -f1 || echo "?")
  log ""
  log "CANDIDATE: ${full_path}"
  log "  Size: ${FSIZE}  Type: ${type}  Reason: ${reason}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "  [DRY-RUN] Would delete: ${full_path}"
    DELETED+=("${full_path}")
  else
    if rm -f "${full_path}" 2>>"${LOG_FILE}"; then
      log "  DELETED: ${full_path}"
      DELETED+=("${full_path}")
    else
      log "  FAILED: ${full_path}"
      FAILED+=("${full_path}")
    fi
  fi

done < "${TSV}"

log ""
log "=== AFTER: disk and repo size ==="
df -h "${REPO_ROOT}" | tee -a "${LOG_FILE}"
du -sh "${REPO_ROOT}" 2>/dev/null | tee -a "${LOG_FILE}"

log ""
log "=== SUMMARY ==="
log "Deleted (or would delete): ${#DELETED[@]}"
for p in "${DELETED[@]}"; do log "  - ${p}"; done
log "Skipped: ${#SKIPPED[@]}"
for p in "${SKIPPED[@]}"; do log "  - ${p}"; done
log "Failed: ${#FAILED[@]}"
for p in "${FAILED[@]}"; do log "  - ${p}"; done
log ""
log "Log: ${LOG_FILE}"
log "DONE: $(date)"
