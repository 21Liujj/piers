#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
python3 tools/convert_standford2d3d_span8_labels.py \
  --processed_root /root/autodl-tmp/OmniSAM-main/data_extra/processed/Stanford360_SPAN8 \
  --log_root /root/autodl-tmp/OmniSAM-main/logs/Standford2d3d_results \
  --repo_root "$REPO_ROOT"
