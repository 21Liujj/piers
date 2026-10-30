#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
python3 tools/prepare_standford2d3d.py \
  --zip_path /root/autodl-tmp/OmniSAM-main/standford-360.zip \
  --out_root /root/autodl-tmp/OmniSAM-main/data_extra \
  --log_root /root/autodl-tmp/OmniSAM-main/logs/Standford2d3d_results \
  --use_symlink true \
  --force false
