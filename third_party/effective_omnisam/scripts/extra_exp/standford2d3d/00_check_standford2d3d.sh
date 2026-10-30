#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
python3 tools/check_standford2d3d_span8.py \
  --config configs/extra_exp/standford2d3d/standford360_span8_eval.yaml \
  --repo-root "$REPO_ROOT" \
  --vis-dir "$REPO_ROOT/logs/Standford2d3d_results/vis_check"
