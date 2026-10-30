#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
python3 tools/collect_standford2d3d_results.py --repo-root "$REPO_ROOT"
