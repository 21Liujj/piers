#!/bin/bash
# piers-omni env prep (GPU host). Verifies Effective-Omnisam source first.
set -e
python -m pip install -r requirements-omni.txt
echo "NEXT: audit Effective-Omnisam checkout, set SAM2_BUILD_CUDA, see docs/piers/gpu_runbook.md"
