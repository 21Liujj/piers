#!/usr/bin/env bash
# Offline same-checkpoint original versus real DINO/SAM2-I1 complete model.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PIERS_PYTHON:-/root/autodl-tmp/envs/piers/bin/python}"
OUT="${1:-$REPO/runs/INDOOR-MATCHED-DINO-20261112}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ ! -f "$OUT/protocol.json" ]; then
  "$PY" scripts/piers/indoor_matched_run.py prepare --out "$OUT"
fi
trap 'echo "FAILED at line $LINENO" > "$OUT/FAILED.flag"' ERR
for SPLIT in calib heldout; do
  "$PY" -u scripts/piers/indoor_matched_run.py semantic --out "$OUT" --split "$SPLIT" --arm full_core > "$OUT/semantic_${SPLIT}_full_core.log" 2>&1
  "$PY" -u scripts/piers/indoor_matched_run.py text --out "$OUT" --split "$SPLIT" > "$OUT/text_${SPLIT}.log" 2>&1
  if [ "$SPLIT" = calib ]; then
    "$PY" -u scripts/piers/indoor_matched_run.py calibrate --out "$OUT" > "$OUT/calibrate.log" 2>&1
  fi
done
"$PY" -u scripts/piers/indoor_matched_run.py semantic --out "$OUT" --split heldout --arm original > "$OUT/semantic_heldout_original.log" 2>&1
"$PY" -u scripts/piers/indoor_matched_run.py fuse --out "$OUT" --split heldout > "$OUT/fuse.log" 2>&1
"$PY" -u scripts/piers/indoor_matched_run.py score --out "$OUT" --split heldout > "$OUT/score.log" 2>&1
date -Is > "$OUT/DONE.flag"
