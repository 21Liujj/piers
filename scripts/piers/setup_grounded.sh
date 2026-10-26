#!/bin/bash
# piers-grounded env prep (GPU host). HF weights downloaded in explicit step.
set -e
python -m pip install -r requirements-grounded.txt
echo "NEXT: download grounding-dino + sam2 ckpts to local dir (local_files_only at runtime)"
