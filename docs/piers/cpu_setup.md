# cpu_setup (no-GPU acceptance)

Covers: wrap crop/inverse-map (neg start, seam, non-divisible stride),
resize/letterbox/box coords, assoc no-merge, empty/unmapped/ambiguous,
19→13 + ignore 255, IoU/Dice/reject rules, collab-off passthrough,
pseudo conflict fallback + aug alignment note, memory budget/sync,
B=0 degeneration, worker/cache failure handling, synth end-to-end (marked synthetic).

Run:
```
pip install -r requirements-core.txt
pip install -e .
piers_ljj smoke
pytest tests/piers -q
```

No `--help`/import/CPU test may trigger model download, CUDA, wandb, or data scan.
Real-data/model tests needing CUDA are marked pending_gpu with reason.
