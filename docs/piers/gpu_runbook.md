# gpu_runbook (run AFTER env prep; keep updated)

1. Preflight: `nvidia-smi`, nvcc, torch/torchvision/CUDA, arch support (esp. 5090).
2. Per-env import check: record `sys.executable`, torch, `sam2.__file__`, ckpt SHA256.
3. Single-image grounded smoke (object + empty target).
4. Single-ERP omni + text-FoV + seam remap smoke.
5. Closed-new-module parity vs inherited project (declare tol + nondeterminism).
6. Small fixed-set collab/memory/seam smoke incl. one target-train step reading revised pseudo.
7. Fixed-subset ablations confirming configs hit intended modules.
8. Full-list runs per budget, fixed seeds; no unbounded grid.

First-gpu command chain (fill paths from paths.yaml):
```
conda activate piers-omni && piers_ljj doctor
conda activate piers-grounded && piers_ljj doctor
piers_ljj run --config configs/piers/effective_start.yaml --dry-run
```
