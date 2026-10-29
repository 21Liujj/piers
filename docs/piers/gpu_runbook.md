# gpu_runbook (run AFTER env prep; keep updated)

1. Preflight: `nvidia-smi`, nvcc, torch/torchvision/CUDA, arch support (esp. 5090).
2. Per-env import check: record `sys.executable`, torch, `sam2.__file__`, ckpt SHA256.
3. **Prepare the Grounding-DINO weights (REQUIRED, offline, do once).**
   ```bash
   python scripts/piers/convert_grounding_ckpt.py \
       --src  /path/to/grounding-dino-tiny/pytorch_model.bin \
       --dst  /path/to/grounding-dino-tiny-hf --copy-aux
   ```
   Then point `grounded.grounding_model_id` at the CONVERTED directory.
   Skipping this makes the detector return zero candidates with no error
   (see docs/piers/implementation_status.md, bug 2).
4. Single-image grounded smoke (object + empty target) — this step is ALREADY
   covered on CPU by `python scripts/piers/cpu_grounded_check.py`; on GPU just
   re-run it with `--device cuda` semantics and record timings.
5. Single-ERP omni + text-FoV + seam remap smoke.
6. Closed-new-module parity vs inherited project (declare tol + nondeterminism).
7. Small fixed-set collab/memory/seam smoke incl. one target-train step reading revised pseudo.
8. Fixed-subset ablations confirming configs hit intended modules.
9. Full-list runs per budget, fixed seeds; no unbounded grid.

## First-gpu command chain (fill paths from paths.yaml)

```bash
# --- piers-omni env -------------------------------------------------------
conda activate piers-omni && piers_ljj doctor
piers_ljj run --config configs/piers/effective_start.yaml --dry-run

# --- piers-grounded env ---------------------------------------------------
conda activate piers-grounded
pip install -r requirements-grounded.txt
python scripts/piers/convert_grounding_ckpt.py \
    --src "$PIERS_WEIGHTS/grounding-dino-tiny/pytorch_model.bin" \
    --dst "$PIERS_WEIGHTS/grounding-dino-tiny-hf" --copy-aux
python scripts/piers/cpu_grounded_check.py        # real backend sanity, CPU
python scripts/piers/cpu_grounded_check.py        # again after env switch
piers_ljj run --config configs/piers/T1_grounded_fov.yaml --dry-run
```

## Notes / gotchas verified on CPU (2026-10-29)

- `transformers` must be `>=4.57,<5`. The box-threshold kwarg is
  `box_threshold` there; the worker resolves the name at runtime so older
  versions work too, but 4.57.6 is what was verified.
- `sam2` must resolve to `third_party/grounded_sam2/sam2`, NOT any repo-root
  `sam2/` copy. `cpu_grounded_check.py` asserts this and fails loudly.
- The vendored `sam2` needs `hydra-core`, `omegaconf`, `iopath`; it is NOT
  pip-installed as a package (pure-Python import path is enough, no CUDA build
  required for the CPU path).
- Grounding-DINO warns that `model.text_backbone.pooler.*` is unused. That is
  expected (the pooler belongs to a classification head, not the detector).
- A "no candidates" result on a query that should detect something is a
  WEIGHT-LAYOUT symptom, not a threshold symptom. Check for the
  "newly initialized" warning in the load log first.
