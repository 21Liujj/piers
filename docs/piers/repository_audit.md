# repository_audit (v1 scaffold — re-verify against real checkouts on GPU host)

Audited 2026-10-26 (from prompt; NOT re-verified locally — no source checkouts present):

- Effective-Omnisam (private): branch `feature/scm-circular-memory`,
  commit `ef46a4c6007623173ccd6e70b5745e48951f70d`.
- Grounded-SAM-2: main @ `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`.

Key files to confirm in Effective-Omnisam checkout:
`train.py`, `run_adaptation.py`, `validation.py`, `model/model.py`,
`model/memory_bank.py`, `model/modules/memory_compressor.py`,
`model/modules/sparse_memory_retriever.py`, `model/modules/pano_smb_allocator.py`,
`dataset/DensePASS.py`, `dataset/DensePASS_circular.py`, `utils/sam2_bootstrap.py`,
`utils/pl_merge.py`, `utils/pl_sidecar.py`, `adaptation/*`, `configs/experiments/text_modality*`.

Known issues (must re-check, do NOT copy stale README install lines):
1. README python3.8/torch2.0.1 vs `sam/setup.py` python>=3.10/torch>=2.5.1 mismatch.
2. `sam/` bootstrap + `sam2/` dir + Grounded-SAM-2 `sam2` — resolve real `sam2.__file__`
   per process; fail loudly on shadowing.

C-D checkpoint: path/SHA256 unresolved here — owner to supply; no filename guessed.
