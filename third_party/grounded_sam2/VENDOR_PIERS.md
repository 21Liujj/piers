# Grounded-SAM-2 vendor provenance (piers-ljj)

- Upstream: https://github.com/IDEA-Research/Grounded-SAM-2
- Audited upstream commit (prompt record, 2026-10-26): `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28` on `main`.
  Status vs this snapshot: **unverified** (this snapshot came from a source ZIP, not a git clone;
  re-verify via `git clone` + `git rev-parse HEAD` when network allows).
- Source ZIP used: `D:\piers2027\Grounded-SAM-2-main.zip`
  - size: 118638620 bytes, sha256: `b63ad8ec9d576bd05e222b1c258974c24379493385e2597dbadbe0853b398d49`
- Vendored: 2026-10-27 into `third_party/grounded_sam2/` (199 files, ~19.1 MB).
- Excluded from the ZIP (kept out of git to save space; re-extractable from the ZIP):
  `*.mp4/*.mp3/*.avi/*.mov/*.mkv`, `notebooks/`, `demo/frontend/`, `demo/data/gallery/`,
  `sav_dataset/example/`, `__MACOSX` (640 files, mostly demo videos/notebook media).
- Kept intentionally: `LICENSE`, `LICENSE_sam2`, `LICENSE_groundingdino`, `LICENSE_cctorch`,
  `grounded_sam2_hf_model_demo.py` (HF DINO + SAM2ImagePredictor reference for our T-branch),
  `sam2/`, `grounding_dino/`, `checkpoints/download_ckpts.sh`, `gdino_checkpoints/download_ckpts.sh`.
- No nested `.git` was copied; no vendor file was merged over main-project `sam2/`.
  Our adapter lives in `src/piers_ljj/text/grounded_adapter.py` and **never imports the demo module**
  (the demo runs `argparse` + CUDA + weight downloads at import time).
- Weights (`*.pt/*.pth`) and `outputs/` are ignored by the vendor `.gitignore`; fetch via the
  `download_ckpts.sh` scripts or HF hub on the GPU machine, then record hashes in run manifests.
