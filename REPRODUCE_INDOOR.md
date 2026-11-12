# Indoor adaptation and web reproduction — 2026-11-12

This is the active guide for the completed Stanford 2D-3D-S SPAN8 indoor experiment. [Outdoor v0.3](REPRODUCE.md) retains its original protocol and results. Both use the same I1/I2/I3 method family; the indoor task adapts the semantic model to eight indoor classes. Within the indoor original/full pair, checkpoint, LoRA and decoder are identical. The complete method additionally fits I1 reliability calibrators on training-area calib104. It uses Grounding DINO + SAM2; the earlier CLIP Stage2 checkpoint is not used.

## Archived result and evidence

| Method | Images | mIoU % | mAcc % | PixelAcc % |
|---|---:|---:|---:|---:|
| Original OmniSAM inference, shared adapted weights | 373 | 37.6424 | 42.9790 | 83.1710 |
| I2 + I3 semantic core | 373 | 42.0019 | 47.4632 | 84.0946 |
| Complete I1 + I2 + I3 | 373 | 46.5591 | 52.0355 | 86.0581 |

Full-minus-original is +8.9167 percentage points. Paired image-bootstrap95% CI is[+8.3126,+9.4728]pp. The single heldout area is area5; same-room views may be correlated. The training partitions use camera UUIDs rather than verified room identities. Earlier smoke diagnostics inspected heldout, so this is not a wholly untouched external benchmark. The Stage1 checkpoint follows a fixed1200-step training run, not a claimed hyperparameter optimum. The indoor training jointly adapts rank4 image-encoder LoRA and the decoder, so their individual contributions are not isolated.

The dataset preparation contains1413 panoramas at1024x512: fit832/calib104/val104/heldout373. Classes are ceiling, chair, door, floor, sofa, table, wall, window;255 is ignored. Obtain images and labels from their original dataset source/terms. Full datasets are not included. Exact split manifests, inference hashes, calibration models, training loss, integer ledgers and four figures are in [results/indoor_matched20261112](results/indoor_matched20261112). Historical six-arm and CLIP runs must not be mixed into this table.

## CPU verification from a clone

```bash
git clone https://github.com/21Liujj/piers.git
cd piers
git checkout v0.4-indoor-20261112
python -m pip install numpy pillow
python scripts/piers/verify_indoor_results.py
python scripts/paper/reproduce.py verify-ledgers
```

The first verifier recomputes the five metrics and eight class IoUs for all three indoor arms from373 per-image integer confusion matrices. It also checks equal per-image GT support, inference IDs, original experimental source hashes, calibration fingerprints and the recorded web equivalence. It does not run new inference or re-establish web equivalence without its archived predictions.

## Downloadable large artifacts

Use [release v0.4-indoor-20261112](https://github.com/21Liujj/piers/releases/tag/v0.4-indoor-20261112). Sizes and SHA256 values are in [release_assets.json](assets/indoor20261112/release_assets.json); the release also contains `SHA256SUMS.txt`.

- `indoor_stage1_shared.pth`:186,303,773 bytes, shared Stage1 checkpoint with LoRA and8-class decoder.
- `indoor_experiment_evidence.zip`: both main predictions, per-image audits, calibration evidence, source snapshot and packaging checks.
- `indoor_web_examples_20261112.zip`: three fixed real examples, original GT and outputs, nine result screenshots and one experiment dashboard screenshot.

The shared checkpoint SHA256 is `b3bfc7f54181e62dcababce9cc47709adbc655d9ae423b85e3d37e717749ac02`. The separate tensor-level LoRA fingerprint is recorded in `FINAL_VERIFICATION.json`.

## GPU web setup

Use Linux with the recorded PyTorch2.8.0+cu128/transformers4.57.6 environment and vendored SAM2 sources described in [outdoor reproduction](REPRODUCE.md). The recorded indoor versions are in `source_environment_audit.json` and `FINAL_VERIFICATION.json`. A fresh environment installation is not itself a full dataset rerun. Indoor and grounded SAM2 backends run in separate Python processes.

The current implementation retains the original experiment's Linux paths. On a fresh server, clone this release at `/root/autodl-tmp/piers-main`, and provide the base assets at `/root/autodl-tmp/weights`. For example:

```bash
export PIERS_WEIGHTS_ROOT=/root/autodl-tmp/weights
python scripts/paper/reproduce.py fetch-assets
python scripts/piers/prepare_indoor_release.py --download
# Or use an already downloaded, matching checkpoint:
# python scripts/piers/prepare_indoor_release.py --checkpoint /absolute/indoor_stage1_shared.pth
```

The installer verifies checkpoint size/SHA256 and the two calibration models, writes a web runtime protocol, and installs the recorded training split manifests. It refuses to overwrite different existing files. Original archived provenance stays unchanged under `results/`. New runtime protocol hashes can differ because the checkpoint location changes. It does not retrain, copy GT into inference inputs or replace archived scores.

Preserve a separate outdoor frozen checkout at `/root/autodl-tmp/piers-web-frozen-v03` if using the existing workbench defaults. The GPU agent reads indoor code from `/root/autodl-tmp/piers-main`; a different location requires setting `PIERS_INDOOR_REPO` in the remote agent environment before starting it. Local connection settings continue to select the outdoor checkout. See [the workbench README](apps/panorama_web/README.md) for its local Windows setup and SSH connection. Passwords are session-only and must not be committed.

Indoor defaults and subsets of the eight SPAN8 classes use adapted full inference. Requests containing other indoor categories use the separately labelled generic text route. Formal geometry is1024x512; output is resized to the uploaded image dimensions. The archived web equivalence covers the three fixed1024x512 examples at heldout indices0,124,248. Instance IDs/counts are estimates; no instance AP benchmark is claimed.

## New full-dataset rerun

In addition to the installed files above, obtain the recorded1024x512 RGB images under `/root/autodl-tmp/data_extra/Stanford2d3d_Seg/images` and SPAN8 labels under `/root/autodl-tmp/data_extra/SPAN8_PREP/<area>/labels`. The hashes in `inference_calib.json` and `inference_heldout.json` identify the original inputs. The original run-directory manifests are installed from `results/.../training`.

Run on an idle GPU into a fresh output directory:

```bash
export PIERS_PYTHON=/root/autodl-tmp/envs/piers/bin/python
bash scripts/piers/run_indoor_matched.sh "$PWD/runs/INDOOR-MATCHED-RERUN"
```

This prepares inputs; runs calibration/full-core semantics and genuine DINO/SAM2; refits I1 on calib104; runs heldout original/full; and scores against GT after inference. The script does not repeat Stage1 training. Existing archived results are not overwritten by the example. Timings are sums of recorded compute stages excluding model loading, intermediate cache I/O and orchestration, not isolated end-to-end latency. Whole-system speedup is not established.
