# Outdoor reproduction — v0.3-paper-20261109

Current main also contains the completed373-image indoor adaptation and adapted web integration. Use [REPRODUCE_INDOOR.md](REPRODUCE_INDOOR.md) and release `v0.4-indoor-20261112` for that work. This document describes the unchanged historical outdoor v0.3 protocol; checking out its tag omits the later indoor code.

This is the active guide for the frozen complete-v3 outdoor results. It replaces the earlier run-directory-dependent instructions. A clone contains code, source calibrators, exact sample manifests and pixel-count ledgers. The three large inference weights are downloaded from the matching GitHub Release with SHA256 verification. Dataset images/labels must be obtained separately under their original terms; they are not included in Git.

## 1. Clone and environment

```bash
git clone https://github.com/21Liujj/piers.git
cd piers
git checkout v0.3-paper-20261109
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-paper-20261109.txt
pip install -e . --no-deps
python scripts/paper/reproduce.py fetch-assets
python scripts/paper/reproduce.py verify-ledgers
```

The recorded host is Linux x86_64, Python3.12.16, PyTorch2.8.0+cu128, torchvision0.23.0+cu128, transformers4.57.6, RTX6000D. `requirements-paper-20261109.txt` pins the directly used packages from that host; transitive dependencies are resolved by pip. No clean reinstallation of that entire environment is claimed by the GPU smoke check. Jetson needs its platform-specific PyTorch stack; this file is not a Jetson installer. New hardware/library versions can introduce floating-point differences and different timings.

Both semantic and grounded SAM2 sources are vendored. Do not install another `sam2` into this environment. The launcher uses separate processes because those two vendor namespaces are incompatible in one interpreter. Inference runs offline after assets are ready. `fetch-assets` includes the already converted, verified HF DINO checkpoint; do not reconvert it. All files/sizes/SHA256 values are recorded in `assets/paper20261109/weights.json`.

## 2. Dataset layout and exact inputs

Set roots, or place the data under the repository's default `data/` and `data_extra/`:

```bash
export PIERS_DATA_ROOT=/your/data
export PIERS_DATA_EXTRA_ROOT=/your/data_extra
# Optional external weight directory; use the same root when fetching:
# export PIERS_WEIGHTS_ROOT=/your/weights
# python scripts/paper/reproduce.py fetch-assets
```

Expected examples, relative to the chosen roots:

```text
data/DensePASS/leftImg8bit/val/100_.png
data/DensePASS/gtFine/val/100_labelTrainIds.png
data/WildPASS/leftImg8bit/val/cs/1.png
data/WildPASS/gtFine/val/cs/1.png
data/SynPASS/img/cloud/val/MAP_6_point10/000000.jpg
data/SynPASS/semantic/cloud/val/MAP_6_point10/000000_trainID.png
data_extra/WildPASS13/images/1.png    # accepted equivalent WildPASS RGB source
```

The complete filename lists and GT/canonical-pixel hashes are in `configs/piers/paper20261109/datasets/*_samples.json`: DensePASS100, WildPASS500, SynPASS1690. If your official distribution has a different directory layout, organize copies/symlinks to these paths. The prepare step fails if the actual decoded input pixels, geometry or label bytes differ. It permits different RGB file encodings only if canonical pixels are identical. Images become2048x400 RGB with PIL BICUBIC; native-size predictions are scored against the original labels. GT files are not modified. Ground truth is available to preparation/scoring only; inference workers reject GT file access.

WildPASS uses8 native valid classes. SynPASS raw0 is void, leaving22 valid classes. The13-class SynPASS experiment is a separate13-category inference run, followed by the recorded GT mapping in the scorer: RoadLine/Road merge, nonshared categories are ignored. It is not the22-class result with a new denominator. DensePASS uses the recorded standard19-to13 mapping. No new manual labels or target-domain training are required.

## 3. Paired full-dataset inference

```bash
python scripts/paper/reproduce.py prepare --task densepass13 --out runs/inputs_dense13
python scripts/paper/reproduce.py prepare --task wildpass8 --out runs/inputs_wild8
python scripts/paper/reproduce.py prepare --task synpass22 --out runs/inputs_syn22
python scripts/paper/reproduce.py prepare --task synpass13 --out runs/inputs_syn13

python scripts/paper/reproduce.py run --prepared runs/inputs_dense13 --out runs/paired_dense13 --variants original,full --workers-per-variant 3 --text-workers 2 --max-gpu-workers 8
python scripts/paper/reproduce.py run --prepared runs/inputs_wild8 --out runs/paired_wild8 --variants original,full --workers-per-variant 3 --text-workers 2 --max-gpu-workers 8
python scripts/paper/reproduce.py run --prepared runs/inputs_syn22 --out runs/paired_syn22 --variants original,full --workers-per-variant 3 --text-workers 2 --max-gpu-workers 8
python scripts/paper/reproduce.py run --prepared runs/inputs_syn13 --out runs/paired_syn13 --variants original,full --workers-per-variant 3 --text-workers 2 --max-gpu-workers 8
```

Each command parallelizes its workers on one idle GPU and scores after all predictions finish. Run task commands sequentially; they refuse to start over unrelated GPU work. Eight workers are a starting point for the recorded85GB GPU; lower worker counts on smaller cards. Always use fresh output directories. Accuracy can reuse a completed matching text-evidence directory with `--text-cache /path/to/text`; this is not an end-to-end latency measurement. Input hashes, category names, frozen checkpoint hash, memory audit and GT-access status are recorded per prediction. `results/summary.json`, confusion arrays and per-image histograms are produced per run.

For a quick installation check, prepare one sample with `--limit 1` and run original/full with one worker each and one text worker. A small smoke test does not reproduce the complete dataset aggregate.

The inference API receives image, geometry and requested category names. Dataset identities and GT mapping are used by preparation/scoring. The original baseline has a13-class head: requested novel classes have zero source support and can produce explicit abstentions. Full-v3 uses text masks to add their channels. This is an inherited source-head restriction, not proof of arbitrary out-of-domain inability. Frozen weights mean no target training; source LoRA was trained before transfer. Source I1 calibrators are supplied as three small NPZ models under `assets/paper20261109/calibrators`.

## 4. Innovation controls and historical evidence

`configs/piers/paper20261109/matrix.json` registers the original, I1, I2, I3 and combined variants, plus memory/seam controls. For example:

```bash
python scripts/paper/reproduce.py run --prepared runs/inputs_dense13 --out runs/dense_components --variants original,i2,i2_i3,i1_i3,full --workers-per-variant 2 --text-workers 2 --max-gpu-workers 8
```

These controls are registered for DensePASS development validation. The release contains the completed matched original/I2/I2+I3/full accuracy data and the complete-system I2-on/off control. Not every registered factorial/control variant has a completed new measurement. See `docs/piers/PAPER_RELEASE_STATUS_20261109.md`; a runnable variant is not an experimental result. The historical18-row memory selection matrix and original calibrator bytes were unavailable; do not describe them as newly reconstructed. Older results/source snapshots remain historical and have their own protocols.

The active memory budget is1024, inherited from the frozen November6 configuration. The defense-stage2048 configuration is a different historical budget. Default1024 is not reselected on the target datasets. I2 reduces attention inputs32768->1024 while retaining dense stored history. Its matched component timing improves locally; the full-system I2 toggle shows only about0.56% mean latency reduction and no device-memory saving. The current full pipeline is slower than the original baseline.

## 5. Efficiency measurements

For serial real-text timing, use the recorded plans. First prepare all relevant tasks above, then materialize a plan with local paths:

```bash
python scripts/paper/benchmark.py --prepared runs/inputs_dense13 --plan results/paper20261109/densepass_efficiency_protocol.json --out runs/bench_dense
python scripts/paper/benchmark.py --prepared runs/inputs_wild8 runs/inputs_syn22 runs/inputs_syn13 --plan results/paper20261109/native_efficiency_protocol.json --out runs/bench_targets
```

The benchmark loads resident models, runs genuine DINO/SAM2 for every full-model request, excludes weight-loading time, includes image loading, IPC and output writing, and samples whole-device memory on an otherwise idle GPU. It refuses concurrent GPU inference. Frozen plans use3 images x2 measured repeats per task, with warmups; these are descriptive small-sample measurements. Hardware timing is not expected to equal archived seconds bit-for-bit.

`full_cached` is an optional experimental prefusion feature cache. It preserves separate class queries and recomputes the text-conditioned multimodal encoder/decoder. The current evidence is only3 fixed images and2 measured repeats per image on RTX6000D, with exactly matching text outputs; text time fell31–36%. No full-system or Jetson acceleration claim has been established. The frozen main `full` variant keeps this option off.

## 6. Verification and boundaries

```bash
python scripts/paper/reproduce.py verify-ledgers
pytest tests/piers/test_full_model_contract.py tests/piers/test_open_vocab_competition.py tests/piers/test_requested_literal_queries.py tests/piers/test_prefusion_feature_cache.py -q
```

Integer ledgers reconstruct all four main accuracy rows exactly, including per-class metrics and abstention misses. New inference has been checked in an isolated checkout against the existing frozen predictions as recorded in `results/paper20261109/release_validation.json`; refer to the listed sample counts rather than assuming every dataset was re-run during packaging. Source calibrator rebuilding and old model-training runs are not rerun by the commands above. The v0.2 trained checkpoints are historical and are not the checkpoint used in this frozen main table.

Report DensePASS as development validation, retain v1/v2 failed or partial results, and disclose that target diagnostics informed v3 repairs. At the outdoor v0.3 freeze, indoor adaptation was still pending; it is now separately completed in [the v0.4 indoor protocol](REPRODUCE_INDOOR.md). Outdoor target-domain fine-tuning/UDA and a complete-system speed advantage remain unestablished. The outdoor table and its limitations are unchanged.
