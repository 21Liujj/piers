# Reproducing the paper results

This repository is self-contained: cloning it on a fresh machine gives you the
code, the configs, the result ledgers and the trained checkpoints needed to
reproduce every number reported in the paper.

```
git clone https://github.com/21Liujj/piers.git
cd piers
```

Everything below is what the experiments actually ran with. Where a value came
from the machine where the experiments were executed, it is recorded here and
in `results/`.

---

## 0. TL;DR

```bash
git clone https://github.com/21Liujj/piers.git && cd piers
pip install -r requirements.txt

# 1. point the repo at your data / weights
cp configs/piers/paths.example.yaml configs/piers/paths.yaml
$EDITOR configs/piers/paths.yaml

# 2. sanity-check that the shipped result ledgers match the shipped code
python scripts/piers/verify_result_fingerprints.py

# 3. check the suite configs are coherent
python scripts/piers/check_suite_configs.py

# 4. run the CPU acceptance tests
pytest tests -q
```

Step 2 is the important one: it proves that the code snapshot stored in
`results/I1_densepass/source_snapshot/` is byte-for-byte the code that produced
the numbers in `results/I1_densepass/`.

---

## 1. What is in this repository

| Path | Contents |
|---|---|
| `src/piers_ljj/` | The framework: contracts, geometry, collaboration, memory, seam, text, evaluation, worker entry point |
| `configs/piers/` | Suite configs and query manifests. Paths are `${PIERS_*}` placeholders, never absolute |
| `experiments/I1_densepass/code/` | The exact scripts that produced the Innovation-1 numbers |
| `experiments/top_level_scripts/` | Cross-dataset evaluation drivers (`m_eval_*.py`) and shell runners |
| `results/I1_densepass/` | Innovation-1 result ledgers, protocol record, environment freeze, source fingerprints |
| `results/I1_collab/` | Collaboration experiment ledgers (`reports/`, `reports_gate/`) |
| `results/E0/` | Baseline (OmniSAM-S) evaluation output |
| `results/I2_I3/` | Innovation-2 / Innovation-3 run records |
| `docs/piers/` | Architecture, data protocol, GPU runbooks, per-innovation status |
| `third_party/effective_omnisam/` | Vendored inherited engine (PanoSMB / SMR / MemoryBank / MTC / DynamicSCM) |

### The three innovations and where their evidence lives

| Innovation | Code | Results | Status |
|---|---|---|---|
| **I1** — reliable text↔semantic cooperation | `src/piers_ljj/collaboration/` | `results/I1_densepass/`, `results/I1_collab/` | Frozen inference release; see `docs/piers/INNOVATION1_STATUS.json` |
| **I2** — class-aware sparse memory | `src/piers_ljj/memory/`, `configs/piers/innovation2_final.json` | `docs/piers/innov2_results.md`, `results/I2_I3/` | DensePASS100 frozen; cross-domain partially blocked (see ledger) |
| **I3** — adaptive seam budgeting | `src/piers_ljj/seam/` | `results/I2_I3/` | Candidate; `configs/piers/innovation3_candidate.json` |

---

## 2. Environment

Recorded on the experiment machine in
`results/I1_densepass/environment_runtime.txt` and
`results/I1_densepass/environment_pip_freeze.txt`:

```
python 3.12.16
torch 2.8.0+cu128
cuda 12.8
gpu NVIDIA RTX 6000D (85 GB)
```

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-core.txt
pip install -r requirements-grounded.txt   # only needed for T0/T1/T2 text suites
pip install -r requirements-omni.txt       # only needed for the dense/memory suites
```

`requirements.txt` is the union of all three.

`SAM 2` and `Grounding-DINO` are **vendored, not pip-installed**:
`third_party/effective_omnisam/` (SAM2) and `third_party/grounded_sam2/`
(Grounding-DINO). Do not `pip install` them — the vendored copies are the ones
the results were produced with.

> **Note on `upstream_versions.json`** — it records the audited upstream
> commits of Effective-Omnisam and Grounded-SAM-2. Verify them on checkout;
> do not silently fall back to upstream `main`.

---

## 3. External assets you must supply

These are **not** in the repository (weights and images are too large). The
repository stores their checksums and the expected layout instead.

### 3.1 Model weights

Expected under `$PIERS_WEIGHTS_ROOT` (default `/root/autodl-tmp/weights`):

| File | SHA256 | Notes |
|---|---|---|
| `sam2.1_hiera_small.pt` | `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` | SAM 2.1 small backbone |
| `grounding-dino-tiny-hf/` | directory of files, checksums in `results/I1_densepass/weights_checksums.txt` | **converted** HF format — run `scripts/piers/convert_grounding_ckpt.py` once on the raw download |

The baseline OmniSAM-S checkpoint (`omnisam_s.pth`, 177 MB) is the initial
checkpoint for every training run. It is distributed through the upstream
Effective-Omnisam / OmniSAM project rather than through this repository.
**Its expected checksum and the exact download instructions are recorded in
`results/I1_densepass/weights_checksums.txt`.**

Verify what you have:

```bash
sha256sum $PIERS_WEIGHTS_ROOT/sam2.1_hiera_small.pt
```

### 3.2 Trained checkpoints (Innovation 1)

The trained checkpoints used for the reported numbers are published as a
**GitHub Release**, so you do not need to retrain to reproduce the numbers:

```bash
gh release download v0.2-paper-repro --repo 21Liujj/piers --dir ./ckpt
```

| Asset | SHA256 | Role |
|---|---|---|
| `allhard_runs_R0_seed1234_final.pth` | `ff79b0b0f766dbe7ad695bd56535d11cade9b12f608ce0ddb13778ba49f9a1d3` | I1 main arm, seed 1234 |
| `allhard_runs_R1_seed1234_final.pth` | `07fdc14b7bf072edc78c402fd11e940d28a7e3dd73dcd3ff7f5f4c973b6a7232` | I1 main arm, seed 4321 |
| `dense_controls_R0_seed1234_final.pth` | `40159d944c8925cd1c61ce4f8929cbde1fdbda761d0c41fb3bdfe27fec8f5625` | Dense-memory control |

Verify a download with:

```bash
python scripts/piers/verify_release_asset.py
```

Place them as:

```
ckpt/allhard_runs/R0_seed1234/training/final.pth
ckpt/allhard_runs/R1_seed1234/training/final.pth
ckpt/dense_controls/trained/R0_seed1234/training/final.pth
```

What is inside these checkpoints (and which tensors training actually changed)
is documented in [`results/CHECKPOINTS.md`](results/CHECKPOINTS.md).

### 3.3 Datasets

Expected layout (recorded from the experiment machine in
`results/I1_densepass/weights_checksums.txt`):

```
$PIERS_DATA_ROOT/
├── DensePASS/
│   ├── leftImg8bit/val/     # 100 ERP images  (100_.png ... )
│   └── gtFine/val/          # labelTrainIds
├── WildPASS/
│   ├── leftImg8bit/
│   └── gtFine/
├── SynPASS/
│   ├── img/
│   └── semantic/
└── Cityscapes/
    ├── leftImg8bit/
    └── gtFine/

$PIERS_DATA_EXTRA_ROOT/
├── WildPASS13/
│   ├── images/        (500)
│   ├── labels_raw/    (500)
│   ├── labels_trainid/(500)
│   └── lists/
└── Stanford360_SPAN8/
    ├── images/        (1040)
    ├── labels_raw/    (1040)
    ├── labels_trainid/(1040)
    └── lists/
```

The query manifests in `configs/piers/` reference these files by their
`${PIERS_DATA_ROOT}` / `${PIERS_DATA_EXTRA_ROOT}` placeholders, so no edit is
needed as long as `paths.yaml` is correct.

---

## 4. Configuring paths

Committed configs and manifests never contain absolute paths. They use
placeholders that are expanded at load time by `src/piers_ljj/paths.py`.

```bash
cp configs/piers/paths.example.yaml configs/piers/paths.yaml
```

`configs/piers/paths.yaml` is git-ignored, so your local paths never leak into
the repository.

Precedence: explicit overrides → environment variables → `paths.yaml` →
`paths.example.yaml`.

Either edit `paths.yaml`:

```yaml
data_root: "/data/piers"
data_extra_root: "/data/piers-extra"
weights_root: "/data/piers-weights"
```

or use environment variables:

```bash
export PIERS_DATA_ROOT=/data/piers
export PIERS_DATA_EXTRA_ROOT=/data/piers-extra
export PIERS_WEIGHTS_ROOT=/data/piers-weights
```

Check that nothing machine-specific leaked into the configs:

```bash
python scripts/piers/parametrize_paths.py --check
```

---

## 5. Reproducing each result

### 5.1 Innovation 1 — reliable text↔semantic cooperation

Main ledger: `results/I1_densepass/`. Protocol: `results/I1_densepass/protocol.json`.
Headline numbers: `docs/piers/INNOVATION1_STATUS.json`.

```bash
cd experiments/I1_densepass/code

# inference with the published checkpoints
python -u semantic_worker.py --manifest ../../results/I1_densepass/val100.json \
    --out "$PIERS_RUNS_ROOT/i1/val_cache" --ckpt <ckpt>/allhard_runs/R0_seed1234/training/final.pth

# scoring
python score_trained.py ...
```

Key protocol facts (from `protocol.json`):

- train split: 160 images, 4 per city × 40 cities, seed `20261107`, **no GT used**
- validation: DensePASS100 — **development set**, not an independent test
- GT: original `*_labelTrainIds.png`, 19 trainIds mapped to the project 13,
  `255` preserved
- seed: `1234`; final seeds `1234` and `4321`
- optimizer: AdamW, lr `2e-06`, batch size 1, 160 iterations, 9 updates/image
- memory policy: sealed Innovation-2 class-aware uncertainty-1024, both arms
- frozen pseudo threshold `0.85`

### 5.2 Innovation 2 — class-aware sparse memory

Main ledger: `docs/piers/innov2_results.md`.
Config: `configs/piers/innovation2_final.json`.

| Variant | Memory | Tokens | mIoU |
|---|---|---|---|
| dense | baseline | 32768 | 60.46 |
| **ca** | class-aware (rare-class preserving) | 2048 | 60.30 (−0.16) |
| bg | boundary-preserving | 2048 | 60.27 (−0.19) |
| pool / topk / uniform / cams | ablations | 2048 | 60.07 / 59.83 / 59.80 / 59.75 |

Same pipeline as the baseline (`sam2_s`, `num_maskmem=9`, sliding window
1024 / stride 384 / 2 passes / 18 windows, `validation()`); the **only** variable
is how memory is read.

### 5.3 Innovation 3 — adaptive seam budgeting

Config: `configs/piers/innovation3_candidate.json`. Run records in
`results/I2_I3/`. This one is a **candidate**, not a frozen claim — keep the
paper wording aligned with `docs/piers/innov2_results.md`.

### 5.4 Cross-dataset evaluation

```bash
python experiments/top_level_scripts/m_eval_xdom.py        # WildPASS500 + SPAN260
python experiments/top_level_scripts/m_eval_text.py        # text suite
python experiments/top_level_scripts/m_eval_wildpass_native8.py
```

Two known blockers are recorded in `docs/piers/innov2_results.md` §5 and are
**not** model failures:

- XW500-dense: the manifest's flat `labels_trainid` maps void to road, so GT
  is wrong and the number is voided.
- SPAN8 (XS260): the 13-class checkpoint cannot be evaluated on an 8-class
  dataset (`decoder 13 vs 8 mismatch`). Table 3 therefore excludes SPAN8.

---

## 6. Verifying a reproduction

| Check | Command | Expected |
|---|---|---|
| Code matches the results | `python scripts/piers/verify_result_fingerprints.py` | `14/14 files match` |
| Code matches in a fresh clone | `python scripts/piers/verify_snapshot_in_git.py` | `14/14 files match` |
| Release weights are intact | `python scripts/piers/verify_release_asset.py` | `3/3 release assets verified` |
| Configs are machine-independent | `python scripts/piers/parametrize_paths.py --check` | `OK: all configs are machine-independent` |
| Suite configs coherent | `python scripts/piers/check_suite_configs.py` | no errors |
| Unit tests | `pytest tests -q` | all pass |

The fingerprint check is the strong one: it compares the SHA256 of every source
file against the hashes recorded at experiment time, so it proves the code in
this repository is the code that produced the published numbers.

---

## 7. Reproducibility caveats — read this before writing claims

These are stated in the ledgers themselves and must be reflected in the paper:

1. **All DensePASS100 metrics are development-set results.** The existing
   DensePASS100 split was used for model selection. No fresh independent
   target test split is claimed.
2. **Training gain is not established.** The released Innovation-1 numbers are
   an *inference-time* release over the frozen baseline checkpoint.
3. **The 400-image supplement is an existing appendix only** and is not a
   condition for the generalization claim.
4. **Cross-domain (SPAN8 / XW500) is blocked for data and architecture
   reasons**, not model quality — see §5.4.
5. **Single-machine, single-GPU** results. Reproducing the exact numbers
   assumes the same GPU, torch build and seeds.

---

## 8. Server-side helper

On the GPU machine used for the experiments the working copy lives in
`/root/autodl-tmp/piers-main` and the conda env in `/root/autodl-tmp/envs/piers`.
`runs/I1-densepass-20261107_121457/` holds the full run tree including the
per-image prediction caches (not committed — they are large), while
`results/` in this repository holds the compact, versionable ledgers.

`results/I1_densepass/source_snapshot/` is the code snapshot captured during
the run; `results/I1_densepass/source_snapshot.tar.gz` is the same content
as archived on the machine.
