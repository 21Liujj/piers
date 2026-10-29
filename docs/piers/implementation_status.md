# implementation_status

Last updated: 2026-10-29 (after the no-GPU CPU acceptance pass).

## Verified on this machine (CPU, no GPU)

| Item | Result | Evidence |
|---|---|---|
| Unit + integration tests | **52 passed** | `python -m pytest tests/piers -q` |
| `piers_ljj doctor` | OK | py 3.12.3, torch 2.6.0+cpu, numpy 2.2.0 |
| `piers_ljj smoke` | OK | `{"fovs": 4, "max_recon_err": 0.0, "synthetic": true, "status": "passed_cpu"}` |
| `piers_ljj validate-data` | OK | 19->13 map assertion passes |
| `piers_ljj sweep --suite C` | OK | dep order `E0 -> T2 -> T1 -> U1 -> B1 -> M -> S -> C` |
| **REAL Grounded-SAM-2 backend (CPU)** | **12/12 checks passed** | `python scripts/piers/cpu_grounded_check.py` -> `runs/cpu_grounded_check/result.json` |

### Real backend details (2026-10-29)

Real weights, real inference, no mocks:

- grounding-dino-tiny loaded offline (`local_files_only=True`) from
  `D:/piers2027/weights/grounding-dino-tiny-hf`
- sam2.1_hiera_small loaded with `configs/sam2.1/sam2.1_hiera_s.yaml`
  (`image_size=1024`)
- query "car" on `third_party/grounded_sam2/assets/tracking_car_1.jpg`
  (1920x1080) -> **13 candidates**, labels `['car', ...]`, scores 0.53-0.67
- DINO boxes verified in ORIGINAL pixel coords (no double scaling)
- SAM2 mask shape == original image shape `(1080, 1920)`
- SAM2 mask overlaps its own box (coverage 0.867) -> coordinate chain agrees
- 1 image encode per FoV; 13 boxes -> 13 mask-decoder calls
- absent query "airplane" -> 0 boxes, 0 masks, 0 image encodes (no fallback)

## Bugs found and fixed during this pass

1. **`post_process_grounded_object_detection` kwarg rename.**
   `workers/entry.py` passed `threshold=`; transformers renamed it to
   `box_threshold=`. This raised `TypeError` immediately and would have
   blocked every GPU text run. Fixed by resolving the name from the live
   signature (`_post_process_grounded`), with a `**kwargs` fallback and a
   loud error when neither name is accepted.
   Regression tests: `tests/piers/test_grounded_integration.py`.

2. **Grounding-DINO checkpoint layout mismatch (silent weight loss).**
   The published checkpoint stores decoder/text attention FUSED
   (`in_proj_weight`), while every transformers >= 4.40 declares those as
   three separate `nn.Linear` modules. Direct loading dropped ~36 attention
   tensors and randomly re-initialised them -> **the detector returned ZERO
   candidates with no error**. Fixed by
   `scripts/piers/convert_grounding_ckpt.py`, which performs the same
   split that transformers' own `convert_grounding_dino_to_hf.py` assumes.
   18 fused layers converted, 0 leftover.

3. **Adapter only accepted 2-tuple detector output.** The DINO backend now
   also returns detected phrases (not just the raw query), so the adapter
   accepts both 2-tuple and 3-tuple backends.

4. **Suite dependency checks never matched (scheduler safety hole).**
   `cmd_run` passed the RAW config `suite` string (`"T2-semantic-to-text"`)
   to `check_deps`, but `DEPS` is keyed by canonical codes (`"T2"`), so no
   dependency was ever looked up. On top of that, dependency directories were
   resolved as the canonical CODE (`runs/E0`) while configs write results under
   their `run_id` (`runs/E0-effective-start`). Net effect: **T2/U1/B1/C would
   have started without their prerequisites**, producing invalid ablation
   numbers. Fixed by canonicalizing the suite code and resolving dep dirs via
   a new `SUITE_RUN_IDS` registry.
   Regression tests: `tests/piers/test_suite_resolution.py`.

5. **`effective_start.yaml` declared no `suite`.** The E0 baseline — the
   anchor for every other comparison — did not resolve to its registry entry.
   Added `suite: E0`. Also gave T1/T2/M/S explicit `run_id` and `worker`
   fields so the scheduler treats them consistently.
   Verified: `python scripts/piers/check_suite_configs.py` -> 10/10 configs
   resolve; `run --config .../T2_semantic_to_text.yaml --dry-run` now reports
   `BLOCKED, missing deps ['E0']` instead of passing silently.

## Done (code)

contracts, ERP geometry, assoc, collab rules (text_guide + revision),
memory policy + bridge, seam policy + adaptive, metrics, suites, scheduler,
CLI, configs, CPU tests, scripts, docs, vendored Grounded-SAM-2 snapshot,
grounded worker backend, CPU real-backend acceptance script.

## pending_gpu

- All real-model inference/training on GPU (the CPU pass above proves the
  *plumbing*; it does not produce paper numbers).
- OmniSAM/UDA path: E0 baseline, U1 text->pseudo training, B1 bidirectional,
  M (memory attention plug), S (seam real overlap fusion).
- ckpt-gated parity between inherited project and new project.
- Full suites E0..C with real data.

## Blockers (need owner action)

1. **Effective-Omnisam not migrated into `piers-ljj`.** The inherited main
   engine (`model/`, `dataset/`, `sam/`, `sam2/`, `adaptation/`, `utils/`,
   `train.py`, `run_adaptation.py`, `validation.py`) still lives OUTSIDE the
   repo at `D:/piers2027/effective-omnisam-src`. That directory has NO
   `.git`, so the audited commit `ef46a4c6` is still unverified. Until this
   is migrated, `workers/entry.py::omni_main` intentionally refuses to
   fake-load, and the pseudo-label revision patch has no training-side
   consumer.
2. **C-D checkpoint path/SHA256 not supplied.** `ckpt_zoo/files/*.pth` in the
   inherited tree are 100-200 byte broken symlinks/placeholders, not real
   weights.
3. **Cityscapes / DensePASS not present on this machine.**
   `D:/dataset_split` is a facial-expression dataset (angry/disgust/fear/...),
   unrelated to the segmentation protocol.
4. **No CUDA on this host.** `torch.cuda.is_available() == False`; torch is
   the `+cpu` build. A CUDA wheel must be installed on the GPU host.
