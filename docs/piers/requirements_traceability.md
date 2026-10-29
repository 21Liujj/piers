# R1–R6 traceability

Updated 2026-10-29 after the CPU acceptance pass (real Grounded-SAM-2 backend verified).

| Req | Source | Code | Config | CPU test | GPU exp | Status |
|---|---|---|---|---|---|---|
| R1 dual paths | report §3–4, PPT p7–8 | contracts.py, workers/entry.py, geometry/erp.py, text/pipeline.py | effective_start.yaml, T0, T1 | test_geometry, test_text_branch, **cpu_grounded_check (real backend 12/12)** | E0,T0,T1 | text-path code-done + real backend verified; omni path blocked on main-engine migration |
| R2 collab | report §3(1), PPT p9 | collaboration/text_guide.py, collaboration/revision.py | T2, U1, B1 | test_modules (rerank/pseudo), test_innov1 | T2,U1,B1 | code-done; **pseudo-label training consumer still blocked** on main-engine migration |
| R3 memory | report §3(2), PPT p10 | memory/policy.py, memory/bridge.py | M_memory.yaml | test_modules (budget), test_innov23 | M | policy-done; attention plug pending_gpu |
| R4 seam | report §3(3), PPT p11 | seam/policy.py, seam/adaptive.py | S_seam.yaml | test_modules (B=0), test_innov23 | S | code-done; real overlap fusion pending_gpu |
| R5 eval+cost | report §4,6 | evaluation/metrics.py, cli.py | all suites | test_metrics, test_scheduler | all | code-done; real runs pending_gpu |
| R6 existing base | report §5.1 | upstream_versions.json; effective_start preserved | effective_start.yaml | validate-data | E0 | **NOT YET MIGRATED** — inherited engine still outside the repo |

Rules honored: semantic never replaces instance masks; non-detect never forced to
background; seam never fakes equal edge columns; synthetic always labeled.

## Verified real-backend facts (2026-10-29)

`python scripts/piers/cpu_grounded_check.py` -> 12/12 passed, real weights,
CPU inference, no mocks:
grounding-dino-tiny (converted layout) + sam2.1_hiera_small on a 1920x1080
photo; 13 "car" candidates; boxes in original pixel coords; SAM2 mask shape
== image shape and overlapping its own box (coverage 0.867); 1 image encode
per FoV; absent query -> empty list with zero encodes.

## Open blockers (see implementation_status.md for detail)

1. Effective-Omnisam main engine not migrated into `piers-ljj` (no `.git` at
   `D:/piers2027/effective-omnisam-src`, so commit `ef46a4c6` unverified).
2. C-D checkpoint path + SHA256 not supplied (ckpt_zoo files are placeholders).
3. Cityscapes/DensePASS absent (D:/dataset_split is a facial-expression set).
4. No CUDA on host (torch is the +cpu build).
