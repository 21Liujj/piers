# R1–R6 traceability

| Req | Source | Code | Config | CPU test | GPU exp | Status |
|---|---|---|---|---|---|---|
| R1 dual paths | report §3–4, PPT p7–8 | contracts.py, workers/entry.py, geometry/erp.py | effective_start.yaml, T1 | test_geometry, smoke | E0,T0,T1 | code-done, pending_gpu |
| R2 collab | report §3(1), PPT p9 | collaboration/rules.py | T2 | test_modules (rerank/pseudo) | T2,U1,B1 | code-done, pending_gpu |
| R3 memory | report §3(2), PPT p10 | memory/policy.py | M_memory.yaml | test_modules (budget) | M | policy-done, attention-plug pending_gpu |
| R4 seam | report §3(3), PPT p11 | seam/policy.py | S_seam.yaml | test_modules (B=0) | S | code-done, pending_gpu |
| R5 eval+cost | report §4,6 | evaluation/metrics.py, cli | all suites | test_metrics | all | code-done, pending_gpu |
| R6 existing base | report §5.1 | upstream_versions.json; effective_start preserved | effective_start.yaml | validate-data | E0 | recorded, reverify on checkout |

Rules honored: semantic never replaces instance masks; non-detect never forced to
background; seam never fakes equal edge columns; synthetic always labeled.
