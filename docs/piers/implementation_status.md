# implementation_status

- Done (code): contracts, ERP geometry, assoc, collab rules, memory policy,
  seam policy, metrics, suites, CLI, configs, CPU tests, scripts, docs scaffold.
- CPU-tested: pending this-machine run (pytest + smoke) — see status.json after run.
- pending_gpu: all real-model inference/training, MemoryBank attention plug,
  vendor snapshot import, ckpt-gated parity, full suites E0..C.
- Blockers: (1) Effective-Omnisam checkout unavailable here; (2) Grounded-SAM-2
  snapshot not vendored; (3) C-D ckpt path/SHA not supplied; (4) no CUDA on host.
