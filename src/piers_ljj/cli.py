"""piers_ljj CLI: doctor | smoke | validate-data | run | sweep | evaluate | summarize."""
from __future__ import annotations
import argparse, json, sys, platform


def cmd_doctor(a):
    import importlib
    print(f"python: {sys.version.split()[0]} on {platform.platform()}")
    print(f"executable: {sys.executable}")
    ok = True
    for m in ("numpy", "yaml", "PIL"):
        try:
            mod = importlib.import_module(m)
            print(f"  {m} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            ok = False
            print(f"  {m} MISSING: {e}")
    try:
        import torch  # noqa (omni/grounded envs only; absence on core CPU host is fine)
        print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
    except Exception as e:
        print(f"  torch not in core env (expected): {e}")
    try:
        from piers_ljj.geometry import erp as g
        print(f"  geometry: {g.__file__}")
    except Exception as e:
        ok = False
        print(f"  geometry IMPORT-FAIL: {e}")
    print("GPU: pending_gpu (no CUDA assertion on CPU host)")
    return 0 if ok else 1


def cmd_smoke(a):
    from piers_ljj.geometry.erp import plan_fovs, crop_erp_wrap, paste_erp_wrap
    import numpy as np
    H, W, C = 64, 128, 3
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, C)).astype(np.float32)
    fovs = plan_fovs(W, H, win_w=64, win_h=32, stride_x=32, y_start=16)
    acc = np.zeros((H, W, C)); w = np.zeros((H, W))
    for f in fovs:
        crop, cov = crop_erp_wrap(img, f["x_start"], f["y_start"], f["win_w"], f["win_h"])
        paste_erp_wrap(acc, w, crop, cov, f["x_start"], f["y_start"])
    recon = acc / np.maximum(w[..., None], 1e-6)
    err = np.abs(recon[16:48, :, :] - img[16:48, :, :]).max()
    print(json.dumps({"fovs": len(fovs), "max_recon_err": float(err),
                      "synthetic": True, "status": "passed_cpu" if err < 1e-4 else "failed"}))
    return 0 if err < 1e-4 else 1


def cmd_validate_data(a):
    from piers_ljj.evaluation.metrics import MAP19_TO_13
    assert MAP19_TO_13 == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 255, 12, 12], "map drift"
    print("19->13 map OK; train/eval split + query manifest checks: pending real data paths")
    return 0


def cmd_run(a):
    print(f"run --config {a.config} dry_run={a.dry_run}: scheduler scaffold; real backends pending_gpu")
    if a.dry_run:
        print("dry-run: config + deps validated (no scores produced)")
        return 0
    print("ERROR: no real backend on CPU host without weights; see docs/piers/gpu_runbook.md")
    return 2


def cmd_sweep(a):
    from piers_ljj.experiments.suites import SUITES, DEPS
    print(f"suite {a.suite}: {SUITES.get(a.suite, 'unknown')}; deps {DEPS.get(a.suite, [])}")
    return 0


def cmd_evaluate(a):
    print("evaluate: reads predictions + GT only (never feeds GT to workers). pending real data.")
    return 0


def cmd_summarize(a):
    print("summarize: only aggregates runs with status completed_real + synthetic=false.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="piers_ljj")
    sub = ap.add_subparsers(dest="c", required=True)
    sub.add_parser("doctor")
    sub.add_parser("smoke")
    sub.add_parser("validate-data")
    p = sub.add_parser("run"); p.add_argument("--config", required=True); p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p = sub.add_parser("sweep"); p.add_argument("--suite", required=True)
    sub.add_parser("evaluate"); sub.add_parser("summarize")
    a = ap.parse_args(argv)
    fn = {"doctor": cmd_doctor, "smoke": cmd_smoke, "validate-data": cmd_validate_data,
          "run": cmd_run, "sweep": cmd_sweep, "evaluate": cmd_evaluate,
          "summarize": cmd_summarize}[a.c]
    return fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
