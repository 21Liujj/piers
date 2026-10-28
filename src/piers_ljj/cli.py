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
    import os
    import yaml
    from piers_ljj.experiments.scheduler import (config_hash, run_dir, read_status,
                                                 write_status, check_deps,
                                                 launch_worker)
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    suite = cfg.get("suite", "custom")
    run_id = cfg.get("run_id", suite)
    root = cfg.get("runs_root", "runs")
    d = run_dir(root, run_id)
    with open(os.path.join(d, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    st = read_status(d)
    if st.get("status") == "completed_real" and st.get("config_hash") == config_hash(cfg):
        print(f"run {run_id}: resume-skip (completed_real, config unchanged)")
        return 0
    missing = check_deps(suite, root)
    if missing:
        write_status(d, {"status": "blocked", "suite": suite,
                         "missing_deps": missing, "config_hash": config_hash(cfg)})
        print(f"run {run_id}: BLOCKED, missing deps {missing}")
        return 3
    if a.dry_run:
        write_status(d, {"status": "not_run", "suite": suite, "dry_run": True,
                         "config_hash": config_hash(cfg), "synthetic": False})
        print(f"run {run_id}: dry-run OK (config + deps validated, no scores)")
        return 0
    # real dispatch: text suites -> grounded worker (needs manifest + cuda for real)
    import torch  # noqa — ImportError on core-only host is a clear blocked signal
    try:
        import torch as _t
        cuda = _t.cuda.is_available()
    except Exception:
        cuda = False
    if suite.startswith("T") or cfg.get("worker") == "grounded":
        manifest = cfg.get("manifest")
        if not manifest:
            write_status(d, {"status": "blocked", "reason": "no manifest for text suite",
                             "config_hash": config_hash(cfg)})
            print(f"run {run_id}: BLOCKED (text suite needs manifest)")
            return 3
        if not cuda and not cfg.get("allow_cpu_smoke"):
            write_status(d, {"status": "pending_gpu", "suite": suite,
                             "reason": "no CUDA on host; use --dry-run or GPU host",
                             "config_hash": config_hash(cfg)})
            print(f"run {run_id}: pending_gpu (no CUDA here; see gpu_runbook)")
            return 2
        rc = launch_worker("grounded", ["--config", a.config, "--manifest", manifest,
                                        "--out", os.path.join(d, "worker")],
                           os.path.join(d, "worker.log"))
        st2 = read_status(os.path.join(d, "worker"))
        write_status(d, {"status": st2.get("status", "failed" if rc else "blocked"),
                         "suite": suite, "worker_rc": rc,
                         "config_hash": config_hash(cfg), "synthetic": False})
        return rc
    write_status(d, {"status": "pending_gpu", "suite": suite,
                     "reason": "omni/UDA backend runs on GPU host (runbook step 5-6)",
                     "config_hash": config_hash(cfg)})
    print(f"run {run_id}: pending_gpu (omni suite; dispatch on GPU host)")
    return 2


def cmd_sweep(a):
    from piers_ljj.experiments.suites import SUITES, DEPS
    from piers_ljj.experiments.scheduler import check_deps
    print(f"suite {a.suite}: {SUITES.get(a.suite, 'unknown')}; deps {DEPS.get(a.suite, [])}")
    order = [a.suite]
    # expand transitive deps first
    stack = list(DEPS.get(a.suite, []))
    while stack:
        s = stack.pop()
        order.append(s)
        stack.extend(DEPS.get(s, []))
    print("order (deps-first): " + " -> ".join(reversed(order)))
    if getattr(a, "check_only", False):
        missing = check_deps(a.suite, getattr(a, "runs_root", "runs"))
        print(f"missing: {missing or 'none'}")
    return 0


def cmd_evaluate(a):
    import json
    import numpy as np
    from piers_ljj.evaluation.metrics import (binary_scores, confusion_matrix,
                                              scores_from_cm)
    pred = np.load(a.pred_npz)["pred"] if a.pred_npz.endswith(".npz") else np.load(a.pred_npz)
    gt = np.load(a.gt_npz)["gt"] if a.gt_npz.endswith(".npz") else np.load(a.gt_npz)
    task = a.task
    if task == "text":
        area_thr = int(a.area_thr)
        if pred.ndim == 3:  # per-query stack -> per-query scores + macro/micro
            per, inter, union = [], 0, 0
            for p, g in zip(pred, gt):
                r = binary_scores(p, g, area_thr=area_thr)
                per.append(r)
                if r["iou"] is not None:
                    inter += int(np.logical_and(p > 0, g > 0).sum())
                    union += int(np.logical_or(p > 0, g > 0).sum())
            vals = [r["iou"] for r in per if r["iou"] is not None]
            out = {"task": "text", "n": len(per),
                   "macro_iou": float(np.mean(vals)) if vals else None,
                   "micro_iou": float(inter) / float(max(union, 1)) if vals else None,
                   "correct_reject": sum(1 for r in per if r.get("correct_reject")),
                   "false_positive": sum(1 for r in per if r.get("false_positive")),
                   "per_query": per}
        else:
            out = {"task": "text", **binary_scores(pred, gt, area_thr=area_thr)}
    else:  # uda semantic
        cm = confusion_matrix(pred, gt, int(a.n_cls))
        out = {"task": "uda", **scores_from_cm(cm),
               "confusion_matrix": cm.tolist()}
    out["synthetic"] = bool(a.synthetic)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"evaluate ({task}): -> {a.out} synthetic={out['synthetic']}")
    return 0


def cmd_summarize(a):
    import glob
    import json
    import os
    from piers_ljj.experiments.scheduler import read_status
    rows = []
    for st_path in glob.glob(os.path.join(a.runs_root, "*", "status.json")):
        run_id = os.path.basename(os.path.dirname(st_path))
        st = read_status(os.path.dirname(st_path))
        if st.get("synthetic"):
            continue  # synthetic NEVER aggregates
        if st.get("status") != "completed_real":
            rows.append({"run_id": run_id, "status": st.get("status"),
                         "note": "not aggregated (not completed_real)"})
            continue
        met_path = os.path.join(a.runs_root, run_id, "metrics.json")
        met = {}
        if os.path.exists(met_path):
            with open(met_path, encoding="utf-8") as f:
                met = json.load(f)
        rows.append({"run_id": run_id, "status": "completed_real", **met})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    n_real = sum(1 for r in rows if r.get("status") == "completed_real")
    print(f"summarize: {n_real} real runs aggregated -> {a.out} (synthetic excluded)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="piers_ljj")
    sub = ap.add_subparsers(dest="c", required=True)
    sub.add_parser("doctor")
    sub.add_parser("smoke")
    sub.add_parser("validate-data")
    p = sub.add_parser("run"); p.add_argument("--config", required=True); p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p = sub.add_parser("sweep"); p.add_argument("--suite", required=True); p.add_argument("--check-only", dest="check_only", action="store_true"); p.add_argument("--runs-root", dest="runs_root", default="runs")
    p = sub.add_parser("evaluate"); p.add_argument("--pred-npz", required=True); p.add_argument("--gt-npz", required=True); p.add_argument("--task", choices=["text", "uda"], required=True); p.add_argument("--out", required=True); p.add_argument("--n-cls", dest="n_cls", type=int, default=13); p.add_argument("--area-thr", dest="area_thr", type=int, default=16); p.add_argument("--synthetic", action="store_true")
    p = sub.add_parser("summarize"); p.add_argument("--runs-root", dest="runs_root", default="runs"); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fn = {"doctor": cmd_doctor, "smoke": cmd_smoke, "validate-data": cmd_validate_data,
          "run": cmd_run, "sweep": cmd_sweep, "evaluate": cmd_evaluate,
          "summarize": cmd_summarize}[a.c]
    return fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
