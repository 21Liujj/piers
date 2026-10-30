#!/usr/bin/env python3
"""Build timing.json for WildPASS train-time experiment from pd_root outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def read_jsonl_last(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return {}
    return json.loads(lines[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd-root", type=str, required=True)
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--pseudo-label-root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-max-iters", type=int, default=500)
    args = ap.parse_args()

    pd_root = Path(args.pd_root)
    ckpt_dir = pd_root / "checkpoints" / "Cityscapes_13cls" / "sam2_s"
    train_timing = ckpt_dir / "metrics" / "train_timing.jsonl"
    metricsjl = ckpt_dir / "metrics.jsonl"

    tt = read_jsonl_last(train_timing)
    mv = read_jsonl_last(metricsjl)

    train_time = float(tt.get("epoch_time", tt.get("epoch_total_time", 0.0)) or 0.0)
    avg_iter = float(tt.get("step_time", tt.get("iter_total_time", 0.0)) or 0.0)
    ips = float(tt.get("train_images_per_sec", 0.0) or 0.0)

    payload = {
        "dataset": "WildPASS13",
        "method": args.method,
        "pseudo_label_reuse": True,
        "pseudo_label_root": args.pseudo_label_root,
        "val_dataset": "DensePASS",
        "num_epochs": int(args.num_epochs),
        "num_iters": int(args.train_max_iters),
        "batch_size": int(args.batch_size),
        "train_time_sec": round(train_time, 4),
        "avg_iter_time_sec": round(avg_iter, 6),
        "images_per_sec": round(ips, 4),
        "pl_time_sec": 0.0,
        "eval_time_sec": None,
        "total_time_sec": round(train_time, 4),
        "mIoU": mv.get("val_mIoU"),
        "mAcc": mv.get("val_mAcc"),
        "PixAcc": mv.get("val_PixAcc"),
        "FWIoU": mv.get("val_FWIoU"),
        "mDice": mv.get("val_mDice"),
        "raw_train_timing_keys": {k: tt.get(k) for k in ("step_time", "epoch_time", "train_images_per_sec") if tt},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
