#!/usr/bin/env python3
"""Merge paper_ckpt + final_ckpt metrics/timing into val_ckpts_summary.json (used by 01_val_ckpts_wildpass.sh)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    raw = os.environ.get("OUT_ROOT", "").strip()
    if not raw and len(sys.argv) > 1:
        raw = sys.argv[1].strip()
    if not raw:
        print("write_val_ckpts_summary: set OUT_ROOT or pass path", file=sys.stderr)
        return 2
    root = Path(raw).resolve()
    summary: dict = {"out_root": str(root), "runs": []}
    for name in ("paper_ckpt", "final_ckpt"):
        mj = root / name / "metrics.json"
        tj = root / name / "timing.json"
        row: dict = {"name": name, "metrics_json": str(mj) if mj.is_file() else None}
        if mj.is_file():
            m = json.loads(mj.read_text(encoding="utf-8"))
            row["ckpt"] = m.get("ckpt")
            row["mIoU"] = m.get("mIoU")
            row["mAcc"] = m.get("mAcc")
            row["PixAcc"] = m.get("PixAcc")
        if tj.is_file():
            t = json.loads(tj.read_text(encoding="utf-8"))
            row["eval_time_sec"] = t.get("eval_time_sec")
            row["images_per_sec"] = t.get("images_per_sec")
            row["memory_accel"] = t.get("memory_accel")
        if row.get("metrics_json"):
            summary["runs"].append(row)

    out = root / "val_ckpts_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n = len(summary["runs"])
    print(f"[01_val_ckpts_wildpass] wrote {out} ({n} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
