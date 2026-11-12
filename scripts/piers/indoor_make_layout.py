"""Adapt SPAN8_PREP into the layout the base model expects.

The upstream base model (`third_party/effective_omnisam`) already supports
Stanford2D3D / SPAN8 natively:
  * DATASET_NAME2CLASSES["Stanford2D3D"] = 8 classes in SPAN8 order
  * dataset/Stanford360Span8.py reads images/ + labels_trainid/ + lists/*.txt
  * trainIds are exactly 0..7 with 255 = ignore

So instead of writing our own loader we materialise that layout from
SPAN8_PREP and reuse the upstream dataset class verbatim. That keeps the
adaptation surface minimal and keeps the frozen outdoor path untouched.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

SRC = Path("/root/autodl-tmp/data_extra/SPAN8_PREP")
DST = Path("/root/autodl-tmp/data_extra/Stanford2d3d_Seg")
RUNS = Path("/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109")

SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]


def build():
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels_trainid").mkdir(parents=True, exist_ok=True)
    (DST / "lists").mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(SRC / "manifest.jsonl", encoding="utf-8") if l.strip()]
    by_area = {}
    for r in recs:
        by_area.setdefault(r["area"], []).append(r)
    counts = {}
    for area, rs in by_area.items():
        names = []
        for r in rs:
            fn = Path(r["image"]).name
            src_i = SRC / area / r["image"]
            src_l = SRC / area / r["label"]
            dst_i = DST / "images" / fn
            dst_l = DST / "labels_trainid" / fn
            if not dst_i.exists():
                os.symlink(src_i, dst_i)
            if not dst_l.exists():
                os.symlink(src_l, dst_l)
            names.append(fn)
        names.sort()
        counts[area] = len(names)
    # split lists (fold-level), reusing the room-disjoint splits already built
    split_map = {"fit": "train_fit.txt", "calib": "train_calib.txt",
                 "val": "train_val.txt", "heldout": "val.txt"}
    written = {}
    for split, fname in split_map.items():
        p = RUNS / f"split_{split}.jsonl"
        if not p.exists():
            continue
        rs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        names = sorted(Path(r["image"]).name for r in rs)
        (DST / "lists" / fname).write_text("\n".join(names) + "\n", encoding="utf-8")
        written[fname] = len(names)
    summary = {
        "dst": str(DST),
        "per_area": counts,
        "lists": written,
        "layout": {"images": "images/", "labels": "labels_trainid/", "lists": "lists/"},
        "note": "symlinks into SPAN8_PREP; upstream Stanford360Span8 loader reused verbatim",
    }
    (DST / "LAYOUT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
