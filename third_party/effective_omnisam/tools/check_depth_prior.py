#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.depth_prior import find_depth_path, DEFAULT_DEPTH_EXTS


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Check WildPASS RGB-depth alignment.")
    ap.add_argument("--data_root", type=str, default="data/WildPASS", help="WildPASS root with leftImg8bit/")
    ap.add_argument("--list_path", type=str, default="dataset/densepass_list/train.txt")
    ap.add_argument("--depth_dir", type=str, required=True)
    ap.add_argument("--file_exts", type=str, default=",".join(DEFAULT_DEPTH_EXTS))
    args = ap.parse_args()

    exts = [x.strip() for x in str(args.file_exts).split(",") if x.strip()]
    names = _read_list(args.list_path)
    total = 0
    found = 0
    missing_samples = []
    for name in names:
        rel = name.lstrip("/")
        rgb = os.path.join(args.data_root, "leftImg8bit", rel)
        if not os.path.isfile(rgb):
            continue
        total += 1
        dpath = find_depth_path(rgb, args.depth_dir, exts)
        if dpath is not None:
            found += 1
        elif len(missing_samples) < 20:
            missing_samples.append(rel)

    ratio = (found / max(1, total))
    print(f"rgb_total={total}")
    print(f"depth_found={found}")
    print(f"depth_missing={max(0, total - found)}")
    print(f"depth_available_ratio={ratio:.6f}")
    if missing_samples:
        print("missing_examples:")
        for x in missing_samples:
            print(f"  - {x}")


if __name__ == "__main__":
    main()

