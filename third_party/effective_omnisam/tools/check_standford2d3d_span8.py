#!/usr/bin/env python3
"""Sanity checks for Stanford360_SPAN8 processed data + overlays under logs/Standford2d3d_results/vis_check."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.Stanford360Span8 import Stanford360PanoramaVal  # noqa: E402


def colorize(mask: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1234)
    cmap = rng.integers(0, 255, size=(16, 3), dtype=np.uint8)
    cmap[8 % 16] = [200, 200, 200]
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for u in np.unique(mask):
        ui = int(u) % 16
        rgb[mask == u] = cmap[ui]
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--repo-root", type=str, default="")
    ap.add_argument(
        "--vis-dir",
        type=str,
        default="/root/autodl-tmp/OmniSAM-main/logs/Standford2d3d_results/vis_check",
    )
    ap.add_argument("--report-mismatch", type=str, default="")
    ap.add_argument("--num-overlay", type=int, default=10)
    args = ap.parse_args()

    repo = Path(args.repo_root or REPO_ROOT).resolve()
    cfg = OmegaConf.load(args.config if os.path.isabs(args.config) else str(repo / args.config))
    proc = cfg.get("processed_root") or cfg.get("root")
    proc_path = Path(proc)
    if not proc_path.is_absolute():
        proc_path = (repo / proc_path).resolve()

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    val_rel = str(cfg.get("val_list", "lists/val.txt"))
    val_list = proc_path / val_rel if not os.path.isabs(val_rel) else Path(val_rel)
    image_dir = str(cfg.get("image_dir", "images"))
    label_dir = str(cfg.get("label_dir", "labels_trainid"))

    lines = [ln.strip() for ln in val_list.open(encoding="utf-8") if ln.strip()]
    print(f"[check] val entries: {len(lines)}")
    if not lines:
        print("[error] empty val list", file=sys.stderr)
        return 2

    mismatch_path = Path(
        args.report_mismatch
        or str(repo / "logs/Standford2d3d_results/reports/standford2d3d_size_mismatch.txt")
    )

    ok = True
    sizes_img = []
    sizes_lbl = []
    mm_lines: list[str] = []
    allowed = set(range(8)) | {255}
    for i, name in enumerate(lines[: min(200, len(lines))]):
        ip = proc_path / image_dir / name
        stem = Path(name).stem
        lp = proc_path / label_dir / name
        if not lp.is_file():
            for p in (proc_path / label_dir).glob(stem + ".*"):
                lp = p
                break
        if not ip.is_file() or not lp.is_file():
            print(f"[error] missing pair for {name}: img={ip.is_file()} lbl={lp.is_file()}")
            ok = False
            continue
        im = Image.open(ip).convert("RGB")
        lb = Image.open(lp)
        sizes_img.append(im.size)
        sizes_lbl.append(lb.size)
        arr = np.array(lb).astype(np.int32)
        if im.size != lb.size:
            msg = f"{name}: image {im.size} vs label {lb.size}"
            mm_lines.append(msg)
            print(f"[error] size mismatch {msg}")
            ok = False
        u = set(np.unique(arr).tolist())
        if not u.issubset(allowed):
            print(f"[warn] label unique outside 0..7,255 for {name}: {sorted(u)[:30]}")
            ok = False

    if mm_lines:
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        mismatch_path.write_text("\n".join(mm_lines) + "\n", encoding="utf-8")

    crop = tuple(cfg.get("crop_size", [3072, 1024]))
    sw = tuple(cfg.get("sliding_window", [256, 1024]))
    legacy = bool(cfg.get("legacy_area_crop", False))
    ds = Stanford360PanoramaVal(
        str(proc_path),
        str(val_list),
        crop_size=(int(crop[0]), int(crop[1])),
        sliding_window=(int(sw[0]), int(sw[1])),
        passes=int(cfg.get("passes", 2)),
        legacy_area_crop=legacy,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"[check] dataloader batch tensors: image {batch[0].shape}, label {batch[1].shape}")

    n_vis = min(args.num_overlay, len(lines))
    for i in range(n_vis):
        name = lines[i]
        ip = proc_path / image_dir / name
        stem = Path(name).stem
        lp = proc_path / label_dir / name
        if not lp.is_file():
            for p in (proc_path / label_dir).glob(stem + ".*"):
                lp = p
                break
        im = np.array(Image.open(ip).convert("RGB"))
        lb = np.array(Image.open(lp).convert("L"))
        cm = colorize(lb.astype(np.int32))
        alpha = 0.45
        blend = (im.astype(np.float32) * (1 - alpha) + cm.astype(np.float32) * alpha).astype(np.uint8)
        out_p = vis_dir / f"overlay_{i:02d}_{stem}.png"
        Image.fromarray(blend).save(out_p)
        print(f"[vis] {out_p}")

    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
