#!/usr/bin/env python3
"""Sanity checks for WildPASS13 processed data + optional overlays."""
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

from dataset.WildPASS import WildPASS13  # noqa: E402


def colorize(mask: np.ndarray, num_classes: int = 13) -> np.ndarray:
    rng = np.random.default_rng(1234)
    cmap = rng.integers(0, 255, size=(max(num_classes, 13), 3), dtype=np.uint8)
    cmap[255 % len(cmap)] = [0, 0, 0]
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for u in np.unique(mask):
        ui = int(u) % len(cmap)
        rgb[mask == u] = cmap[ui]
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--repo-root", type=str, default="")
    ap.add_argument("--vis-dir", type=str, default="logs/wildpass_results/vis_check")
    ap.add_argument("--num-overlay", type=int, default=10)
    args = ap.parse_args()

    repo = Path(args.repo_root or REPO_ROOT).resolve()
    cfg = OmegaConf.load(args.config if os.path.isabs(args.config) else str(repo / args.config))
    proc = cfg.get("processed_root") or cfg.get("root")
    proc_path = Path(proc)
    if not proc_path.is_absolute():
        proc_path = (repo / proc_path).resolve()

    val_rel = str(cfg.get("val_list", "lists/val.txt"))
    val_list = proc_path / val_rel if not os.path.isabs(val_rel) else Path(val_rel)
    image_dir = str(cfg.get("image_dir", "images"))
    label_dir = str(cfg.get("label_dir", "labels_trainid"))

    lines = [ln.strip() for ln in val_list.open(encoding="utf-8") if ln.strip()]
    print(f"[check] val entries: {len(lines)}")
    assert lines, "empty val list"

    ok = True
    sizes_img = []
    sizes_lbl = []
    uniques = []
    for i, name in enumerate(lines[: min(200, len(lines))]):
        ip = proc_path / image_dir / name
        stem = Path(name).stem
        lp = proc_path / label_dir / name
        if not lp.is_file():
            lp = proc_path / label_dir / f"{stem}.png"
        if not ip.is_file() or not lp.is_file():
            print(f"[error] missing pair for {name}: img={ip.is_file()} lbl={lp.is_file()}")
            ok = False
            continue
        im = Image.open(ip).convert("RGB")
        lb = Image.open(lp)
        sizes_img.append(im.size)
        sizes_lbl.append(lb.size)
        arr = np.array(lb).astype(np.int32)
        uniques.append(np.unique(arr).tolist())
        if im.size != lb.size:
            print(f"[error] size mismatch {name}: {im.size} vs {lb.size}")
            ok = False

    allowed = set(range(13)) | {255}
    for name, u in zip(lines[: min(200, len(lines))], uniques):
        if not set(u).issubset(allowed):
            print(f"[warn] label unique outside 0..12,255 for {name}: {u[:20]}")

    syn_w, syn_h = 4096, 1024
    sw_w_stride, sw_size = 384, 1024
    ds = WildPASS13(
        str(proc_path),
        str(val_list),
        crop_size=(syn_w, syn_h),
        sliding_window=(sw_w_stride, sw_size),
        passes=2,
        set_name="val",
        layout="flat",
        image_dir=image_dir,
        label_dir=label_dir,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"[check] dataloader batch ok, tensors: {[x.shape if hasattr(x, 'shape') else type(x) for x in batch]}")

    vis_dir = Path(args.vis_dir)
    if not vis_dir.is_absolute():
        vis_dir = repo / vis_dir
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(lines[: args.num_overlay]):
        ip = proc_path / image_dir / name
        stem = Path(name).stem
        lp = proc_path / label_dir / name
        if not lp.is_file():
            lp = proc_path / label_dir / f"{stem}.png"
        im = np.array(Image.open(ip).convert("RGB"))
        lb = np.array(Image.open(lp)).astype(np.int32)
        col = colorize(lb)
        h, w = min(im.shape[0], col.shape[0]), min(im.shape[1], col.shape[1])
        blend = (0.55 * im[:h, :w] + 0.45 * col[:h, :w]).astype(np.uint8)
        Image.fromarray(blend).save(vis_dir / f"overlay_{i:02d}_{stem}.png")

    print(f"[check] wrote overlays -> {vis_dir}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
