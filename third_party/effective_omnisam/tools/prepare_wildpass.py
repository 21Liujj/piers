#!/usr/bin/env python3
"""
Prepare WildPASS from zip: unzip to data_extra/raw/WildPASS, normalize to data_extra/processed/WildPASS13.
Writes logs/wildpass_results/reports/wildpass_structure.txt and meta.json under processed root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _sanitize_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        raw = os.environ.get(name, "")
        try:
            n = int(str(raw).strip())
            if n < 1:
                raise ValueError
        except (ValueError, TypeError):
            os.environ[name] = "1"


_sanitize_threads()

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
LBL_EXT = {".png", ".tif", ".tiff", ".npy", ".PNG", ".TIF", ".TIFF"}


def _count_by_ext(dir_path: Path) -> Tuple[int, Dict[str, int]]:
    counts: Dict[str, int] = defaultdict(int)
    n = 0
    if not dir_path.is_dir():
        return 0, {}
    for p in dir_path.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix
        if suf in IMG_EXT:
            counts["image"] += 1
            n += 1
        elif suf in LBL_EXT:
            counts["label"] += 1
            n += 1
    return n, dict(counts)


def scan_dataset_roots(raw_root: Path) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Find best image root and label root directories under raw_root.
    Returns (img_root, lbl_root, note).
    """
    best_img: Optional[Tuple[int, Path]] = None
    best_lbl: Optional[Tuple[int, Path]] = None

    for dirpath, _, filenames in os.walk(raw_root):
        p = Path(dirpath)
        imgs = sum(1 for f in filenames if Path(f).suffix in IMG_EXT)
        lbls = sum(1 for f in filenames if Path(f).suffix in LBL_EXT)
        if imgs > 0:
            cur = best_img[0] if best_img else 0
            if imgs > cur:
                best_img = (imgs, p)
        if lbls > 0:
            cur = best_lbl[0] if best_lbl else 0
            if lbls > cur:
                best_lbl = (lbls, p)

    note_parts = []
    if best_img:
        note_parts.append(f"image_dir candidate: {best_img[1]} ({best_img[0]} files)")
    if best_lbl:
        note_parts.append(f"label_dir candidate: {best_lbl[1]} ({best_lbl[0]} files)")

    # Cityscapes layout override
    cs_img = None
    cs_lbl = None
    for cand in raw_root.rglob("leftImg8bit"):
        if cand.is_dir():
            cs_img = cand
            break
    for cand in raw_root.rglob("gtFine"):
        if cand.is_dir():
            cs_lbl = cand
            break
    if cs_img is not None and cs_lbl is not None:
        # Prefer deepest val split that has files
        for split in ("val", "train"):
            for city in sorted(cs_img.glob(f"{split}/*")):
                if city.is_dir():
                    n_img, _ = _count_by_ext(city)
                    if n_img > 0:
                        matching_gt = cs_lbl / split / city.name
                        if matching_gt.is_dir():
                            n_lbl, _ = _count_by_ext(matching_gt)
                            if n_lbl > 0:
                                return city, matching_gt, f"cityscapes_style:{split}/{city.name}"

    img_r = best_img[1] if best_img else None
    lbl_r = best_lbl[1] if best_lbl else None
    return img_r, lbl_r, "; ".join(note_parts) if note_parts else "no_files_found"


def try_symlink(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
        return True
    except OSError:
        return False


def link_or_copy(src: Path, dst: Path) -> str:
    if try_symlink(src, dst):
        return "symlink"
    shutil.copy2(src, dst)
    return "copy"


def gather_pairs(img_root: Path, lbl_root: Path) -> List[Tuple[str, Path, Path]]:
    """Match image and label by basename stem."""
    imgs: Dict[str, Path] = {}
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXT:
            imgs[p.stem] = p
    lbls: Dict[str, Path] = {}
    for p in lbl_root.rglob("*"):
        if p.is_file() and p.suffix in LBL_EXT:
            lbls[p.stem] = p
    keys = sorted(set(imgs.keys()) & set(lbls.keys()))
    out: List[Tuple[str, Path, Path]] = []
    for k in keys:
        rel_img = imgs[k].relative_to(img_root)
        rel_lbl = lbls[k].relative_to(lbl_root)
        out.append((f"{rel_img.as_posix()}", imgs[k], lbls[k]))
    return out


def stat_sizes(paths: List[Path], sample_limit: int = 500) -> Dict[str, float]:
    from PIL import Image

    ws, hs = [], []
    for p in paths[:sample_limit]:
        try:
            with Image.open(p) as im:
                w, h = im.size
                ws.append(w)
                hs.append(h)
        except Exception:
            continue
    if not ws:
        return {"w_min": 0, "w_max": 0, "h_min": 0, "h_max": 0, "w_p50": 0, "h_p50": 0}

    def pct(xs: List[float], q: float) -> float:
        ys = sorted(xs)
        return ys[int(q * (len(ys) - 1))]

    return {
        "w_min": float(min(ws)),
        "w_max": float(max(ws)),
        "h_min": float(min(hs)),
        "h_max": float(max(hs)),
        "w_p50": pct(ws, 0.5),
        "h_p50": pct(hs, 0.5),
    }


def write_report(
    report_path: Path,
    n_img: int,
    n_lbl: int,
    img_sample: str,
    lbl_sample: str,
    img_sz: Dict[str, float],
    lbl_sz: Dict[str, float],
    paired: bool,
    label_unique_sample: str,
    val_lines_sample: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "WildPASS structure report",
        "========================",
        f"image_count: {n_img}",
        f"label_count: {n_lbl}",
        f"image_example: {img_sample}",
        f"label_example: {lbl_sample}",
        f"image_size_distribution: {json.dumps(img_sz, ensure_ascii=False)}",
        f"label_size_distribution: {json.dumps(lbl_sz, ensure_ascii=False)}",
        f"image_label_paired_one_to_one: {paired}",
        f"label_unique_values_sample: {label_unique_sample}",
        "val.txt_sample:",
        val_lines_sample,
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default="", help="Path to WildPASS.zip")
    ap.add_argument("--repo-root", type=str, default="", help="Repository root")
    ap.add_argument(
        "--raw-parent",
        type=str,
        default="data_extra/raw",
        help="Directory to unzip into (creates raw-parent/WildPASS/ when zip has top-level WildPASS/).",
    )
    ap.add_argument("--processed-out", type=str, default="data_extra/processed/WildPASS13")
    ap.add_argument("--reports-dir", type=str, default="logs/wildpass_results/reports")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).resolve()
    raw_parent = (repo / args.raw_parent).resolve()
    proc_root = (repo / args.processed_out).resolve()
    reports_dir = (repo / args.reports_dir).resolve()
    zip_path = Path(args.zip).resolve() if args.zip else repo / "WildPASS.zip"

    need_confirm = reports_dir / "NEED_CONFIRM_WILDPASS_STRUCTURE.md"

    if args.dry_run:
        print("[dry-run] would unzip:", zip_path, "->", raw_parent)
        print("[dry-run] would layout:", proc_root)
        print("[dry-run] report ->", reports_dir / "wildpass_structure.txt")
        return 0

    raw_parent.mkdir(parents=True, exist_ok=True)
    if zip_path.is_file():
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(raw_parent)],
            check=True,
        )
    else:
        print(f"[warn] zip not found: {zip_path}, assuming raw data exists under {raw_parent / 'WildPASS'}")

    scan_root = raw_parent / "WildPASS"
    if not scan_root.is_dir():
        for name in ("WildPASS", "wildpass"):
            if (raw_parent / name).is_dir():
                scan_root = raw_parent / name
                break

    img_root, lbl_root, note = scan_dataset_roots(scan_root)
    if img_root is None or lbl_root is None:
        need_confirm.parent.mkdir(parents=True, exist_ok=True)
        need_confirm.write_text(
            "# NEED_CONFIRM_WILDPASS_STRUCTURE\n\n"
            f"Could not detect image and/or label directories under `{scan_root}`.\n"
            f"scan_note: {note}\n",
            encoding="utf-8",
        )
        print(f"[error] wrote {need_confirm}", file=sys.stderr)
        return 2

    pairs = gather_pairs(img_root, lbl_root)
    if not pairs:
        need_confirm.parent.mkdir(parents=True, exist_ok=True)
        need_confirm.write_text(
            "# NEED_CONFIRM_WILDPASS_STRUCTURE\n\n"
            f"No matching image/label basenames under:\n- {img_root}\n- {lbl_root}\n",
            encoding="utf-8",
        )
        print(f"[error] wrote {need_confirm}", file=sys.stderr)
        return 2

    if need_confirm.exists():
        need_confirm.unlink()

    # Build processed layout: images/, labels_raw/, lists/val.txt
    images_dir = proc_root / "images"
    labels_raw_dir = proc_root / "labels_raw"
    lists_dir = proc_root / "lists"
    for d in (images_dir, labels_raw_dir, lists_dir):
        d.mkdir(parents=True, exist_ok=True)

    val_names: List[str] = []
    link_modes: List[str] = []
    for rel_key, ip, lp in pairs:
        base_name = Path(rel_key).name
        val_names.append(base_name)
        r1 = link_or_copy(ip, images_dir / base_name)
        r2 = link_or_copy(lp, labels_raw_dir / base_name)
        link_modes.extend([r1, r2])

    val_names = sorted(set(val_names))
    val_txt = lists_dir / "val.txt"
    val_txt.write_text("\n".join(val_names) + "\n", encoding="utf-8")

    meta = {
        "scan_root": str(scan_root),
        "image_source_dir": str(img_root),
        "label_source_dir": str(lbl_root),
        "scan_note": note,
        "processed_root": str(proc_root),
        "num_images": len(val_names),
        "link_modes": {"symlink": link_modes.count("symlink"), "copy": link_modes.count("copy")},
    }
    (proc_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    img_paths = [images_dir / n for n in val_names]
    lbl_paths = [labels_raw_dir / n for n in val_names]
    img_sz = stat_sizes(img_paths)
    lbl_sz = stat_sizes(lbl_paths)

    # Label unique sample (first label)
    import numpy as np

    uniq_sample = ""
    try:
        p0 = lbl_paths[0]
        if p0.suffix.lower() == ".npy":
            arr = np.load(str(p0))
            u = np.unique(arr)
        else:
            from PIL import Image

            arr = np.array(Image.open(p0))
            u = np.unique(arr)
        uniq_sample = str(u[:32].tolist()) + ("..." if len(u) > 32 else "")
    except Exception as e:
        uniq_sample = f"(failed: {e})"

    write_report(
        reports_dir / "wildpass_structure.txt",
        n_img=len(val_names),
        n_lbl=len(val_names),
        img_sample=str(img_paths[0]),
        lbl_sample=str(lbl_paths[0]),
        img_sz=img_sz,
        lbl_sz=lbl_sz,
        paired=len(val_names) == len(pairs),
        label_unique_sample=uniq_sample,
        val_lines_sample="\n".join(val_names[:10]),
    )

    print(f"[ok] processed -> {proc_root}")
    print(f"[ok] report -> {reports_dir / 'wildpass_structure.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
