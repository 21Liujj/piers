#!/usr/bin/env python3
"""
Prepare Stanford-360 from standford-360.zip: raw under data_extra/raw/Stanford360,
processed layout data_extra/processed/Stanford360_SPAN8 (images, labels_raw, labels_trainid, lists, meta.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
LBL_EXT = {".png", ".tif", ".tiff", ".npy", ".PNG", ".TIF", ".TIFF", ".NPY"}
IMG_KW = ("image", "images", "img", "rgb", "pano", "panorama", "erp")
LBL_KW = ("label", "labels", "semantic", "mask", "annotation", "ann", "gt")


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


def try_symlink(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except OSError:
            return False
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
        return True
    except OSError:
        return False


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> str:
    if use_symlink and try_symlink(src, dst):
        return "symlink"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copy"


def _keyword_bonus(dirname: str, keys: Tuple[str, ...]) -> int:
    low = dirname.lower()
    return sum(10 for k in keys if k in low)


def scan_best_dirs(raw_root: Path) -> Tuple[Optional[Path], Optional[Path], str]:
    best_img: Optional[Tuple[float, Path]] = None
    best_lbl: Optional[Tuple[float, Path]] = None

    for dirpath, _, filenames in os.walk(raw_root):
        p = Path(dirpath)
        imgs = sum(1 for f in filenames if Path(f).suffix in IMG_EXT)
        lbls = sum(1 for f in filenames if Path(f).suffix in LBL_EXT)
        name = p.name
        if imgs > 0:
            score = imgs + _keyword_bonus(name, IMG_KW)
            if best_img is None or score > best_img[0]:
                best_img = (score, p)
        if lbls > 0:
            score = lbls + _keyword_bonus(name, LBL_KW)
            if best_lbl is None or score > best_lbl[0]:
                best_lbl = (score, p)

    note_parts = []
    if best_img:
        note_parts.append(f"image_dir: {best_img[1]} (score={best_img[0]:.0f})")
    if best_lbl:
        note_parts.append(f"label_dir: {best_lbl[1]} (score={best_lbl[0]:.0f})")

    return (
        best_img[1] if best_img else None,
        best_lbl[1] if best_lbl else None,
        "; ".join(note_parts) if note_parts else "no_files_found",
    )


def gather_pairs(img_root: Path, lbl_root: Path) -> List[Tuple[str, Path, Path]]:
    imgs: Dict[str, Path] = {}
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXT:
            if p.stem not in imgs:
                imgs[p.stem] = p
    lbls: Dict[str, Path] = {}
    for p in lbl_root.rglob("*"):
        if p.is_file() and p.suffix in LBL_EXT:
            if p.stem not in lbls:
                lbls[p.stem] = p
    keys = sorted(set(imgs.keys()) & set(lbls.keys()))
    return [(imgs[k].relative_to(img_root).as_posix(), imgs[k], lbls[k]) for k in keys]


def find_official_splits(raw_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    train_p = val_p = None
    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if n in ("train.txt", "train_list.txt") and train_p is None:
            train_p = p
        if n in ("val.txt", "valid.txt", "validation.txt", "val_list.txt") and val_p is None:
            val_p = p
    return train_p, val_p


def read_split_basenames(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(Path(line).name)
    return out


def stat_sizes(paths: List[Path], sample_limit: int = 500) -> Dict[str, float]:
    from PIL import Image

    ws, hs = [], []
    for p in paths[:sample_limit]:
        try:
            if p.suffix.lower() == ".npy":
                import numpy as np

                arr = np.load(str(p))
                if arr.ndim >= 2:
                    h, w = arr.shape[-2], arr.shape[-1]
                    ws.append(float(w))
                    hs.append(float(h))
                continue
            with Image.open(p) as im:
                w, h = im.size
                ws.append(float(w))
                hs.append(float(h))
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


def unique_sample_label(path: Path) -> str:
    import numpy as np

    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(str(path))
            u = np.unique(arr)
        else:
            from PIL import Image

            arr = np.array(Image.open(path))
            u = np.unique(arr)
        return str(u[:48].tolist()) + ("..." if len(u) > 48 else "")
    except Exception as e:
        return f"(failed: {e})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--log_root", type=str, required=True)
    ap.add_argument("--use_symlink", type=str, default="true")
    ap.add_argument("--force", type=str, default="false")
    args = ap.parse_args()

    use_symlink = str(args.use_symlink).lower() in ("1", "true", "yes")
    force = str(args.force).lower() in ("1", "true", "yes")

    out_root = Path(args.out_root).resolve()
    log_root = Path(args.log_root).resolve()
    zip_path = Path(args.zip_path).resolve()

    raw_root = out_root / "raw" / "Stanford360"
    proc_root = out_root / "processed" / "Stanford360_SPAN8"
    reports_dir = log_root / "reports"
    need_confirm = reports_dir / "NEED_CONFIRM_STANFORD360_STRUCTURE.md"

    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file():
        need_confirm.write_text(
            f"# NEED_CONFIRM_STANFORD360_STRUCTURE\n\nZip not found: `{zip_path}`\n",
            encoding="utf-8",
        )
        print(f"[error] missing zip: {zip_path}", file=sys.stderr)
        return 2

    has_content = raw_root.is_dir() and any(raw_root.iterdir())
    if not has_content or force:
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(raw_root)], check=True)
    else:
        print("[info] raw Stanford360 already populated; skip unzip (use --force true to re-unzip)")

    scan_root = raw_root
    subs = [p for p in raw_root.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if len(subs) == 1:
        scan_root = subs[0]

    img_root, lbl_root, scan_note = scan_best_dirs(scan_root)
    if img_root is None or lbl_root is None:
        need_confirm.write_text(
            "# NEED_CONFIRM_STANFORD360_STRUCTURE\n\n"
            f"Could not detect image and/or label directories under `{scan_root}`.\n\n"
            f"scan_note: {scan_note}\n",
            encoding="utf-8",
        )
        print(f"[error] wrote {need_confirm}", file=sys.stderr)
        return 2

    pairs = gather_pairs(img_root, lbl_root)
    if not pairs:
        need_confirm.write_text(
            "# NEED_CONFIRM_STANFORD360_STRUCTURE\n\n"
            f"No matching image/label stems under:\n- `{img_root}`\n- `{lbl_root}`\n",
            encoding="utf-8",
        )
        print(f"[error] wrote {need_confirm}", file=sys.stderr)
        return 2

    if need_confirm.exists():
        try:
            need_confirm.unlink()
        except OSError:
            pass

    stem_to_pair: Dict[str, Tuple[Path, Path]] = {}
    for _, ip, lp in pairs:
        stem_to_pair[ip.stem] = (ip, lp)

    train_txt_path, val_txt_path = find_official_splits(scan_root)

    images_dir = proc_root / "images"
    labels_raw_dir = proc_root / "labels_raw"
    labels_trainid_dir = proc_root / "labels_trainid"
    lists_dir = proc_root / "lists"

    if force:
        for d in (images_dir, labels_raw_dir, labels_trainid_dir):
            if d.is_dir():
                shutil.rmtree(d)
    for d in (images_dir, labels_raw_dir, labels_trainid_dir, lists_dir):
        d.mkdir(parents=True, exist_ok=True)

    link_modes: List[str] = []
    used_dst: Set[str] = set()
    coll = 0

    def unique_bn(ip: Path, lp: Path) -> Tuple[str, str]:
        nonlocal coll
        ib, lb = ip.name, lp.name
        if ib in used_dst:
            coll += 1
            stem = f"{ip.stem}__{coll}"
            ib = stem + ip.suffix
            lb = stem + lp.suffix
        used_dst.add(ib)
        return ib, lb

    stem_to_flat: Dict[str, Tuple[str, str]] = {}
    for stem in sorted(stem_to_pair.keys()):
        ip, lp = stem_to_pair[stem]
        ib, lb = unique_bn(ip, lp)
        stem_to_flat[stem] = (ib, lb)
        link_modes.append(link_or_copy(ip, images_dir / ib, use_symlink))
        link_modes.append(link_or_copy(lp, labels_raw_dir / lb, use_symlink))

    stem_order = sorted(stem_to_pair.keys())

    train_names: List[str] = []
    val_names: List[str] = []
    split_source = ""

    use_official = False
    if train_txt_path and val_txt_path and train_txt_path.is_file() and val_txt_path.is_file():
        tr = read_split_basenames(train_txt_path)
        va = read_split_basenames(val_txt_path)
        tr_set = {Path(x).name for x in tr}
        va_set = {Path(x).name for x in va}
        train_stems: Set[str] = set()
        val_stems: Set[str] = set()
        for st in stem_order:
            ip, _ = stem_to_pair[st]
            ib = stem_to_flat[st][0]
            oname = ip.name
            if oname in tr_set or st in tr_set:
                train_stems.add(st)
            elif oname in va_set or st in va_set:
                val_stems.add(st)
        if train_stems and val_stems and not (train_stems & val_stems):
            use_official = True
            split_source = (
                f"official_or_packaged_split_files train={train_txt_path.relative_to(scan_root)} "
                f"val={val_txt_path.relative_to(scan_root)}"
            )
            missed = set(stem_order) - train_stems - val_stems
            for st in sorted(missed):
                train_stems.add(st)
            train_names = sorted(stem_to_flat[st][0] for st in train_stems)
            val_names = sorted(stem_to_flat[st][0] for st in val_stems)

    if not use_official:
        sorted_names = sorted(stem_to_flat[st][0] for st in stem_order)
        n = len(sorted_names)
        if n == 1413:
            train_names = sorted_names[:1040]
            val_names = sorted_names[1040:]
            split_source = "non-official deterministic split (sorted filenames; n=1413 -> 1040 train / 373 val)"
        else:
            order = sorted_names[:]
            random.Random(42).shuffle(order)
            n_val = max(1, n // 4)
            val_names = sorted(order[-n_val:])
            train_names = sorted(order[:-n_val])
            split_source = (
                "non-official deterministic split (seed=42 shuffle after sort; ~75% train / ~25% val)"
            )

    (lists_dir / "train.txt").write_text("\n".join(train_names) + ("\n" if train_names else ""), encoding="utf-8")
    (lists_dir / "val.txt").write_text("\n".join(val_names) + ("\n" if val_names else ""), encoding="utf-8")

    meta = {
        "zip_path": str(zip_path),
        "raw_extract_path": str(raw_root),
        "scan_root": str(scan_root),
        "image_source_dir": str(img_root),
        "label_source_dir": str(lbl_root),
        "processed_root": str(proc_root),
        "scan_note": scan_note,
        "split_source": split_source,
        "num_pairs": len(pairs),
        "num_train_list": len(train_names),
        "num_val_list": len(val_names),
        "link_mode_counts": {
            "symlink": sum(1 for m in link_modes if m == "symlink"),
            "copy": sum(1 for m in link_modes if m == "copy"),
        },
        "use_symlink": use_symlink,
    }
    (proc_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    img_paths = sorted(p for p in images_dir.glob("*") if p.suffix in IMG_EXT)
    lbl_paths: List[Path] = []
    for ip in img_paths:
        st = ip.stem.split("__")[0]  # collision stem
        cand = [p for p in labels_raw_dir.glob(ip.stem + ".*") if p.suffix in LBL_EXT]
        if not cand:
            cand = [p for p in labels_raw_dir.glob(st + ".*") if p.suffix in LBL_EXT]
        lbl_paths.append(cand[0] if cand else labels_raw_dir / (ip.stem + ".png"))

    n_img = len(img_paths)
    n_lbl = sum(1 for p in lbl_paths if p.is_file())
    img_sz = stat_sizes(img_paths)
    lbl_sz = stat_sizes([p for p in lbl_paths if p.is_file()])
    paired = n_img == n_lbl and n_img > 0
    uniq_txt = unique_sample_label(lbl_paths[0]) if lbl_paths and lbl_paths[0].is_file() else ""

    img_samples = [str(p) for p in img_paths[:10]]
    lbl_samples = [str(p) for p in lbl_paths[:10] if p.is_file()]

    meta_summary = {
        "split_source": split_source,
        "num_train": len(train_names),
        "num_val": len(val_names),
        "link_counts": meta["link_mode_counts"],
    }

    report_lines = [
        "Stanford360 / Stanford2D3D-Panoramic structure report",
        "======================================================",
        f"1. unzip_path (raw): {raw_root}",
        f"2. processed_path: {proc_root}",
        f"3. image_count: {n_img}",
        f"4. label_count (labels_raw): {n_lbl}",
        f"5. train_count (train.txt lines): {len(train_names)}",
        f"6. val_count (val.txt lines): {len(val_names)}",
        "7. image_path_examples (10):",
        *[f"   - {x}" for x in img_samples],
        "8. label_path_examples (10):",
        *[f"   - {x}" for x in lbl_samples],
        f"9. image_size_distribution: {json.dumps(img_sz, ensure_ascii=False)}",
        f"10. label_size_distribution: {json.dumps(lbl_sz, ensure_ascii=False)}",
        f"11. image_label_one_to_one_paired: {paired}",
        f"12. label_unique_values_sample (first label): {uniq_txt}",
        f"13. split_source: {split_source}",
        f"14. meta.json_summary: {json.dumps(meta_summary, ensure_ascii=False)}",
    ]
    (reports_dir / "standford2d3d_structure.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[ok] processed -> {proc_root}")
    print(f"[ok] report -> {reports_dir / 'standford2d3d_structure.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
