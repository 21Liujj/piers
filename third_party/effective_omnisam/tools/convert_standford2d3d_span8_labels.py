#!/usr/bin/env python3
"""Convert Stanford360 labels_raw to SPan8 trainId (0-7,255) using repo OmniSAM mapping."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

_MAP13_TO_8 = np.array([255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 5, 6, 7], dtype=np.int32)

# colors.npy row order matches name2label coarse classes (see dataset/s2d3d_pin_list/colors.npy).
KEYS14 = [
    "<UNK>",
    "beam",
    "board",
    "bookcase",
    "ceiling",
    "chair",
    "clutter",
    "column",
    "door",
    "floor",
    "sofa",
    "table",
    "wall",
    "window",
]


def _scan_repo_for_mapping_hints(repo: Path) -> List[str]:
    hints: List[str] = []
    for p in (repo / "dataset" / "StanfordPin.py", repo / "dataset" / "StanfordPan.py"):
        if p.is_file():
            hints.append(str(p))
    d = repo / "dataset" / "s2d3d_pin_list"
    if d.is_dir():
        hints.extend(str(x) for x in sorted(d.glob("*")))
    return hints


def load_semantic_tables(repo: Path) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
    import json as js

    sem_json = repo / "dataset" / "s2d3d_pin_list" / "semantic_labels.json"
    name_json = repo / "dataset" / "s2d3d_pin_list" / "name2label.json"
    colors_np = np.load(str(repo / "dataset" / "s2d3d_pin_list" / "colors.npy"))
    with open(sem_json, encoding="utf-8") as f:
        id2name = [name.split("_")[0] for name in js.load(f)] + ["<UNK>"]
    with open(name_json, encoding="utf-8") as f:
        name2id = js.load(f)
    id2label = np.array([name2id[name] for name in id2name], dtype=np.uint8)
    return id2label, name2id, colors_np


def rgb_semantic_to_intermediate(
    sem_rgb: np.ndarray,
    img_rgb: Optional[np.ndarray],
    id2label: np.ndarray,
    colors_np: np.ndarray,
    name2id: Dict[str, int],
) -> np.ndarray:
    """
    StanfordPin-style RGB semantic -> raw category id, then sem-1.
    If palette index exceeds legacy id2label table length, fall back to nearest colors.npy row.
    """
    sem = np.array(sem_rgb, dtype=np.int32)
    if img_rgb is not None:
        rgb = np.array(img_rgb, dtype=np.int32)
    else:
        rgb = np.zeros((*sem.shape[:2], 3), dtype=np.int32)
    h, w, _ = sem.shape
    idx = sem[..., 1] * 256 + sem[..., 2]
    valid = idx < len(id2label)
    raw = np.zeros((h, w), dtype=np.int32)
    raw[valid] = id2label[idx[valid]].astype(np.int32)
    id_vec = np.array([name2id[k] for k in KEYS14], dtype=np.int32)
    cc = colors_np.astype(np.float32)
    inv = ~valid
    if np.any(inv):
        pix = sem[..., :3][inv].astype(np.float32)
        dist = ((pix[:, None, :] - cc[None, :, :]) ** 2).sum(axis=2)
        r_inv = np.argmin(dist, axis=1)
        raw[inv] = id_vec[r_inv]
    unk = sem[..., 0] != 0
    raw = raw.astype(np.int32)
    raw[unk] = 0
    raw[rgb.sum(-1) == 0] = 0
    raw = raw - 1
    return raw


def map13to8(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int32).copy()
    for v in np.unique(m):
        vi = int(v)
        if vi == 255:
            continue
        if vi < 0 or vi >= len(_MAP13_TO_8):
            m[m == vi] = 255
        else:
            m[m == vi] = int(_MAP13_TO_8[vi])
    return m


def convert_one(
    path: Path,
    out_path: Path,
    paired_image: Optional[Path],
    id2label: np.ndarray,
    colors_np: np.ndarray,
    name2id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path))
        if arr.ndim == 3:
            arr = arr[0] if arr.shape[0] == 1 else arr[..., 0]
    else:
        im = Image.open(path)
        arr = np.array(im)

    before_u = np.unique(arr)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        img_rgb = None
        if paired_image and paired_image.is_file():
            img_rgb = np.array(Image.open(paired_image).convert("RGB"))
        elif path.suffix.lower() != ".npy":
            img_rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.int32)
        sem_int = rgb_semantic_to_intermediate(
            arr[:, :, :3], img_rgb, id2label, colors_np, name2id
        )
        after = map13to8(sem_int)
    elif arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        m = arr.astype(np.int32)
        if arr.ndim == 3:
            m = m[:, :, 0]
        u = set(int(x) for x in np.unique(m).tolist())
        allowed8 = set(range(8)) | {255}
        allowed13 = set(range(13)) | {255}
        if u.issubset(allowed8):
            after = m.copy()
            after[(after > 7) & (after != 255)] = 255
        elif u.issubset(allowed13):
            after = map13to8(m)
        else:
            raise ValueError(f"unrecognized_label_values:{sorted(list(u))[:40]}")
    else:
        raise ValueError(f"unsupported_label_shape:{arr.shape}")

    after_u = np.unique(after.astype(np.int64))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(after.astype(np.uint8)).save(out_path)
    return before_u.astype(np.int64), after_u.astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", type=str, required=True)
    ap.add_argument("--log_root", type=str, required=True)
    ap.add_argument("--repo_root", type=str, default=str(REPO_ROOT))
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    proc = Path(args.processed_root).resolve()
    log_root = Path(args.log_root).resolve()
    reports = log_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    raw_dir = proc / "labels_raw"
    out_dir = proc / "labels_trainid"
    img_dir = proc / "images"

    need_map = reports / "NEED_CONFIRM_STANFORD360_MAPPING.md"
    mapping_json = reports / "standford2d3d_label_mapping.json"
    unique_json = reports / "standford2d3d_label_unique_before_after.json"

    hints = _scan_repo_for_mapping_hints(repo)
    id2label, name2id, colors_np = load_semantic_tables(repo)

    if not raw_dir.is_dir():
        need_map.write_text(f"labels_raw not found: {raw_dir}\n", encoding="utf-8")
        return 2

    class_names = [
        "ceiling",
        "chair",
        "door",
        "floor",
        "sofa",
        "table",
        "wall",
        "window",
    ]

    before_all: Set[int] = set()
    after_all: Set[int] = set()
    per_file: List[Dict[str, Any]] = []
    errors: List[str] = []

    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".tif", ".tiff", ".npy"}:
            continue
        rel = p.relative_to(raw_dir)
        out_p = out_dir / rel.with_suffix(".png")
        stem = p.stem.split("__")[0]
        paired = None
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            cand = img_dir / (p.stem + ext)
            if cand.is_file():
                paired = cand
                break
        if paired is None:
            for ext in (".png", ".jpg", ".jpeg"):
                cand = img_dir / (stem + ext)
                if cand.is_file():
                    paired = cand
                    break

        try:
            bu, au = convert_one(p, out_p, paired, id2label, colors_np, name2id)
            before_all.update(int(x) for x in bu.tolist())
            after_all.update(int(x) for x in au.tolist())
            per_file.append({"file": str(rel), "before": bu.tolist()[:64], "after": au.tolist()[:64]})
        except Exception as e:
            errors.append(f"{rel}: {e}")

    ok_after = after_all.issubset(set(range(8)) | {255}) and len(errors) == 0

    payload_map = {
        "sources_searched": hints,
        "map13_to_8": _MAP13_TO_8.tolist(),
        "ignore_index": 255,
        "class_names_span8": class_names,
        "semantic_label_table_len": int(len(id2label)),
        "notes": "RGB decode: legacy id2label index when in-range; else NN fallback to dataset/s2d3d_pin_list/colors.npy rows "
        "(aligned with KEYS14 / name2label). Then _map13_to_8.",
    }
    mapping_json.write_text(json.dumps(payload_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    unique_payload = {
        "unique_before_union": sorted(before_all),
        "unique_after_union": sorted(after_all),
        "per_file_sample": per_file[:50],
        "errors": errors,
    }
    unique_json.write_text(json.dumps(unique_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if errors or not ok_after:
        tbl = "\n".join(f"| {i} | ? |" for i in sorted(before_all))
        need_map.write_text(
            "# NEED_CONFIRM_STANFORD360_MAPPING\n\n"
            f"- Raw union unique values: {sorted(before_all)}\n"
            f"- After attempt union: {sorted(after_all)}\n"
            f"- Errors:\n"
            + "\n".join(errors[:50])
            + "\n\n## Candidate mapping files in repo\n"
            + "\n".join(f"- `{h}`" for h in hints[:40])
            + "\n\n## Recommended 8-class names (OmniSAM Stanford2D3D)\n"
            + "\n".join(f"{i}: {n}" for i, n in enumerate(class_names))
            + "\n\n## Please confirm raw_id -> trainId\n\n"
            + "| raw_id | trainId |\n| --- | --- |\n"
            + tbl
            + "\n",
            encoding="utf-8",
        )
        print(f"[error] wrote {need_map}", file=sys.stderr)
        return 2

    print(f"[ok] labels_trainid -> {out_dir}")
    print(f"[ok] reports -> {mapping_json}, {unique_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
