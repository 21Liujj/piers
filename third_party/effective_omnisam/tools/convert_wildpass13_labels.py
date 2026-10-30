#!/usr/bin/env python3
"""
将 WildPASS labels_raw 转为 Cityscapes-13 trainIds（0..12, 255 ignore）。

v2：按图推断 raw 为 Cityscapes labelId 还是 19-class trainId（见 utils.wildpass_label_convert），
避免「值 ⊆ {0..18}」时整条误走 trainId 分支导致 road/sky 等错位。

输出对比报告：logs/wildpass_results/reports/wildpass_label_conversion_fix_report.json
（及可选 wildpass_label_conversion_fix_report.md）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.WildPASS import WildPASS13  # noqa: E402
from utils.label_spaces import CITYSCAPES13_CLASSES, CITYSCAPES19_CLASSES  # noqa: E402
from utils.wildpass_label_convert import convert_wildpass_raw_to_train13  # noqa: E402


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


def export_mapping_json(out_path: Path) -> None:
    inst = WildPASS13.__new__(WildPASS13)
    inst._key = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 255, 12, 12])
    mapping_lines = []
    for tid in range(19):
        mapped = int(inst._key[tid])
        mapping_lines.append(
            {
                "cityscapes_train_id": tid,
                "cityscapes_train_name": CITYSCAPES19_CLASSES[tid],
                "mapped_train_id_13": mapped,
                "mapped_name_13": CITYSCAPES13_CLASSES[mapped] if mapped != 255 else "ignore",
            }
        )
    payload = {
        "ignore_index": 255,
        "description": "Map 19-class Cityscapes TrainIds to legacy 13-class space (same as WildPASS13._key).",
        "classes_13": CITYSCAPES13_CLASSES,
        "train_id_19_to_13": mapping_lines,
        "note": "Raw labelId -> trainId 使用 torchvision.datasets.Cityscapes 与 WildPASS13._cityscapes_label_id_to_train_id 一致。",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_label_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(str(path)).astype(np.int32)
    return np.array(Image.open(path)).astype(np.int32)


def _accumulate_hist(h: Counter, arr: np.ndarray) -> None:
    u, c = np.unique(arr.astype(np.int64).ravel(), return_counts=True)
    for v, n in zip(u.tolist(), c.tolist()):
        h[int(v)] += int(n)


def _hist_to_ratio(h: Counter, total: int) -> Dict[str, float]:
    if total <= 0:
        return {}
    return {str(k): round(v / total, 8) for k, v in sorted(h.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=str, required=True, help="WildPASS13 processed root")
    ap.add_argument("--repo-root", type=str, default="")
    ap.add_argument("--reports-dir", type=str, default="logs/wildpass_results/reports")
    ap.add_argument("--list-file", type=str, default="lists/val.txt", help="相对 processed-root 的列表")
    ap.add_argument("--backup", action="store_true", help="覆盖前备份整个 labels_trainid 目录")
    ap.add_argument("--dry-run", action="store_true", help="不写盘，只写报告（若可统计）")
    args = ap.parse_args()

    repo = Path(args.repo_root or REPO_ROOT).resolve()
    proc = (repo / args.processed_root).resolve() if not os.path.isabs(args.processed_root) else Path(args.processed_root)
    reports = (repo / args.reports_dir).resolve()
    labels_raw = proc / "labels_raw"
    labels_out = proc / "labels_trainid"
    list_path = proc / args.list_file
    need_confirm = reports / "NEED_CONFIRM_WILDPASS_MAPPING.md"
    mapping_json = reports / "wildpass_label_mapping.json"
    unique_json = reports / "wildpass_label_unique_before_after_v2.json"
    fix_report_json = reports / "wildpass_label_conversion_fix_report.json"
    fix_report_md = reports / "wildpass_label_conversion_fix_report.md"

    export_mapping_json(mapping_json)

    if not labels_raw.is_dir():
        reports.mkdir(parents=True, exist_ok=True)
        need_confirm.write_text(f"# NEED_CONFIRM_WILDPASS_MAPPING\n\nMissing labels_raw: `{labels_raw}`\n", encoding="utf-8")
        print(f"[error] {need_confirm}", file=sys.stderr)
        return 2
    if not list_path.is_file():
        need_confirm.write_text(f"# NEED_CONFIRM_WILDPASS_MAPPING\n\nMissing list: `{list_path}`\n", encoding="utf-8")
        return 2

    names = [ln.strip() for ln in list_path.open(encoding="utf-8") if ln.strip()]
    hist_before: Counter = Counter()
    hist_after: Counter = Counter()
    enc_counts: Counter = Counter()
    per_image: List[Dict[str, Any]] = []
    ambiguous = False

    if not args.dry_run and args.backup and labels_out.is_dir():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = proc / f"labels_trainid.backup_{ts}"
        shutil.copytree(labels_out, bak)
        print(f"[backup] {labels_out} -> {bak}")

    for name in names:
        src = labels_raw / name
        if not src.is_file():
            stem, _ = os.path.splitext(name)
            alt = labels_raw / (stem + ".png")
            src = alt if alt.is_file() else src
        if not src.is_file():
            ambiguous = True
            continue

        raw = _load_label_array(src)
        uniq_b = np.unique(raw).tolist()[:80]

        prev_path = labels_out / Path(name).with_suffix(".png").name
        if prev_path.is_file():
            try:
                prev = np.array(Image.open(prev_path)).astype(np.int64)
                _accumulate_hist(hist_before, prev)
            except OSError:
                pass

        try:
            out, enc = convert_wildpass_raw_to_train13(raw)
        except Exception as e:
            need_confirm.parent.mkdir(parents=True, exist_ok=True)
            need_confirm.write_text(f"# NEED_CONFIRM_WILDPASS_MAPPING\n\nFailed on `{src}`: {e}\n", encoding="utf-8")
            print(f"[error] {e}", file=sys.stderr)
            return 2

        uniq_a = np.unique(out).tolist()[:80]
        enc_counts[enc] += 1
        _accumulate_hist(hist_after, out)
        if len(per_image) < 200:
            per_image.append(
                {
                    "sample_id": name,
                    "encoding": enc,
                    "unique_raw": uniq_b,
                    "unique_train13": uniq_a,
                }
            )

        if not args.dry_run:
            out_path = labels_out / Path(name).with_suffix(".png").name
            labels_out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out.astype(np.uint8)).save(out_path)

    total_b = sum(hist_before.values())
    total_a = sum(hist_after.values())

    report = {
        "processed_root": str(proc),
        "list_file": str(list_path),
        "num_list_entries": len(names),
        "encoding_counts": dict(enc_counts),
        "histogram_train13_before_on_list": {str(k): hist_before[k] for k in sorted(hist_before.keys())},
        "histogram_train13_after_on_list": {str(k): hist_after[k] for k in sorted(hist_after.keys())},
        "ratio_train13_before": _hist_to_ratio(hist_before, total_b),
        "ratio_train13_after": _hist_to_ratio(hist_after, total_a),
        "total_pixels_before": int(total_b),
        "total_pixels_after": int(total_a),
        "per_image_sample": per_image,
        "classes_13_names": CITYSCAPES13_CLASSES,
        "dry_run": bool(args.dry_run),
    }

    reports.mkdir(parents=True, exist_ok=True)
    unique_json.write_text(json.dumps(per_image[:30], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fix_report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# WildPASS labels_trainid 转换修复报告（v2：labelId / trainId 按图推断）",
        "",
        f"- processed_root: `{proc}`",
        f"- 列表: `{list_path}`",
        f"- dry_run: `{args.dry_run}`",
        "",
        "## 每张图选用的编码分支（encoding_counts）",
        "",
        json.dumps(dict(enc_counts), indent=2, ensure_ascii=False),
        "",
        "## 13 类像素占比（列表内样本，修复前 = 当前 labels_trainid 若存在）",
        "",
        "| label | name | ratio_before | ratio_after |",
        "| ---: | --- | ---: | ---: |",
    ]
    for i in range(13):
        rb = hist_before.get(i, 0) / total_b if total_b else 0.0
        ra = hist_after.get(i, 0) / total_a if total_a else 0.0
        lines.append(f"| {i} | {CITYSCAPES13_CLASSES[i]} | {rb:.6f} | {ra:.6f} |")
    lines.append(f"| 255 | ignore | {hist_before.get(255,0)/total_b if total_b else 0:.6f} | {hist_after.get(255,0)/total_a if total_a else 0:.6f} |")
    fix_report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if ambiguous:
        need_confirm.write_text(
            "# NEED_CONFIRM_WILDPASS_MAPPING\n\nSome list entries missing under labels_raw.\n", encoding="utf-8"
        )
        return 2

    if args.dry_run:
        print(f"[dry-run] report -> {fix_report_json}")
        return 0

    if need_confirm.exists():
        need_confirm.unlink()

    print(f"[ok] labels_trainid -> {labels_out}")
    print(f"[ok] report -> {fix_report_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
