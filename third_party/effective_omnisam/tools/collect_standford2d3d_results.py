#!/usr/bin/env python3
"""Aggregate Stanford360 experiment outputs into CSV/Markdown under logs/Standford2d3d_results/summaries/."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scan_val_ckpts(base: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = base / "logs" / "Standford2d3d_results" / "val_ckpts"
    if not root.is_dir():
        return rows
    for ts_dir in sorted(root.iterdir()):
        if not ts_dir.is_dir():
            continue
        for sub in ("paper_ckpt", "final_ckpt"):
            d = ts_dir / sub
            if not d.is_dir():
                continue
            m = read_json(d / "metrics.json")
            t = read_json(d / "timing.json")
            ckpt_txt = d / "checkpoint.txt"
            ckpt = ckpt_txt.read_text(encoding="utf-8").strip() if ckpt_txt.is_file() else ""
            rows.append(
                {
                    "Dataset": "Stanford360_SPAN8",
                    "Eval_Set": "val",
                    "Method": sub,
                    "Checkpoint": ckpt,
                    "mIoU": m.get("mIoU"),
                    "mAcc": m.get("mAcc"),
                    "PixAcc": m.get("PixAcc"),
                    "FWIoU": m.get("FWIoU"),
                    "mDice": m.get("mDice"),
                    "Eval_Time_sec": t.get("eval_time_sec"),
                    "Images_per_sec": t.get("images_per_sec"),
                    "Timestamp": ts_dir.name,
                    "Notes": "",
                }
            )
    return rows


def scan_train_time(base: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = base / "logs" / "Standford2d3d_results" / "train_time_reuse_pl"
    if not root.is_dir():
        return rows
    for ts_dir in sorted(root.iterdir()):
        if not ts_dir.is_dir():
            continue
        base_row: Optional[Dict[str, Any]] = None
        final_row: Optional[Dict[str, Any]] = None
        for name, key in (("base", "base"), ("final_optimized", "final_optimized")):
            d = ts_dir / name
            if not d.is_dir():
                continue
            t = read_json(d / "timing.json")
            pl_txt = d / "pseudo_label_root.txt"
            pl = pl_txt.read_text(encoding="utf-8").strip() if pl_txt.is_file() else ""
            row = {
                "Dataset": "Stanford360_SPAN8",
                "Method": key,
                "Pseudo_Label_Reuse": True,
                "Pseudo_Label_Root": pl,
                "Train_Time_sec": t.get("train_time_sec"),
                "Avg_Iter_Time_sec": t.get("avg_iter_time_sec"),
                "Images_per_sec": t.get("images_per_sec"),
                "Eval_mIoU": t.get("mIoU"),
                "Eval_mAcc": t.get("mAcc"),
                "Eval_PixAcc": t.get("PixAcc"),
                "Delta_mIoU": "",
                "Speedup_train_pct": "",
                "Speedup_iter_pct": "",
                "Timestamp": ts_dir.name,
                "Notes": "",
            }
            if key == "base":
                base_row = row
            else:
                final_row = row
            rows.append(row)

        if base_row and final_row:
            bt = float(base_row.get("Train_Time_sec") or 0)
            ft = float(final_row.get("Train_Time_sec") or 0)
            if bt > 0:
                base_row["Speedup_train_pct"] = round((bt - ft) / bt * 100.0, 4)
                final_row["Speedup_train_pct"] = base_row["Speedup_train_pct"]
            bi = float(base_row.get("Avg_Iter_Time_sec") or 0)
            fi = float(final_row.get("Avg_Iter_Time_sec") or 0)
            if bi > 0:
                sp = round((bi - fi) / bi * 100.0, 4)
                base_row["Speedup_iter_pct"] = sp
                final_row["Speedup_iter_pct"] = sp
            bm = base_row.get("Eval_mIoU")
            fm = final_row.get("Eval_mIoU")
            if bm is not None and fm is not None:
                dmi = round(float(fm) - float(bm), 4)
                base_row["Delta_mIoU"] = dmi
                final_row["Delta_mIoU"] = dmi
    return rows


def write_csv(path: Path, headers: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def try_write_xlsx(path: Path, headers: List[str], rows: List[Dict[str, Any]]) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False
    try:
        df = pd.DataFrame(rows, columns=headers)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=False)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default="")
    args = ap.parse_args()
    base = Path(args.repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).resolve()
    summ = base / "logs" / "Standford2d3d_results" / "summaries"
    summ.mkdir(parents=True, exist_ok=True)

    val_rows = scan_val_ckpts(base)
    train_rows = scan_train_time(base)

    vh = [
        "Dataset",
        "Eval_Set",
        "Method",
        "Checkpoint",
        "mIoU",
        "mAcc",
        "PixAcc",
        "FWIoU",
        "mDice",
        "Eval_Time_sec",
        "Images_per_sec",
        "Timestamp",
        "Notes",
    ]
    th = [
        "Dataset",
        "Method",
        "Pseudo_Label_Reuse",
        "Pseudo_Label_Root",
        "Train_Time_sec",
        "Avg_Iter_Time_sec",
        "Images_per_sec",
        "Eval_mIoU",
        "Eval_mAcc",
        "Eval_PixAcc",
        "Delta_mIoU",
        "Speedup_train_pct",
        "Speedup_iter_pct",
        "Timestamp",
        "Notes",
    ]

    write_csv(summ / "standford2d3d_val_ckpts.csv", vh, val_rows)
    write_csv(summ / "standford2d3d_train_time.csv", th, train_rows)
    try_write_xlsx(summ / "standford2d3d_val_ckpts.xlsx", vh, val_rows)
    try_write_xlsx(summ / "standford2d3d_train_time.xlsx", th, train_rows)

    md = summ / "standford2d3d_summary.md"
    zh = summ / "standford2d3d_summary_text.md"
    md.write_text(
        "# Stanford360 results summary\n\n## Table 1: val checkpoints\n\n"
        + json.dumps(val_rows, indent=2, ensure_ascii=False)
        + "\n\n## Table 2: train time\n\n"
        + json.dumps(train_rows, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    zh.write_text(
        "# Stanford360 / Stanford2D3D-Panoramic 补充实验（中文摘要）\n\n"
        "1. **设置**：室内全景语义分割，OmniSAM SPin8→SPan8 口径；数据位于 `data_extra/processed/Stanford360_SPAN8`。\n"
        "2. **验证**：对比原论文 indoor checkpoint 与优化后 indoor checkpoint 的 val 指标（需 8 类 head）。\n"
        "3. **训练时间**：在复用同一 Stanford360/SPan8 伪标签目录下，对比 base 与 final optimized 配置。\n"
        "4. **限制**：不能使用 Cityscapes13/DensePASS13/WildPASS13 的 13 类 checkpoint 直接评估 8 类 Stanford360。\n"
        "5. **缺失项**：若缺少 SPin8 source 根目录或 Stanford SPan8 pseudo labels，请在对应 MISSING_*.md 中补齐路径。\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote summaries -> {summ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
