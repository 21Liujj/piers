#!/usr/bin/env python3
"""Aggregate WildPASS experiment outputs into CSV/MD under logs/wildpass_results/summaries/."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scan_val_ckpts(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = root / "logs" / "wildpass_results" / "val_ckpts"
    if not base.is_dir():
        return rows
    for ts_dir in sorted(base.iterdir()):
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
                    "Dataset": "WildPASS13",
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


def scan_train_time(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = root / "logs" / "wildpass_results" / "train_time_reuse_pl"
    if not base.is_dir():
        return rows
    for ts_dir in sorted(base.iterdir()):
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
                "Dataset": "WildPASS13",
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
                "Notes": "val_dataset=" + str(t.get("val_dataset", "")),
            }
            if key == "base":
                base_row = row
            else:
                final_row = row
            rows.append(row)

        if base_row and final_row:
            bt = float(base_row.get("Train_Time_sec") or 0)
            ft = float(final_row.get("Train_Time_sec") or 0)
            bi = float(base_row.get("Avg_Iter_Time_sec") or 0)
            fi = float(final_row.get("Avg_Iter_Time_sec") or 0)
            bm = base_row.get("Eval_mIoU")
            fm = final_row.get("Eval_mIoU")
            try:
                bm_f = float(bm) if bm is not None else float("nan")
                fm_f = float(fm) if fm is not None else float("nan")
                delta = fm_f - bm_f
            except (TypeError, ValueError):
                delta = float("nan")
            speedup = ((bt - ft) / bt * 100.0) if bt > 0 else float("nan")
            iter_speedup = ((bi - fi) / bi * 100.0) if bi > 0 else float("nan")
            rows.append(
                {
                    "Dataset": "WildPASS13",
                    "Method": "comparison_same_timestamp",
                    "Pseudo_Label_Reuse": True,
                    "Pseudo_Label_Root": base_row.get("Pseudo_Label_Root"),
                    "Train_Time_sec": "",
                    "Avg_Iter_Time_sec": "",
                    "Images_per_sec": "",
                    "Eval_mIoU": "",
                    "Eval_mAcc": "",
                    "Eval_PixAcc": "",
                    "Delta_mIoU": delta,
                    "Speedup_train_pct": speedup,
                    "Speedup_iter_pct": iter_speedup,
                    "Timestamp": ts_dir.name,
                    "Notes": "ΔmIoU final-base; speedup_train=(base_train-final_train)/base_train",
                }
            )
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_md(
    out_md: Path,
    val_rows: List[Dict[str, Any]],
    train_rows: List[Dict[str, Any]],
) -> None:
    lines = [
        "# WildPASS results summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Table 1 — Checkpoint validation (WildPASS13)",
        "",
        "| Dataset | Eval Set | Method | mIoU | mAcc | PixAcc | Eval Time | Img/s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in val_rows:
        if r.get("Method") in ("paper_ckpt", "final_ckpt"):
            lines.append(
                f"| {r.get('Dataset')} | {r.get('Eval_Set')} | {r.get('Method')} | "
                f"{r.get('mIoU')} | {r.get('mAcc')} | {r.get('PixAcc')} | "
                f"{r.get('Eval_Time_sec')} | {r.get('Images_per_sec')} |"
            )
    lines.extend(["", "## Table 2 — Train time (reuse PL)", ""])
    lines.append("| Method | Train s | iter s | img/s | val mIoU | Notes |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in train_rows:
        if str(r.get("Method")).endswith("optimized") or r.get("Method") == "base":
            lines.append(
                f"| {r.get('Method')} | {r.get('Train_Time_sec')} | {r.get('Avg_Iter_Time_sec')} | "
                f"{r.get('Images_per_sec')} | {r.get('Eval_mIoU')} | {r.get('Notes')} |"
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_text_cn(out_path: Path) -> None:
    text = """# WildPASS 补充实验 — 中文简述（组会 / 论文素材）

## 1. WildPASS 补充验证集设置

本仓库将 WildPASS 处理为 `WildPASS13`（13 类 outdoor、ignore=255），验证流程复用 `validation.py` 与全景滑窗逻辑。

## 2. 默认 checkpoint（WildPASS val 对比）

- **base / paper**：\`<repo>/exp2/Cityscapes/best_sam2_s_10.pth\`
- **final（DensePASS val ~60.26）**：\`<repo>/data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth\`

路径集中在 \`scripts/extra_exp/wildpass/wildpass_default_ckpts.inc.sh\`，\`01_val_ckpts_wildpass.sh\` 无参数时即用上述两支。

**加速分工**：WildPASS **独立 val** 默认 **不开** memory/sparse（\`MEMORY_ACCEL=off\`）；**UDA 训练**侧加速用 \`wildpass_train_final_optimized.yaml\`，无加速对照用 \`wildpass_train_base.yaml\`（\`02_train_time_wildpass_reuse_pl.sh\`）。

## 3. 原论文 / Original checkpoint vs 最终优化 checkpoint

对比指标见 \`wildpass_val_ckpts.csv\`（mIoU、mAcc、PixAcc、FWIoU、mDice 等）。

## 4. 复用相同伪标签下的训练时间对比

在 `--reuse_pl_dir` 固定同一套伪标签、关闭伪标签再生成的前提下，对比 base 配置与 final optimized 配置的训练耗时与吞吐。

训练日志中的验证集仍为 **DensePASS**（与当前 `Engine` / `DatasetBuilder` 的 UDA 默认一致）；请在 `timing.json` 中查看 `val_dataset` 字段。

## 5. 速度收益与精度变化

见汇总表中的 Speedup（相对训练时间与迭代时间）以及 ΔmIoU（若两侧在同一时间戳目录下可自动估算）。

## 6. 标签与数据范围说明

正式实验约定：**伪标签 / 标签转换在 WildPASS2K**（与 `dataset/densepass_list/train.txt` 的 2000 行列表一致）上完成；
**WildPASS 全量 val（500）** 上仅做两支 checkpoint（原始 vs 约 60.26 mIoU 的最终权重）的 **独立验证**，不在此流程内做 WildPASS 全量 UDA 训练。

若你本地实际只做部分步骤，请在组会材料中注明与上述约定的差异。

"""
    out_path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default="")
    args = ap.parse_args()
    root = Path(args.repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).resolve()
    summ = root / "logs" / "wildpass_results" / "summaries"
    summ.mkdir(parents=True, exist_ok=True)

    val_rows = scan_val_ckpts(root)
    train_rows = scan_train_time(root)

    write_csv(
        summ / "wildpass_val_ckpts.csv",
        [
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
        ],
        val_rows,
    )

    write_csv(
        summ / "wildpass_train_time.csv",
        [
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
        ],
        train_rows,
    )

    try:
        import pandas as pd

        if val_rows:
            pd.DataFrame(val_rows).to_excel(summ / "wildpass_val_ckpts.xlsx", index=False)
        if train_rows:
            pd.DataFrame(train_rows).to_excel(summ / "wildpass_train_time.xlsx", index=False)
    except Exception as e:
        (summ / "xlsx_note.txt").write_text(f"pandas/openpyxl xlsx export skipped: {e}\n", encoding="utf-8")

    write_summary_md(summ / "wildpass_summary.md", val_rows, train_rows)
    write_summary_text_cn(summ / "wildpass_summary_text.md")

    print(f"[collect] wrote -> {summ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
