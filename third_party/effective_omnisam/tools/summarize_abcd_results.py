#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SETTING_DEFAULTS = {
    "A": {
        "use_text_proto": False,
        "use_text_gate_ssl": False,
        "ssl_gate_mode": "hard",
        "use_language_adapter": False,
    },
    "B": {
        "use_text_proto": True,
        "use_text_gate_ssl": False,
        "ssl_gate_mode": "hard",
        "use_language_adapter": False,
    },
    "C": {
        "use_text_proto": True,
        "use_text_gate_ssl": True,
        "ssl_gate_mode": "hard",
        "use_language_adapter": False,
    },
    "D": {
        "use_text_proto": True,
        "use_text_gate_ssl": True,
        "ssl_gate_mode": "hard",
        "use_language_adapter": True,
    },
}


@dataclass
class Row:
    exp_name: str
    setting: str
    use_text_proto: str
    use_text_gate_ssl: str
    ssl_gate_mode: str
    use_language_adapter: str
    visual_mask_ratio: str
    text_mask_ratio: str
    text_sim_mean: str
    text_sim_std: str
    unsup_kept_ratio: str
    val_mIoU: str
    best_mIoU: str
    training_time: str
    validation_time: str
    per_class_iou: str
    delta_mIoU: str
    delta_text_mask_ratio: str
    notes: str
    log_file: str


def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def _fmt(v: Optional[float]) -> str:
    return "NA" if v is None else f"{v:.4f}"


def _extract_setting_and_exp(log_name: str) -> Tuple[str, str]:
    # Example: multimodal_abcd_A_baseline.log
    m = re.match(r"multimodal_(.+?)_([A-D])(?:_|\.log)", log_name)
    if m:
        return m.group(2), m.group(1)

    # Fallback: infer from filename letters
    m2 = re.search(r"(^|[_-])([A-D])([_-]|$)", log_name)
    setting = m2.group(2) if m2 else "NA"
    exp_name = log_name.replace(".log", "")
    return setting, exp_name


def _parse_epoch_metrics(text: str) -> Tuple[Optional[float], Optional[float]]:
    # Epoch n: val mIoU=60.31, ... best=60.36
    matches = re.findall(r"Epoch\s+\d+:\s+val mIoU=([0-9.]+).*?best=([0-9.]+)", text)
    if not matches:
        return None, None
    val_s, best_s = matches[-1]
    return _safe_float(val_s), _safe_float(best_s)


def _parse_last_metric_from_key(text: str, key: str) -> Optional[float]:
    # key could appear like "visual_mask_ratio': 0.123" or "visual_mask_ratio=0.123"
    patterns = [
        rf"{re.escape(key)}['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        rf"{re.escape(key)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    out = None
    for p in patterns:
        ms = re.findall(p, text)
        if ms:
            out = _safe_float(ms[-1])
    return out


def _parse_per_class_iou(text: str) -> str:
    # trainer.validate output: "road           : 80.74"
    pairs = re.findall(r"^\s*([A-Za-z _-]+?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text, flags=re.MULTILINE)
    if not pairs:
        # alternative style: "===> class : IoU xx"
        pairs2 = re.findall(r"===>\s*([A-Za-z _-]+?)\s*:\s*IoU\s*([0-9]+(?:\.[0-9]+)?)", text)
        pairs = pairs2
    if not pairs:
        return "NA"
    # keep latest class block by class uniqueness
    d: Dict[str, str] = {}
    for k, v in pairs:
        d[k.strip()] = v.strip()
    return "; ".join([f"{k}:{v}" for k, v in sorted(d.items())])


def _detect_time_fields(text: str) -> Tuple[str, str]:
    # Best-effort placeholders; fill NA if no explicit time logs.
    train_t = None
    val_t = None
    m_train = re.findall(r"Train(?:ing)?\s*time\s*[:=]\s*([0-9:.]+)", text, flags=re.IGNORECASE)
    m_val = re.findall(r"Val(?:idation)?\s*time\s*[:=]\s*([0-9:.]+)", text, flags=re.IGNORECASE)
    if m_train:
        train_t = m_train[-1]
    if m_val:
        val_t = m_val[-1]
    return train_t or "NA", val_t or "NA"


def parse_log_file(path: Path) -> Row:
    text = path.read_text(encoding="utf-8", errors="ignore")
    setting, exp_name = _extract_setting_and_exp(path.name)

    default_cfg = SETTING_DEFAULTS.get(setting, {})
    use_text_proto = str(default_cfg.get("use_text_proto", "NA"))
    use_text_gate_ssl = str(default_cfg.get("use_text_gate_ssl", "NA"))
    ssl_gate_mode = str(default_cfg.get("ssl_gate_mode", "NA"))
    use_language_adapter = str(default_cfg.get("use_language_adapter", "NA"))

    val_miou, best_miou = _parse_epoch_metrics(text)
    visual_mask_ratio = _parse_last_metric_from_key(text, "visual_mask_ratio")
    text_mask_ratio = _parse_last_metric_from_key(text, "text_mask_ratio")
    text_sim_mean = _parse_last_metric_from_key(text, "text_sim_mean")
    text_sim_std = _parse_last_metric_from_key(text, "text_sim_std")
    unsup_kept_ratio = _parse_last_metric_from_key(text, "unsup_kept_ratio")
    train_t, val_t = _detect_time_fields(text)
    per_cls = _parse_per_class_iou(text)

    return Row(
        exp_name=exp_name,
        setting=setting,
        use_text_proto=use_text_proto,
        use_text_gate_ssl=use_text_gate_ssl,
        ssl_gate_mode=ssl_gate_mode,
        use_language_adapter=use_language_adapter,
        visual_mask_ratio=_fmt(visual_mask_ratio),
        text_mask_ratio=_fmt(text_mask_ratio),
        text_sim_mean=_fmt(text_sim_mean),
        text_sim_std=_fmt(text_sim_std),
        unsup_kept_ratio=_fmt(unsup_kept_ratio),
        val_mIoU=_fmt(val_miou),
        best_mIoU=_fmt(best_miou),
        training_time=train_t,
        validation_time=val_t,
        per_class_iou=per_cls,
        delta_mIoU="NA",
        delta_text_mask_ratio="NA",
        notes="",
        log_file=str(path),
    )


def _rows_to_dicts(rows: List[Row]) -> List[dict]:
    return [r.__dict__.copy() for r in rows]


def add_deltas(rows: List[Row]) -> None:
    # baseline by exp_name + setting A
    baseline_by_exp: Dict[str, Row] = {}
    for r in rows:
        if r.setting == "A":
            baseline_by_exp[r.exp_name] = r

    for r in rows:
        b = baseline_by_exp.get(r.exp_name)
        if b is None:
            continue
        rv = _safe_float(r.val_mIoU) if r.val_mIoU != "NA" else None
        bv = _safe_float(b.val_mIoU) if b.val_mIoU != "NA" else None
        if rv is not None and bv is not None:
            r.delta_mIoU = _fmt(rv - bv)

        rt = _safe_float(r.text_mask_ratio) if r.text_mask_ratio != "NA" else None
        bt = _safe_float(b.text_mask_ratio) if b.text_mask_ratio != "NA" else None
        if rt is not None and bt is not None:
            r.delta_text_mask_ratio = _fmt(rt - bt)


def write_csv(rows: List[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dict_rows = _rows_to_dicts(rows)
    fieldnames = list(dict_rows[0].keys()) if dict_rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in dict_rows:
            w.writerow(row)


def write_md(rows: List[Row], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "exp_name",
        "setting",
        "use_text_proto",
        "use_text_gate_ssl",
        "ssl_gate_mode",
        "use_language_adapter",
        "visual_mask_ratio",
        "text_mask_ratio",
        "text_sim_mean",
        "text_sim_std",
        "unsup_kept_ratio",
        "val_mIoU",
        "best_mIoU",
        "delta_mIoU",
        "delta_text_mask_ratio",
        "training_time",
        "validation_time",
        "notes",
    ]
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# ABCD 实验自动汇总\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals = [getattr(r, h, "NA") for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

        f.write("\n## Per-class IoU（若日志可解析）\n\n")
        for r in rows:
            f.write(f"- `{Path(r.log_file).name}` ({r.setting}): {r.per_class_iou}\n")


def collect_logs(log_dirs: List[Path], pattern: str) -> List[Path]:
    out: List[Path] = []
    for d in log_dirs:
        if not d.exists():
            continue
        out.extend(sorted(d.rglob(pattern)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser("Summarize ABCD experiment logs to CSV/Markdown")
    parser.add_argument(
        "--log_dirs",
        nargs="+",
        default=["./logs"],
        help="One or more directories containing experiment logs.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="multimodal_*_*.log",
        help="Glob pattern for selecting log files.",
    )
    parser.add_argument("--out_csv", type=str, default="./logs/abcd_summary.csv")
    parser.add_argument("--out_md", type=str, default="./logs/abcd_summary.md")
    args = parser.parse_args()

    log_dirs = [Path(p).resolve() for p in args.log_dirs]
    files = collect_logs(log_dirs, args.glob)
    if not files:
        print(f"[WARN] No log files found with pattern '{args.glob}' under {log_dirs}")
        # still create empty outputs
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_csv).write_text("", encoding="utf-8")
        Path(args.out_md).write_text("# ABCD 实验自动汇总\n\n无匹配日志。\n", encoding="utf-8")
        return

    rows = [parse_log_file(p) for p in files]
    # Stable order: exp_name then A/B/C/D
    setting_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    rows.sort(key=lambda r: (r.exp_name, setting_rank.get(r.setting, 99), Path(r.log_file).name))
    add_deltas(rows)

    write_csv(rows, Path(args.out_csv).resolve())
    write_md(rows, Path(args.out_md).resolve())
    print(f"[OK] CSV: {Path(args.out_csv).resolve()}")
    print(f"[OK] MD : {Path(args.out_md).resolve()}")
    print(f"[OK] Rows: {len(rows)}")


if __name__ == "__main__":
    main()
