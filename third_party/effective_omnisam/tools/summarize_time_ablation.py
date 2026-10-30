#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_last_jsonl(path: Path):
    if not path.exists():
        return {}
    last = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    return last


def collect_runs(root: Path):
    runs = []
    for p in root.rglob("metrics"):
        pl = read_last_jsonl(p / "pl_timing.jsonl")
        tr = read_last_jsonl(p / "train_timing.jsonl")
        va = read_last_jsonl(p.parent / "metrics.jsonl")
        if not pl and not tr and not va:
            continue
        runs.append((p.parent, pl, tr, va))
    return runs


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "NA") for k in header})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Ablation root directory")
    ap.add_argument("--out_dir", required=True, help="Output summary directory")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir)
    runs = collect_runs(root)

    pl_header = [
        "Method", "CompressorMode", "CompressLocation", "SchedulerMode", "MemoryBankSize",
        "AvgRawPatchesPerImage", "AvgEffectivePatchesPerImage", "PatchReductionRatio",
        "PLTotalTime", "PLTimePerImage", "PLImagesPerSec", "PeakMemPL", "HighConfRatio", "UncertainRatio",
    ]
    train_header = [
        "Method", "CompressorMode", "CompressLocation", "SchedulerMode", "MemoryBankSize",
        "IterTotalTime", "TrainImagesPerSec", "TrainPatchesPerSec", "EpochTrainTime", "EpochTotalTime", "PeakMemTrain",
    ]
    trade_header = [
        "Method", "mIoU", "Delta_mIoU", "PLSpeedup", "EpochSpeedup", "PeakMemTrain",
        "AvgEffectivePatchesPerImage", "RawMemoryTokens", "CompressedMemoryTokens", "MemoryCompressionRatio",
    ]

    pl_rows, train_rows, trade_rows = [], [], []
    baseline_pl, baseline_epoch, baseline_miou = None, None, None

    for run_dir, pl, tr, va in runs:
        method = run_dir.name
        comp = tr.get("compressor_mode", pl.get("compressor_mode", "none"))
        loc = tr.get("compress_location", pl.get("compress_location", "before_attention"))
        sched = "on" if float(pl.get("patch_reduction_ratio", 0.0) or 0.0) > 0 else "off"
        bank = tr.get("memory_bank_size", pl.get("memory_bank_size", "NA"))

        pl_total = float(pl.get("pl_total_time", 0.0) or 0.0)
        epoch_total = float(tr.get("epoch_total_time", 0.0) or 0.0)
        miou = float(va.get("val_mIoU", 0.0) or 0.0)
        if method == "baseline":
            baseline_pl = pl_total if pl_total > 0 else baseline_pl
            baseline_epoch = epoch_total if epoch_total > 0 else baseline_epoch
            baseline_miou = miou if miou > 0 else baseline_miou

        pl_rows.append({
            "Method": method,
            "CompressorMode": comp,
            "CompressLocation": loc,
            "SchedulerMode": sched,
            "MemoryBankSize": bank,
            "AvgRawPatchesPerImage": pl.get("avg_raw_patches_per_image", "NA"),
            "AvgEffectivePatchesPerImage": pl.get("avg_effective_patches_per_image", "NA"),
            "PatchReductionRatio": pl.get("patch_reduction_ratio", "NA"),
            "PLTotalTime": pl.get("pl_total_time", "NA"),
            "PLTimePerImage": pl.get("pl_time_per_image", "NA"),
            "PLImagesPerSec": pl.get("pl_images_per_sec", "NA"),
            "PeakMemPL": pl.get("peak_gpu_mem_pl", "NA"),
            "HighConfRatio": pl.get("high_conf_ratio", "NA"),
            "UncertainRatio": pl.get("uncertain_ratio", "NA"),
        })

        train_rows.append({
            "Method": method,
            "CompressorMode": comp,
            "CompressLocation": loc,
            "SchedulerMode": sched,
            "MemoryBankSize": bank,
            "IterTotalTime": tr.get("iter_total_time", "NA"),
            "TrainImagesPerSec": tr.get("train_images_per_sec", "NA"),
            "TrainPatchesPerSec": tr.get("train_patches_per_sec", "NA"),
            "EpochTrainTime": tr.get("epoch_train_time", "NA"),
            "EpochTotalTime": tr.get("epoch_total_time", "NA"),
            "PeakMemTrain": tr.get("peak_gpu_mem_train", tr.get("gpu_mem_peak", "NA")),
        })

        trade_rows.append({
            "Method": method,
            "mIoU": va.get("val_mIoU", "NA"),
            "Delta_mIoU": (miou - baseline_miou) if (baseline_miou is not None and miou > 0) else "NA",
            "PLSpeedup": (baseline_pl / pl_total) if (baseline_pl and pl_total > 0) else "NA",
            "EpochSpeedup": (baseline_epoch / epoch_total) if (baseline_epoch and epoch_total > 0) else "NA",
            "PeakMemTrain": tr.get("peak_gpu_mem_train", tr.get("gpu_mem_peak", "NA")),
            "AvgEffectivePatchesPerImage": pl.get("avg_effective_patches_per_image", "NA"),
            "RawMemoryTokens": pl.get("raw_memory_tokens", tr.get("memory_tokens_before", "NA")),
            "CompressedMemoryTokens": pl.get("compressed_memory_tokens", tr.get("memory_tokens_after", "NA")),
            "MemoryCompressionRatio": pl.get("memory_compression_ratio", tr.get("memory_token_ratio", "NA")),
        })

    write_csv(out / "pl_efficiency_summary.csv", pl_header, pl_rows)
    write_csv(out / "train_efficiency_summary.csv", train_header, train_rows)
    write_csv(out / "accuracy_efficiency_tradeoff_summary.csv", trade_header, trade_rows)
    print(f"Summaries written to: {out}")


if __name__ == "__main__":
    main()
