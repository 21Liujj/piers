#!/usr/bin/env python3
"""Aggregate multi-seed TeCMA run results: mean +/- std for mIoU / BFScore / timing.

Usage (run on remote server, repo root):
    python aggregate_multiseed.py <config_name> [<config_name> ...]

Scans experiments/tecma/runs/<config_name>_seed*_* dirs, reads run_summary.json
for mIoU/timing and run.log for BFScore, prints a summary table and writes
a JSON file with raw per-seed values + mean/std for later LaTeX table generation.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

RUNS_ROOT = "experiments/tecma/runs"


def find_seed_dirs(config_name):
    pattern = os.path.join(RUNS_ROOT, f"{config_name}_seed*_*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    return sorted(dirs)


def parse_seed_from_dirname(dirname):
    m = re.search(r"_seed(\d+)_", os.path.basename(dirname))
    return int(m.group(1)) if m else None


def read_bfscore_from_log(run_log_path):
    if not os.path.isfile(run_log_path):
        return None
    bf = None
    with open(run_log_path, "r", errors="ignore") as f:
        for line in f:
            m = re.search(r"\[ValSummary\]\s*BFScore=([0-9.]+)", line)
            if m:
                bf = float(m.group(1))
    return bf


def read_run_summary(run_dir):
    path = os.path.join(run_dir, "run_summary.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect_config(config_name):
    seed_dirs = find_seed_dirs(config_name)
    records = []
    for d in seed_dirs:
        seed = parse_seed_from_dirname(d)
        summary = read_run_summary(d)
        if summary is None:
            print(f"  [WARN] no run_summary.json in {d}, skipping", file=sys.stderr)
            continue
        if summary.get("train_return_code", 1) != 0:
            print(f"  [WARN] {d} train_return_code != 0, skipping", file=sys.stderr)
            continue
        run_log = summary.get("run_log", os.path.join(d, "run.log"))
        bf = read_bfscore_from_log(run_log)
        miou = summary.get("metrics", {}).get("val_mIoU")
        step_time = summary.get("profile", {}).get("step_time")
        records.append({
            "seed": seed,
            "dir": d,
            "mIoU": miou,
            "BFScore": bf,
            "step_time": step_time,
        })
    return records


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize(config_name, records):
    mious = [r["mIoU"] for r in records]
    bfs = [r["BFScore"] for r in records]
    steps = [r["step_time"] for r in records]
    miou_mean, miou_std = mean_std(mious)
    bf_mean, bf_std = mean_std(bfs)
    step_mean, step_std = mean_std(steps)
    return {
        "config": config_name,
        "n_seeds": len(records),
        "records": records,
        "mIoU_mean": miou_mean, "mIoU_std": miou_std,
        "BFScore_mean": bf_mean, "BFScore_std": bf_std,
        "step_time_mean": step_mean, "step_time_std": step_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed TeCMA results")
    parser.add_argument("configs", nargs="+", help="Config names (e.g. baseline_omnisam A_Dtrain_Etrain)")
    parser.add_argument("--out", default="multiseed_summary.json", help="JSON output path")
    args = parser.parse_args()

    all_summaries = []
    for cfg in args.configs:
        print(f"=== {cfg} ===")
        recs = collect_config(cfg)
        if not recs:
            print(f"  No valid runs found for {cfg}\n")
            continue
        summ = summarize(cfg, recs)
        all_summaries.append(summ)
        print(f"  n_seeds: {summ['n_seeds']}")
        if summ['mIoU_mean'] is not None:
            print(f"  mIoU: {summ['mIoU_mean']:.2f} ± {summ['mIoU_std']:.2f}")
        if summ['BFScore_mean'] is not None:
            print(f"  BFScore: {summ['BFScore_mean']:.2f} ± {summ['BFScore_std']:.2f}")
        if summ['step_time_mean'] is not None:
            print(f"  step_time: {summ['step_time_mean']:.4f} ± {summ['step_time_std']:.4f} s")
        print()

    with open(args.out, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

