#!/usr/bin/env python3
"""
tools/tecma_analysis/collect_profile_summary.py
Phase 6-1: Read-only profile/timing collection from TeCMA clean ablation and
           token budget runs. Generates CSV tables to experiments/tecma/tables/.

Usage:
  cd /root/autodl-tmp/OmniSAM-main
  python tools/tecma_analysis/collect_profile_summary.py

Does NOT start any training, validation, or PL generation.
"""
import os, json, csv, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Run directories to collect ----
RUNS = [
    # Phase 4-clean ablation
    ("ablation", "experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834"),
    ("ablation", "experiments/tecma/runs/A_ccpm_seed1_20260616_104911"),
    ("ablation", "experiments/tecma/runs/A_Dtrain_seed1_20260616_111019"),
    ("ablation", "experiments/tecma/runs/A_Etrain_seed1_20260616_113023"),
    ("ablation", "experiments/tecma/runs/Dtrain_Etrain_seed1_20260616_115113"),
    ("ablation", "experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135"),
    # Phase 5 token budget
    ("token_budget", "experiments/tecma/runs/token_budget/A_Etrain_keep075_seed1_20260617_112455"),
    ("token_budget", "experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629"),
]

COLS = [
    "group", "config", "mIoU", "BFScore",
    "mAcc", "PixAcc", "FWIoU", "mDice",
    "step_time", "iter_compute_time", "memory_attention_time",
    "gpu_mem_peak",
    "memory_tokens_before_final", "memory_tokens_after_final",
    "compression_tokens_before", "compression_tokens_after", "compression_ratio",
    "sparse_tokens_before", "sparse_tokens_after", "sparse_keep_ratio",
    "memory_tokens_scope", "pl_reuse_evidence", "run_dir",
]


def extract_bfscore(log_path):
    """Extract BFScore from [ValSummary] line in run.log."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in reversed(f.readlines()):
            m = re.search(r'\[ValSummary\] BFScore=([\d.]+)', line)
            if m:
                return float(m.group(1))
    return None


def extract_pl_reuse(log_path):
    """Check run.log for PL reuse evidence."""
    if not os.path.exists(log_path):
        return "no run.log"
    with open(log_path) as f:
        for line in f:
            if "Reuse fixed pseudo-label dir" in line:
                return line.strip()
    return "canonical PL (no reuse line; clean ablation)"


def collect_run(group, rel_dir):
    run_dir = os.path.join(ROOT, rel_dir)
    summary_path = os.path.join(run_dir, "run_summary.json")
    log_path = os.path.join(run_dir, "run.log")

    if not os.path.exists(summary_path):
        print(f"  WARNING: missing run_summary.json in {rel_dir}", file=sys.stderr)
        return None

    with open(summary_path) as f:
        s = json.load(f)

    prof = s.get("profile", {})
    metrics = s.get("metrics", {})

    row = {
        "group": group,
        "config": s.get("config_name", ""),
        "mIoU": metrics.get("val_mIoU"),
        "BFScore": extract_bfscore(log_path),
        "mAcc": metrics.get("val_mAcc"),
        "PixAcc": metrics.get("val_PixAcc"),
        "FWIoU": metrics.get("val_FWIoU"),
        "mDice": metrics.get("val_mDice"),
        "step_time": prof.get("step_time"),
        "iter_compute_time": prof.get("iter_compute_time"),
        "memory_attention_time": prof.get("memory_attention_time"),
        "gpu_mem_peak": prof.get("gpu_mem_peak"),
        "memory_tokens_before_final": prof.get("memory_tokens_before_final"),
        "memory_tokens_after_final": prof.get("memory_tokens_after_final"),
        "compression_tokens_before": prof.get("compression_tokens_before"),
        "compression_tokens_after": prof.get("compression_tokens_after"),
        "compression_ratio": prof.get("compression_ratio"),
        "sparse_tokens_before": prof.get("sparse_tokens_before"),
        "sparse_tokens_after": prof.get("sparse_tokens_after"),
        "sparse_keep_ratio": prof.get("sparse_keep_ratio"),
        "memory_tokens_scope": prof.get("memory_tokens_scope"),
        "pl_reuse_evidence": extract_pl_reuse(log_path),
        "run_dir": rel_dir,
    }
    return row


def main():
    rows = []
    for group, rel_dir in RUNS:
        print(f"collecting: {rel_dir}")
        row = collect_run(group, rel_dir)
        if row:
            rows.append(row)

    out_path = os.path.join(ROOT, "experiments/tecma/tables/profile_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for row in rows:
            w.writerow({k: (row.get(k, "") if row.get(k) is not None else "") for k in COLS})

    print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
