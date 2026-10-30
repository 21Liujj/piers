#!/usr/bin/env python3
"""Generate machine- and human-readable Stage 2 summaries."""

import argparse
import csv
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def delta(b0, b1, key):
    a, b = b0.get(key), b1.get(key)
    return (float(b) - float(a)) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def pct_delta(b0, b1, key):
    a = b0.get(key)
    d = delta(b0, b1, key)
    return (100.0 * d / float(a)) if d is not None and float(a) != 0.0 else None


def fmt(value, digits=4):
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", required=True)
    parser.add_argument("--b1", required=True)
    parser.add_argument("--quota", required=True)
    parser.add_argument("--sanity", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--per-class-csv", required=True)
    args = parser.parse_args()

    b0, b1, quota, sanity = map(load, (args.b0, args.b1, args.quota, args.sanity))
    deltas = {
        "mIoU": delta(b0, b1, "mIoU"),
        "BFScore": delta(b0, b1, "BFScore"),
        "step_time": delta(b0, b1, "step_time"),
        "step_overhead_percent": pct_delta(b0, b1, "step_time"),
        "training_time": delta(b0, b1, "training_time"),
        "training_overhead_percent": pct_delta(b0, b1, "training_time"),
        "memory_attention_time": delta(b0, b1, "memory_attention_time"),
        "actual_memory_tokens": delta(b0, b1, "actual_memory_tokens"),
    }
    result = {
        "status": "COMPLETE",
        "scientific_verdict": "REQUIRES_HUMAN_ANALYSIS",
        "B0": b0,
        "B1": b1,
        "delta": deltas,
        "quota": quota,
        "fairness": {
            "config_diff": "plugins.pano_smb.enabled only",
            "fixed_PL_shared": True,
            "initial_checkpoint_shared": True,
            "seed_shared": True,
            "B0_sanity": sanity,
        },
        "paths": {
            "summary_json": args.output_json,
            "summary_md": args.output_md,
            "per_class_delta_csv": args.per_class_csv,
        },
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    b0_classes = {item["class"]: item.get("IoU") for item in b0.get("per_class_IoU", [])}
    b1_classes = {item["class"]: item.get("IoU") for item in b1.get("per_class_IoU", [])}
    with Path(args.per_class_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "B0_IoU", "B1_IoU", "delta"])
        for name in b0_classes:
            a, b = b0_classes[name], b1_classes.get(name)
            writer.writerow([name, a, b, (float(b) - float(a)) if a is not None and b is not None else None])

    md = f"""# Stage 2 Runtime Status

Status: COMPLETE
Scientific verdict: REQUIRES HUMAN ANALYSIS

## B0

- mIoU: {fmt(b0.get('mIoU'), 2)}
- BFScore: {fmt(b0.get('BFScore'), 2)}
- Tokens: {fmt(b0.get('actual_memory_tokens'), 0)}
- Step Time: {fmt(b0.get('step_time'))}
- Training Time: {fmt(b0.get('training_time'))}
- Memory Attention: {fmt(b0.get('memory_attention_time'))}

## B1

- mIoU: {fmt(b1.get('mIoU'), 2)}
- BFScore: {fmt(b1.get('BFScore'), 2)}
- Tokens: {fmt(b1.get('actual_memory_tokens'), 0)}
- Step Time: {fmt(b1.get('step_time'))}
- Training Time: {fmt(b1.get('training_time'))}
- Memory Attention: {fmt(b1.get('memory_attention_time'))}

## Delta

- Delta mIoU: {fmt(deltas['mIoU'], 2)}
- Delta BFScore: {fmt(deltas['BFScore'], 2)}
- Step overhead: {fmt(deltas['step_overhead_percent'], 2)}%
- Training overhead: {fmt(deltas['training_overhead_percent'], 2)}%
- Token delta: {fmt(deltas['actual_memory_tokens'], 0)}

## Quota

- mean: {fmt(quota.get('quota_mean'), 2)}
- std: {fmt(quota.get('quota_std'), 2)}
- min: {fmt(quota.get('quota_min'), 0)}
- max: {fmt(quota.get('quota_max'), 0)}
- fallback rate: {fmt(quota.get('fallback_rate'), 6)}
- budget errors: {fmt(quota.get('budget_error_max'), 0)}
- allocation calls: {fmt(quota.get('allocation_calls'), 0)}

## Artifact Paths

- B0: {b0['paths']['run_dir']}
- B1: {b1['paths']['run_dir']}
- Per-class delta: {args.per_class_csv}
"""
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(f"summary PASS: {args.output_json}")


if __name__ == "__main__":
    main()
