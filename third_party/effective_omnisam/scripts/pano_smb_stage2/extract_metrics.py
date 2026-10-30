#!/usr/bin/env python3
"""Extract Stage 2 metrics without changing the evaluator."""

import argparse
import json
import math
import re
from pathlib import Path


CLASS_IOU_RE = re.compile(r"^(.+?)\s*:\s*(NA|[-+]?\d+(?:\.\d+)?)\s*$")
BF_RE = re.compile(r"\[ValSummary\]\s+BFScore=([-+]?\d+(?:\.\d+)?)")


def last_json(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty JSONL: {path}")
    return json.loads(lines[-1])


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def parse_log(log_path):
    bfscore = None
    class_iou = []
    in_iou = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "========== per-class IoU (0-100) ==========" in line:
            in_iou = True
            class_iou = []
            continue
        if in_iou and line.startswith("=========================================="):
            in_iou = False
            continue
        if in_iou:
            match = CLASS_IOU_RE.match(line.strip())
            if match:
                name, value = match.groups()
                class_iou.append({"class": name.strip(), "IoU": None if value == "NA" else float(value)})
        match = BF_RE.search(line)
        if match:
            bfscore = float(match.group(1))
    return bfscore, class_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    save_dir = run_dir / "checkpoints" / "Cityscapes_13cls" / "sam2_s"
    metrics_path = save_dir / "metrics.jsonl"
    timing_path = save_dir / "metrics" / "train_timing.jsonl"
    run_log = run_dir / "run.log"
    for path in (metrics_path, timing_path, run_log):
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = last_json(metrics_path)
    timing = last_json(timing_path)
    bfscore, per_class_iou = parse_log(run_log)
    if not finite_number(bfscore):
        raw = metrics.get("BFScore")
        best = metrics.get("best_metrics") or {}
        bfscore = raw if finite_number(raw) else best.get("BFScore")

    result = {
        "label": args.label,
        "mIoU": metrics.get("val_mIoU"),
        "BFScore": bfscore,
        "mAcc": metrics.get("val_mAcc"),
        "PixAcc": metrics.get("val_PixAcc"),
        "FWIoU": metrics.get("val_FWIoU"),
        "mDice": metrics.get("val_mDice"),
        "per_class_IoU": per_class_iou,
        "actual_memory_tokens": timing.get("memory_tokens_after_final", timing.get("memory_tokens_after")),
        "step_time": timing.get("step_time"),
        "training_time": timing.get("epoch_total_time", timing.get("epoch_time")),
        "memory_attention_time": timing.get("memory_attention_time"),
        "peak_gpu_memory_gib": timing.get("gpu_mem_peak"),
        "best_checkpoint": metrics.get("best_ckpt_path"),
        "paths": {
            "run_dir": str(run_dir),
            "run_log": str(run_log),
            "metrics_jsonl": str(metrics_path),
            "train_timing_jsonl": str(timing_path),
        },
    }
    required = ("mIoU", "BFScore", "actual_memory_tokens", "step_time", "training_time", "memory_attention_time")
    invalid = [key for key in required if not finite_number(result.get(key))]
    if invalid:
        raise ValueError(f"invalid or missing metrics: {', '.join(invalid)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"metrics PASS: {output}")


if __name__ == "__main__":
    main()
