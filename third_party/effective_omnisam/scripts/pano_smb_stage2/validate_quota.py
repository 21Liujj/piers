#!/usr/bin/env python3
"""Validate and copy the Stage 2 B1 aggregate FoV quota artifact."""

import argparse
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    data = json.loads(source.read_text(encoding="utf-8"))
    calls = data.get("allocation_calls")
    required_numbers = (
        "quota_mean",
        "quota_std",
        "quota_min",
        "quota_max",
        "fallback_rate",
        "budget_error_max",
        "allocation_time_total",
        "allocation_time_per_call",
    )
    invalid = [key for key in required_numbers if not finite(data.get(key))]
    if not isinstance(calls, int) or calls <= 0:
        invalid.append("allocation_calls")
    if data.get("quota_min", 0) < 256 or data.get("quota_max", 1025) > 1024:
        invalid.append("quota_bounds")
    if data.get("budget_error_max") != 0:
        invalid.append("budget_error_max")
    if not isinstance(data.get("history_fov_counts"), dict) or not data["history_fov_counts"]:
        invalid.append("history_fov_counts")
    if not isinstance(data.get("representative"), list) or not data["representative"]:
        invalid.append("representative")
    for index, record in enumerate(data.get("representative", [])):
        quotas = record.get("quotas", [])
        history_fovs = record.get("history_fovs")
        if len(quotas) != history_fovs or sum(quotas) != 512 * history_fovs:
            invalid.append(f"representative[{index}]_budget")
    if invalid:
        raise ValueError("invalid quota aggregate: " + ", ".join(invalid))

    data["validation"] = {
        "status": "PASS",
        "quota_bounds": [256, 1024],
        "fixed_budget_per_history_fov": 512,
        "maximum_absolute_budget_error": 0,
        "source": str(source),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"quota aggregate PASS: {output}")


if __name__ == "__main__":
    main()
