#!/usr/bin/env python3
"""Apply the conservative B0 protocol sanity gate."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--miou-anchor", type=float, default=60.02)
    parser.add_argument("--miou-tolerance", type=float, default=0.30)
    parser.add_argument("--bf-anchor", type=float, default=39.23)
    parser.add_argument("--bf-tolerance", type=float, default=0.50)
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    miou = metrics.get("mIoU")
    bfscore = metrics.get("BFScore")
    valid = all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (miou, bfscore))
    miou_delta = abs(float(miou) - args.miou_anchor) if valid else None
    bf_delta = abs(float(bfscore) - args.bf_anchor) if valid else None
    passed = bool(valid and miou_delta <= args.miou_tolerance and bf_delta <= args.bf_tolerance)
    record = {
        "status": "PASS" if passed else "B0_SANITY_FAIL",
        "purpose": "protocol sanity gate; not a scientific verdict",
        "observed": {"mIoU": miou, "BFScore": bfscore},
        "anchor": {"mIoU": args.miou_anchor, "BFScore": args.bf_anchor},
        "tolerance": {"mIoU": args.miou_tolerance, "BFScore": args.bf_tolerance},
        "absolute_delta": {"mIoU": miou_delta, "BFScore": bf_delta},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(record["status"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
