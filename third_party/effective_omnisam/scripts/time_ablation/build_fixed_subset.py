#!/usr/bin/env python3
import argparse
from pathlib import Path
import random


def main():
    ap = argparse.ArgumentParser(description="Build fixed target subset list for time ablation.")
    ap.add_argument("--input_list", required=True, help="Input list file, one relative path per line.")
    ap.add_argument("--output_list", required=True, help="Output subset list path.")
    ap.add_argument("--num_images", type=int, required=True, help="Number of images to keep.")
    ap.add_argument("--offset", type=int, default=0, help="Start offset in input list.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before selecting subset (reproducible with --seed).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed used when --shuffle is set.")
    args = ap.parse_args()

    inp = Path(args.input_list)
    out = Path(args.output_list)
    lines = [x.strip() for x in inp.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(lines)
    offset = max(0, int(args.offset))
    num = max(0, int(args.num_images))
    subset = lines[offset : offset + num]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(subset) + ("\n" if subset else ""), encoding="utf-8")
    print(f"subset_size={len(subset)} shuffle={bool(args.shuffle)} seed={int(args.seed)} -> {out}")


if __name__ == "__main__":
    main()
