#!/usr/bin/env python3
"""Validate frozen Stage 2 configs and write the allowed-only diff."""

import argparse
import difflib
from pathlib import Path

import yaml

ALLOWED_DIFFS = {"plugins.pano_smb.enabled"}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, path))
        return out
    return {prefix: value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", required=True)
    parser.add_argument("--b1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    b0_path = Path(args.b0)
    b1_path = Path(args.b1)
    b0 = yaml.safe_load(b0_path.read_text(encoding="utf-8"))
    b1 = yaml.safe_load(b1_path.read_text(encoding="utf-8"))
    flat0 = flatten(b0)
    flat1 = flatten(b1)
    changed = {
        key: (flat0.get(key), flat1.get(key))
        for key in sorted(set(flat0) | set(flat1))
        if flat0.get(key) != flat1.get(key)
    }
    unexpected = sorted(set(changed) - ALLOWED_DIFFS)
    missing = sorted(ALLOWED_DIFFS - set(changed))
    enabled_pair = changed.get("plugins.pano_smb.enabled")
    valid_switch = enabled_pair == (False, True)
    passed = not unexpected and not missing and valid_switch

    text_diff = "".join(
        difflib.unified_diff(
            b0_path.read_text(encoding="utf-8").splitlines(True),
            b1_path.read_text(encoding="utf-8").splitlines(True),
            fromfile=str(b0_path),
            tofile=str(b1_path),
        )
    )
    lines = [
        f"CONFIG_DIFF_CHECK={'PASS' if passed else 'FAIL'}",
        f"allowed_paths={','.join(sorted(ALLOWED_DIFFS))}",
        f"observed_paths={','.join(changed)}",
        f"unexpected_paths={','.join(unexpected) if unexpected else 'NONE'}",
        f"missing_paths={','.join(missing) if missing else 'NONE'}",
        "",
        text_diff,
    ]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    if not passed:
        raise SystemExit(1)
    print(f"config diff PASS: {args.output}")


if __name__ == "__main__":
    main()
