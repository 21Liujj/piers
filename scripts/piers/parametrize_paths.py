"""Rewrite the machine-specific absolute paths in configs/piers/* into
${PIERS_*} placeholders so the committed configs are machine-independent.

Idempotent: running it twice is a no-op.

Usage:  python scripts/piers/parametrize_paths.py [--check]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from piers_ljj.paths import PLACEHOLDER_RE, default_paths  # noqa: E402

# Longest prefix first so nested roots are replaced correctly.
SUBS = [
    ("/root/autodl-tmp/data_extra", "${PIERS_DATA_EXTRA_ROOT}"),
    ("/root/autodl-tmp/data", "${PIERS_DATA_ROOT}"),
    ("/root/autodl-tmp/weights", "${PIERS_WEIGHTS_ROOT}"),
    ("/root/autodl-tmp/piers-main", "${PIERS_REPO_ROOT}"),
]


def rewrite(text: str) -> str:
    for literal, placeholder in SUBS:
        text = text.replace(literal, placeholder)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any hardcoded path remains")
    args = ap.parse_args()

    targets = sorted(
        glob.glob(os.path.join(REPO, "configs", "piers", "*.yaml"))
        + glob.glob(os.path.join(REPO, "configs", "piers", "*.json"))
    )
    changed = []
    for path in targets:
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
        updated = rewrite(original)
        if updated != original:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(updated)
            changed.append(os.path.relpath(path, REPO))

    if changed:
        print("rewritten:")
        for rel in changed:
            print(f"  {rel}")
    else:
        print("no changes needed")

    if args.check:
        leftovers = []
        for path in targets:
            with open(path, encoding="utf-8") as fh:
                body = fh.read()
            for literal, _ in SUBS:
                if literal in body:
                    leftovers.append(f"{os.path.relpath(path, REPO)}: {literal}")
            for name in PLACEHOLDER_RE.findall(body):
                if name not in default_paths():
                    leftovers.append(f"{os.path.relpath(path, REPO)}: unknown {name}")
        if leftovers:
            print("\nFAIL: unresolved paths remain")
            for line in leftovers[:50]:
                print("  " + line)
            return 1
        print("\nOK: all configs are machine-independent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
