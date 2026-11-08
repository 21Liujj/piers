"""Verify the snapshot AS STORED IN GIT (not the working tree).

This is the check that matters for "clone on a new machine and reproduce":
it reads the blob objects straight out of the git index/HEAD, so it validates
what a fresh `git clone` would actually give you, ignoring any local
line-ending conversion the working tree may have applied.

Usage:
    python scripts/piers/verify_snapshot_in_git.py [--rev HEAD]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results", "I1_densepass")
FINGERPRINTS = os.path.join(RESULTS, "source_fingerprints.json")
SNAPSHOT_PREFIX = "results/I1_densepass/source_snapshot/"
SERVER_PREFIX = "/root/autodl-tmp/piers-main/"


def git_blob(rev: str, path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return proc.stdout if proc.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    args = ap.parse_args()

    recorded = json.load(open(FINGERPRINTS, encoding="utf-8"))
    ok = bad = missing = 0
    problems = []

    for server_path, expected in recorded.items():
        rel = server_path.replace(SERVER_PREFIX, "")
        git_path = SNAPSHOT_PREFIX + rel
        blob = git_blob(args.rev, git_path)
        if blob is None:
            missing += 1
            problems.append(f"MISSING  {git_path}")
            continue
        got = hashlib.sha256(blob).hexdigest()
        if got == expected:
            ok += 1
        else:
            bad += 1
            problems.append(f"DIFFER   {git_path}\n         expected {expected}\n         got      {got}")

    for line in problems:
        print(line)

    total = len(recorded)
    print(f"\n{ok}/{total} snapshot files match the recorded hashes "
          f"(as stored in git at {args.rev})")
    if bad or missing:
        print("FAIL: a fresh clone would not reproduce the recorded source state")
        return 1
    print("OK: a fresh git clone carries exactly the code that produced the results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
