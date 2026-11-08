"""Verify that the source snapshot stored under results/ reproduces the exact
file hashes recorded in source_fingerprints.json (the hashes of the code that
actually produced the paper numbers).

The recorded hashes were computed on Linux (LF line endings). When this repo is
checked out on Windows the working tree may use CRLF, so the comparison
normalises line endings before hashing. Content must otherwise be identical.

Run:  python scripts/piers/verify_result_fingerprints.py
Exit code 0 => the snapshot in this repo is the code that produced the results.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results", "I1_densepass")
SNAPSHOT = os.path.join(RESULTS, "source_snapshot")
FINGERPRINTS = os.path.join(RESULTS, "source_fingerprints.json")

# Recorded on the GPU machine under /root/autodl-tmp/piers-main/
SERVER_PREFIX = "/root/autodl-tmp/piers-main/"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_raw(path: str) -> str:
    with open(path, "rb") as fh:
        return _sha256_bytes(fh.read())


def sha256_lf(path: str) -> str:
    """Hash the file with line endings normalised to LF."""
    with open(path, "rb") as fh:
        data = fh.read()
    return _sha256_bytes(data.replace(b"\r\n", b"\n"))


def main() -> int:
    if not os.path.exists(FINGERPRINTS):
        print(f"MISSING fingerprints file: {FINGERPRINTS}")
        return 2
    if not os.path.isdir(SNAPSHOT):
        print(f"MISSING source snapshot dir: {SNAPSHOT}")
        return 2

    recorded = json.load(open(FINGERPRINTS, encoding="utf-8"))
    matched = differed = missing = 0
    failures = []

    for server_path, expected in recorded.items():
        rel = server_path.replace(SERVER_PREFIX, "").replace("/", os.sep)
        cand = os.path.join(SNAPSHOT, rel)
        if not os.path.exists(cand):
            missing += 1
            failures.append(("MISSING", rel, expected, ""))
            continue
        raw = sha256_raw(cand)
        lf = sha256_lf(cand)
        if raw == expected:
            matched += 1
        elif lf == expected:
            # Same content, different line endings (Windows checkout).
            matched += 1
            print(f"NOTE    {rel} (matches after LF normalisation)")
        else:
            differed += 1
            failures.append(("DIFFER", rel, expected, raw))

    for kind, rel, expected, got in failures:
        print(f"{kind:8s} {rel}")
        print(f"         expected {expected}")
        if got:
            print(f"         got      {got}")

    total = len(recorded)
    print(f"\n{total - differed - missing}/{total} files match the recorded hashes")
    if differed or missing:
        print("FAIL: snapshot does not reproduce the recorded source state")
        return 1
    print("OK: the snapshot in results/I1_densepass/source_snapshot is exactly")
    print("    the code that produced the numbers in results/I1_densepass/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
