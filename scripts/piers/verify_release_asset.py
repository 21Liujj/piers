"""Download a GitHub release asset and verify its SHA256 against the value
documented in REPRODUCE.md.

Usage: python scripts/piers/verify_release_asset.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RELEASE = "v0.2-paper-repro"
BASE = f"https://github.com/21Liujj/piers/releases/download/{RELEASE}"

EXPECTED = {
    "allhard_runs_R0_seed1234_final.pth":
        "ff79b0b0f766dbe7ad695bd56535d11cade9b12f608ce0ddb13778ba49f9a1d3",
    "allhard_runs_R1_seed1234_final.pth":
        "07fdc14b7bf072edc78c402fd11e940d28a7e3dd73dcd3ff7f5f4c973b6a7232",
    "dense_controls_R0_seed1234_final.pth":
        "40159d944c8925cd1c61ce4f8929cbde1fdbda761d0c41fb3bdfe27fec8f5625",
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)
    url = f"{BASE}/{name}"
    print(f"GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "piers-verify"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            print(f"  {total/1e6:.1f} MB", end="\r", flush=True)
    print()
    return dest


def main() -> int:
    only = sys.argv[1:] or list(EXPECTED)
    dest_dir = os.path.join(REPO, "runs", "_release_verify")
    failures = 0
    for name in only:
        expected = EXPECTED.get(name)
        if expected is None:
            print(f"SKIP {name}: not in the documented list")
            continue
        try:
            path = download(name, dest_dir)
        except Exception as exc:
            print(f"FAIL {name}: download error {exc}")
            failures += 1
            continue
        got = sha256_of(path)
        size = os.path.getsize(path)
        ok = got == expected
        print(f"{'OK  ' if ok else 'FAIL'} {name}  {size/1e6:.1f} MB")
        print(f"     expected {expected}")
        print(f"     got      {got}")
        if not ok:
            failures += 1
        os.remove(path)
    print(f"\n{len(only) - failures}/{len(only)} release assets verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
