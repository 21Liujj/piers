"""Upload the reproduction assets (trained checkpoints) to a GitHub Release.

Usage:
    python scripts/piers/upload_release_assets.py --release v0.2-paper-repro

Reads the token from the git credential store (or $GITHUB_TOKEN).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    cred = os.path.expanduser("~/.git-credentials")
    if os.path.exists(cred):
        with open(cred, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if "github.com" in line and "@" in line:
                    userinfo = line.split("://", 1)[1].split("@", 1)[0]
                    if ":" in userinfo:
                        return userinfo.split(":", 1)[1]
    raise SystemExit("no GitHub token found (set GITHUB_TOKEN or use a credential store)")


def api(method: str, url: str, token: str, body: bytes | None = None,
        content_type: str | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="release tag")
    ap.add_argument("--assets", nargs="*", default=[], help="local files to upload")
    args = ap.parse_args()

    token = get_token()
    status, payload = api(
        "GET",
        f"https://api.github.com/repos/21Liujj/piers/releases/tags/{args.release}",
        token,
    )
    if status != 200:
        print(f"cannot read release: {status} {payload[:300]!r}")
        return 1
    release = json.loads(payload)
    existing = {a["name"] for a in release.get("assets", [])}
    upload_base = release["upload_url"].split("{")[0]
    print(f"release {release['tag_name']} id={release['id']} assets={len(existing)}")

    ok = True
    for path in args.assets:
        name = os.path.basename(path)
        if name in existing:
            print(f"SKIP  {name} (already uploaded)")
            continue
        size = os.path.getsize(path)
        print(f"UPLOAD {name} ({size/1e6:.1f} MB) ...", flush=True)
        with open(path, "rb") as fh:
            data = fh.read()
        query = f"?name={name}"
        st, body = api(
            "POST",
            upload_base + query,
            token,
            body=data,
            content_type="application/octet-stream",
        )
        if st == 201:
            print(f"  OK  {name}")
        else:
            ok = False
            print(f"  FAIL {name}: {st} {body[:300]!r}")
        time.sleep(1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
