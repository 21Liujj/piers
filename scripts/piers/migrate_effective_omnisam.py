"""Deterministic migration of Effective-Omnisam into piers-ljj/third_party.

Replaces ad-hoc robocopy (which produced nested dirs) with an explicit
allow-list copy + verification against the source tree.

Usage:  python scripts/piers/migrate_effective_omnisam.py [--apply]
        (default = dry run, prints the plan + verification result)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

SRC = r"D:\piers2027\effective-omnisam-src"
DST = r"D:\piers2027\piers-ljj\third_party\effective_omnisam"

# Top-level directories copied verbatim (whole tree).
DIRS = [
    "model",
    "adaptation",
    "utils",
    "dataset",
    "train",
    "configs",
    "assets",
    "tools",
    "scripts",
    "tests",
    "docs",
]

# Individual files copied at the vendor root.
FILES = [
    "train.py",
    "run_adaptation.py",
    "validation.py",
    "validation_muti.py",
    "requirements.txt",
    "license.txt",
    "README.md",
    "QUICK_START.txt",
    "CLAUDE.md",
    "GRADUATION_PLAN.md",
    "INNOVATION_SUMMARY.md",
    "INNOVATION_DETAILED_EXPLANATION.md",
    "PROJECT_SUMMARY.md",
    "RESEARCH_ROADMAP_2026.md",
    "aggregate_multiseed.py",
    "analyze_tonight_results.py",
    "debug_wildpass.py",
    "summarize_routes.py",
    "test_cams_unit.py",
    "test_sgdrf_integration.py",
]

# sam/: only the real SAM2 package + the local LoRA patch. Demo/notebooks/
# videos/assets are upstream examples, not required to run training.
SAM_FILES = [
    "sam_lora_image_encoder.py",
    "setup.py",
    "LICENSE",
    "LICENSE_cctorch",
    "MANIFEST.in",
    "pyproject.toml",
    ".gitignore",
]
SAM_DIRS = ["sam"]

# ckpt_zoo: metadata only. files/*.pth are 150-byte pointer stubs that
# reference /root/autodl-tmp paths and are useless without the real weights.
CKPT_FILES = ["CKPT_INDEX.txt", "index.json", "README.txt"]

# Never migrate: bulk artifacts / upstream clutter.
EXCLUDED_TREE = [
    "experiments",      # 164 MB of result figures + logs
    "paper",            # 53 MB of LaTeX/PDF artifacts
    "markdown_archive",
    ".claude",
]

IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".ipynb_checkpoints"
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: str, dst: str) -> int:
    """Copy src tree onto dst, replacing dst. Returns file count."""
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
        rel = os.path.relpath(root, src)
        out_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out_root, exist_ok=True)
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                continue
            shutil.copy2(os.path.join(root, name), os.path.join(out_root, name))
            n += 1
    return n


def copy_file(src: str, dst: str) -> int:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return 1


def build(apply: bool) -> int:
    if not os.path.isdir(SRC):
        print(f"FATAL source missing: {SRC}")
        return 2

    if apply:
        if os.path.isdir(DST):
            shutil.rmtree(DST)
        os.makedirs(DST, exist_ok=True)

    total = 0
    for d in DIRS:
        s = os.path.join(SRC, d)
        if not os.path.isdir(s):
            print(f"  SKIP  {d} (not in source)")
            continue
        if apply:
            n = copy_tree(s, os.path.join(DST, d))
        else:
            n = sum(len(f) for _, _, f in os.walk(s))
        total += n
        print(f"  dir   {d:<12} {n:>4} files")

    for f in FILES:
        s = os.path.join(SRC, f)
        if not os.path.isfile(s):
            print(f"  SKIP  {f} (not in source)")
            continue
        if apply:
            total += copy_file(s, os.path.join(DST, f))
        else:
            total += 1
        print(f"  file  {f:<28} 1")

    # sam/
    sam_src = os.path.join(SRC, "sam")
    for d in SAM_DIRS:
        s = os.path.join(sam_src, d)
        if not os.path.isdir(s):
            continue
        if apply:
            n = copy_tree(s, os.path.join(DST, "sam", d))
        else:
            n = sum(len(f) for _, _, f in os.walk(s))
        total += n
        print(f"  dir   sam/{d:<8} {n:>4} files")
    for f in SAM_FILES:
        s = os.path.join(sam_src, f)
        if not os.path.isfile(s):
            continue
        if apply:
            total += copy_file(s, os.path.join(DST, "sam", f))
        else:
            total += 1
        print(f"  file  sam/{f:<24} 1")

    # sam2/ compat shim (points its __path__ at <vendor>/sam/sam)
    s = os.path.join(SRC, "sam2", "__init__.py")
    if os.path.isfile(s):
        if apply:
            total += copy_file(s, os.path.join(DST, "sam2", "__init__.py"))
        else:
            total += 1
        print("  file  sam2/__init__.py       1")

    for f in CKPT_FILES:
        s = os.path.join(SRC, "ckpt_zoo", f)
        if not os.path.isfile(s):
            continue
        if apply:
            total += copy_file(s, os.path.join(DST, "ckpt_zoo", f))
        else:
            total += 1
        print(f"  file  ckpt_zoo/{f:<20} 1")

    print(f"\n  TOTAL {total} files -> {DST}")
    return 0


def verify() -> int:
    """Compare every migrated file against the source (size + sha256)."""
    bad, checked = [], 0
    for d in DIRS:
        s = os.path.join(SRC, d)
        if not os.path.isdir(s):
            continue
        for root, dirs, files in os.walk(s):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", ".pytest_cache")]
            for name in files:
                if name.endswith((".pyc", ".pyo")):
                    continue
                sp = os.path.join(root, name)
                rel = os.path.relpath(sp, s)
                dp = os.path.join(DST, d, rel)
                if not os.path.isfile(dp):
                    bad.append(f"MISSING {d}/{rel}")
                    continue
                if os.path.getsize(sp) != os.path.getsize(dp):
                    bad.append(f"SIZE    {d}/{rel}")
                    continue
                if sha256(sp) != sha256(dp):
                    bad.append(f"HASH    {d}/{rel}")
                    continue
                checked += 1
    print(f"  verified {checked} files, {len(bad)} mismatches")
    for b in bad[:30]:
        print(f"    {b}")
    return 1 if bad else 0


def structure_check() -> int:
    """Fail if any directory is nested one level too deep."""
    problems = []
    for d in DIRS + ["sam", "sam2", "ckpt_zoo"]:
        p = os.path.join(DST, d)
        if not os.path.isdir(p):
            continue
        nested = os.path.join(p, d)
        if os.path.isdir(nested):
            problems.append(f"NESTED  {d}/{d}")
    # stray pycache
    for root, dirs, _ in os.walk(DST):
        for x in dirs:
            if x == "__pycache__":
                problems.append(f"PYCACHE {os.path.relpath(root, DST)}")
    print(f"  structure problems: {len(problems)}")
    for p in problems[:20]:
        print(f"    {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        build(a.apply)
    if a.apply:
        print("\nverify:")
        rc = verify()
        print("structure:")
        rc |= structure_check()
        sys.exit(rc)
    print("\n(dry run — pass --apply to write)")
