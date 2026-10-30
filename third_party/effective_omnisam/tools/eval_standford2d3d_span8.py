#!/usr/bin/env python3
"""Evaluate indoor SPan8 checkpoint on Stanford360_SPAN8 via validation.py; enforce num_classes==8."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _strip_module(state):
    return {k.replace("module.", ""): v for k, v in state.items()}


def decoder_num_classes(ckpt_path: str) -> Optional[int]:
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        return None
    state = _strip_module(raw)
    w = state.get("decoder.linear_pred.weight")
    if w is None:
        return None
    return int(w.shape[0])


def write_num_class_mismatch(out_dir: Path, ckpt: str, got: int | None, expect: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "NUM_CLASS_MISMATCH.md"
    p.write_text(
        "# NUM_CLASS_MISMATCH\n\n"
        f"- checkpoint_path: `{ckpt}`\n"
        f"- decoder.linear_pred weight rows (num_classes): **{got}**\n"
        f"- dataset num_classes: **{expect}**\n\n"
        "## Why evaluation was skipped\n\n"
        "OmniSAM Stanford360 / SPan8 uses **8** semantic classes. "
        "A Cityscapes-13 / DensePASS-13 / WildPASS-13 head cannot be evaluated on this dataset "
        "without a compatible indoor checkpoint.\n\n"
        "## Required\n\n"
        "Provide an **indoor SPin8 / SPan8** OmniSAM checkpoint whose decoder outputs **8** classes.\n",
        encoding="utf-8",
    )


def write_missing_ckpt(out_dir: Path, reason: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MISSING_CKPT.md").write_text(
        f"# MISSING_CKPT\n\n{reason}\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    ap.add_argument("--method-name", type=str, default="run")
    ap.add_argument("--backbone", type=str, default="sam2_s")
    ap.add_argument("--use-mem-bank", action="store_true", default=True)
    ap.add_argument("--no-use-mem-bank", action="store_false", dest="use_mem_bank")
    ap.add_argument("--num-maskmem", type=int, default=9)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ckpt = Path(args.ckpt).resolve()
    repo = Path(args.repo_root).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint.txt").write_text(str(ckpt) + "\n", encoding="utf-8")

    if not ckpt.is_file():
        write_missing_ckpt(out_dir, f"Checkpoint not found: {ckpt}")
        print(f"[error] missing ckpt -> {out_dir / 'MISSING_CKPT.md'}", file=sys.stderr)
        return 2

    ncls = decoder_num_classes(str(ckpt))
    if ncls is not None and ncls != 8:
        write_num_class_mismatch(out_dir, str(ckpt), ncls, 8)
        print(f"[error] num_classes={ncls} -> {out_dir / 'NUM_CLASS_MISMATCH.md'}", file=sys.stderr)
        return 2

    proc = repo / "data_extra/processed/Stanford360_SPAN8"
    metrics_json = out_dir / "metrics.json"
    timing_json = out_dir / "timing.json"
    log_file = out_dir / "eval.log"

    cfg_stub = repo / "configs/extra_exp/standford2d3d/standford360_span8_eval.yaml"
    (out_dir / "config_used.yaml").write_text(cfg_stub.read_text(encoding="utf-8") if cfg_stub.is_file() else "", encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo / "validation.py"),
        "--dataset",
        "Stanford2D3D",
        "--stanford_processed_root",
        str(proc),
        "--backbone",
        args.backbone,
        "--batch_size",
        "1",
        "--ckpt",
        str(ckpt),
        "--metrics_json",
        str(metrics_json),
        "--timing_json",
        str(timing_json),
        "--log_file",
        str(log_file),
    ]
    if args.use_mem_bank:
        cmd.append("--use_mem_bank")
        cmd.extend(["--num_maskmem", str(args.num_maskmem)])
    else:
        cmd.extend(["--num_maskmem", "0"])

    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    patch_path = out_dir / "git_diff.patch"
    try:
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--", "."],
            stdout=open(patch_path, "w", encoding="utf-8"),
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        patch_path.write_text("(git diff unavailable)\n", encoding="utf-8")

    if args.dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0

    env = os.environ.copy()
    rc = subprocess.run(cmd, cwd=str(repo), env=env)
    if rc.returncode != 0:
        return int(rc.returncode)

    summary = out_dir / "summary.md"
    met = {}
    if metrics_json.is_file():
        try:
            met = json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    tim = {}
    if timing_json.is_file():
        try:
            tim = json.loads(timing_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    lines = [
        f"# Stanford360 SPan8 eval ({args.method_name})",
        "",
        f"- ckpt: `{ckpt}`",
        f"- mIoU: {met.get('mIoU')}",
        f"- mAcc: {met.get('mAcc')}",
        f"- PixAcc: {met.get('PixAcc')}",
        f"- FWIoU: {met.get('FWIoU')}",
        f"- mDice: {met.get('mDice')}",
        f"- eval_time_sec: {tim.get('eval_time_sec')}",
        f"- images_per_sec: {tim.get('images_per_sec')}",
        "",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
