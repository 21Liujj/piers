#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ckpt_zoo.py

目标：
- 扫描/归档本地训练产生的 ckpt（推荐默认用 symlink，节省空间）
- 生成可读索引（CKPT_INDEX.txt）与机器可读清单（index.json）
- 一键选择某个 ckpt 跑 validation.py（DensePASS / WildPASS）

用法示例：
  # 1) 归档（会写入 OmniSAM-main/ckpt_zoo/）
  python tools/ckpt_zoo.py collect

  # 2) 列出可用 ckpt
  python tools/ckpt_zoo.py list
  python tools/ckpt_zoo.py list --grep iou60

  # 3) 选择 ckpt 跑 val（通过 id / 名称关键词）
  python tools/ckpt_zoo.py val --id 3 --dataset DensePASS --class_set 13
  python tools/ckpt_zoo.py val --match best_iou60.36 --dataset DensePASS
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ZOO_DIR = os.path.join(REPO_ROOT, "ckpt_zoo")
ZOO_FILES_DIR = os.path.join(ZOO_DIR, "files")
ZOO_INDEX_JSON = os.path.join(ZOO_DIR, "index.json")
ZOO_INDEX_TXT = os.path.join(ZOO_DIR, "CKPT_INDEX.txt")


CKPT_EXTS = (".pth", ".pt", ".ckpt", ".pkl")
DEFAULT_SCAN_ROOTS = [
    # 项目内训练产物一般会在这里
    REPO_ROOT,
    # 官方/外置 ckpt 常被放在 repo 同级目录
    os.path.abspath(os.path.join(REPO_ROOT, "..")),
]


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _safe_slug(s: str) -> str:
    s = s.strip().replace(os.sep, "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "ckpt"


def _is_ckpt_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    base = os.path.basename(path)
    if base.startswith("."):
        return False
    return any(base.endswith(ext) for ext in CKPT_EXTS)


def _iter_ckpts(scan_roots: Iterable[str]) -> Iterable[str]:
    seen = set()
    for root in scan_roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # 跳过明显无关/巨大的目录
            bn = os.path.basename(dirpath)
            if bn in {"__pycache__", ".git", "wandb"}:
                dirnames[:] = []
                continue
            if os.path.abspath(dirpath).startswith(os.path.abspath(ZOO_DIR)):
                dirnames[:] = []
                continue

            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if not _is_ckpt_file(p):
                    continue
                ap = os.path.abspath(p)
                if ap in seen:
                    continue
                seen.add(ap)
                yield ap


@dataclass
class CkptEntry:
    id: int
    name: str
    src_path: str
    zoo_path: str
    size_mb: float
    mtime: str


def _format_entry_line(e: CkptEntry) -> str:
    return f"[{e.id:03d}] {e.name}  ({e.size_mb:.1f} MB, {e.mtime})\n      src={e.src_path}\n      zoo={e.zoo_path}"


def _write_index(entries: list[CkptEntry]) -> None:
    os.makedirs(ZOO_DIR, exist_ok=True)
    with open(ZOO_INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("CKPT Zoo Index\n")
    lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Repo: {REPO_ROOT}\n")
    lines.append("\nHow to use:\n")
    lines.append("  python tools/ckpt_zoo.py list\n")
    lines.append("  python tools/ckpt_zoo.py val --id <ID> --dataset DensePASS\n")
    lines.append("  python tools/ckpt_zoo.py val --match <keyword> --dataset DensePASS\n")
    lines.append("\nEntries:\n")
    for e in entries:
        lines.append(_format_entry_line(e) + "\n")

    with open(ZOO_INDEX_TXT, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def _load_entries() -> list[CkptEntry]:
    if not os.path.isfile(ZOO_INDEX_JSON):
        raise FileNotFoundError(f"index not found: {ZOO_INDEX_JSON}. Run: python tools/ckpt_zoo.py collect")
    with open(ZOO_INDEX_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: list[CkptEntry] = []
    for r in raw:
        out.append(CkptEntry(**r))
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    scan_roots = args.scan_root or DEFAULT_SCAN_ROOTS
    scan_roots = [os.path.abspath(p) for p in scan_roots]

    os.makedirs(ZOO_FILES_DIR, exist_ok=True)

    copied = 0
    entries: list[CkptEntry] = []
    for i, p in enumerate(sorted(_iter_ckpts(scan_roots))):
        rel = os.path.relpath(p, start=os.path.abspath(os.path.join(REPO_ROOT, "..")))
        base = os.path.basename(p)
        name = _safe_slug(os.path.splitext(base)[0])

        # 用相对路径 hash 避免重名
        uniq = _sha1_short(rel)
        zoo_name = f"{name}__{uniq}{os.path.splitext(base)[1]}"
        zoo_path = os.path.join(ZOO_FILES_DIR, zoo_name)

        if os.path.lexists(zoo_path):
            pass
        else:
            if args.mode == "copy":
                import shutil
                shutil.copy2(p, zoo_path)
                copied += 1
            elif args.mode == "link":
                os.symlink(p, zoo_path)
            else:
                raise ValueError(f"unknown mode: {args.mode}")

        st = os.stat(p)
        size_mb = st.st_size / (1024 * 1024)
        mtime = _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        entries.append(
            CkptEntry(
                id=i,
                name=zoo_name,
                src_path=p,
                zoo_path=zoo_path,
                size_mb=size_mb,
                mtime=mtime,
            )
        )

    _write_index(entries)
    print(f"Collected {len(entries)} ckpts into: {ZOO_FILES_DIR}")
    if args.mode == "copy":
        print(f"Copied new files: {copied}")
    print(f"Index: {ZOO_INDEX_TXT}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    entries = _load_entries()
    pat = args.grep
    for e in entries:
        s = _format_entry_line(e)
        if pat and (pat not in s):
            continue
        print(s)
    return 0


def _pick_entry(entries: list[CkptEntry], ckpt_id: Optional[int], match: Optional[str]) -> CkptEntry:
    if ckpt_id is not None:
        for e in entries:
            if e.id == ckpt_id:
                return e
        raise ValueError(f"ckpt id not found: {ckpt_id}")
    if match:
        match_l = match.lower()
        hits = [e for e in entries if match_l in e.name.lower() or match_l in e.src_path.lower()]
        if not hits:
            raise ValueError(f"no ckpt matched: {match}")
        if len(hits) == 1:
            return hits[0]
        # 多命中时取 mtime 最新
        hits.sort(key=lambda x: x.mtime, reverse=True)
        return hits[0]
    raise ValueError("must provide --id or --match")


def cmd_val(args: argparse.Namespace) -> int:
    entries = _load_entries()
    e = _pick_entry(entries, args.id, args.match)

    ckpt_path = e.zoo_path
    if not os.path.isfile(ckpt_path):
        # symlink 可能断了，尝试回退 src
        if os.path.isfile(e.src_path):
            ckpt_path = e.src_path
        else:
            raise FileNotFoundError(f"ckpt file missing: {e.zoo_path} (and src missing: {e.src_path})")

    logs_dir = os.path.join(ZOO_DIR, "val_logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = _now_stamp()
    log_file = os.path.join(logs_dir, f"val_{args.dataset}_{_safe_slug(args.stage)}_{e.id:03d}_{stamp}.log")

    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "validation.py"),
        "--dataset",
        args.dataset,
        "--data_root",
        args.data_root,
        "--backbone",
        args.backbone,
        "--batch_size",
        str(args.batch_size),
        "--ckpt",
        ckpt_path,
        "--log_file",
        log_file,
    ]
    if not args.no_use_mem_bank:
        cmd.append("--use_mem_bank")
    if args.dataset == "DensePASS":
        cmd.extend(["--class_set", str(args.class_set)])

    print("==========================================")
    print("ckpt_zoo val")
    print(f"Selected: [{e.id:03d}] {e.name}")
    print(f"CKPT: {ckpt_path}")
    print(f"Log : {log_file}")
    print("Cmd : " + " ".join(cmd))
    print("==========================================", flush=True)

    p = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(p.returncode)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect and manage ckpts + run validation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="scan and collect ckpts into ckpt_zoo/")
    p_collect.add_argument(
        "--scan_root",
        action="append",
        default=[],
        help="extra scan root (repeatable). default scans repo and its parent",
    )
    p_collect.add_argument("--mode", choices=["link", "copy"], default="link", help="store as symlink or copy")
    p_collect.set_defaults(func=cmd_collect)

    p_list = sub.add_parser("list", help="list collected ckpts")
    p_list.add_argument("--grep", default="", help="simple substring filter")
    p_list.set_defaults(func=cmd_list)

    p_val = sub.add_parser("val", help="run validation.py with a selected ckpt")
    g = p_val.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, help="ckpt id from list")
    g.add_argument("--match", type=str, help="substring match against name/src_path (picks newest if multiple)")
    p_val.add_argument("--dataset", choices=["DensePASS", "WildPASS"], default="DensePASS")
    p_val.add_argument("--class_set", type=int, default=13, help="DensePASS class set (13 or 19)")
    p_val.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    p_val.add_argument("--backbone", default="sam2_s")
    p_val.add_argument("--batch_size", type=int, default=1)
    p_val.add_argument("--no_use_mem_bank", action="store_true", help="disable --use_mem_bank")
    p_val.add_argument("--stage", default="ckpt_zoo")
    p_val.set_defaults(func=cmd_val)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

