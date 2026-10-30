#!/usr/bin/env python3
"""
WildPASS 数据与配对可视化诊断（不加载模型）。
  gt_distribution  — 全量 labels_trainid 像素分布 + 每图 valid ratio
  pairing_vis      — 随机 N 张 RGB / GT 彩图 / overlay
  raw_format       — labels_raw 像素 unique / dtype / shape / mode 推断格式
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf  # noqa: E402
from utils.seg_vis import label_ids_to_rgb, overlay_rgb_on_label  # noqa: E402


def _load_wildpass_paths(repo: Path, eval_config: Path):
    cfg = OmegaConf.load(str(eval_config))
    proc = cfg.get("processed_root") or cfg.get("root")
    p = Path(proc)
    if not p.is_absolute():
        p = (repo / p).resolve()
    val_list = p / str(cfg.get("val_list", "lists/val.txt"))
    image_dir = str(cfg.get("image_dir", "images"))
    label_dir = str(cfg.get("label_dir", "labels_trainid"))
    return p, val_list, image_dir, label_dir


def cmd_gt_distribution(repo: Path, eval_config: Path, out_json: Path, out_md: Path) -> None:
    root, val_list, image_dir, label_dir = _load_wildpass_paths(repo, eval_config)
    lines = [x.strip() for x in val_list.read_text(encoding="utf-8").splitlines() if x.strip()]
    total_pixels = 0
    label_counts: Counter = Counter()
    per_image: list = []
    for rel in lines:
        lab_path = root / label_dir / rel.lstrip("/")
        if not lab_path.is_file():
            stem, _ = os.path.splitext(rel)
            alt = root / label_dir / (stem + ".png")
            lab_path = alt if alt.is_file() else lab_path
        if not lab_path.is_file():
            per_image.append({"sample_id": rel, "error": "label_missing", "path": str(lab_path)})
            continue
        arr = np.array(Image.open(lab_path), dtype=np.int64).ravel()
        n = arr.size
        total_pixels += n
        u, c = np.unique(arr, return_counts=True)
        for v, cnt in zip(u.tolist(), c.tolist()):
            label_counts[int(v)] += int(cnt)
        n255 = int((arr == 255).sum())
        valid_ratio = float((n - n255) / n) if n else 0.0
        uvals = np.unique(arr)
        per_image.append(
            {
                "sample_id": rel,
                "path": str(lab_path),
                "shape_hw": list(Image.open(lab_path).size[::-1]),
                "num_pixels": n,
                "valid_ratio_non255": round(valid_ratio, 6),
                "unique_values": uvals[:64].tolist(),
            }
        )

    dist = {str(k): {"pixels": v, "ratio_of_all_pixels": round(v / total_pixels, 8)} for k, v in sorted(label_counts.items())}
    payload = {
        "processed_root": str(root),
        "label_dir": label_dir,
        "val_list": str(val_list),
        "num_list_entries": len(lines),
        "total_labeled_pixels": total_pixels,
        "label_id_distribution": dist,
        "per_image_valid_ratio": per_image,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines_md = [
        "# WildPASS13 labels_trainid — 全量像素分布",
        "",
        f"- processed_root: `{root}`",
        f"- 总像素（所有已读 label 图）: {total_pixels}",
        "",
        "| label_id | pixels | ratio |",
        "| --- | ---: | ---: |",
    ]
    for k in sorted(label_counts.keys(), key=lambda x: (x == 255, x)):
        v = label_counts[k]
        r = v / total_pixels if total_pixels else 0.0
        lines_md.append(f"| {k} | {v} | {r:.6f} |")
    lines_md.extend(["", "## 每图 valid ratio（非 255 像素比例）", ""])
    lines_md.append("| sample_id | valid_ratio |")
    lines_md.append("| --- | ---: |")
    for row in per_image[:30]:
        if "error" in row:
            lines_md.append(f"| `{row['sample_id']}` | ERR: {row['error']} |")
        else:
            lines_md.append(f"| `{row['sample_id']}` | {row['valid_ratio_non255']:.4f} |")
    if len(per_image) > 30:
        lines_md.append(f"| ... | （共 {len(per_image)} 条，见 JSON） |")
    out_md.write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    print(f"[gt_distribution] wrote {out_json} and {out_md}")


def cmd_pairing_vis(repo: Path, eval_config: Path, out_dir: Path, n: int, seed: int) -> None:
    root, val_list, image_dir, label_dir = _load_wildpass_paths(repo, eval_config)
    lines = [x.strip() for x in val_list.read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = random.Random(seed)
    pick = lines if len(lines) <= n else rng.sample(lines, n)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel in pick:
        safe_id = rel.replace("/", "_").replace("\\", "_")
        img_path = root / image_dir / rel.lstrip("/")
        lab_path = root / label_dir / rel.lstrip("/")
        if not lab_path.is_file():
            stem, _ = os.path.splitext(rel)
            lab_path = root / label_dir / (stem + ".png")
        if not img_path.is_file():
            print(f"[pairing_vis] skip missing image: {img_path}", flush=True)
            continue
        if not lab_path.is_file():
            print(f"[pairing_vis] skip missing label: {lab_path}", flush=True)
            continue
        rgb = np.array(Image.open(img_path).convert("RGB"))
        lab = np.array(Image.open(lab_path), dtype=np.int64)
        # resize label to rgb size if needed (nearest)
        if rgb.shape[:2] != lab.shape[:2]:
            lab_p = Image.fromarray(lab.astype(np.uint8)).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
            lab = np.array(lab_p, dtype=np.int64)
        col = label_ids_to_rgb(lab, num_classes=13, ignore_index=255)
        over = overlay_rgb_on_label(rgb, col, alpha=0.45)
        Image.fromarray(rgb).save(out_dir / f"sample_{safe_id}_rgb.png")
        Image.fromarray(col).save(out_dir / f"sample_{safe_id}_gt_color.png")
        Image.fromarray(over).save(out_dir / f"sample_{safe_id}_overlay.png")
    print(f"[pairing_vis] wrote {len(pick)} triples under {out_dir}")


def cmd_raw_format(repo: Path, eval_config: Path, out_md: Path, max_files: int) -> None:
    root, val_list, image_dir, label_dir = _load_wildpass_paths(repo, eval_config)
    raw_dir = root / "labels_raw"
    lines = [x.strip() for x in val_list.read_text(encoding="utf-8").splitlines() if x.strip()]
    sections: list[str] = [
        "# WildPASS 原始标签格式诊断（labels_raw）",
        "",
        f"- processed_root: `{root}`",
        f"- labels_raw 存在: `{raw_dir.is_dir()}`",
        "",
    ]
    if not raw_dir.is_dir():
        sections.append("未找到 `labels_raw/`，跳过像素级诊断。请确认预处理目录结构。")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(sections) + "\n", encoding="utf-8")
        print(f"[raw_format] wrote {out_md} (no labels_raw)")
        return

    examined = 0
    agg_uniques: set = set()
    sample_rows: list[str] = []
    for rel in lines:
        if examined >= max_files:
            break
        raw_path = raw_dir / rel.lstrip("/")
        if not raw_path.is_file():
            stem, _ = os.path.splitext(rel)
            raw_path = raw_dir / (stem + ".png")
        if not raw_path.is_file():
            continue
        examined += 1
        pil = Image.open(raw_path)
        mode = pil.mode
        arr = np.array(pil)
        shape = list(arr.shape)
        dtype = str(arr.dtype)
        if arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[-1])
            if flat.shape[0] > 50000:
                rng = np.random.default_rng(0)
                idx = rng.choice(flat.shape[0], size=50000, replace=False)
                flat = flat[idx]
            uniq = np.unique(flat, axis=0)
            uvals = uniq[:5].tolist()
            fmt_guess = "likely **RGB / color mask** (multi-channel)"
        else:
            u = np.unique(arr)
            uvals = u.tolist()[:48]
            agg_uniques.update(int(x) for x in u.tolist())
            umin, umax = int(u.min()), int(u.max())
            in13 = set(range(13))
            in19 = set(range(19)) | {255}
            if set(u.tolist()).issubset(in13):
                fmt_guess = "像素值 ⊆ {0..12} → 可能已是 **13 类 trainId**"
            elif set(u.tolist()).issubset(in19):
                fmt_guess = "像素值 ⊆ {0..18}∪{255} → 可能为 **Cityscapes 19 trainId**"
            elif umax <= 33 and umin >= 0:
                fmt_guess = "数值落在 labelId 常见范围 → 可能为 **Cityscapes labelId**（需 trainId 转换）"
            else:
                fmt_guess = "无法仅凭 unique 归入上述类别，需对照官方文档"

        sample_rows.append(
            f"| `{rel}` | `{raw_path.name}` | `{mode}` | `{dtype}` | `{shape}` | `{str(uvals)[:120]}...` | {fmt_guess} |"
        )

    sections.append("## 抽样文件（不凭空判断；以下由读取结果归纳）")
    sections.append("")
    sections.append("| sample_id | file | PIL mode | dtype | shape | unique(摘要) | 格式推断 |")
    sections.append("| --- | --- | --- | --- | --- | --- | --- |")
    sections.extend(sample_rows if sample_rows else ["| — | — | — | — | — | 无可读文件 | — |"])
    sections.append("")
    sections.append(f"## 合并 unique（单通道子集，最多 {max_files} 张）")
    sections.append("")
    sections.append(f"- sorted unique: `{sorted(agg_uniques)[:80]}`")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[raw_format] examined={examined}, wrote {out_md}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=str(REPO))
    ap.add_argument(
        "--eval-config",
        type=str,
        default="configs/extra_exp/wildpass/wildpass13_eval.yaml",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("gt_distribution", help="全量 GT 像素分布")
    p1.add_argument("--out-json", type=str, default="logs/wildpass_results/reports/wildpass_gt_distribution.json")
    p1.add_argument("--out-md", type=str, default="logs/wildpass_results/reports/wildpass_gt_distribution.md")

    p2 = sub.add_parser("pairing_vis", help="随机配对可视化")
    p2.add_argument("--n", type=int, default=20)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--out-dir", type=str, default="logs/wildpass_results/debug_pairing_vis")

    p3 = sub.add_parser("raw_format", help="labels_raw 格式诊断")
    p3.add_argument("--max-files", type=int, default=50)
    p3.add_argument("--out-md", type=str, default="logs/wildpass_results/reports/wildpass_raw_label_format_diagnosis.md")

    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    cfg = Path(args.eval_config)
    if not cfg.is_absolute():
        cfg = repo / cfg

    if args.cmd == "gt_distribution":
        cmd_gt_distribution(repo, cfg, repo / args.out_json, repo / args.out_md)
    elif args.cmd == "pairing_vis":
        cmd_pairing_vis(repo, cfg, repo / args.out_dir, args.n, args.seed)
    elif args.cmd == "raw_format":
        cmd_raw_format(repo, cfg, repo / args.out_md, args.max_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
