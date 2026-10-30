import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure repo root is on sys.path when running as a script from tools/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CITYSCAPES13 = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "car",
]


@dataclass
class Stats:
    total_px: int = 0
    ignore_px: int = 0
    per_class_px: Dict[int, int] = None

    def __post_init__(self):
        if self.per_class_px is None:
            self.per_class_px = defaultdict(int)

    def update_from_mask(self, mask: np.ndarray, num_classes: int, ignore_index: int):
        self.total_px += int(mask.size)
        self.ignore_px += int(np.sum(mask == ignore_index))
        for cid in range(num_classes):
            self.per_class_px[cid] += int(np.sum(mask == cid))

    def merge(self, other: "Stats"):
        self.total_px += other.total_px
        self.ignore_px += other.ignore_px
        for k, v in other.per_class_px.items():
            self.per_class_px[k] += v


def _read_label_png(path: Path) -> np.ndarray:
    # labelTrainIds are saved as single-channel (L) PNG.
    with Image.open(path) as im:
        im = im.convert("L")
        return np.array(im, dtype=np.uint8)


def _pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def _paths_from_used_path(pl_dir: Path) -> List[str]:
    used_path = pl_dir.parent / "used_path.txt"
    if not used_path.is_file():
        return []
    return [ln.strip() for ln in used_path.read_text().splitlines() if ln.strip()]


def _resolve_label_file(pl_dir: Path, name: str) -> Path:
    # name examples seen in this repo: "/Amsterdam/xxx.jpg" or "Amsterdam/xxx.jpg"
    rel = name.lstrip("/").replace(".jpg", "_labelTrainIds.png")
    return pl_dir / "gtFine" / rel


def analyze_files(
    files: List[Path],
    class_names: List[str],
    ignore_index: int,
    show_progress: bool = True,
) -> Tuple[Stats, Dict[str, Stats], Dict[str, Stats], List[Tuple[float, str]]]:
    if not files:
        raise FileNotFoundError("No pseudo-label PNGs to analyze.")

    num_classes = len(class_names)
    total = Stats()
    by_city: Dict[str, Stats] = defaultdict(Stats)
    by_img: Dict[str, Stats] = {}
    ignore_ratios: List[Tuple[float, str]] = []

    it = tqdm(files, desc="analyze", total=len(files)) if show_progress else files
    for p in it:
        mask = _read_label_png(p)
        s = Stats()
        s.update_from_mask(mask, num_classes=num_classes, ignore_index=ignore_index)
        total.merge(s)

        city = p.parent.name
        by_city[city].merge(s)
        by_img[str(p)] = s

        ign = s.ignore_px / max(1, s.total_px)
        ignore_ratios.append((float(ign), str(p)))

    return total, by_city, by_img, ignore_ratios


def render_table(stats: Stats, class_names: List[str], ignore_index: int) -> str:
    num_classes = len(class_names)
    labeled_px = max(0, stats.total_px - stats.ignore_px)
    lines = []
    lines.append(f"total_px={stats.total_px:,}")
    lines.append(f"ignore_px={stats.ignore_px:,} ({_pct(stats.ignore_px / max(1, stats.total_px))})  [ignore_index={ignore_index}]")
    lines.append(f"labeled_px={labeled_px:,} ({_pct(labeled_px / max(1, stats.total_px))})")
    lines.append("")
    lines.append("per-class pixel distribution (over labeled_px):")
    lines.append(f"{'class':<15} {'px':>12} {'share':>9}")
    for cid in range(num_classes):
        px = int(stats.per_class_px.get(cid, 0))
        share = (px / labeled_px) if labeled_px > 0 else 0.0
        lines.append(f"{class_names[cid]:<15} {px:>12,} {_pct(share):>9}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Analyze DensePASS/WildPASS pseudo-label PNGs (distribution + ignore ratio).")
    ap.add_argument(
        "--pl_dir",
        type=str,
        required=True,
        help="Pseudo-label epoch dir that contains gtFine/, e.g. pd_uda_densepass/DensePASS/sam2_s/Cityscapes/epoch1",
    )
    ap.add_argument("--ignore_index", type=int, default=255)
    ap.add_argument("--dataset13", type=str, default="Cityscapes", choices=["Cityscapes"])
    ap.add_argument("--topk_cities", type=int, default=10, help="Print top-K cities by ignore ratio.")
    ap.add_argument("--topk_images", type=int, default=20, help="Print top-K images by ignore ratio.")
    ap.add_argument(
        "--use_used_path",
        action="store_true",
        help="If set, only analyze PNGs referenced by used_path.txt next to epoch dir.",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If >0, randomly sample this many label files for a quick estimate (recommended: 200~500).",
    )
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    pl_dir = Path(args.pl_dir)
    label_root = pl_dir / "gtFine"
    class_names = CITYSCAPES13

    if args.use_used_path:
        names = _paths_from_used_path(pl_dir)
        if not names:
            raise FileNotFoundError(f"--use_used_path set but used_path.txt not found next to: {pl_dir}")
        all_files = [_resolve_label_file(pl_dir, n) for n in names]
        missing = [str(p) for p in all_files if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} pseudo-label PNGs. Examples: {missing[:3]}")
    else:
        all_files = sorted(label_root.rglob("*_labelTrainIds.png"))

    if not all_files:
        raise FileNotFoundError(f"No *_labelTrainIds.png under: {label_root}")

    # Optional sampling (full set can be large; sampling gives quick estimates).
    if args.max_files and args.max_files > 0 and len(all_files) > args.max_files:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(all_files), size=args.max_files, replace=False)
        picked = [all_files[int(i)] for i in sorted(pick)]
        print(f"[sampling] analyzed {len(picked)}/{len(all_files)} files (seed={args.seed})")
        files = picked
    else:
        files = all_files

    total, by_city, _, ignore_ratios = analyze_files(
        files=files, class_names=class_names, ignore_index=args.ignore_index, show_progress=True
    )

    print("== Overall ==")
    print(render_table(total, class_names=class_names, ignore_index=args.ignore_index))
    print("")

    # City-wise ignore ratio (useful to see domain splits / difficult cities)
    city_rows = []
    for city, st in by_city.items():
        ign = st.ignore_px / max(1, st.total_px)
        city_rows.append((ign, city, st))
    city_rows.sort(reverse=True, key=lambda x: x[0])

    k = max(0, int(args.topk_cities))
    if k:
        print(f"== Top {k} cities by ignore ratio ==")
        for ign, city, st in city_rows[:k]:
            print(f"{city:<15} ignore={_pct(ign)}  total_px={st.total_px:,}")

    ki = max(0, int(args.topk_images))
    if ki:
        ignore_ratios.sort(reverse=True, key=lambda x: x[0])
        print("")
        print(f"== Top {ki} images by ignore ratio ==")
        for ign, p in ignore_ratios[:ki]:
            print(f"ignore={_pct(ign)}  {p}")


if __name__ == "__main__":
    main()

