"""Build a query manifest from a DensePASS-style image + GT layout.

Usage:
    python scripts/piers/make_manifest.py \
        --img-dir D:/datasets/DensePASS/val/img \
        --gt-dir  D:/datasets/DensePASS/val/gtFine \
        --split dev-calib --out configs/piers/manifest.dev-calib.json \
        [--queries queries.txt] [--limit 20]

Default query set (no GT reading required to build the manifest itself):
  category_union : car, person, road, vegetation, sky
  instance_referring : "the leftmost car", "the nearest person"
  negatives (need verified missing_basis): "train" (DensePASS has no rail
  class in 13-space; annotator must still confirm absence per image).

GT masks are recorded as PATHS ONLY (worker_view strips them before inference).
Evaluator resolves them later via eval_view.
"""
from __future__ import annotations
import argparse
import json
import os


DEFAULT_QUERIES = [
    ("car", "category_union", {"map13": 12, "relation": "coarse_compatible"}),
    ("person", "category_union", {"map13": 11, "relation": "coarse_compatible"}),
    ("road", "category_union", {"map13": 0, "relation": "exact"}),
    ("vegetation", "category_union", {"map13": 8, "relation": "exact"}),
    ("sky", "category_union", {"map13": 10, "relation": "exact"}),
    ("the leftmost car", "instance_referring", None),
    ("the nearest person", "instance_referring", None),
]

DEFAULT_NEGATIVES = [
    ("train", "taxonomy-missing in densepass13 (id 255): confirm absent per image"),
]


def load_queries(path: str | None):
    if not path:
        return [(t, k, o, False, "") for t, k, o in DEFAULT_QUERIES] + [
            (t, "category_union", None, True, b) for t, b in DEFAULT_NEGATIVES]
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            text = parts[0]
            kind = parts[1] if len(parts) > 1 else "category_union"
            neg = len(parts) > 2 and parts[2].lower() == "negative"
            basis = parts[3] if len(parts) > 3 else ""
            out.append((text, kind, None, neg, basis))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt-dir", default="")
    ap.add_argument("--split", default="dev-calib",
                    choices=["train", "dev-calib", "final-eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--query-set-version", default="v0")
    a = ap.parse_args()

    names = sorted(f for f in os.listdir(a.img_dir)
                   if f.lower().endswith((".jpg", ".png")))
    if a.limit:
        names = names[:a.limit]
    if not names:
        print(f"FATAL: no images in {a.img_dir}")
        return 2

    from PIL import Image
    queries = load_queries(a.queries)
    images, qlist = [], []
    for i, fn in enumerate(names):
        stem = os.path.splitext(fn)[0]
        ipath = os.path.join(a.img_dir, fn)
        with Image.open(ipath) as im:
            w, h = im.size
        iid = f"dpass_{a.split}_{i:06d}"
        images.append({"image_id": iid, "path": ipath, "width": w, "height": h,
                       "dataset": "DensePASS", "split": a.split, "src_file": fn})
        for j, (text, kind, onto, neg, basis) in enumerate(queries):
            q = {"query_id": f"{iid}_q{j:03d}", "image_id": iid,
                 "raw_text": text, "task_type": kind, "language": "en",
                 "query_set_version": a.query_set_version}
            if onto:
                q["ontology"] = onto
            if neg:
                q["negative"] = True
                q["missing_basis"] = basis or "VERIFY-MANUALLY"
            if a.gt_dir:
                q["gt_mask"] = os.path.join(a.gt_dir, stem + f"_q{j:03d}.png")
            qlist.append(q)

    man = {"split": a.split, "query_set_version": a.query_set_version,
           "images": images, "queries": qlist}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    n_neg = sum(1 for q in qlist if q.get("negative"))
    print(f"manifest: {len(images)} images, {len(qlist)} queries "
          f"({n_neg} negatives) -> {a.out}")
    print("NOTE: negative missing_basis must be human-verified per image "
          "(taxonomy-missing != absent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
