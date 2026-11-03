"""Text-branch evaluation: predictions.json (+masks.npz) vs GT masks.

Usage:
  python m_eval_text.py --run runs/T1-100 --manifest configs/piers/manifest.dev-calib-100.json \
      [--out runs/T1-100/metrics_text.json]

Reports per task_type (category_union / instance_referring), macro/micro IoU,
correct_reject, false_positive. GT is read from the manifest (eval side only);
the worker never sees it.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def load_masks(run_dir):
    """Return {query_id: HxW bool mask} from masks.npz (or empty)."""
    p = os.path.join(run_dir, "masks.npz")
    if not os.path.exists(p):
        return {}
    d = np.load(p)
    qids = [str(x) for x in d["query_ids"]]
    masks = d["masks"]
    return {q: masks[i].astype(bool) for i, q in enumerate(qids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with predictions.json/masks.npz")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--area-thr", dest="area_thr", type=int, default=16)
    a = ap.parse_args()

    man = json.load(open(a.manifest, encoding="utf-8"))
    gt_map = {q["query_id"]: q.get("gt_mask") for q in man["queries"]}
    task_map = {q["query_id"]: q.get("task_type", "unknown") for q in man["queries"]}

    preds = json.load(open(os.path.join(a.run, "predictions.json"), encoding="utf-8"))
    pmask = load_masks(a.run)

    per_task = {}
    overall = []
    for e in preds:
        qid = e["query_id"]
        gt_path = gt_map.get(qid)
        if not gt_path:
            continue
        g = np.array(Image.open(gt_path).convert("L")) > 0
        p = pmask.get(qid)
        if p is None:
            p = np.zeros_like(g, dtype=bool)
        if p.shape != g.shape:
            # ERP-global mask should match image size; resize defensively
            p = np.array(Image.fromarray(p.astype(np.uint8) * 255).resize(
                (g.shape[1], g.shape[0]), Image.NEAREST)) > 0

        if not g.any() and not p.any():
            rec = {"iou": None, "correct_reject": True}
        elif not g.any() and p.any():
            rec = {"iou": 0.0, "false_positive": bool(p.sum() >= a.area_thr),
                   "fp_area": int(p.sum())}
        else:
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            rec = {"iou": float(inter) / float(max(union, 1)),
                   "dice": float(2 * inter) / float(max(p.sum() + g.sum(), 1)),
                   "recall": float(inter) / float(max(g.sum(), 1)),
                   "pred_area": int(p.sum()), "gt_area": int(g.sum())}

        t = task_map.get(qid, "unknown")
        per_task.setdefault(t, []).append(rec)
        overall.append(rec)

    def agg(recs):
        vals = [r["iou"] for r in recs if r.get("iou") is not None]
        inter = sum(r.get("pred_area", 0) for r in recs if "pred_area" in r)
        return {"n": len(recs),
                "n_scored": len(vals),
                "macro_iou": float(np.mean(vals)) if vals else None,
                "correct_reject": sum(1 for r in recs if r.get("correct_reject")),
                "false_positive": sum(1 for r in recs if r.get("false_positive")),
                "empty_pred_on_nonempty": sum(
                    1 for r in recs if r.get("iou") is not None and r.get("gt_area", 0) > 0
                    and r.get("pred_area", 0) == 0)}

    out = {"run": a.run, "task": "text", "synthetic": False,
           "overall": agg(overall),
           "by_task_type": {k: agg(v) for k, v in sorted(per_task.items())},
           "per_query": [{"query_id": e["query_id"],
                          "task_type": task_map.get(e["query_id"]),
                          **{k: v for k, v in r.items()}}
                         for e, r in zip(preds, overall)
                         if gt_map.get(e["query_id"])]}
    o = a.out or os.path.join(a.run, "metrics_text.json")
    with open(o, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"overall": out["overall"],
                      "by_task_type": out["by_task_type"]}, indent=2))
    print(f"-> {o}")


if __name__ == "__main__":
    main()
