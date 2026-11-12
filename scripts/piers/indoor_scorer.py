"""SPAN8 mIoU scorer (CPU). Reads predictions + GT, never touches model code."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]
NUM_CLASSES = 8
IGNORE = 255


def confusion(pred: np.ndarray, gt: np.ndarray, num_classes=NUM_CLASSES):
    """Return the confusion matrix [num_classes, num_classes+1] (last col = ignore)."""
    C = num_classes
    cm = np.zeros((C, C + 1), dtype=np.int64)
    valid = (gt != IGNORE) & (gt >= 0) & (gt < C)
    if valid.any():
        idx = gt[valid].astype(np.int64) * (C + 1) + pred[valid].astype(np.int64)
        cm += np.bincount(idx, minlength=C * (C + 1)).reshape(C, C + 1)
    return cm


def miou_from_confusion(cms) -> dict:
    """Aggregate per-image confusion matrices into mIoU / mAcc / per-class IoU."""
    cm = np.sum(cms, axis=0)
    C = cm.shape[0]
    ious = []
    per_class = {}
    for c in range(C):
        inter = cm[c, c]
        union = cm[c, :].sum() + cm[:, c].sum() - inter
        iou = float(inter / union) if union > 0 else float("nan")
        per_class[SPAN8[c]] = iou
        if union > 0:
            ious.append(iou)
    valid_px = cm[:, :C].sum()
    acc = float(np.trace(cm[:, :C]) / valid_px) if valid_px else float("nan")
    return {
        "miou": float(np.mean(ious)) if ious else float("nan"),
        "mAcc": acc,
        "per_class_iou": per_class,
        "n_images": len(cms),
        "valid_pixels": int(valid_px),
        "per_class_support": {SPAN8[c]: int(cm[:, c].sum()) for c in range(C)},
    }


def score_predictions(pred_dir, manifest_path, label_root, area_key="area",
                      num_classes=NUM_CLASSES, out_json=None):
    """pred_dir contains one .npy/.png per image id (prediction as class ids).

    label_root/<area>/<label path from manifest>
    """
    root = Path(label_root)
    recs = [json.loads(l) for l in open(manifest_path, encoding="utf-8") if l.strip()]
    cms = []
    per_image = []
    for r in recs:
        pid = r["id"]
        pp = Path(pred_dir) / (pid + ".npy")
        if not pp.exists():
            pp = Path(pred_dir) / (pid + ".png")
        if not pp.exists():
            continue
        if pp.suffix == ".npy":
            pred = np.load(pp)
        else:
            pred = np.array(Image.open(pp))
        gt = np.array(Image.open(root / r["area"] / r["label"]))
        if gt.ndim == 3:
            gt = gt[..., 0]
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch {pid}: {pred.shape} vs {gt.shape}")
        cm = confusion(pred, gt, num_classes)
        cms.append(cm)
        inter = np.diag(cm)[:num_classes]
        union = cm.sum(1)[:num_classes] + cm.sum(0)[:num_classes] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        per_image.append({"id": pid, "area": r["area"],
                          "miou": float(np.nanmean(iou)) if np.isfinite(iou).any() else None})
    res = miou_from_confusion(cms)
    res["per_image"] = per_image
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label-root", default="/root/autodl-tmp/data_extra/SPAN8_PREP")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = score_predictions(a.pred_dir, a.manifest, a.label_root, out_json=a.out)
    r.pop("per_image", None)
    print(json.dumps(r, indent=2))
