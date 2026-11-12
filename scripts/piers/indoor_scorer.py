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
    """Return the confusion matrix [num_classes, num_classes+1].

    Column `num_classes` counts "predicted outside the SPAN8 label set" (255).
    Those pixels are wrong for their GT class (they become false negatives)
    but must not create false positives on any other class.  This is the
    standard ignore-prediction protocol and it is what makes the frozen
    street head's structurally-unreachable classes visible as zero recall
    instead of being silently absorbed.
    """
    C = num_classes
    cm = np.zeros((C, C + 1), dtype=np.int64)
    valid = (gt != IGNORE) & (gt >= 0) & (gt < C)
    if valid.any():
        g = gt[valid].astype(np.int64)
        p = pred[valid].astype(np.int64)
        pred_ok = (p >= 0) & (p < C)
        idx = g * (C + 1) + np.where(pred_ok, p, C)
        cm += np.bincount(idx, minlength=C * (C + 1)).reshape(C, C + 1)
    return cm


def miou_from_confusion(cms) -> dict:
    """Aggregate per-image confusion matrices into mIoU / mAcc / per-class IoU.

    Column `num_classes` of the confusion matrix holds the pixels whose GT is
    valid but whose prediction fell outside the SPAN8 label set (only the
    frozen street 13-head can produce those).  Such a pixel is a miss for its
    GT class but is never a false positive on another class.
    """
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
    # correct pixels = diagonal over the 8 SPAN8 classes
    correct = float(np.trace(cm[:, :C]))
    gt_valid = float(cm.sum())
    acc = float(correct / gt_valid) if gt_valid else float("nan")
    return {
        "miou": float(np.mean(ious)) if ious else float("nan"),
        "mAcc": acc,
        "per_class_iou": per_class,
        "n_images": len(cms),
        "gt_valid_pixels": int(gt_valid),
        "correct_pixels": int(correct),
        # per-class GT support = all pixels whose GT is that class
        "per_class_gt_support": {SPAN8[c]: int(cm[:, c].sum()) for c in range(C)},
        # per-class predicted support = all pixels predicted as that class
        "per_class_pred_support": {SPAN8[c]: int(cm[c, :C].sum()) for c in range(C)},
        "predictions_outside_label_set": int(cm[:, C].sum()),
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
    res["manifest"] = str(manifest_path)
    res["label_root"] = str(root)
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
