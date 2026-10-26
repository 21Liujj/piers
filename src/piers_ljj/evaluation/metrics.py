"""R5 evaluation: UDA semantic + text target, honest aggregation."""
from __future__ import annotations
import numpy as np

IGNORE = 255


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, n_cls: int, ignore_id: int = IGNORE):
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    m = (gt != ignore_id)
    p, g = pred[m].astype(int), gt[m].astype(int)
    valid = (p >= 0) & (p < n_cls) & (g >= 0) & (g < n_cls)
    np.add.at(cm, (g[valid], p[valid]), 1)
    return cm


def scores_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    denom_iou = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, np.maximum(denom_iou, 1), out=np.zeros_like(tp), where=denom_iou > 0)
    acc = np.divide(tp, np.maximum(cm.sum(1), 1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    present = cm.sum(1) > 0
    return {"miou": float(iou[present].mean()) if present.any() else 0.0,
            "per_class_iou": iou.tolist(), "macc": float(acc[present].mean()) if present.any() else 0.0,
            "pixacc": float(tp.sum() / max(cm.sum(), 1))}


DENSEPASS19 = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
               "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
               "truck", "bus", "train", "motorcycle", "bicycle"]
MAP19_TO_13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 255, 12, 12]


def map19_to_13(ids: np.ndarray) -> np.ndarray:
    lut = np.array(MAP19_TO_13, dtype=np.int64)
    return lut[np.asarray(ids)]


def binary_scores(pred: np.ndarray, gt: np.ndarray | None, area_thr: int = 16):
    """Text task: IoU/Dice per query; empty-pred on non-empty GT counts 0;
    both-empty -> null + correct_reject flag (never counted as perfect IoU)."""
    if gt is None:
        return {"iou": None, "dice": None, "reason": "no verifiable GT (qualitative only)"}
    p = (np.asarray(pred) > 0)
    g = (np.asarray(gt) > 0)
    if not g.any() and not p.any():
        return {"iou": None, "dice": None, "correct_reject": True}
    if not g.any() and p.any():
        return {"iou": 0.0, "dice": 0.0, "false_positive": bool(p.sum() >= area_thr),
                "fp_area": int(p.sum())}
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = float(inter) / float(max(union, 1))
    dice = float(2 * inter) / float(max(p.sum() + g.sum(), 1))
    return {"iou": iou, "dice": dice, "recall": float(inter) / float(max(g.sum(), 1))}
