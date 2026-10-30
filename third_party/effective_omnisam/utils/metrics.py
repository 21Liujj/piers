import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import torch
import cv2

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def per_class_dice(hist):
    """Dice / F1 per class: 2*TP / (2*TP + FP + FN)."""
    n = hist.shape[0]
    diag = np.diag(hist)
    denom = hist.sum(1) + hist.sum(0)
    return np.divide(2.0 * diag, denom, out=np.full(n, np.nan), where=denom > 0)


def freq_weighted_dice(hist: np.ndarray) -> float:
    """
    Frequency-weighted Dice (0–100):
      sum_c freq_c * dice_c
    where freq_c = gt_pixels_c / total_pixels.
    """
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return float("nan")
    dices = per_class_dice(hist)  # (K,)
    freq = hist.sum(1) / total
    return float(np.nansum(freq * dices) * 100.0)


def _seg_to_boundary(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    Convert a semantic mask (H,W) to a binary boundary map.
    This is class-agnostic boundary (union of all class boundaries).
    """
    m = mask.astype(np.int32)
    valid = (m != int(ignore_index))
    # Mark invalid as a large negative label to avoid creating boundaries with neighbors.
    m2 = m.copy()
    m2[~valid] = -100000
    # Boundary where label changes with 4-neighborhood.
    b = np.zeros_like(m2, dtype=np.uint8)
    b[1:, :] |= (m2[1:, :] != m2[:-1, :]).astype(np.uint8)
    b[:-1, :] |= (m2[:-1, :] != m2[1:, :]).astype(np.uint8)
    b[:, 1:] |= (m2[:, 1:] != m2[:, :-1]).astype(np.uint8)
    b[:, :-1] |= (m2[:, :-1] != m2[:, 1:]).astype(np.uint8)
    b[~valid] = 0
    return b


def boundary_f_score(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_index: int = 255,
    tolerance: int = 2,
) -> float:
    """
    Boundary F-score (BF1, 0–100) for semantic segmentation (class-agnostic).
    Uses morphological dilation with pixel tolerance.
    """
    pred_b = _seg_to_boundary(pred, ignore_index=ignore_index)
    gt_b = _seg_to_boundary(gt, ignore_index=ignore_index)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 100.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    tol = max(0, int(tolerance))
    if tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        pred_d = cv2.dilate(pred_b, k, iterations=1)
        gt_d = cv2.dilate(gt_b, k, iterations=1)
    else:
        pred_d, gt_d = pred_b, gt_b
    # precision: pred boundary matched by gt (within tol)
    tp_p = float((pred_b & gt_d).sum())
    fp = float(pred_b.sum() - tp_p)
    # recall: gt boundary matched by pred (within tol)
    tp_r = float((gt_b & pred_d).sum())
    fn = float(gt_b.sum() - tp_r)
    tp = 0.5 * (tp_p + tp_r)
    denom = (2.0 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return float((2.0 * tp / denom) * 100.0)


def scores_from_hist(hist: np.ndarray) -> dict:
    """
    Standard semantic segmentation metrics from confusion matrix hist[i,j] = |true i, pred j|.

    Returns percentages (0–100) for: mIoU, mAcc, PixAcc, FWIoU, mDice.
    """
    hist = np.asarray(hist, dtype=np.float64)
    n = hist.shape[0]
    total = hist.sum()
    if total <= 0:
        return {
            "mIoU": float("nan"),
            "mAcc": float("nan"),
            "PixAcc": float("nan"),
            "FWIoU": float("nan"),
            "mDice": float("nan"),
        }

    diag = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - diag
    ious = np.divide(diag, union, out=np.full(n, np.nan), where=union > 0)
    miou = float(np.nanmean(ious) * 100.0)

    row_sum = hist.sum(1)
    accs = np.divide(diag, row_sum, out=np.full(n, np.nan), where=row_sum > 0)
    macc = float(np.nanmean(accs) * 100.0)

    pix_acc = float(diag.sum() / total * 100.0)

    freq = row_sum / total
    fwiou = float(np.nansum(freq * ious) * 100.0)

    dices = per_class_dice(hist)
    mdice = float(np.nanmean(dices) * 100.0)
    fwdice = freq_weighted_dice(hist)

    return {
        "mIoU": miou,
        "mAcc": macc,
        "PixAcc": pix_acc,
        "FWIoU": fwiou,
        "mDice": mdice,
        "FWDice": fwdice,
    }
