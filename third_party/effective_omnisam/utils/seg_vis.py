"""13-class Cityscapes-style palette for trainId 0..12 (ignore 255 -> black)."""
from __future__ import annotations

import numpy as np

# Cityscapes trainId colors (first 13 semantic classes used in DensePASS13 / WildPASS13)
CS13_PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [0, 0, 142],
    ],
    dtype=np.uint8,
)


def label_ids_to_rgb(mask_hw: np.ndarray, num_classes: int = 13, ignore_index: int = 255) -> np.ndarray:
    """mask_hw int; H×W×3 uint8, ignore -> black."""
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    m = mask_hw.astype(np.int64)
    valid = (m >= 0) & (m < num_classes)
    out[valid] = CS13_PALETTE[m[valid]]
    if ignore_index is not None:
        out[m == int(ignore_index)] = 0
    return out


def overlay_rgb_on_label(rgb_hwc: np.ndarray, label_rgb_hwc: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """rgb and label color both uint8 H×W×3."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return np.clip(rgb_hwc.astype(np.float32) * (1.0 - a) + label_rgb_hwc.astype(np.float32) * a, 0, 255).astype(np.uint8)
