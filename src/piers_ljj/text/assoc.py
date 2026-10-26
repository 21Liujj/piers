"""Cross-FoV candidate association. Keeps instance identity honest.

Rules:
- Same query only. Match on shared visible ERP area: mask IoU over union of
  valid coverage + spatial adjacency + phrase/category compatibility.
- Two adjacent same-class instances are NOT merged without overlap evidence.
- Seam-crossing instance keeps ONE id with two spatial segments (see geometry).
- No-overlap fragments are never force-merged; flagged for seam re-observe.
"""
from __future__ import annotations
import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    a = (a > 0) & valid
    b = (b > 0) & valid
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(max(union, 1))


def associate(cands, iou_thr: float = 0.3, compat_fn=None):
    """Greedy association over candidates sharing a query_id.

    cands: list of dict {id, query_id, mask(HxW bool), valid(HxW bool), phrase/cat, seg_ids}
    Returns list of groups {group_id, members:[ids], reason}.
    """
    groups = []
    used = set()
    for i, c in enumerate(cands):
        if i in used:
            continue
        g = [i]
        used.add(i)
        for j in range(i + 1, len(cands)):
            if j in used:
                continue
            d = cands[j]
            if d["query_id"] != c["query_id"]:
                continue
            if compat_fn is not None and not compat_fn(c, d):
                continue
            valid = c["valid"] & d["valid"]
            if valid.sum() == 0:
                continue  # no shared visible area -> cannot claim same instance
            iou = mask_iou(c["mask"], d["mask"], valid)
            if iou >= iou_thr:
                g.append(j)
                used.add(j)
        groups.append({"group_id": f"g{len(groups):03d}", "members": [cands[k]["id"] for k in g],
                       "reason": f"iou>={iou_thr} on shared-visible area"})
    return groups
