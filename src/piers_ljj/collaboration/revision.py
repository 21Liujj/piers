"""Innovation-1 backward: reliable text masks -> versioned pseudo-label revision. R2.

Produces a *revision patch* that the inherited Effective-Omnisam chain
(pl_generator -> pl_merge -> trainer) consumes WITHOUT modification:

patch NPZ keys (per image, ERP grid HxW):
  revise_mask   bool   True where the text mask writes target_cls
  ignore_mask   bool   True where uncertain edges are set to ignore_id
  target_cls    int    class id in the SAME class space as pseudo (e.g. densepass13)
  quality       float  candidate quality that passed qual_thr
  query_id      str    source query (provenance)
  reason        str    why each pixel class was changed

Merge rule applied downstream (unchanged old code path):
  new_pseudo[revise_mask] = target_cls ; new_pseudo[ignore_mask] = ignore_id.
Non-detected areas NEVER forced to background: everything outside the two
masks keeps the original pseudo-label. Text branch never sees target GT.

Aug alignment note: caller must warp revise/ignore masks with the SAME
geometric transform (crop/flip/resize) as the training augmentation; this
module stores masks in ERP coords + records the FoV/aug params so the
trainer-side warp is exact.
"""
from __future__ import annotations
import numpy as np


def erode(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Binary erosion by square kernel (numpy-only, CPU-safe)."""
    m = np.asarray(mask, bool)
    for _ in range(width):
        padded = np.pad(m, 1)
        neigh = (padded[:-2, :-2] & padded[:-2, 1:-1] & padded[:-2, 2:] &
                 padded[1:-1, :-2] & padded[1:-1, 1:-1] & padded[1:-1, 2:] &
                 padded[2:, :-2] & padded[2:, 1:-1] & padded[2:, 2:])
        m = neigh
    return m


def build_revision(pseudo_shape: tuple, masks: list, target_cls: int,
                   qual_thr: float = 0.8, edge_width: int = 2,
                   query_id: str = "", reason: str = "") -> dict:
    """Build one image revision patch from reliable text masks.

    masks: list of {mask (HxW bool, ERP grid), quality float}.
    Only quality>=thr pixels are written; eroded edges -> ignore.
    Overlapping low-quality / conflicting candidates fall back to keep.
    """
    H, W = pseudo_shape
    revise = np.zeros((H, W), bool)
    ignore = np.zeros((H, W), bool)
    stats = {"written": 0, "ignored_edge": 0, "skipped_lowq": 0, "conflict": 0}
    for m in masks:
        mm = np.asarray(m["mask"], bool)
        assert mm.shape == (H, W), f"mask {mm.shape} != pseudo {H, W}"
        if float(m.get("quality", 0.0)) < qual_thr:
            stats["skipped_lowq"] += int(mm.sum())
            continue
        inner = erode(mm, edge_width)
        edge = mm & (~inner)
        overlap = revise & inner
        if overlap.any():
            # conflict: two reliable masks claim same pixel for one class —
            # keep first, mark overlap as conflict (trainer keeps original).
            stats["conflict"] += int(overlap.sum())
            inner = inner & (~revise)
        revise |= inner
        ignore |= (edge & (~revise))
    stats["written"] = int(revise.sum())
    stats["ignored_edge"] = int(ignore.sum())
    return {"revise_mask": revise, "ignore_mask": ignore,
            "target_cls": int(target_cls), "query_id": query_id,
            "reason": reason or f"qual>={qual_thr},edge={edge_width}",
            "stats": stats}


def save_revision(path: str, patch: dict):
    """Save patch NPZ (versioned pseudo-label input for the old chain)."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, revise_mask=patch["revise_mask"],
                        ignore_mask=patch["ignore_mask"],
                        target_cls=np.int64(patch["target_cls"]),
                        query_id=np.array(patch["query_id"], dtype=object),
                        reason=np.array(patch["reason"], dtype=object))
    return path


def load_revision(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {"revise_mask": d["revise_mask"].astype(bool),
            "ignore_mask": d["ignore_mask"].astype(bool),
            "target_cls": int(d["target_cls"]),
            "query_id": str(d["query_id"]),
            "reason": str(d["reason"])}
