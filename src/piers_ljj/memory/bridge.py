"""Innovation-2 bridge: ERP position metadata -> inherited MemoryBank/SMR. R3.

piers-ljj owns ONLY:
- per-history-token ERP footprint (image_id, FoV, token grid -> ERP rect)
- overlap/adjacency decisions (which buckets/FoVs are redundant, which are
  neighborhood of the current FoV)
- variant -> old-chain parameter mapping + honest token/timing accounting

All heavy selection runs the INHERITED modules (never reimplemented):
  effective-omnisam-src/model/modules/pano_smb_allocator.py  (FoV quotas)
  effective-omnisam-src/model/modules/sparse_memory_retriever.py (token select)
  effective-omnisam-src/model/memory_bank.py (sparse_mode wiring)

Variant mapping (M suite):
  current            -> sparse_mode as in effective_start (no override)
  nocompress_ref     -> MTC mode=none + SMR mode=none (accuracy/cost reference,
                        token counts NOT comparable; reported separately)
  uniform_budget     -> PanoSMB skipped (equal quotas) + SMR feature_norm topk
  topk_fixed         -> SMR feature_norm, same budget
  pos_dedup          -> SMR feature_norm + piers overlap pre-filter (M1)
  pos_dedup_priority -> SMR boundary_guarded/class_aware via piers masks (M2)
  neighborhood       -> SMR cams/hybrid with piers adjacency order (M3)

Frame-index is NEVER used as ERP position: every token carries its
(image_id, fov_id, erp_rect). Different images/batches never share cache:
cache key = (image_id, memory_id, strategy, budget).
"""
from __future__ import annotations
import time

VARIANT_TO_SMR = {
    "current": {"sparse_mode": None},          # keep effective_start
    "nocompress_ref": {"sparse_mode": "none"},
    "uniform_budget": {"sparse_mode": "feature_norm"},
    "topk_fixed": {"sparse_mode": "feature_norm"},
    "pos_dedup": {"sparse_mode": "feature_norm", "prefilter": "overlap"},
    "pos_dedup_priority": {"sparse_mode": "boundary_guarded",
                           "fallback": "class_aware"},
    "neighborhood": {"sparse_mode": "cams", "fallback": "hybrid"},
}


def token_erp_rect(tok_col: int, tok_row: int, grid_w: int, grid_h: int,
                   fov: dict, erp_w: int, erp_h: int) -> tuple:
    """Token grid cell -> ERP pixel rect (x0,y0,x1,y1), wrap-aware.

    fov: {x_start, y_start, win_w, win_h}. Token grid covers the FoV window.
    Returns possibly two segments if the FoV itself crosses the seam.
    """
    cw, ch = fov["win_w"] / grid_w, fov["win_h"] / grid_h
    lx0, ly0 = tok_col * cw, fov["y_start"] + tok_row * ch
    gx0 = (fov["x_start"] + lx0) % erp_w
    gx1 = gx0 + cw
    if gx1 <= erp_w:
        return (gx0, ly0, gx1, ly0 + ch)
    # wraps: two honest segments
    return ((gx0, ly0, float(erp_w), ly0 + ch),
            (0.0, ly0, gx1 - erp_w, ly0 + ch))


def rect_overlap(a: tuple, b: tuple) -> float:
    """IoU of two ERP rects (single-segment; wrapped handled as two segs by caller)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(ua, 1e-9)


def footprint_overlap(fa, fb) -> float:
    """Max IoU over segment pairs (handles wrapped footprints)."""
    sa = fa if isinstance(fa[0], tuple) else (fa,)
    sb = fb if isinstance(fb[0], tuple) else (fb,)
    return max([rect_overlap(a, b) for a in sa for b in sb] + [0.0])


def plan_selection(items: list, variant: str, budget: int,
                   cur_footprint=None) -> dict:
    """CPU-side plan: which history items survive under a fixed token budget.

    items: [{id, footprint, stable, compat, score, boundary, rare}].
    Returns {kept_ids, dropped_ids, smr_params, report}.
    Real tensor gather runs in the omni process via inherited SMR.
    """
    t0 = time.perf_counter()
    if variant not in VARIANT_TO_SMR:
        raise ValueError(f"unknown memory variant {variant}")
    if variant == "nocompress_ref":
        kept = [it["id"] for it in items]
        return {"kept_ids": kept, "dropped_ids": [],
                "smr_params": VARIANT_TO_SMR[variant],
                "report": {"kept": len(kept), "participating": len(kept),
                           "note": "reference; token count not comparable",
                           "plan_ms": (time.perf_counter() - t0) * 1000}}
    pool = list(items)
    dropped = []
    if variant == "pos_dedup" or variant == "pos_dedup_priority":
        # M1: same-bucket/overlap dedup, stable+compatible only, keep best score
        seen: dict = {}
        kept_pool = []
        for it in sorted(pool, key=lambda x: -x.get("score", 0.0)):
            b = it.get("bucket")
            if not it.get("stable", False) or b is None:
                kept_pool.append(it)
                continue
            if b in seen and it.get("compat", True) and seen[b].get("compat", True):
                if footprint_overlap(it["footprint"], seen[b]["footprint"]) > 0.5:
                    dropped.append(it["id"])
                    continue
            seen[b] = it
            kept_pool.append(it)
        pool = kept_pool
    if variant == "pos_dedup_priority":
        # M2: reserve quota for boundary/rare, fill rest by score
        pri = [it for it in pool if it.get("boundary") or it.get("rare")]
        rest = [it for it in pool if it not in pri]
        pri = sorted(pri, key=lambda x: -x.get("score", 0.0))
        rest = sorted(rest, key=lambda x: -x.get("score", 0.0))
        pool = pri + rest
    if variant == "neighborhood" and cur_footprint is not None:
        # M3: adjacency to current FoV first, keep global tail
        pool = sorted(pool,
                      key=lambda it: footprint_overlap(it["footprint"], cur_footprint),
                      reverse=True)
    elif variant != "pos_dedup_priority":
        # pos_dedup_priority already ordered as pri+rest; do not re-sort.
        pool = sorted(pool, key=lambda x: -x.get("score", 0.0))
    kept = [it["id"] for it in pool[:budget]]
    dropped += [it["id"] for it in pool[budget:]]
    return {"kept_ids": kept, "dropped_ids": dropped,
            "smr_params": VARIANT_TO_SMR[variant],
            "report": {"kept": len(kept), "participating": len(kept),
                       "budget": budget, "total": len(items),
                       "plan_ms": (time.perf_counter() - t0) * 1000}}
