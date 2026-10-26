"""R3 position-constrained memory policy (works on metadata; plugs into MemoryBank).

M1 dedup-only: within same ERP spatial bucket / overlap, keep reliable representative.
M2 dedup+priority: reserve quota for small/low-freq/boundary/uncertain, then fill.
M3 +neighborhood: rank history by adjacency/overlap to current FoV + global context.

Budget: same actual token budget vs baselines (uniform, topk). Counts honest:
we report kept/participating tokens; if impl only zeros dense tensor, report no speedup.
"""
from __future__ import annotations
import numpy as np


def erp_bucket(x: float, y: float, W: int, H: int, bw: int = 128, bh: int = 128):
    return (int(x // bw), int(y // bh), W // bw + 1)


def dedup_positions(items, overlap_fn=None, keep: str = "reliable"):
    """items: list of dict {id, bucket/footprint, stable:bool, compat, score}.
    Returns (kept, dropped). Only stable+compatible candidates deduped.
    """
    seen = {}
    kept, dropped = [], []
    for it in items:
        b = it.get("bucket")
        if not it.get("stable", False) or b in (None,):
            kept.append(it)
            continue
        if b in seen:
            prev = seen[b]
            compat = True if overlap_fn is None else bool(overlap_fn(prev, it))
            if compat and it.get("compat", True) and prev.get("compat", True):
                # keep higher-score representative, never average features blindly
                if it.get("score", 0) > prev.get("score", 0):
                    dropped.append(prev)
                    seen[b] = it
                    kept = [k for k in kept if k["id"] != prev["id"]] + [it]
                else:
                    dropped.append(it)
            else:
                kept.append(it)  # conflict -> keep both
        else:
            seen[b] = it
            kept.append(it)
    return kept, dropped


def allocate_with_priority(items, budget: int, priority_fn):
    """Reserve-then-fill under fixed budget. priority_fn(item)->(is_priority, rank)."""
    assert budget >= 0
    scored = [(priority_fn(it), it) for it in items]
    pri = sorted([it for (p, _), it in zip(scored, items) if p[0]], key=lambda it: priority_fn(it)[1])
    rest = sorted([it for (p, _), it in zip(scored, items) if not p[0]], key=lambda it: priority_fn(it)[1])
    chosen = (pri + rest)[:budget]
    return chosen, {"budget": budget, "kept": len(chosen), "total": len(items)}


def neighborhood_rank(items, cur_footprint, adjacency_fn):
    """Sort history by adjacency/overlap to current FoV; keep global context tail."""
    scored = sorted(items, key=lambda it: adjacency_fn(it, cur_footprint), reverse=True)
    return scored
