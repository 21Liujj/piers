"""Innovation-1 forward: ERP-level semantic_to_text guidance. R2.

Takes T1 text predictions (per-FoV, remapped to ERP via pipeline.run_erp_text)
plus a frozen E0 semantic cache (class probs on the same ERP grid), and
re-ranks / filters text candidates by semantic support.

Rules (deterministic v1, mirrors collaboration/rules.py single-image logic):
- support = mean P(class) over candidate's valid ERP pixels for mapped class.
- soft rerank: score' = score * (1 + alpha * support); always kept.
- hard filter (separate ablation): drop if support < thr, UNLESS relation
  is unmapped / unknown category / ambiguous -> keep original + reason.
- NEVER replace an instance mask with the whole class mask.

Cache contract: semantic cache NPZ/PNG keyed by image_id + class_space
version + ckpt_sha256 (see manifest cache-key rule). This module only reads.
"""
from __future__ import annotations
import numpy as np


def erp_support(mask: np.ndarray, prob: np.ndarray, valid: np.ndarray) -> float:
    """Mean class prob over candidate's valid ERP pixels."""
    m = (np.asarray(mask) > 0) & (np.asarray(valid) > 0)
    if m.sum() == 0:
        return 0.0
    return float(np.asarray(prob)[m].mean())


def guide_query(predictions: list, probs: dict, alpha: float = 0.5,
                hard_thr: float | None = None):
    """Re-rank one query's T1 predictions using the semantic cache.

    predictions: list of dict {id, score|detector_score, mask (ERP HxW bool),
      valid (ERP HxW bool), class, relation}.
    probs: {class_name_or_id: ERP HxW prob}.
    Returns (guided, log). Input list is never mutated.
    """
    out, log = [], []
    for p in predictions:
        score = float(p.get("score", p.get("detector_score", 0.0)))
        rel = p.get("relation", "unmapped")
        cls = p.get("class")
        if rel == "unmapped" or cls is None or cls not in probs:
            out.append({**p, "score_guided": score, "kept": True})
            log.append({"id": p.get("id"), "action": "keep",
                        "reason": "unmapped/no-prob fallback"})
            continue
        s = erp_support(p["mask"], probs[cls], p["valid"])
        sg = score * (1.0 + alpha * s)
        if hard_thr is not None and s < hard_thr:
            out.append({**p, "score_guided": sg, "kept": False,
                        "support": s})
            log.append({"id": p.get("id"), "action": "drop",
                        "reason": f"support {s:.3f} < {hard_thr}"})
        else:
            out.append({**p, "score_guided": sg, "kept": True,
                        "support": s})
            log.append({"id": p.get("id"), "action": "keep",
                        "reason": f"support {s:.3f}"})
    return out, log


def apply_mode(predictions: list, probs: dict, mode: str,
               alpha: float = 0.5, hard_thr: float | None = 0.15):
    """Collab switch: none | semantic_to_text (+hard-filter ablation).

    mode='none' returns predictions untouched (with score_guided=score).
    """
    if mode == "none":
        out = [{**p, "score_guided": float(p.get("score", p.get("detector_score", 0.0))),
                "kept": True} for p in predictions]
        return out, [{"id": p.get("id"), "action": "keep",
                      "reason": "collab off"} for p in predictions]
    if mode == "semantic_to_text":
        return guide_query(predictions, probs, alpha=alpha, hard_thr=None)
    if mode == "semantic_to_text_hard":
        return guide_query(predictions, probs, alpha=alpha, hard_thr=hard_thr)
    raise ValueError(f"unknown semantic guide mode {mode}")
