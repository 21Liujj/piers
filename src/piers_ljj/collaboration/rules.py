"""R2 bidirectional collaboration (deterministic, configurable rules v1).

A. semantic_to_text: re-rank / filter text candidates by semantic support.
   support = mean P(class) over candidate's valid pixels for mapped class set.
   - soft rerank: score' = score * (1 + alpha*support)
   - hard filter (separate ablation): drop if support < thr AND mapped relation != unmapped-handled.
   Unknown/unmapped categories or ambiguous multi-instance -> keep original + reason.
   NEVER replace an instance mask with the whole class mask.

B. text_to_pseudo: revise target-domain pseudo-labels with reliable masks.
   Only inside mask; uncertain edges -> ignore(255); conflicts -> fallback keep.
   Non-detected areas NEVER forced to background.
Switches: none | semantic_to_text | text_to_pseudo | bidirectional.
"""
from __future__ import annotations
import numpy as np


def semantic_support(mask: np.ndarray, prob: np.ndarray, valid: np.ndarray) -> float:
    """Mean class probability over candidate's valid pixels."""
    m = (mask > 0) & valid
    if m.sum() == 0:
        return 0.0
    return float(prob[m].mean())


def rerank_candidates(cands, probs: dict, alpha: float = 0.5, hard_thr: float | None = None):
    """cands: list of dict {id, score, mask, valid, class, relation}.
    probs: {class: HxW prob}. Returns (new_cands, log) — originals preserved by caller.
    """
    out, log = [], []
    for c in cands:
        rel = c.get("relation", "unmapped")
        cls = c.get("class")
        if rel == "unmapped" or cls is None or cls not in probs:
            out.append({**c, "score_new": c["score"], "kept": True})
            log.append({"id": c["id"], "action": "keep", "reason": "unmapped/no-prob fallback"})
            continue
        s = semantic_support(np.asarray(c["mask"]), np.asarray(probs[cls]), np.asarray(c["valid"]))
        ns = c["score"] * (1.0 + alpha * s)
        if hard_thr is not None and s < hard_thr:
            out.append({**c, "score_new": ns, "kept": False})
            log.append({"id": c["id"], "action": "drop", "reason": f"support {s:.3f} < {hard_thr}"})
        else:
            out.append({**c, "score_new": ns, "kept": True})
            log.append({"id": c["id"], "action": "keep", "reason": f"support {s:.3f}"})
    return out, log


def revise_pseudo(pseudo: np.ndarray, conf: np.ndarray, masks: list, target_cls: int,
                  qual_thr: float = 0.8, edge_width: int = 2, ignore_id: int = 255):
    """Revise pseudo-labels inside reliable masks.

    pseudo HxW int, conf HxW float, masks: list of {mask bool, quality float}.
    Returns (new_pseudo, stats). Only pixels with quality>=thr are written;
    eroded edges set to ignore; conflicts (overlap of different target classes —
    here single class, so overlapping low-quality) fall back to keep.
    """
    new = pseudo.copy()
    stats = {"written": 0, "ignored_edge": 0, "skipped_lowq": 0}
    for m in masks:
        if m["quality"] < qual_thr:
            stats["skipped_lowq"] += int(m["mask"].sum())
            continue
        try:
            import cv2  # optional; numpy fallback below if missing
            k = np.ones((3, 3), np.uint8)
            er = cv2.erode(m["mask"].astype(np.uint8), k, iterations=edge_width)
            inner = er > 0
        except Exception:
            inner = m["mask"]
        edge = m["mask"] & (~inner)
        new[inner] = target_cls
        new[edge] = ignore_id
        stats["written"] += int(inner.sum())
        stats["ignored_edge"] += int(edge.sum())
    return new, stats
