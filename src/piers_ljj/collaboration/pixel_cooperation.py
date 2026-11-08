"""Innovation-1 bidirectional cooperation at the PIXEL level (PPT slide 9).

The semantic->text direction is handled by `reliability_gate` (candidate level).
This module handles the text->semantic direction: a reliable text mask may
revise the semantic pseudo-label of the corresponding class, but ONLY where
the two branches do not actively disagree.

Why pixel level matters
-----------------------
The earlier implementation only wrote pixels whose teacher hard label already
EQUALLED the text class. On a frozen teacher that changes almost nothing, so
"cooperation" produced a zero-effect result. This module instead:

  * KEEPS the teacher's confident pixels untouched (no needless flips).
  * In the teacher's UNCERTAIN region (low max-prob) a reliable, agreeing
    text mask may write the class -> this is where text masks can actually
    ADD supervision that the semantic model was unsure about.
  * In the teacher's IGNORE region (255) a reliable, agreeing text mask may
    also write the class -> this recovers supervision the teacher discarded.
  * Where the teacher is CONFIDENT about a DIFFERENT class, the text mask is
    BLOCKED (conflict -> independent fallback).

So the observable effect is:
  (a) more (and better) supervised pixels at the same teacher quality, and
  (b) fewer noisy pixels, because text masks override confident teacher
      errors only when the text mask itself is reliable and the teacher is
      NOT confident about a different class.

All quantities are reported so the mechanism can be audited:
written / changed / blocked / recovered pixels, coverage, and the pixel
distribution over the three teacher-confidence regimes.
"""

from __future__ import annotations

import numpy as np

from .reliability_gate import AGREE, CONFLICT, UNCERTAIN

# Teacher confidence regimes.
CONF_HIGH = 0.8      # teacher is confident
CONF_LOW = 0.5       # below this the teacher is considered unreliable


def _max_prob(probs) -> np.ndarray:
    """Teacher confidence per pixel; accepts (C,H,W) array or {cls: (H,W)}."""
    if isinstance(probs, dict):
        if not probs:
            return np.zeros((1,), np.float32)
        keys = sorted(probs.keys())
        return np.stack([np.asarray(probs[k], np.float32) for k in keys], axis=0).max(axis=0)
    arr = np.asarray(probs, np.float32)
    return arr.max(axis=0) if arr.ndim == 3 else arr


def _as_prob_array(probs) -> np.ndarray:
    if isinstance(probs, dict):
        if not probs:
            return np.zeros((0,), np.float32)
        keys = sorted(probs.keys())
        return np.stack([np.asarray(probs[k], np.float32) for k in keys], axis=0)
    return np.asarray(probs, np.float32)


def teacher_regimes(probs: np.ndarray,
                    ignore_mask: np.ndarray | None = None) -> dict:
    """Split ERP pixels into confident / uncertain / unreliable regimes."""
    conf = _max_prob(probs)
    reg = {
        "confident": conf >= CONF_HIGH,
        "uncertain": (conf < CONF_HIGH) & (conf >= CONF_LOW),
        "unreliable": conf < CONF_LOW,
    }
    if ignore_mask is not None:
        reg["ignore"] = np.asarray(ignore_mask, bool)
        reg["uncertain"] &= ~reg["ignore"]
        reg["unreliable"] &= ~reg["ignore"]
    return reg


def _erode(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Binary erosion by a square kernel (numpy only, CPU safe)."""
    m = np.asarray(mask, bool)
    if width <= 0:
        return m.copy()
    for _ in range(int(width)):
        p = np.pad(m, 1)
        neigh = (p[:-2, :-2] & p[:-2, 1:-1] & p[:-2, 2:] &
                 p[1:-1, :-2] & p[1:-1, 1:-1] & p[1:-1, 2:] &
                 p[2:, :-2] & p[2:, 1:-1] & p[2:, 2:])
        m = neigh
    return m


def revise_pseudo_by_text(teacher: np.ndarray,
                          probs: np.ndarray,
                          text_masks: list,
                          *,
                          target_cls: int,
                          qual_thr: float = 0.8,
                          edge_width: int = 2,
                          semantic_support_thr: float = 0.5,
                          allow_confident_write: bool = False,
                          require_gate_agree: bool = True) -> dict:
    """Revise a frozen semantic pseudo-label with reliable text masks.

    Parameters
    ----------
    teacher : (H,W) int teacher hard labels; 255 means ignore.
    probs   : (C,H,W) frozen teacher class probabilities.
    text_masks : list of dicts {mask:(H,W) bool, quality:float, support:float,
                                state:str, class:int|None}.
    target_cls : class id written into the pseudo-label.
    allow_confident_write : if False (default), a pixel whose teacher is
        confident about a DIFFERENT class is never overwritten (conflict
        fallback).
    require_gate_agree : if True (default), only masks whose gate state is
        `agree` may write. Uncertain masks keep the teacher.

    Returns a dict with the revised label plus full statistics.
    """
    teacher = np.asarray(teacher)
    H, W = teacher.shape
    probs_arr = _as_prob_array(probs)
    if probs_arr.ndim != 3:
        raise ValueError("probs must be (C,H,W) or {cls: (H,W)}")
    # NOTE: a {cls: (H,W)} dict must be indexed by the ORIGINAL class id,
    # but the stacked array is indexed by sorted key position. We keep the
    # sorted-key mapping explicit so class ids stay correct.
    if isinstance(probs, dict):
        class_ids = sorted(probs.keys())
    else:
        class_ids = list(range(probs_arr.shape[0]))

    ignore_in = teacher == 255
    reg = teacher_regimes(probs, ignore_in)

    proposal = np.full((H, W), -1, np.int16)
    quality = np.zeros((H, W), np.float32)
    support = np.zeros((H, W), np.float32)
    cross_conflict = np.zeros((H, W), bool)
    edge = np.zeros((H, W), bool)

    stats = {
        "masks_seen": len(text_masks),
        "masks_accepted": 0,
        "skipped_low_quality": 0,
        "skipped_not_agree": 0,
        "skipped_unmapped": 0,
        "skipped_low_support": 0,
    }

    for m in text_masks:
        mask = np.asarray(m.get("mask"), bool)
        if mask.shape != (H, W):
            raise ValueError("Text mask shape does not match the teacher grid")
        q = float(m.get("quality", 0.0))
        s = float(m.get("support", 0.0))
        state = m.get("state", UNCERTAIN)
        cls = m.get("class")
        if cls is None or int(cls) != int(target_cls):
            stats["skipped_unmapped"] += 1
            continue
        if q < qual_thr:
            stats["skipped_low_quality"] += 1
            continue
        if require_gate_agree and state != AGREE:
            stats["skipped_not_agree"] += 1
            continue
        if s < semantic_support_thr:
            stats["skipped_low_support"] += 1
            continue

        inner = _erode(mask, edge_width)
        if not inner.any():
            stats["skipped_low_quality"] += 1
            continue

        if allow_confident_write:
            writable = inner
        else:
            teacher_cls_here = probs_arr.argmax(axis=0)
            # map stacked position -> real class id
            real_cls = np.asarray([class_ids[k] for k in range(len(class_ids))])
            different = (real_cls[teacher_cls_here] != int(target_cls)) & reg["confident"]
            cross_conflict |= different & inner
            writable = inner & ~different

        writable &= proposal < 0
        proposal[writable] = int(target_cls)
        quality[writable] = q
        support[writable] = s
        edge |= mask & ~inner
        stats["masks_accepted"] += 1

    write = proposal >= 0
    revised = teacher.copy()
    revised[write] = proposal[write]

    edge_ignore = edge & ~write & reg["uncertain"] & ~cross_conflict
    revised[edge_ignore] = 255

    changed = write & (teacher != proposal)
    recovered = changed & (teacher == 255)
    in_uncertain = changed & reg["uncertain"]

    stats.update({
        "written_pixels": int(write.sum()),
        "changed_pixels": int(changed.sum()),
        "recovered_ignore_pixels": int(recovered.sum()),
        "uncertain_region_changes": int(in_uncertain.sum()),
        "blocked_conflict_pixels": int(cross_conflict.sum()),
        "edge_ignore_pixels": int(edge_ignore.sum()),
        "coverage": float(write.sum()) / float(H * W),
    })
    return {
        "revised": revised,
        "write_mask": write,
        "changed_mask": changed,
        "conflict_mask": cross_conflict,
        "edge_ignore_mask": edge_ignore,
        "target_labels": proposal,
        "quality": quality,
        "support": support,
        "stats": stats,
    }


def compare_revisions(teacher: np.ndarray, variants: dict,
                      gt: np.ndarray | None = None) -> dict:
    """Compare several revised pseudo-labels against the teacher and (if
    available, evaluation side only) the ground truth.

    `gt` is ONLY used by the independent evaluator; inference never sees it.
    """
    teacher = np.asarray(teacher)
    out = {}
    for name, rev in variants.items():
        rev = np.asarray(rev)
        changed = (rev != teacher)
        rec = {
            "changed_pixels": int(changed.sum()),
            "coverage": float((rev != 255).sum()) / float(teacher.size),
        }
        if gt is not None:
            g = np.asarray(gt)
            eval_mask = (g != 255)
            tp = int(((rev == g) & (rev != 255) & eval_mask).sum())
            fp = int(((rev != 255) & (rev != g) & eval_mask).sum())
            rec["pseudo_precision_on_gt"] = tp / max(tp + fp, 1)
            teacher_tp = int(((teacher == g) & (teacher != 255) & eval_mask).sum())
            teacher_fp = int(((teacher != 255) & (teacher != g) & eval_mask).sum())
            rec["teacher_precision_on_gt"] = teacher_tp / max(teacher_tp + teacher_fp, 1)
            good = int((changed & (rev == g) & eval_mask).sum())
            bad = int((changed & (rev != g) & eval_mask).sum())
            rec["correct_changes"] = good
            rec["wrong_changes"] = bad
            rec["net_correct_changes"] = good - bad
        out[name] = rec
    return out
