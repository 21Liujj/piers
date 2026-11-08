"""Innovation-1 reliability gate (PPT slide 9): tri-state semantic/text consistency.

Requirement being implemented
-----------------------------
PPT slide 9: "设计可靠性门控——预测一致时双向增强（语义辅助文本减少误检，
文本掩码修正语义伪标签），不一致时各自独立回退，避免错误传播。"

The previous implementation (`text_guide.apply_mode`) used a HARD FILTER:
a candidate whose semantic support was below `hard_thr` was DELETED. That
violates the requirement, because a semantic model MISSING a true object
must not veto a text detector that correctly found it.

This module replaces that with an explicit three-state gate evaluated
PER CANDIDATE:

  agree     -> both branches agree (semantic support high, teacher confident).
                Cooperation is allowed: semantic support boosts the candidate
                score, and the text mask may revise the pseudo-label.
  conflict  -> both branches positively disagree (semantic support high but
                the teacher is confident about a DIFFERENT class, or the
                candidate sits on a confidently different class).
                Cooperation is DISABLED for that region/candidate; each
                branch keeps its own output (independent fallback).
  uncertain -> evidence is insufficient (semantic support low / teacher
                unconfident / class not mappable / open-vocabulary query).
                Cooperation is DISABLED, but the text candidate is KEPT
                (never vetoed) and the independent text branch decides.

Key design points
-----------------
* The gate NEVER deletes a text candidate on semantic grounds alone. A
  candidate can only be dropped by the text branch's OWN confidence rule.
* A missing semantic class (low support) is treated as "uncertain", which
  falls back to the independent text branch -- never as "conflict".
* `agree` requires POSITIVE agreement, so a semantic miss cannot create
  false agreement either.
* All state transitions are logged so the gate's behaviour is auditable:
  gate pass rate, conflict rate, fallback rate, and per-class distribution.

Status vocabulary (kept short and stable for downstream aggregation):
    agree | conflict | uncertain
"""

from __future__ import annotations

import numpy as np

# Gate states
AGREE = "agree"
CONFLICT = "conflict"
UNCERTAIN = "uncertain"

# Relations that carry a usable class correspondence.
MAPPED_RELATIONS = frozenset({"exact", "coarse_compatible"})


def _confidence(prob: np.ndarray) -> np.ndarray:
    """Teacher confidence per pixel = max class probability."""
    arr = np.asarray(prob, np.float32)
    return arr.max(axis=0) if arr.ndim == 3 else arr


def _as_prob_array(probs) -> np.ndarray:
    """Accept either a (C,H,W) array or a {class_id: (H,W)} mapping.

    `FrozenSemanticCache.image()` returns a DICT of per-class 2D maps, while
    some call sites pass a stacked (C,H,W) array. Both must work, otherwise
    the gate silently reads the wrong axis (or crashes on `shape[0]`).
    """
    if isinstance(probs, dict):
        if not probs:
            return np.zeros((0,), np.float32)
        keys = sorted(probs.keys())
        return np.stack([np.asarray(probs[k], np.float32) for k in keys], axis=0)
    arr = np.asarray(probs, np.float32)
    return arr


def _class_prob(probs, cls: int) -> np.ndarray:
    """Per-pixel probability of class `cls` from either representation."""
    if isinstance(probs, dict):
        if cls not in probs:
            return np.zeros((1,), np.float32)
        return np.asarray(probs[cls], np.float32)
    arr = np.asarray(probs, np.float32)
    if cls < 0 or cls >= arr.shape[0]:
        return np.zeros((1,), np.float32)
    return arr[cls]


def candidate_support(mask: np.ndarray, prob: np.ndarray,
                      valid: np.ndarray) -> float:
    """Mean probability of the mapped class inside the candidate's valid pixels."""
    m = (np.asarray(mask) > 0) & (np.asarray(valid) > 0)
    if not m.any():
        return 0.0
    return float(np.asarray(prob)[m].mean())


def teacher_top_class(teacher_probs, mask: np.ndarray,
                      valid: np.ndarray) -> tuple[int, float]:
    """(class id, mean prob) of the teacher's argmax over the candidate region."""
    m = (np.asarray(mask) > 0) & (np.asarray(valid) > 0)
    if not m.any():
        return -1, 0.0
    arr = _as_prob_array(teacher_probs)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return -1, 0.0
    p = arr[:, m]
    mean = p.mean(axis=1)
    return int(mean.argmax()), float(mean.max())


def evaluate_candidate(cand: dict, teacher_probs: np.ndarray,
                       valid: np.ndarray, *,
                       support_agree: float = 0.5,
                       teacher_conf_agree: float = 0.5,
                       teacher_conf_conflict: float = 0.8) -> dict:
    """Classify one candidate into agree / conflict / uncertain.

    `cand` must carry: mask (bool HxW), class (int|None), relation (str).
    `teacher_probs` is (C,H,W) class probabilities of the frozen teacher.

    Decision rule (deterministic, no learned parameters):
      * not mappable (open vocabulary / unmapped relation) -> uncertain
      * teacher top class over the candidate != mapped class:
          - teacher very confident about a different class -> conflict
          - otherwise                               -> uncertain
      * teacher top class == mapped class:
          - support >= support_agree AND teacher conf >= teacher_conf_agree
              -> agree
          - support low but teacher confident about the mapped class
              -> uncertain (semantic support is weak; do not boost, do not
                 veto; independent text branch decides)
    """
    rel = cand.get("relation", "unmapped")
    cls = cand.get("class")
    out = {
        "id": cand.get("id"),
        "state": UNCERTAIN,
        "reason": "",
        "support": 0.0,
        "teacher_class": -1,
        "teacher_conf": 0.0,
        "class": cls,
        "relation": rel,
    }
    if rel not in MAPPED_RELATIONS or cls is None:
        out["reason"] = "unmapped_or_open_vocabulary"
        return out
    cls = int(cls)
    mask = np.asarray(cand["mask"], bool)
    valid = np.asarray(valid, bool)
    n_classes = _as_prob_array(teacher_probs).shape[0]
    if cls < 0 or cls >= n_classes:
        out["reason"] = "class_out_of_teacher_range"
        return out

    support = candidate_support(mask, _class_prob(teacher_probs, cls), valid)
    tcls, tconf = teacher_top_class(teacher_probs, mask, valid)
    out["support"] = support
    out["teacher_class"] = tcls
    out["teacher_conf"] = tconf

    if tcls < 0:
        out["reason"] = "empty_candidate_region"
        return out

    if tcls != cls:
        # Teacher points at a different class here.
        if tconf >= teacher_conf_conflict:
            out["state"] = CONFLICT
            out["reason"] = (f"teacher_confident_different_class({tcls}@{tconf:.3f})")
        else:
            out["state"] = UNCERTAIN
            out["reason"] = f"teacher_weak_different_class({tcls}@{tconf:.3f})"
        return out

    # Teacher agrees on the class.
    if support >= support_agree and tconf >= teacher_conf_agree:
        out["state"] = AGREE
        out["reason"] = f"agree(support={support:.3f},conf={tconf:.3f})"
    else:
        out["state"] = UNCERTAIN
        out["reason"] = (f"agree_class_but_weak_evidence"
                         f"(support={support:.3f},conf={tconf:.3f})")
    return out


def gate_candidates(candidates: list, teacher_probs: np.ndarray, valids: list,
                    *, support_agree: float = 0.5,
                    teacher_conf_agree: float = 0.5,
                    teacher_conf_conflict: float = 0.8,
                    boost_alpha: float = 0.5) -> tuple[list, list]:
    """Apply the tri-state gate to every candidate of one query.

    Returns (gated_candidates, log).

    * `agree`    -> score boosted by semantic support (cooperation).
    * `conflict` -> score UNCHANGED (no semantic influence at all).
    * `uncertain`-> score UNCHANGED (no semantic influence at all).

    In every non-agree state the candidate is KEPT. The text branch's own
    score/threshold remains the only thing allowed to remove a candidate.
    """
    gated, log = [], []
    for cand, valid in zip(candidates, valids):
        verdict = evaluate_candidate(
            cand, teacher_probs, valid,
            support_agree=support_agree,
            teacher_conf_agree=teacher_conf_agree,
            teacher_conf_conflict=teacher_conf_conflict)
        base = float(cand.get("score", cand.get("detector_score", 0.0)))
        if verdict["state"] == AGREE:
            boosted = base * (1.0 + boost_alpha * verdict["support"])
            kept = True
        else:
            boosted = base
            kept = True
        gated.append({
            **cand,
            "score_guided": boosted,
            "score": base,
            "kept": kept,
            "gate_state": verdict["state"],
            "gate_reason": verdict["reason"],
            "support": verdict["support"],
            "teacher_class": verdict["teacher_class"],
            "teacher_conf": verdict["teacher_conf"],
        })
        log.append({
            "id": cand.get("id"),
            "action": "boost" if verdict["state"] == AGREE else "fallback",
            "state": verdict["state"],
            "reason": verdict["reason"],
            "support": verdict["support"],
            "teacher_conf": verdict["teacher_conf"],
            "score_before": base,
            "score_after": boosted,
        })
    return gated, log


def gate_summary(logs: list) -> dict:
    """Aggregate gate statistics over many queries (auditable counters)."""
    total = len(logs)
    counts = {AGREE: 0, CONFLICT: 0, UNCERTAIN: 0}
    per_class: dict[str, dict[str, int]] = {}
    for entry in logs:
        st = entry.get("state", UNCERTAIN)
        counts[st] = counts.get(st, 0) + 1
        cls = str(entry.get("class", "unmapped"))
        per_class.setdefault(cls, {AGREE: 0, CONFLICT: 0, UNCERTAIN: 0})
        per_class[cls][st] = per_class[cls].get(st, 0) + 1
    denom = max(total, 1)
    return {
        "candidates": total,
        "agree": counts[AGREE],
        "conflict": counts[CONFLICT],
        "uncertain": counts[UNCERTAIN],
        "agree_rate": counts[AGREE] / denom,
        "conflict_rate": counts[CONFLICT] / denom,
        "fallback_rate": (counts[CONFLICT] + counts[UNCERTAIN]) / denom,
        "per_class": per_class,
    }
