"""Tests for the PPT slide-9 reliability gate and pixel-level cooperation."""
import numpy as np
import pytest

from piers_ljj.collaboration.reliability_gate import (
    AGREE, CONFLICT, UNCERTAIN,
    evaluate_candidate, gate_candidates, gate_summary, candidate_support,
)
from piers_ljj.collaboration.pixel_cooperation import (
    revise_pseudo_by_text, teacher_regimes, compare_revisions,
)


def _probs(shape=(16, 16), cls=3, value=0.9, other=0.1, ncls=13):
    p = np.full((ncls,) + shape, other / (ncls - 1), np.float32)
    p[cls] = value
    return p


def _mask(shape=(16, 16), y0=2, x0=2, n=5):
    m = np.zeros(shape, bool)
    m[y0:y0 + n, x0:x0 + n] = True
    return m


def test_candidate_support_mean_prob():
    p = _probs()
    v = np.ones((16, 16), bool)
    assert candidate_support(_mask(), p[3], v) == pytest.approx(0.9)


def test_semantic_miss_is_uncertain_not_conflict():
    """A semantic model that MISSES the object must not veto the text hit."""
    p = _probs(cls=5, value=0.9)      # teacher says "building" here
    v = np.ones((16, 16), bool)
    cand = {"id": "a", "mask": _mask(), "class": 3, "relation": "exact",
            "score": 0.8}
    verdict = evaluate_candidate(cand, p, v)
    # teacher top class != mapped class and teacher is confident -> conflict
    assert verdict["state"] == CONFLICT
    # but the candidate is still kept by the gate
    gated, log = gate_candidates([cand], p, [v])
    assert gated[0]["kept"] is True
    assert gated[0]["score_guided"] == pytest.approx(0.8)   # NOT boosted
    assert log[0]["state"] == CONFLICT


def test_low_support_never_deletes_candidate():
    """Low semantic support -> uncertain -> independent text branch decides."""
    p = _probs(cls=3, value=0.2)      # teacher unsure
    v = np.ones((16, 16), bool)
    cand = {"id": "a", "mask": _mask(), "class": 3, "relation": "exact",
            "score": 0.6}
    gated, log = gate_candidates([cand], p, [v])
    assert gated[0]["kept"] is True
    assert log[0]["state"] == UNCERTAIN
    assert gated[0]["score_guided"] == pytest.approx(0.6)


def test_agree_boosts_score():
    p = _probs(cls=3, value=0.95)
    v = np.ones((16, 16), bool)
    cand = {"id": "a", "mask": _mask(), "class": 3, "relation": "exact",
            "score": 0.5}
    gated, log = gate_candidates([cand], p, [v], boost_alpha=0.5)
    assert log[0]["state"] == AGREE
    assert gated[0]["score_guided"] == pytest.approx(0.5 * (1 + 0.5 * 0.95))
    assert gated[0]["kept"] is True


def test_open_vocabulary_falls_back():
    """tire -> car must NOT be forced; unmapped query keeps text result."""
    p = _probs()
    v = np.ones((16, 16), bool)
    cand = {"id": "a", "mask": _mask(), "class": None, "relation": "unmapped",
            "score": 0.4}
    gated, log = gate_candidates([cand], p, [v])
    assert log[0]["state"] == UNCERTAIN
    assert "open_vocabulary" in log[0]["reason"]
    assert gated[0]["score_guided"] == pytest.approx(0.4)
    assert gated[0]["kept"] is True


def test_instance_query_without_mapping_falls_back():
    p = _probs()
    v = np.ones((16, 16), bool)
    cand = {"id": "a", "mask": _mask(), "class": None,
            "relation": "unmapped", "score": 0.7}
    gated, _ = gate_candidates([cand], p, [v])
    assert gated[0]["kept"] and gated[0]["score_guided"] == pytest.approx(0.7)


def test_gate_summary_counts_states():
    p_agree = _probs(cls=3, value=0.95)
    p_conf = _probs(cls=5, value=0.95)
    v = np.ones((16, 16), bool)
    cands = [
        {"id": "a", "mask": _mask(), "class": 3, "relation": "exact", "score": 0.5},
        {"id": "b", "mask": _mask(), "class": 3, "relation": "exact", "score": 0.5},
        {"id": "c", "mask": _mask(), "class": None, "relation": "unmapped", "score": 0.5},
    ]
    valids = [v, v, v]
    # evaluate b against a conflicting teacher
    logs = []
    for c, pr in zip(cands, [p_agree, p_conf, p_agree]):
        _, lg = gate_candidates([c], pr, [v])
        logs.extend(lg)
    s = gate_summary(logs)
    assert s["candidates"] == 3
    assert s["agree"] == 1 and s["conflict"] == 1 and s["uncertain"] == 1
    assert s["fallback_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------- pixel level

def test_pixel_cooperation_writes_in_uncertain_region():
    """Teacher unsure on the region -> the text mask may write the class."""
    H = W = 16
    teacher = np.full((H, W), 3, np.int64)     # teacher says "road" (wrong)
    probs = np.full((13, H, W), 0.01, np.float32)
    probs[12, 4:8, 4:8] = 0.45                # teacher unsure on the mask
    probs /= probs.sum(0)
    m = np.zeros((H, W), bool)
    m[3:9, 3:9] = True
    masks = [{"mask": m, "quality": 0.9, "support": 0.9, "state": AGREE,
              "class": 12}]
    out = revise_pseudo_by_text(teacher, probs, masks, target_cls=12,
                                qual_thr=0.8, edge_width=1,
                                semantic_support_thr=0.5)
    assert out["stats"]["written_pixels"] > 0
    assert out["stats"]["changed_pixels"] > 0
    assert out["stats"]["uncertain_region_changes"] > 0
    # The write happened where the teacher was unsure (not confidently wrong).
    assert (out["revised"][4:8, 4:8] == 12).all()


def test_pixel_cooperation_blocks_confident_conflict():
    H = W = 16
    teacher = np.full((H, W), 5, np.int64)   # teacher says class 5 everywhere
    probs = np.full((13, H, W), 0.01, np.float32)
    probs[5] = 0.95
    probs[12] = 0.02
    probs /= probs.sum(0)
    m = np.zeros((H, W), bool)
    m[4:8, 4:8] = True
    masks = [{"mask": m, "quality": 0.9, "support": 0.9, "state": AGREE,
              "class": 12}]
    out = revise_pseudo_by_text(teacher, probs, masks, target_cls=12,
                                qual_thr=0.8, edge_width=1,
                                semantic_support_thr=0.5)
    assert out["stats"]["blocked_conflict_pixels"] > 0
    assert not (out["changed_mask"] & (teacher == 5)).any()


def test_pixel_cooperation_uncertain_mask_cannot_write():
    H = W = 16
    teacher = np.zeros((H, W), np.int64)
    probs = np.full((13, H, W), 0.01, np.float32)
    probs[12, 4:8, 4:8] = 0.45
    probs /= probs.sum(0)
    m = np.zeros((H, W), bool)
    m[3:9, 3:9] = True
    masks = [{"mask": m, "quality": 0.9, "support": 0.9, "state": UNCERTAIN,
              "class": 12}]
    out = revise_pseudo_by_text(teacher, probs, masks, target_cls=12,
                                qual_thr=0.8, edge_width=1,
                                semantic_support_thr=0.5)
    assert out["stats"]["masks_accepted"] == 0
    assert out["stats"]["changed_pixels"] == 0


def test_pixel_cooperation_never_touches_unrelated_pixels():
    H = W = 16
    teacher = np.zeros((H, W), np.int64)
    probs = np.full((13, H, W), 0.01, np.float32)
    probs[12, 4:8, 4:8] = 0.45
    probs /= probs.sum(0)
    m = np.zeros((H, W), bool)
    m[4:8, 4:8] = True
    masks = [{"mask": m, "quality": 0.9, "support": 0.9, "state": AGREE,
              "class": 12}]
    out = revise_pseudo_by_text(teacher, probs, masks, target_cls=12,
                                qual_thr=0.8, edge_width=0,
                                semantic_support_thr=0.5)
    untouched = ~out["write_mask"] & ~out["edge_ignore_mask"]
    assert np.array_equal(out["revised"][untouched], teacher[untouched])


def test_pixel_cooperation_reports_net_effect_vs_gt():
    H = W = 8
    teacher = np.zeros((H, W), np.int64)
    gt = np.zeros((H, W), np.int64)
    gt[2:6, 2:6] = 12
    probs = np.full((13, H, W), 0.01, np.float32)
    probs[12, 2:6, 2:6] = 0.45
    probs /= probs.sum(0)
    m = np.zeros((H, W), bool)
    m[2:6, 2:6] = True
    masks = [{"mask": m, "quality": 0.9, "support": 0.9, "state": AGREE,
              "class": 12}]
    out = revise_pseudo_by_text(teacher, probs, masks, target_cls=12,
                                qual_thr=0.8, edge_width=0,
                                semantic_support_thr=0.5)
    cmp = compare_revisions(teacher, {"coop": out["revised"]}, gt)
    assert cmp["coop"]["net_correct_changes"] > 0
