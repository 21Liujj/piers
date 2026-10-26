import numpy as np
from piers_ljj.text.assoc import associate
from piers_ljj.collaboration.rules import rerank_candidates, revise_pseudo
from piers_ljj.memory.policy import dedup_positions, allocate_with_priority
from piers_ljj.seam.policy import select_extra, fuse_overlap


def _sq(x0, y0, n=8, W=64, H=32):
    m = np.zeros((H, W), bool)
    m[y0:y0 + n, x0:x0 + n] = True
    return m


def test_no_merge_adjacent_same_class():
    V = np.ones((32, 64), bool)
    c = [{"id": "a", "query_id": "q", "mask": _sq(0, 0), "valid": V, "phrase": "car"},
         {"id": "b", "query_id": "q", "mask": _sq(20, 0), "valid": V, "phrase": "car"}]
    g = associate(c)
    assert len(g) == 2, "adjacent same-class instances must NOT merge without overlap"


def test_merge_overlap_same_query():
    V = np.ones((32, 64), bool)
    c = [{"id": "a", "query_id": "q", "mask": _sq(0, 0), "valid": V, "phrase": "car"},
         {"id": "b", "query_id": "q", "mask": _sq(2, 0), "valid": V, "phrase": "car"}]
    g = associate(c)
    assert len(g) == 1 and set(g[0]["members"]) == {"a", "b"}


def test_collab_off_keeps_original():
    V = np.ones((8, 8), bool)
    c = [{"id": "a", "score": 0.5, "mask": np.ones((8, 8), bool), "valid": V,
          "class": "car", "relation": "unmapped"}]
    out, log = rerank_candidates(c, {}, alpha=0.5)
    assert out[0]["score_new"] == 0.5 and log[0]["action"] == "keep"


def test_pseudo_never_forces_background():
    pseudo = np.full((8, 8), 3, int)
    conf = np.full((8, 8), 0.9)
    new, st = revise_pseudo(pseudo, conf, [], target_cls=12)
    assert (new == 3).all() and st["written"] == 0


def test_budget_conserved():
    items = [{"id": str(i), "bucket": i % 3, "stable": True, "score": i / 10} for i in range(9)]
    kept, dropped = dedup_positions(items)
    assert len(kept) + len(dropped) == 9
    chosen, rep = allocate_with_priority(items, 4, lambda it: (it["score"] > 0.5, -it["score"]))
    assert rep["kept"] == 4 and rep["budget"] == 4


def test_seam_B0_degrades():
    assert select_extra({"a": 1.0}, 0) == []
    base = np.ones((4, 4)); w = np.ones((4, 4))
    out, _ = fuse_overlap(base, w, np.zeros((4, 4)), np.zeros((4, 4)))
    assert abs(out - base).max() < 1e-9
