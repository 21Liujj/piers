import numpy as np
from piers_ljj.collaboration.text_guide import guide_query, apply_mode, erp_support
from piers_ljj.collaboration.revision import build_revision, save_revision, load_revision
from piers_ljj.text.manifest import make_manifest, worker_view, subset_ids, negative_ids
from piers_ljj.text.pipeline import run_erp_text
from piers_ljj.text.grounded_adapter import GroundedConfig
from PIL import Image


def _sq(x0, y0, n=4, W=16, H=16):
    m = np.zeros((H, W), bool)
    m[y0:y0 + n, x0:x0 + n] = True
    return m


def test_erp_support_and_soft_rerank():
    V = np.ones((16, 16), bool)
    prob = np.zeros((16, 16)); prob[0:4, 0:4] = 0.9
    assert erp_support(_sq(0, 0), prob, V) == 0.9
    preds = [{"id": "a", "score": 0.5, "mask": _sq(0, 0), "valid": V,
              "class": "car", "relation": "exact"}]
    out, log = guide_query(preds, {"car": prob}, alpha=0.5)
    assert out[0]["kept"] and abs(out[0]["score_guided"] - 0.5 * 1.45) < 1e-9


def test_hard_filter_and_unmapped_fallback():
    V = np.ones((16, 16), bool)
    prob = np.zeros((16, 16))  # no support
    preds = [{"id": "a", "score": 0.5, "mask": _sq(0, 0), "valid": V,
              "class": "car", "relation": "exact"}]
    out, _ = guide_query(preds, {"car": prob}, hard_thr=0.15)
    assert out[0]["kept"] is False  # dropped by hard filter
    # unmapped keeps original
    preds2 = [{"id": "b", "score": 0.4, "mask": _sq(0, 0), "valid": V,
               "class": None, "relation": "unmapped"}]
    out2, log2 = guide_query(preds2, {}, hard_thr=0.15)
    assert out2[0]["kept"] and out2[0]["score_guided"] == 0.4


def test_collab_none_passthrough():
    V = np.ones((8, 8), bool)
    preds = [{"id": "a", "detector_score": 0.7, "mask": np.ones((8, 8), bool),
              "valid": V, "class": "car", "relation": "exact"}]
    out, _ = apply_mode(preds, {}, mode="none")
    assert out[0]["score_guided"] == 0.7 and out[0]["kept"]


def test_revision_never_forces_background_and_edge_ignore():
    H = W = 16
    m = np.zeros((H, W), bool); m[4:12, 4:12] = True
    p = build_revision((H, W), [{"mask": m, "quality": 0.95}],
                       target_cls=12, query_id="q1")
    assert p["target_cls"] == 12 and p["query_id"] == "q1"
    inner, edge = p["revise_mask"], p["ignore_mask"]
    assert not (inner & edge).any()
    assert (inner | edge).sum() == m.sum()  # every mask px either written or ignored
    assert inner.sum() < m.sum()  # edges eroded to ignore


def test_revision_lowq_skipped_and_shape_checked(tmp_path):
    import pytest
    H = W = 8
    m = np.ones((H, W), bool)
    p = build_revision((H, W), [{"mask": m, "quality": 0.1}], target_cls=3)
    assert p["stats"]["skipped_lowq"] == 64 and not p["revise_mask"].any()
    with pytest.raises(AssertionError):
        build_revision((H, W), [{"mask": np.ones((4, 4), bool), "quality": 1.0}],
                       target_cls=3)
    fp = str(tmp_path / "rev.npz")
    save_revision(fp, p)
    q = load_revision(fp)
    assert q["target_cls"] == 3 and not q["revise_mask"].any()


def test_manifest_gt_isolation():
    man = make_manifest(
        [{"image_id": "i1", "path": "a.png", "width": 64, "height": 32}],
        [{"query_id": "q1", "image_id": "i1", "raw_text": "car",
          "task_type": "category_union", "gt_mask": "secret.png"},
         {"query_id": "q2", "image_id": "i1", "raw_text": "nothing",
          "task_type": "instance_referring", "negative": True,
          "missing_basis": "verified-absent"}],
        split="dev")
    w = worker_view(man)
    assert "gt_mask" not in w["queries"][0]
    assert subset_ids(man, "category_union") == ["q1"]
    assert negative_ids(man) == ["q2"]


class _Dino:
    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, img, text, bt, tt):
        return self.boxes, [0.9] * len(self.boxes)


class _Sam:
    def __init__(self):
        self.enc = 0

    def set_image(self, img):
        self.enc += 1
        return object()

    def predict_boxes(self, enc, boxes):
        m = np.zeros((32, 32), bool); m[2:6, 2:6] = True
        return [m] * len(boxes), [0.8] * len(boxes)


def test_pipeline_t1_fov_loop_counts_encodes():
    sam = _Sam()
    img = Image.new("RGB", (64, 32))
    r = run_erp_text(img, "q1", "car",
                     {"win_w": 32, "win_h": 32, "stride_x": 32},
                     GroundedConfig(),
                     backend_factory=lambda: (_Dino([np.array([2, 2, 6, 6])]), sam),
                     mode="fov")
    assert r["fovs"] == 2 and r["calls"]["sam_image_encodes"] == 2
    assert len(r["predictions"]) == 2 and len(r["groups"]) >= 1


def test_pipeline_direct_single_window():
    sam = _Sam()
    img = Image.new("RGB", (64, 32))
    r = run_erp_text(img, "q1", "car", {}, GroundedConfig(),
                     backend_factory=lambda: (_Dino([]), sam), mode="direct")
    assert r["fovs"] == 1 and r["predictions"] == []
