import numpy as np
from piers_ljj.memory.bridge import (VARIANT_TO_SMR, token_erp_rect,
                                      rect_overlap, footprint_overlap,
                                      plan_selection)
from piers_ljj.seam.adaptive import (seam_bands, score_uda, score_text,
                                      plan_cross_seam_windows,
                                      adaptive_extras, fuse_extra_into_base,
                                      cost_report)
from piers_ljj.geometry.erp import crop_erp_wrap


def _item(i, x, stable=True, score=None, boundary=False, rare=False):
    return {"id": f"t{i}", "bucket": (int(x // 128), 0, 9),
            "footprint": (float(x), 0.0, float(x + 64), 32.0),
            "stable": stable, "compat": True,
            "score": 0.1 * i if score is None else score,
            "boundary": boundary, "rare": rare}


def test_variant_map_covers_all_suites():
    from piers_ljj.experiments.suites import MEMORY_VARIANTS
    assert set(VARIANT_TO_SMR) == set(MEMORY_VARIANTS)


def test_token_rect_wrap_two_segments():
    fov = {"x_start": 90, "y_start": 0, "win_w": 20, "win_h": 10}
    r = token_erp_rect(0, 0, grid_w=1, grid_h=1, fov=fov, erp_w=100, erp_h=32)
    assert isinstance(r[0], tuple) and len(r) == 2  # wrapped -> two segs
    fov2 = {"x_start": 10, "y_start": 0, "win_w": 20, "win_h": 10}
    r2 = token_erp_rect(0, 0, grid_w=1, grid_h=1, fov=fov2, erp_w=100, erp_h=32)
    assert len(r2) == 4 and r2[0] == 10.0  # no wrap -> single rect


def test_pos_dedup_keeps_best_and_conserves_budget():
    items = [_item(0, 0, score=0.9), _item(1, 10, score=0.1)]  # same bucket+overlap
    items += [_item(i, 300 + i * 100, score=0.5) for i in range(2, 6)]
    r = plan_selection(items, "pos_dedup", budget=4)
    assert "t0" in r["kept_ids"] and "t1" in r["dropped_ids"]
    assert r["report"]["kept"] == 4 and r["report"]["budget"] == 4


def test_priority_reserves_boundary_rare():
    items = ([_item(i, i * 100, boundary=(i < 2), rare=(i == 2)) for i in range(6)])
    r = plan_selection(items, "pos_dedup_priority", budget=3)
    assert r["smr_params"]["sparse_mode"] == "boundary_guarded"
    assert set(r["kept_ids"]) == {"t0", "t1", "t2"}


def test_neighborhood_ranks_by_current_fov():
    cur = (0.0, 0.0, 64.0, 32.0)
    items = [_item(0, 500, score=0.9), _item(1, 0, score=0.1)]
    r = plan_selection(items, "neighborhood", budget=1, cur_footprint=cur)
    assert r["kept_ids"] == ["t1"]  # adjacent wins over high-score far token


def test_nocompress_ref_flagged_not_comparable():
    items = [_item(i, i * 100) for i in range(4)]
    r = plan_selection(items, "nocompress_ref", budget=2)
    assert r["report"]["kept"] == 4 and "not comparable" in r["report"]["note"]


def test_seam_b0_strict_degradation():
    assert adaptive_extras({"x000": 1.0}, 0) == []
    base = np.ones((4, 4)); w = np.ones((4, 4))
    out, _ = fuse_extra_into_base(base, w, np.zeros((4, 4)), np.zeros((4, 4)))
    assert abs(out - base).max() < 1e-9


def test_cross_seam_windows_are_true_wrap():
    wins = plan_cross_seam_windows(W=100, H=32, win_w=40, win_h=32, n=2)
    assert len(wins) == 2
    img = np.arange(32 * 100, dtype=np.float32).reshape(32, 100)
    for wn in wins:
        crop, cov = crop_erp_wrap(img, wn["x_start"], wn["y_start"],
                                  wn["win_w"], wn["win_h"])
        assert crop.shape == (32, 40) and cov.all()
    # seam column 0 and W-1 both appear in at least one window
    assert any((wn["x_start"] % 100) + 40 > 100 for wn in wins)


def test_score_fns_separate_tasks():
    assert score_uda(0.5, True) > score_uda(0.5, False)
    assert score_text(True, 0.2, False) > score_text(False, 0.9, False)


def test_cost_report_counts_inputs():
    c = cost_report(4, 2, 64, 32, dino_calls=6, sam_enc=6, sam_dec=10,
                    latency_ms=123.0)
    assert c["windows"] == 6 and c["extra_windows"] == 2
    assert c["model_input_px"] == 6 * 64 * 32
