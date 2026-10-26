import numpy as np
from piers_ljj.geometry.erp import (plan_fovs, crop_erp_wrap, paste_erp_wrap,
                                    box_local_to_global, local_to_global)


def test_wrap_crop_and_roundtrip():
    H, W = 32, 64
    img = np.arange(H * W, dtype=np.float32).reshape(H, W)
    crop, cov = crop_erp_wrap(img, -8, 0, 16, 8)  # negative start wraps
    assert crop.shape == (8, 16) and cov.all()
    # first col of crop == global col (W-8)
    assert (crop[:, 0] == img[0:8, W - 8]).all()


def test_nondivisible_stride_coverage():
    fovs = plan_fovs(100, 32, win_w=40, win_h=32, stride_x=30, y_start=0)
    assert len(fovs) == 4  # ceil(100/30)
    covered = np.zeros(100, bool)
    for f in fovs:
        for lx in range(f["win_w"]):
            covered[(f["x_start"] + lx) % 100] = True
    assert covered.all(), "every column must be covered"


def test_box_seam_two_segments():
    # local box (0,0,20,10) viewed at x_start=90 on W=100 -> globals 90..110 wraps
    r = box_local_to_global((0, 0, 20, 10), x_start=90, y_start=0, W=100)
    assert r["crosses"] is True and len(r["segs"]) == 2
    (a, b) = r["segs"]
    assert a[0] >= 90 and b[2] <= 100 and a[2] == 100.0 and b[0] == 0.0


def test_scale_inverse():
    # model coord 60 with scale .5 pad 10 -> infer coord 100
    from piers_ljj.geometry.erp import inverse_scale_coord
    assert inverse_scale_coord(60, 0.5, 10) == 100.0
