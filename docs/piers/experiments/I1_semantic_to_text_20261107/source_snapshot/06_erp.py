"""ERP horizontal-wrap FoV geometry. R1/R3/R4 shared base.

Coordinate spaces (must not be confused):
  orig ERP -> uniform infer ERP (scale s) -> FoV crop (x_start,y_start,win) -> model input (resize/letterbox)
Base relation (no extra perspective projection in v1):
  x_global = (x_start + x_local) mod W ; y_global = y_start + y_local
With scale: first inverse-scale + inverse-pad, then apply the relation.
Pixel convention: centers at (i+0.5), half-open boxes [x0,x1).
"""
from __future__ import annotations
import math


def mod_w(x: int, W: int) -> int:
    return x % W


def local_to_global(x_local: float, y_local: float, x_start: int, y_start: int, W: int):
    return ((x_start + x_local) % W, y_start + y_local)


def global_to_local(x_global: float, y_global: float, x_start: int, y_start: int, W: int):
    return ((x_global - x_start) % W, y_global - y_start)


def inverse_scale_coord(c: float, scale: float, pad_before: float = 0.0) -> float:
    """Model-input coord -> infer-ERP coord (undo letterbox+resize)."""
    return (c - pad_before) / scale


def plan_fovs(W: int, H: int, win_w: int, win_h: int, stride_x: int, y_start: int = 0,
              order: str = "left_to_right"):
    """Plan FoV windows covering width W with wrap; handles non-divisible stride.

    Returns list of dicts {view_id, x_start, y_start, win_w, win_h}.
    Coverage check: every column covered >=1 (guaranteed by wrap stepping).
    """
    assert win_w > 0 and win_h > 0 and stride_x > 0
    assert 0 <= y_start <= H - win_h
    n = math.ceil(W / stride_x)
    fovs = []
    for i in range(n):
        xs = i * stride_x
        # last window may exceed W -> wrap crop (valid; see crop_erp_wrap)
        fovs.append({"view_id": f"v{i:03d}", "x_start": xs, "y_start": y_start,
                     "win_w": win_w, "win_h": win_h, "order": i})
    if order == "right_to_left":
        fovs = fovs[::-1]
    return fovs


def crop_erp_wrap(img, x_start: int, y_start: int, win_w: int, win_h: int):
    """Crop HxWxC (numpy) or PIL image with horizontal wrap. Supports negative x_start.

    Returns (crop, coverage_mask HxW bool all-True here; kept for API symmetry).
    """
    import numpy as np
    from PIL import Image
    if isinstance(img, Image.Image):
        a = np.array(img)
        pil = True
    else:
        a = img
        pil = False
    H, W = a.shape[0], a.shape[1]
    xs = x_start % W
    idx = [(xs + i) % W for i in range(win_w)]
    crop = a[y_start:y_start + win_h, :, ...][:, idx, ...] if a.ndim >= 3 else a[y_start:y_start + win_h, :][:, idx]
    cov = np.ones((win_h, win_w), dtype=bool)
    if pil:
        crop = Image.fromarray(crop)
    return crop, cov


def paste_erp_wrap(canvas, weight, patch, cov, x_start: int, y_start: int):
    """Accumulate patch (HxW[C]) into canvas with wrap; weight accumulates coverage.

    canvas, weight: HxW[C or 1]; patch: win_h x win_w [C]; cov bool mask.
    Returns updated (canvas, weight). Caller divides canvas/weight for mean fusion.
    """
    import numpy as np
    H, W = weight.shape[0], weight.shape[1]
    win_h, win_w = patch.shape[0], patch.shape[1]
    xs = x_start % W
    for lx in range(win_w):
        gx = (xs + lx) % W
        ys = slice(y_start, y_start + win_h)
        w = cov[:, lx].astype(weight.dtype) if cov.ndim == 2 else cov
        if patch.ndim == 3:
            canvas[ys, gx, :] += patch[:, lx, :] * w[:, None]
            weight[ys, gx] += w
        else:
            canvas[ys, gx] += patch[:, lx] * w
            weight[ys, gx] += w
    return canvas, weight


def box_local_to_global(box, x_start: int, y_start: int, W: int, scale: float = 1.0,
                        pad_left: float = 0.0, pad_top: float = 0.0):
    """xyxy in model-input/fov-local coords -> global ERP coords.

    Applies inverse scale/pad first, then wrap map. Returns dict with possibly
    two segments if the box crosses the seam: {'segs': [(x0,y0,x1,y1), ...], 'crosses': bool}.
    Never returns a full-width box to fake a seam crossing.
    """
    x0, y0, x1, y1 = box
    gx0 = (x_start + (x0 - pad_left) / scale) % W
    gx1 = (x_start + (x1 - pad_left) / scale) % W
    gy0 = y_start + (y0 - pad_top) / scale
    gy1 = y_start + (y1 - pad_top) / scale
    if gx1 >= gx0:
        return {"segs": [(gx0, gy0, gx1, gy1)], "crosses": False}
    # wraps: two honest segments
    return {"segs": [(gx0, gy0, float(W), gy1), (0.0, gy0, gx1, gy1)], "crosses": True}
