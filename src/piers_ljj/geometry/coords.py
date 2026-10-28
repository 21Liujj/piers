"""Resize/letterbox + box/mask coordinate transforms. R1/R5 shared base.

Spaces: orig ERP (Wo,Ho) -> uniform infer ERP (Wi,Hi, scale s) -> FoV crop
-> model input (resize to mw,mh with optional letterbox pad).
All boxes half-open [x0,x1); pixel centers at (i+0.5) noted, integer math
kept explicit so synthetic tests can hand-compute expectations.
"""
from __future__ import annotations


def infer_scale(orig_w: int, orig_h: int, infer_w: int, infer_h: int) -> tuple[float, float]:
    return infer_w / orig_w, infer_h / orig_h


def letterbox_params(fov_w: int, fov_h: int, model_w: int, model_h: int) -> dict:
    """Fit preserving aspect; returns {scale, pad_left, pad_top, pad_right, pad_bottom}."""
    s = min(model_w / fov_w, model_h / fov_h)
    new_w, new_h = fov_w * s, fov_h * s
    dw, dh = model_w - new_w, model_h - new_h
    return {"scale": s, "pad_left": dw / 2.0, "pad_top": dh / 2.0,
            "pad_right": dw / 2.0, "pad_bottom": dh / 2.0,
            "new_w": new_w, "new_h": new_h}


def box_model_to_fov(box, scale: float, pad_left: float, pad_top: float) -> tuple:
    """Model-input xyxy -> FoV-local xyxy (inverse resize+pad)."""
    x0, y0, x1, y1 = box
    return ((x0 - pad_left) / scale, (y0 - pad_top) / scale,
            (x1 - pad_left) / scale, (y1 - pad_top) / scale)


def box_fov_to_model(box, scale: float, pad_left: float, pad_top: float) -> tuple:
    x0, y0, x1, y1 = box
    return (x0 * scale + pad_left, y0 * scale + pad_top,
            x1 * scale + pad_left, y1 * scale + pad_top)


def valid_fov_region(fov_w: int, fov_h: int, model_w: int, model_h: int,
                     pad: dict) -> tuple:
    """Valid FoV-local rect actually covered by model input (excludes pad)."""
    s = pad["scale"]
    # model [pad_left, pad_left+new_w] maps back to [0, fov_w]; clamp numeric noise
    x0 = max(0.0, (0 - pad["pad_left"]) / s + pad["pad_left"] * 0)  # = 0 by construction
    return (0.0, 0.0, float(fov_w), float(fov_h))


def clip_box(box, w: float, h: float):
    x0, y0, x1, y1 = box
    x0c, y0c = max(0.0, x0), max(0.0, y0)
    x1c, y1c = min(float(w), x1), min(float(h), y1)
    if x1c <= x0c or y1c <= y0c:
        return None  # fully outside -> empty (legal, never forced to background)
    return (x0c, y0c, x1c, y1c)
