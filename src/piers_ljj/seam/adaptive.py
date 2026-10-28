"""Innovation-3 policy layer: on-demand seam re-observe. R4.

fixed_scm (inherited DensePASS_circular + DynamicSCM) stays the baseline and
is NOT reimplemented here. This module adds ONLY the adaptive_extra policy:

1. score candidates in the seam band + candidate extra windows.
   UDA: entropy / boundary-gap / cross-FoV disagreement.
   Text: mask touch-seam / sam quality / assoc uncertainty.
2. take <=B non-redundant extras (skip already-well-covered).
3. generate TRUE cross-seam continuous windows (wrap crop, never a full-width
   fake box) for the real backend.
4. fuse in true overlap by coverage x confidence; uncovered stays base.

B=0 strictly degrades to base output (no extra windows, fusion = identity).
Same-budget comparison reuses policy.select_extra + policy.fuse_overlap.
"""
from __future__ import annotations
import numpy as np

from piers_ljj.seam.policy import select_extra, fuse_overlap


def seam_bands(W: int, band_w: int) -> dict:
    """Left/right seam band column ranges (spatial neighbours, not same pixels)."""
    band_w = max(1, min(int(band_w), W // 2))
    return {"left": (0, band_w), "right": (W - band_w, W), "band_w": band_w}


def score_uda(entropy: float, truncated: bool, disagreement: float = 0.0,
              w_e: float = 1.0, w_t: float = 1.0, w_d: float = 1.0) -> float:
    return w_e * float(entropy) + w_t * (1.0 if truncated else 0.0) + w_d * float(disagreement)


def score_text(touch: bool, sam_quality: float, assoc_uncertain: bool,
               w_t: float = 1.0, w_q: float = 1.0, w_a: float = 0.5) -> float:
    return (w_t * (1.0 if touch else 0.0)
            + w_q * (1.0 - float(sam_quality))
            + w_a * (1.0 if assoc_uncertain else 0.0))


def plan_cross_seam_windows(W: int, H: int, win_w: int, win_h: int,
                            y_start: int = 0, n: int = 2) -> list:
    """Candidate extra windows centred on the seam (true wrap windows).

    Centres at x=W (the seam) +/- small offsets so each window is continuous
    in wrap-crop space. x_start may be negative or >=W — geometry handles it.
    """
    cands = []
    for i in range(max(1, n)):
        off = (i - (n - 1) / 2.0) * (win_w / 4.0)
        xs = int(round(W - win_w / 2.0 + off))
        cands.append({"view_id": f"x{i:03d}", "x_start": xs, "y_start": y_start,
                      "win_w": win_w, "win_h": win_h, "order": 900 + i})
    return cands


def adaptive_extras(scores: dict, B: int, covered_fn=None) -> list:
    """<=B extra window ids. B=0 -> [] (strict degradation)."""
    if B <= 0:
        return []
    return select_extra(scores, B, covered_fn=covered_fn)


def fuse_extra_into_base(base: np.ndarray, base_w: np.ndarray,
                         extra: np.ndarray, extra_w: np.ndarray,
                         extra_conf: float = 1.0):
    """Fuse one extra window into base in true overlap only."""
    return fuse_overlap(base, base_w, extra, extra_w, extra_conf=extra_conf)


def cost_report(n_base: int, n_extra: int, win_w: int, win_h: int,
                dino_calls: int = 0, sam_enc: int = 0, sam_dec: int = 0,
                latency_ms: float = 0.0) -> dict:
    return {"windows": n_base + n_extra, "base_windows": n_base,
            "extra_windows": n_extra, "window_px": win_w * win_h,
            "model_input_px": (n_base + n_extra) * win_w * win_h,
            "dino_calls": dino_calls, "sam_image_encodes": sam_enc,
            "sam_mask_decoder_calls": sam_dec, "latency_ms": latency_ms}
