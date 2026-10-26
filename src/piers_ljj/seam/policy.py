"""R4 seam on-demand re-observe.

Strategies: linear | fixed_scm | uniform_extra | adaptive_extra.
adaptive_extra: score seam-band candidates (entropy/boundary-gap for UDA;
touch-quality-assoc-uncertainty for text), take <=B non-redundant extras,
run backend on cross-seam continuous windows, fuse in true overlap by
coverage & confidence. B=0 strictly degrades to base output.
"""
from __future__ import annotations
import numpy as np


def score_seam_candidate(uncertainty: float, truncated: bool, w_u: float = 1.0, w_t: float = 1.0) -> float:
    return w_u * float(uncertainty) + w_t * (1.0 if truncated else 0.0)


def select_extra(scores: dict, B: int, covered_fn=None):
    """scores: {cand_id: score}. Skip already-well-covered. Returns <=B ids."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for cid, s in ranked:
        if len(out) >= B:
            break
        if covered_fn is not None and covered_fn(cid):
            continue
        out.append(cid)
    return out


def fuse_overlap(base: np.ndarray, base_w: np.ndarray, extra: np.ndarray,
                 extra_w: np.ndarray, extra_conf: float = 1.0):
    """Fuse in true overlap only. base/extra: HxW[C] or HxW; weights coverage."""
    tot_w = base_w + extra_w * extra_conf
    tot = base + extra * extra_conf
    out = np.divide(tot, np.maximum(tot_w, 1e-6), out=np.zeros_like(tot, dtype=float),
                    where=(tot_w > 1e-9))
    return out, tot_w
