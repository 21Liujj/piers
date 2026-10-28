"""T0/T1 end-to-end text pipeline: FoV loop + remap + assoc. R1.

T0 grounded_direct: single FoV covering the whole ERP (records model resize cost).
T1 grounded_fov: plan_fovs loop, per-FoV text branch, box remap via
geometry.box_local_to_global, coverage accumulation, cross-FoV association.

CPU-testable: backend_factory injects (dino, sam) fakes; real backend comes
from workers/piers-grounded (HF transformers + SAM2ImagePredictor).
"""
from __future__ import annotations
import time
from typing import Callable

import numpy as np
from PIL import Image

from piers_ljj.geometry.erp import plan_fovs, crop_erp_wrap, box_local_to_global
from piers_ljj.text.assoc import associate
from piers_ljj.text.grounded_adapter import run_fov_text_branch, GroundedConfig


def run_erp_text(erp: Image.Image | np.ndarray, query_id: str, raw_text: str,
                 fov_cfg: dict, gcfg: GroundedConfig,
                 backend_factory: Callable[[], tuple] | None = None,
                 mode: str = "fov"):
    """Run text branch over an ERP image.

    mode="direct": one FoV = whole image (T0).
    mode="fov": plan_fovs loop (T1).
    backend_factory: () -> (dino, sam) pair, called once per FoV (lets tests
      count image-encodes; real worker passes a factory building the HF pair).
    Returns dict with predictions (global-ERP), assoc groups, call counts.
    """
    t0 = time.perf_counter()
    if isinstance(erp, Image.Image):
        W, H = erp.size
        erp_np = np.array(erp)
    else:
        H, W = erp.shape[0], erp.shape[1]
        erp_np = erp
    if mode == "direct":
        fovs = [{"view_id": "v000", "x_start": 0, "y_start": 0,
                 "win_w": W, "win_h": H, "order": 0}]
    else:
        fovs = plan_fovs(W, H, win_w=fov_cfg["win_w"], win_h=fov_cfg.get("win_h", H),
                         stride_x=fov_cfg.get("stride_x", fov_cfg["win_w"]),
                         y_start=fov_cfg.get("y_start", 0))
    preds, cands_for_assoc = [], []
    n_enc, n_dec = 0, 0
    for f in fovs:
        crop, cov = crop_erp_wrap(erp_np, f["x_start"], f["y_start"], f["win_w"], f["win_h"])
        pil = Image.fromarray(crop) if not isinstance(crop, Image.Image) else crop
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        backend = backend_factory() if backend_factory else None
        r = run_fov_text_branch(pil, raw_text, gcfg, backend=backend)
        n_enc += r["meta"]["image_encodes"]
        n_dec += r["meta"]["mask_decoder_calls"]
        for k, c in enumerate(r["candidates"]):
            g = box_local_to_global(c.box_xyxy, f["x_start"], f["y_start"], W)
            # local mask -> global ERP canvas (for assoc IoU on shared-visible area)
            gh, gw = H, W
            gm = np.zeros((gh, gw), bool)
            valid = np.zeros((gh, gw), bool)
            lm = np.asarray(c.mask)
            xs = f["x_start"] % W
            for lx in range(f["win_w"]):
                gx = (xs + lx) % W
                ys = slice(f["y_start"], f["y_start"] + f["win_h"])
                if lx < lm.shape[1] and lm.shape[0] == f["win_h"]:
                    gm[ys, gx] |= lm[:, lx] > 0
                valid[ys, gx] = True
            pid = f"{query_id}:{f['view_id']}:{k}"
            preds.append({"id": pid, "query_id": query_id, "fov_id": f["view_id"],
                          "box_segs": g["segs"], "crosses": g["crosses"],
                          "detector_score": c.detector_score, "sam_quality": c.sam_quality,
                          "phrase": c.phrase, "norm_text": r["norm_text"],
                          "synthetic": getattr(c.mask, "synthetic", False)})
            cands_for_assoc.append({"id": pid, "query_id": query_id, "mask": gm,
                                    "valid": valid, "phrase": c.phrase})
    groups = associate(cands_for_assoc)
    dt_ms = (time.perf_counter() - t0) * 1000
    return {"query_id": query_id, "mode": mode, "fovs": len(fovs),
            "predictions": preds, "groups": groups,
            "calls": {"fov_windows": len(fovs), "sam_image_encodes": n_enc,
                      "sam_mask_decoder_calls": n_dec, "latency_ms": dt_ms}}
