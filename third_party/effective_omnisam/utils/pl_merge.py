"""
utils/pl_merge.py

Patch-sequence merge utilities for PseudoLabelGenerator.

Supports three modes:
  legacy_sum      – original behaviour: scatter_add_ without coverage division.
                    Bit-exact with the old merge_width_scatter_np_uncertainty().
  normalized_mean – divides accumulated prob by coverage count before argmax.
                    Fixes the coverage-inflation bias.
  paper_strict    – stub (raises NotImplementedError); reserved for future
                    consensus-vote implementation.

All computation is vectorised on GPU; no Python pixel loops.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401 – kept for potential callers


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PlMergeConfig:
    mode: str = "legacy_sum"          # legacy_sum | normalized_mean | paper_strict
    threshold: float = 0.85
    ignore_index: int = 255
    save_debug_maps: bool = False
    debug_max_images: int = 10

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PlMergeConfig":
        if not isinstance(d, dict):
            return cls()
        return cls(
            mode=str(d.get("mode", "legacy_sum")),
            threshold=float(d.get("threshold", 0.85)),
            ignore_index=int(d.get("ignore_index", 255)),
            save_debug_maps=bool(d.get("save_debug_maps", False)),
            debug_max_images=int(d.get("debug_max_images", 10)),
        )

    @classmethod
    def legacy(cls) -> "PlMergeConfig":
        """Return config equivalent to the old hard-coded behaviour."""
        return cls(mode="legacy_sum", threshold=0.85)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    pred: np.ndarray                          # (N, H, W)  uint8 / int; 255 = ignore
    confidence: Optional[np.ndarray] = None  # (N, H, W)  float32 in [0,1] for norm; [0,C] for legacy
    coverage: Optional[np.ndarray] = None    # (N, H, W)  uint8  – times a pixel was covered
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

@torch.no_grad()
def merge_patch_sequence(
    logits_np: np.ndarray,   # (N, B, C, H_patch, W_patch)
    orig_size: Tuple[int, int],  # (H_full, W_full)
    W_patch: int,
    stride_w: int,
    passes: int = 2,
    device: str = "cuda",
    cfg: Optional[PlMergeConfig] = None,
) -> MergeResult:
    """
    Merge patch-sequence logits back to a full panorama.

    Returns MergeResult with pred (numpy uint8), plus optional confidence /
    coverage arrays depending on mode.
    """
    if cfg is None:
        cfg = PlMergeConfig()

    mode = cfg.mode
    threshold = cfg.threshold
    ignore_index = cfg.ignore_index

    if mode == "paper_strict":
        raise NotImplementedError(
            "paper_strict mode is not yet implemented. "
            "Use legacy_sum or normalized_mean."
        )

    t0 = time.perf_counter()

    N, B, C, H_patch, Wp = logits_np.shape
    H_full, W_full = orig_size
    assert Wp == W_patch, f"W_patch mismatch: {Wp} vs {W_patch}"
    assert H_patch == H_full, "H_patch must equal H_full (no vertical splitting)"

    L = (W_full - W_patch) // stride_w + 1
    assert B == passes * L, (
        f"B={B} must equal passes*L={passes}*{L}={passes*L}"
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    logits = torch.from_numpy(logits_np).to(device=dev, dtype=torch.float32)  # (N,B,C,H,Wp)
    probs  = torch.softmax(logits, dim=2)                                       # (N,B,C,H,Wp)

    # ── reshape + average passes ─────────────────────────────────────────────
    probs = probs.view(N, passes, L, C, H_patch, W_patch)  # (N,passes,L,C,H,Wp)
    if passes == 1:
        probs = probs[:, 0]                              # (N,L,C,H,Wp)
    elif passes == 2:
        forward  = probs[:, 0]                           # (N,L,C,H,Wp)
        backward = probs[:, 1].flip(1)                   # reverse L to align with forward
        probs    = 0.5 * (forward + backward)            # (N,L,C,H,Wp)
    else:
        probs = probs.mean(dim=1)                        # (N,L,C,H,Wp)

    # ── build scatter index (identical for both modes) ───────────────────────
    starts   = torch.arange(0, W_full - W_patch + 1, stride_w, device=dev)  # (L,)
    offs     = torch.arange(W_patch, device=dev)                              # (Wp,)
    idx      = starts[:, None] + offs[None, :]                                # (L, Wp)

    vals     = probs.permute(0, 2, 3, 1, 4)                                   # (N,C,H,L,Wp)
    idx_exp  = idx.view(1, 1, 1, L, W_patch).expand_as(vals)                  # (N,C,H,L,Wp)
    vals_r   = vals.reshape(N, C, H_full, L * W_patch)
    idx_r    = idx_exp.reshape(N, C, H_full, L * W_patch)

    # ── accumulate ──────────────────────────────────────────────────────────
    prob_sum = torch.zeros((N, C, H_full, W_full), device=dev, dtype=torch.float32)
    prob_sum.scatter_add_(3, idx_r, vals_r)

    # ── coverage (needed for normalized_mean; also returned as metadata) ─────
    coverage_map = torch.zeros((N, 1, H_full, W_full), device=dev, dtype=torch.float32)
    ones_r = torch.ones((N, 1, H_full, L * W_patch), device=dev, dtype=torch.float32)
    coverage_map.scatter_add_(3, idx_r[:, :1, :, :], ones_r)   # (N,1,H,W)

    # ── mode-specific score ──────────────────────────────────────────────────
    if mode == "legacy_sum":
        # Original behaviour: threshold on accumulated (unnormalised) sum.
        score_map = prob_sum                               # (N,C,H,W); values in [0, coverage*1]
    else:  # normalized_mean
        score_map = prob_sum / coverage_map.clamp_min(1.0) # (N,C,H,W); values in [0,1]

    probs_max, pred = score_map.max(dim=1)                 # (N,H,W), (N,H,W)

    # ── threshold + coverage-zero masking ───────────────────────────────────
    pred = pred.long()
    pred[probs_max < threshold] = ignore_index
    if mode == "normalized_mean":
        pred[coverage_map[:, 0] == 0] = ignore_index

    t1 = time.perf_counter()

    # ── build coverage histogram ─────────────────────────────────────────────
    cov_cpu = coverage_map[:, 0].cpu().numpy().astype(np.uint8)   # (N,H,W)
    cov_hist = {}
    for v, cnt in Counter(cov_cpu.ravel().tolist()).items():
        cov_hist[int(v)] = int(cnt)

    # ── confidence map ───────────────────────────────────────────────────────
    if mode == "normalized_mean":
        conf_cpu = probs_max.cpu().numpy().astype(np.float32)      # (N,H,W)  in [0,1]
    else:
        conf_cpu = probs_max.cpu().numpy().astype(np.float32)      # (N,H,W)  unnorm

    pred_cpu = pred.cpu().numpy().astype(np.int32)  # (N,H,W)

    # ── stats ────────────────────────────────────────────────────────────────
    valid_mask = pred_cpu != ignore_index
    total_px   = int(pred_cpu.size)
    valid_px   = int(valid_mask.sum())
    valid_conf = conf_cpu[valid_mask]

    stats: dict = {
        "merge_mode":         mode,
        "threshold":          threshold,
        "merge_time_sec":     float(t1 - t0),
        "valid_pixel_ratio":  float(valid_px / max(1, total_px)),
        "ignore_ratio":       float(1.0 - valid_px / max(1, total_px)),
        "confidence_mean":    float(valid_conf.mean())  if len(valid_conf) > 0 else 0.0,
        "confidence_std":     float(valid_conf.std())   if len(valid_conf) > 0 else 0.0,
        "coverage_histogram": cov_hist,
    }

    # per-coverage valid ratio
    vcr: dict = {}
    for cov_v in sorted(cov_hist.keys()):
        mask_c = cov_cpu.ravel() == cov_v
        pred_c = pred_cpu.ravel()[mask_c]
        vcr[int(cov_v)] = float((pred_c != ignore_index).mean()) if len(pred_c) > 0 else 0.0
    stats["valid_ratio_by_coverage"] = vcr

    return MergeResult(
        pred=pred_cpu,
        confidence=conf_cpu,
        coverage=cov_cpu,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Convenience: legacy drop-in
# ---------------------------------------------------------------------------

@torch.no_grad()
def legacy_merge(
    logits_np: np.ndarray,
    orig_size: Tuple[int, int],
    W_patch: int,
    stride_w: int,
    passes: int = 2,
    device: str = "cuda",
    threshold: float = 0.85,
    ignore_index: int = 255,
) -> np.ndarray:
    """Drop-in replacement for merge_width_scatter_np_uncertainty (return_probs=False)."""
    result = merge_patch_sequence(
        logits_np=logits_np,
        orig_size=orig_size,
        W_patch=W_patch,
        stride_w=stride_w,
        passes=passes,
        device=device,
        cfg=PlMergeConfig(mode="legacy_sum", threshold=threshold, ignore_index=ignore_index),
    )
    return result.pred
