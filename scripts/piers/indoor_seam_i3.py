"""Indoor I3 (Innovation-3) seam layer -- ISOLATED from the frozen outdoor code.

The outdoor implementation lives in
``runs/I3-seam-20261107/seam_core.py`` + ``seam_cuda.py`` and hard-codes the
street geometry (4096-wide tensor, 384 stride, 512-wide extra windows, seam
band at H=400/W=2048). Those files stay untouched.

This module re-implements the SAME registered formulas on the indoor canvas
(1024x512, window 512, stride 64) with no hard-coded street constants:

  centrality(width)      identical formula
  support(start,width,W) identical formula
  semantic_risk(p)       identical formula
  choose(...)            identical algorithm, indoor candidate starts + band
  fuse(...)              identical algorithm, indoor base coverage
  fuse_cuda(...)         identical CUDA algorithm

B=0 strictly degrades to the base output (no extra windows, fusion identity),
matching the outdoor contract.
"""
from __future__ import annotations

import numpy as np

try:  # optional; CPU-only paths work without torch
    import torch
except Exception:  # pragma: no cover
    torch = None

from indoor_geometry import (
    INDOOR_W, INDOOR_H, INDOOR_WIN, INDOOR_STRIDE,
    semantic_starts, i3_candidate_starts, i3_base_coverage,
)


def centrality(width):
    """Same formula as outdoor seam_core.centrality."""
    x = (np.arange(width, dtype=np.float32) + .5) / width
    return .25 + .75 * np.sin(np.pi * x) ** 2


def support(start, width, W):
    """Same formula as outdoor seam_core.support (wraps modulo W)."""
    return (int(start) + np.arange(width)) % W


def seam_mask(H=INDOOR_H, W=INDOOR_W, band=None):
    """Left/right seam band column ranges.

    Outdoor uses band=128 on W=2048 (6.25% of width). We keep the same
    fraction for the indoor canvas so the risk weighting is comparable.
    """
    if band is None:
        band = max(1, int(round(W * 128 / 2048)))
    a = np.zeros((H, W), bool)
    a[:, :band] = True
    a[:, -band:] = True
    return a


def semantic_risk(p):
    """Same formula as outdoor seam_core.semantic_risk."""
    q = p.astype(np.float32)
    q /= np.maximum(q.sum(0, keepdims=True), 1e-8)
    ent = -(q * np.log(np.maximum(q, 1e-8))).sum(0) / np.log(q.shape[0])
    top = np.partition(q, -2, axis=0)[-2:]
    margin = top[-1] - top[-2]
    return .5 * ent + .5 * (1 - margin)


def choose(risk, starts, width, B=2, band=None, min_separation=0):
    """Identical algorithm to outdoor seam_core.choose."""
    H, W = risk.shape
    rr = np.asarray(risk, np.float32) * seam_mask(H, W, band)
    residual = rr.mean(0)
    quality = centrality(width)
    selected = []
    scores = []
    for k in range(min(B, len(starts))):
        choices = []
        for s in starts:
            if s in selected or any(min((s - t) % W, (t - s) % W) < min_separation for t in selected):
                continue
            cols = support(s, width, W)
            choices.append((float((residual[cols] * quality).sum()),
                            -abs(((s + width / 2 + W / 2) % W) - W / 2), -s, s))
        if not choices:
            break
        val, _, _, s = max(choices)
        selected.append(s)
        scores.append(val)
        cols = support(s, width, W)
        residual[cols] *= (1 - quality)
    return selected, {'selected': selected, 'marginal_scores': scores, 'GT_reads': 0}


def fuse(base, base_label, extras, base_coverage=None):
    """Identical algorithm to outdoor seam_core.fuse (coverage-aware)."""
    p = base.astype(np.float32)
    p /= np.maximum(p.sum(0, keepdims=True), 1e-8)
    if not extras:
        return p, base_label.copy(), {'B': 0, 'identity': True, 'changed_pixels': 0}
    C, H, W = p.shape
    bw = (np.ones((H, W), np.float32) if base_coverage is None
          else np.broadcast_to(base_coverage, (H, W)).astype(np.float32).copy())
    num = p * bw
    den = bw.copy()
    covered = np.zeros((H, W), bool)
    for s, ep in extras:
        e = ep.astype(np.float32)
        e /= np.maximum(e.sum(0, keepdims=True), 1e-8)
        ww = e.shape[-1]
        cols = support(s, ww, W)
        conf = np.clip(1 + ((e * np.log(np.maximum(e, 1e-8))).sum(0) / np.log(C)), .05, 1)
        w = centrality(ww)[None, :] * conf
        num[:, :, cols] += e * w
        den[:, cols] += w
        covered[:, cols] = True
    out = num / np.maximum(den, 1e-8)
    label = base_label.copy()
    label[covered] = out.argmax(0)[covered].astype(np.uint8)
    assert np.array_equal(label[~covered], base_label[~covered])
    return out, label, {'B': len(extras), 'changed_pixels': int((label != base_label).sum()),
                        'covered_pixels': int(covered.sum()), 'outside_coverage_changes': 0}


def base_coverage():
    """Indoor analogue of outdoor seam_core.base_coverage()."""
    return np.asarray(i3_base_coverage(), dtype=np.float32)[None, :]


def indoor_plan(budget=2, strategy='adaptive', risk=None):
    """I3 window plan for the indoor canvas.

    strategy: adaptive (risk-driven) | uniform | fixed
    Returns (starts, extra_window_width). For 'adaptive' the risk map must be
    supplied (it comes from the base prediction); without it we fall back to
    the uniform plan so the caller always gets a valid window count.
    """
    cands = i3_candidate_starts()
    if budget == 0:
        return [], INDOOR_WIN
    if strategy == 'fixed':
        return [cands[len(cands) // 2]] * budget, INDOOR_WIN
    if strategy == 'uniform':
        return ([cands[len(cands) // 2]] if budget == 1
                else [cands[0], cands[-1]]), INDOOR_WIN
    if risk is None:
        return ([cands[len(cands) // 2]] if budget == 1
                else [cands[0], cands[-1]]), INDOOR_WIN
    selected, _ = choose(risk, cands, INDOOR_WIN, B=budget,
                         min_separation=max(1, INDOOR_WIN // 4))
    return selected, INDOOR_WIN


def risk_cuda(base):
    """CUDA twin of the outdoor risk_cuda (identical formula)."""
    if torch is None:
        raise RuntimeError("torch unavailable")
    p = torch.as_tensor(base, device='cuda', dtype=torch.float32)
    p = p / p.sum(0, keepdim=True).clamp_min(1e-8)
    entropy = -(p * p.clamp_min(1e-8).log()).sum(0) / np.log(p.shape[0])
    top = p.topk(2, dim=0).values
    return (.5 * entropy + .5 * (1 - (top[0] - top[1]))).cpu().numpy()


def fuse_cuda(base, base_label, extras, base_coverage=None):
    """CUDA twin of the outdoor fuse_cuda (identical formula)."""
    if torch is None:
        raise RuntimeError("torch unavailable")
    p = torch.as_tensor(base, device='cuda', dtype=torch.float32)
    p = p / p.sum(0, keepdim=True).clamp_min(1e-8)
    if not extras:
        return p.cpu().numpy(), base_label.copy(), {'B': 0, 'identity': True}
    C, H, W = p.shape
    bw = torch.as_tensor(np.broadcast_to(base_coverage, (H, W)).copy(),
                         device='cuda', dtype=torch.float32)
    num = p * bw
    den = bw.clone()
    covered = torch.zeros(W, dtype=torch.bool, device='cuda')
    for s, ep in extras:
        e = torch.as_tensor(ep, device='cuda', dtype=torch.float32)
        e = e / e.sum(0, keepdim=True).clamp_min(1e-8)
        ww = e.shape[-1]
        cols = torch.as_tensor(support(s, ww, W), device='cuda', dtype=torch.long)
        confidence = (1 + (e * e.clamp_min(1e-8).log()).sum(0) / np.log(C)).clamp(.05, 1)
        w = torch.as_tensor(centrality(ww), device='cuda')[None, :] * confidence
        num.index_add_(2, cols, e * w)
        den.index_add_(1, cols, w)
        covered[cols] = True
    out = num / den.clamp_min(1e-8)
    label = torch.as_tensor(base_label.copy(), device='cuda')
    prediction = out.argmax(0).to(label.dtype)
    label[:, covered] = prediction[:, covered]
    return out.cpu().numpy(), label.cpu().numpy(), {'B': len(extras), 'identity': False}


def check_indoor_i3() -> dict:
    """CPU self-test: B=0 identity, coverage completeness, plan sanity."""
    rng = np.random.default_rng(0)
    C, H, W = 8, INDOOR_H, INDOOR_W
    base = rng.random((C, H, W)).astype(np.float32)
    base /= base.sum(0, keepdims=True)
    label = base.argmax(0).astype(np.uint8)
    cov = base_coverage()
    # B=0 must be identity
    p0, l0, info0 = fuse(base, label, [], cov)
    assert np.array_equal(l0, label), "B=0 must not change labels"
    assert info0['identity'] is True
    # plan
    risk = semantic_risk(base)
    starts, win = indoor_plan(budget=2, strategy='adaptive', risk=risk)
    assert len(starts) == 2 and win == INDOOR_WIN
    assert all(0 <= s < INDOOR_W for s in starts)
    # coverage: every column covered by base windows
    assert min(cov[0]) > 0
    return {
        "canvas": [W, H],
        "window": win,
        "B0_identity": True,
        "plan_b2": starts,
        "candidate_starts": i3_candidate_starts(),
        "coverage_min": float(cov.min()),
        "coverage_max": float(cov.max()),
        "seam_band_px": int(round(INDOOR_W * 128 / 2048)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(check_indoor_i3(), indent=2))
