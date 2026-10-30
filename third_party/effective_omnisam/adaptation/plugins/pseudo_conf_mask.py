from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F


@dataclass
class MaskedUnsupLossOut:
    loss_raw: torch.Tensor
    loss_masked: torch.Tensor
    valid_ratio: float
    mean_confidence: float
    filtered_ratio: float
    valid_pixels: int
    total_pixels: int
    fallback_used: Optional[str] = None


def masked_unsup_ce(
    pred_logits: torch.Tensor,  # (B,C,H,W)
    pseudo_label: torch.Tensor,  # (B,H,W) int64/long
    conf_map: torch.Tensor,  # (B,H,W) float
    base_mask: Optional[torch.Tensor],  # (B,H,W) bool, e.g. text-gated mask; None means all True
    threshold: Union[float, torch.Tensor],
    ignore_index: int,
    min_valid_ratio: float,
    min_valid_pixels: int,
    fallback: str,  # skip|original|zero
    weight_map: Optional[torch.Tensor] = None,  # (B,H,W) float
) -> MaskedUnsupLossOut:
    device = pred_logits.device
    ignore_index = int(ignore_index)
    thr_t: Optional[torch.Tensor] = None
    if torch.is_tensor(threshold):
        thr_t = threshold.to(device=device, dtype=torch.float32)
    else:
        thr_t = None
        thr = float(threshold)

    if base_mask is None:
        base_mask = torch.ones_like(pseudo_label, dtype=torch.bool, device=device)
    else:
        base_mask = base_mask.to(device=device, dtype=torch.bool)

    pseudo_label = pseudo_label.to(device=device)
    conf_map = conf_map.to(device=device, dtype=torch.float32)

    valid = (pseudo_label != ignore_index) & base_mask
    if thr_t is not None and thr_t.shape == conf_map.shape:
        conf_keep = conf_map >= thr_t
    else:
        conf_keep = conf_map >= float(thr if "thr" in locals() else 0.0)
    keep = valid & conf_keep

    # Raw loss: preserve baseline behavior (mask via ignore + default reduction)
    # Here we compute it explicitly for logging and fallback.
    pseudo_filtered = pseudo_label.clone()
    pseudo_filtered[~keep] = ignore_index
    loss_raw = F.cross_entropy(pred_logits, pseudo_filtered.long(), ignore_index=ignore_index, reduction="mean")

    ce = F.cross_entropy(pred_logits, pseudo_label.long(), ignore_index=ignore_index, reduction="none")  # (B,H,W)
    if weight_map is not None:
        wm = weight_map.to(device=device, dtype=ce.dtype)
        if wm.shape == ce.shape:
            ce = ce * wm
    ce = ce * keep.to(dtype=ce.dtype)

    total = int(pseudo_label.numel())
    valid_pixels = int(keep.sum().item())
    denom = max(1, total)
    valid_ratio = float(valid_pixels / denom)
    mean_conf = float(conf_map[valid].mean().item()) if valid.any() else 0.0
    filtered_ratio = float(1.0 - (valid_pixels / max(1, int(valid.sum().item())))) if valid.any() else 1.0

    # Safe normalized masked loss
    if valid_pixels > 0:
        loss_masked = ce.sum() / float(valid_pixels)
        fallback_used = None
    else:
        loss_masked = pred_logits.new_tensor(0.0)
        fallback_used = "zero"

    # Fallback control if too few valid pixels
    too_few = (valid_ratio < float(min_valid_ratio)) or (valid_pixels < int(min_valid_pixels))
    fb = str(fallback or "skip").lower()
    if too_few:
        if fb == "original":
            loss_masked = loss_raw
            fallback_used = "original"
        elif fb == "zero":
            loss_masked = pred_logits.new_tensor(0.0)
            fallback_used = "zero"
        else:
            # skip => treat unsup as zero (but report it distinctly)
            loss_masked = pred_logits.new_tensor(0.0)
            fallback_used = "skip"

    return MaskedUnsupLossOut(
        loss_raw=loss_raw,
        loss_masked=loss_masked,
        valid_ratio=valid_ratio,
        mean_confidence=mean_conf,
        filtered_ratio=filtered_ratio,
        valid_pixels=valid_pixels,
        total_pixels=total,
        fallback_used=fallback_used,
    )

