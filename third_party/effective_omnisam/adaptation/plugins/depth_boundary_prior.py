from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DepthPriorOut:
    weight_map: torch.Tensor  # (B,H,W) float32
    inconsistent_ratio: float
    semantic_boundary_ratio: float
    depth_boundary_ratio: float
    weight_mean: float
    used: bool
    reason: str = ""


def semantic_boundary_map(pseudo_label: torch.Tensor, ignore_index: int, dilate: int = 1) -> torch.Tensor:
    """
    Compute a cheap semantic boundary map from an integer label map by neighbor differences.
    Returns (B,H,W) bool.
    """
    pl = pseudo_label
    if pl.ndim != 3:
        raise ValueError(f"pseudo_label must be (B,H,W), got {tuple(pl.shape)}")
    ign = int(ignore_index)

    # valid interior diffs (ignore_index excluded)
    valid = pl != ign
    b = torch.zeros_like(valid)

    # horizontal diffs
    v1 = valid[:, :, 1:] & valid[:, :, :-1]
    d1 = (pl[:, :, 1:] != pl[:, :, :-1]) & v1
    b[:, :, 1:] |= d1
    b[:, :, :-1] |= d1

    # vertical diffs
    v2 = valid[:, 1:, :] & valid[:, :-1, :]
    d2 = (pl[:, 1:, :] != pl[:, :-1, :]) & v2
    b[:, 1:, :] |= d2
    b[:, :-1, :] |= d2

    if int(dilate) and int(dilate) > 0:
        r = int(dilate)
        k = 2 * r + 1
        # bool -> float for maxpool, then back to bool
        b = F.max_pool2d(b.float().unsqueeze(1), kernel_size=k, stride=1, padding=r).squeeze(1) > 0.0

    return b


def build_depth_boundary_weight(
    pseudo_label: torch.Tensor,  # (B,H,W) int
    depth_boundary: Optional[torch.Tensor],  # (B,H,W) uint8/bool/float in {0,1}
    ignore_index: int,
    weight_on_inconsistent: float,
    semantic_boundary_dilate: int = 1,
) -> DepthPriorOut:
    device = pseudo_label.device
    if depth_boundary is None:
        wm = torch.ones_like(pseudo_label, dtype=torch.float32, device=device)
        return DepthPriorOut(
            weight_map=wm,
            inconsistent_ratio=0.0,
            semantic_boundary_ratio=0.0,
            depth_boundary_ratio=0.0,
            weight_mean=1.0,
            used=False,
            reason="depth_boundary_missing",
        )

    db = depth_boundary.to(device=device)
    if db.dtype != torch.bool:
        db = db.to(dtype=torch.float32)
        db = db > 0.5
    db = db & (pseudo_label != int(ignore_index))  # only consider valid pixels

    sem_b = semantic_boundary_map(pseudo_label, ignore_index=ignore_index, dilate=semantic_boundary_dilate)
    sem_b = sem_b & (pseudo_label != int(ignore_index))

    inconsistent = sem_b & (~db)
    wm = torch.ones_like(pseudo_label, dtype=torch.float32, device=device)
    w_bad = float(weight_on_inconsistent)
    if w_bad < 0.0:
        w_bad = 0.0
    if w_bad > 1.0:
        w_bad = 1.0
    wm[inconsistent] = w_bad

    total = int((pseudo_label != int(ignore_index)).sum().item())
    denom = max(1, total)
    sem_ratio = float(sem_b.sum().item()) / float(denom)
    db_ratio = float(db.sum().item()) / float(denom)
    inc_ratio = float(inconsistent.sum().item()) / float(denom)
    w_mean = float(wm[pseudo_label != int(ignore_index)].mean().item()) if total > 0 else 1.0

    return DepthPriorOut(
        weight_map=wm,
        inconsistent_ratio=inc_ratio,
        semantic_boundary_ratio=sem_ratio,
        depth_boundary_ratio=db_ratio,
        weight_mean=w_mean,
        used=True,
        reason="ok",
    )

