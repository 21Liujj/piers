from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DepthConfidenceOut:
    weight_map: torch.Tensor
    weight_mean: float
    weight_min: float
    weight_max: float
    edge_conflict_ratio: float
    smooth_noisy_ratio: float
    used_depth: bool


def _normalize_per_image(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    x_flat = x.view(b, -1)
    x_min = x_flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    x_max = x_flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    denom = (x_max - x_min).clamp_min(1e-6)
    return (x - x_min) / denom


def compute_depth_edges(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim != 4:
        raise ValueError(f"depth must be (B,1,H,W), got {tuple(depth.shape)}")
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=depth.device, dtype=depth.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=depth.device, dtype=depth.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(depth, sobel_x, padding=1)
    gy = F.conv2d(depth, sobel_y, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return _normalize_per_image(grad)


def compute_label_boundary(pseudo_label: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    if pseudo_label.ndim != 3:
        raise ValueError(f"pseudo_label must be (B,H,W), got {tuple(pseudo_label.shape)}")
    pl = pseudo_label
    valid = pl != int(ignore_index)
    bnd = torch.zeros_like(valid)

    v1 = valid[:, :, 1:] & valid[:, :, :-1]
    d1 = (pl[:, :, 1:] != pl[:, :, :-1]) & v1
    bnd[:, :, 1:] |= d1
    bnd[:, :, :-1] |= d1

    v2 = valid[:, 1:, :] & valid[:, :-1, :]
    d2 = (pl[:, 1:, :] != pl[:, :-1, :]) & v2
    bnd[:, 1:, :] |= d2
    bnd[:, :-1, :] |= d2
    return bnd


def _adaptive_binary_mask(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    thr = x.view(b, -1).mean(dim=1, keepdim=True).view(b, 1, 1, 1)
    return x > thr


def compute_edge_consistency_weight(
    depth: torch.Tensor,
    pseudo_label: torch.Tensor,
    ignore_index: int = 255,
    edge_beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth_edges = compute_depth_edges(depth)
    depth_edge_mask = _adaptive_binary_mask(depth_edges)
    depth_edge_mask = F.max_pool2d(depth_edge_mask.float(), kernel_size=3, stride=1, padding=1) > 0.0

    sem_boundary = compute_label_boundary(pseudo_label, ignore_index=ignore_index)
    sem_boundary = F.max_pool2d(sem_boundary.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0.0
    sem_boundary = sem_boundary & (pseudo_label != int(ignore_index))

    conflict = sem_boundary & (~depth_edge_mask.squeeze(1))
    w_edge = 1.0 - float(edge_beta) * conflict.float()
    return w_edge, conflict


def compute_smooth_consistency_weight(
    depth: torch.Tensor,
    pseudo_label: torch.Tensor,
    ignore_index: int = 255,
    smooth_beta: float = 0.5,
    kernel_size: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad = kernel_size // 2
    mean_d = F.avg_pool2d(depth, kernel_size=kernel_size, stride=1, padding=pad)
    mean_d2 = F.avg_pool2d(depth * depth, kernel_size=kernel_size, stride=1, padding=pad)
    var_d = (mean_d2 - mean_d * mean_d).clamp_min(0.0)

    b = var_d.shape[0]
    thr = var_d.view(b, -1).mean(dim=1, keepdim=True).view(b, 1, 1, 1)
    smooth_mask = var_d < thr

    boundary = compute_label_boundary(pseudo_label, ignore_index=ignore_index).float().unsqueeze(1)
    boundary_density = F.avg_pool2d(boundary, kernel_size=kernel_size, stride=1, padding=pad)
    bd_thr = boundary_density.view(b, -1).mean(dim=1, keepdim=True).view(b, 1, 1, 1)
    high_var = boundary_density > bd_thr

    noisy = smooth_mask & high_var & (pseudo_label.unsqueeze(1) != int(ignore_index))
    w_smooth = 1.0 - float(smooth_beta) * noisy.float().squeeze(1)
    return w_smooth, noisy.squeeze(1)


def compute_depth_confidence_weight(
    depth: Optional[torch.Tensor],
    pseudo_label: torch.Tensor,
    cfg: dict,
) -> DepthConfidenceOut:
    device = pseudo_label.device
    base = torch.ones_like(pseudo_label, dtype=torch.float32, device=device)
    valid = pseudo_label != int(cfg.get("ignore_index", 255))
    if depth is None:
        return DepthConfidenceOut(
            weight_map=base,
            weight_mean=1.0,
            weight_min=1.0,
            weight_max=1.0,
            edge_conflict_ratio=0.0,
            smooth_noisy_ratio=0.0,
            used_depth=False,
        )

    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.shape[-2:] != pseudo_label.shape[-2:]:
        depth = F.interpolate(depth, size=pseudo_label.shape[-2:], mode="bilinear", align_corners=False)
    depth = depth.to(device=device, dtype=torch.float32)

    weight = base
    conflict = torch.zeros_like(valid)
    noisy = torch.zeros_like(valid)

    if bool(cfg.get("edge_consistency", True)):
        w_edge, conflict = compute_edge_consistency_weight(
            depth=depth,
            pseudo_label=pseudo_label,
            ignore_index=int(cfg.get("ignore_index", 255)),
            edge_beta=float(cfg.get("edge_beta", 0.5)),
        )
        weight = weight * w_edge

    if bool(cfg.get("smooth_region_consistency", True)):
        w_smooth, noisy = compute_smooth_consistency_weight(
            depth=depth,
            pseudo_label=pseudo_label,
            ignore_index=int(cfg.get("ignore_index", 255)),
            smooth_beta=float(cfg.get("smooth_beta", 0.5)),
            kernel_size=int(cfg.get("smooth_kernel", 7)),
        )
        weight = weight * w_smooth

    wmin = float(cfg.get("weight_min", 0.3))
    wmax = float(cfg.get("weight_max", 1.0))
    weight = weight.clamp(min=wmin, max=wmax)
    weight = torch.where(valid, weight, torch.ones_like(weight))

    valid_denom = max(1, int(valid.sum().item()))
    w_valid = weight[valid]
    return DepthConfidenceOut(
        weight_map=weight,
        weight_mean=float(w_valid.mean().item()) if w_valid.numel() > 0 else 1.0,
        weight_min=float(w_valid.min().item()) if w_valid.numel() > 0 else 1.0,
        weight_max=float(w_valid.max().item()) if w_valid.numel() > 0 else 1.0,
        edge_conflict_ratio=float((conflict & valid).sum().item() / valid_denom),
        smooth_noisy_ratio=float((noisy & valid).sum().item() / valid_denom),
        used_depth=True,
    )

