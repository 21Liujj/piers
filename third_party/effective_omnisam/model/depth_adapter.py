from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_norm(num_channels: int, norm: str = "gn") -> nn.Module:
    mode = str(norm).lower()
    if mode == "bn":
        return nn.BatchNorm2d(num_channels)
    # default: GroupNorm, stable for small-batch training
    groups = 8 if num_channels % 8 == 0 else 1
    return nn.GroupNorm(groups, num_channels)


class LightDepthAdapter(nn.Module):
    def __init__(
        self,
        out_channels: int,
        mid_channels: int = 32,
        gate_init: float = 0.0,
        norm: str = "gn",
    ):
        super().__init__()
        c1 = int(mid_channels)
        c2 = int(mid_channels) * 2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            _build_norm(c1, norm=norm),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            _build_norm(c2, norm=norm),
            nn.GELU(),
            nn.Conv2d(c2, int(out_channels), kernel_size=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def forward(self, feature: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        if depth is None:
            return feature
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4:
            return feature

        depth = depth.to(device=feature.device, dtype=feature.dtype)
        if depth.shape[-2:] != feature.shape[-2:]:
            depth = F.interpolate(depth, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        depth_feat = self.encoder(depth)
        return feature + self.alpha.to(dtype=feature.dtype) * depth_feat

