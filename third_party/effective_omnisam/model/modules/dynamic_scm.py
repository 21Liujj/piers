"""Semantic-density guided circular margin padding."""

import torch
import torch.nn as nn


class DynamicSCM(nn.Module):
    """Apply density-guided horizontal circular padding on feature maps.

    ``min_pad_width`` and ``max_pad_width`` are expressed in source-image
    pixels, while the module operates on a feature map. ``feature_stride``
    converts them to feature columns and prevents source-scale values from
    accidentally being sliced as feature columns.
    """

    def __init__(self, base_pad_width=1024, min_pad_width=512, max_pad_width=1536,
                 in_channels=256, feature_stride=16):
        super().__init__()
        if min_pad_width <= 0 or min_pad_width > base_pad_width or base_pad_width > max_pad_width:
            raise ValueError("padding widths must satisfy 0 < min <= base <= max")
        self.base_pad_width = int(base_pad_width)
        self.min_pad_width = int(min_pad_width)
        self.max_pad_width = int(max_pad_width)
        self.feature_stride = max(1, int(feature_stride))
        self.in_channels = int(in_channels)
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Start the boundary branch as an identity residual.
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        self.last_attention_mask = None
        self.last_pad_widths = None

    def compute_boundary_density(self, density_map, boundary_width=32):
        if density_map.ndim != 4 or density_map.shape[1] != 1:
            raise ValueError("density_map must have shape (B, 1, H, W)")
        width = density_map.shape[-1]
        boundary_width = min(max(1, int(boundary_width)), max(1, width // 2))
        left = density_map[..., :boundary_width].mean(dim=(1, 2, 3))
        right = density_map[..., -boundary_width:].mean(dim=(1, 2, 3))
        return ((left + right) * 0.5).clamp(0.0, 1.0)

    def compute_adaptive_pad_width(self, boundary_density, feature_width=None, in_feature_units=False):
        density = boundary_density.to(dtype=torch.float32).clamp(0.0, 1.0)
        source_pad = self.base_pad_width + (density - 0.5) * (self.max_pad_width - self.min_pad_width)
        source_pad = source_pad.clamp(self.min_pad_width, self.max_pad_width)
        if in_feature_units:
            pad = torch.ceil(source_pad / self.feature_stride).to(dtype=torch.long)
            if feature_width is not None:
                # Circular indexing supports pads wider than W, but limiting it
                # avoids needless memory for tiny test feature maps.
                pad = pad.clamp(min=1, max=max(1, int(feature_width)))
            return pad
        return torch.round(source_pad).to(dtype=torch.long)

    @staticmethod
    def _circular_columns(features, pad_width):
        _, _, _, width = features.shape
        indices = torch.arange(-pad_width, width + pad_width, device=features.device)
        indices = indices.remainder(width)
        return features.index_select(-1, indices)

    def apply_circular_padding(self, features, pad_width):
        if features.ndim != 4 or features.shape[-1] < 1:
            raise ValueError("features must be a non-empty BCHW tensor")
        return self._circular_columns(features, int(pad_width))

    def apply_batch_circular_padding(self, features, pad_widths):
        if pad_widths.ndim != 1 or pad_widths.shape[0] != features.shape[0]:
            raise ValueError("pad_widths must contain one value per batch sample")
        max_pad = int(pad_widths.max().item())
        padded = self.apply_circular_padding(features, max_pad)
        # True denotes a real token; consumers can convert it to an attention
        # additive mask without guessing which columns belong to the image.
        mask = torch.zeros((features.shape[0], 1, features.shape[2], features.shape[3] + 2 * max_pad),
                           dtype=torch.bool, device=features.device)
        mask[..., max_pad:max_pad + features.shape[3]] = True
        self.last_attention_mask = mask
        self.last_pad_widths = pad_widths.detach().clone()
        return padded, max_pad

    def pad_tensor(self, tensor, pad_width):
        """Pad a BCHW feature or position tensor using the same SCM geometry."""
        return self.apply_circular_padding(tensor, int(pad_width))

    def remove_circular_padding(self, padded_features, pad_width):
        pad_width = int(pad_width)
        if pad_width <= 0:
            return padded_features
        width = padded_features.shape[-1]
        if width <= 2 * pad_width:
            raise ValueError("padded feature width is too small for the requested padding")
        return padded_features[..., pad_width:-pad_width]

    def forward(self, features, density_map, return_pad_width=False, return_attention_mask=False):
        if features.ndim != 4 or density_map.ndim != 4:
            raise ValueError("features and density_map must be BCHW tensors")
        if features.shape[0] != density_map.shape[0] or features.shape[2:] != density_map.shape[2:]:
            raise ValueError("features and density_map must have matching batch and spatial sizes")
        density = self.compute_boundary_density(density_map)
        pad_widths = self.compute_adaptive_pad_width(density, features.shape[-1], in_feature_units=True)
        padded, actual_pad = self.apply_batch_circular_padding(features, pad_widths)
        boundary_delta = self.boundary_conv(padded)
        padded = padded + torch.tanh(self.residual_gate) * boundary_delta
        if return_attention_mask:
            return padded, actual_pad, self.last_attention_mask
        if return_pad_width:
            return padded, actual_pad
        return padded

    def forward_with_removal(self, features, density_map):
        padded, pad_width, mask = self.forward(features, density_map, return_attention_mask=True)
        output = self.remove_circular_padding(padded, pad_width)
        density = self.compute_boundary_density(density_map)
        stats = {
            "original_width": int(features.shape[-1]),
            "padded_width": int(padded.shape[-1]),
            "pad_width": int(pad_width),
            "pad_widths": self.last_pad_widths.tolist(),
            "attention_mask_shape": tuple(mask.shape),
            "avg_boundary_density": float(density.mean().item()),
            "min_boundary_density": float(density.min().item()),
            "max_boundary_density": float(density.max().item()),
        }
        return output, stats
