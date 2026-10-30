from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryCompressStats:
    original_num_tokens: int
    compressed_num_tokens: int
    compression_ratio: float
    original_hw: Tuple[int, int]
    compressed_hw: Tuple[int, int]
    compression_mode: str
    compression_location: str
    skipped: bool = False
    skip_reason: str = ""


class SpatialMemoryCompressor(nn.Module):
    """EfficientTAM-style non-learnable spatial memory pooling."""

    def __init__(
        self,
        embed_dim: int,
        mode: str = "none",
        kernel_size: int = 2,
        stride: int = 2,
        target_size: Optional[Tuple[int, int]] = None,
        keep_global_tokens: int = 0,
        apply_to_pos_enc: bool = True,
        min_tokens: int = 64,
        compression_location: str = "before_attention",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.mode = str(mode or "none").lower()
        self.kernel_size = max(1, int(kernel_size))
        self.stride = max(1, int(stride))
        self.target_size = target_size
        self.keep_global_tokens = max(0, int(keep_global_tokens))
        self.apply_to_pos_enc = bool(apply_to_pos_enc)
        self.min_tokens = max(0, int(min_tokens))
        self.compression_location = str(compression_location or "before_attention").lower()

    def configure(
        self,
        mode: Optional[str] = None,
        kernel_size: Optional[int] = None,
        stride: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
        keep_global_tokens: Optional[int] = None,
        apply_to_pos_enc: Optional[bool] = None,
        min_tokens: Optional[int] = None,
        compression_location: Optional[str] = None,
        downsample_ratio: Optional[int] = None,  # backward compatibility
        window_size: Optional[int] = None,  # backward compatibility
        global_tokens: Optional[int] = None,  # backward compatibility
    ) -> None:
        if mode is not None:
            self.mode = str(mode).lower()
        if kernel_size is not None:
            self.kernel_size = max(1, int(kernel_size))
        elif downsample_ratio is not None:
            self.kernel_size = max(1, int(downsample_ratio))
        if stride is not None:
            self.stride = max(1, int(stride))
        elif downsample_ratio is not None:
            self.stride = max(1, int(downsample_ratio))
        if target_size is not None:
            self.target_size = target_size
        if keep_global_tokens is not None:
            self.keep_global_tokens = max(0, int(keep_global_tokens))
        elif global_tokens is not None:
            self.keep_global_tokens = max(0, int(global_tokens))
        if apply_to_pos_enc is not None:
            self.apply_to_pos_enc = bool(apply_to_pos_enc)
        if min_tokens is not None:
            self.min_tokens = max(0, int(min_tokens))
        if compression_location is not None:
            self.compression_location = str(compression_location).lower()
        if window_size is not None and self.mode in {"spatial", "hybrid"}:
            self.kernel_size = max(1, int(window_size))
            self.stride = max(1, int(window_size))

    @staticmethod
    def _tokens_to_map(tokens_lbc: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = int(hw[0]), int(hw[1])
        l, b, c = tokens_lbc.shape
        if h * w != l:
            raise ValueError(f"Memory token length mismatch: L={l}, HxW={h*w}, hw={hw}")
        return tokens_lbc.permute(1, 2, 0).reshape(b, c, h, w)

    @staticmethod
    def _map_to_tokens(feat_bchw: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat_bchw.shape
        return feat_bchw.reshape(b, c, h * w).permute(2, 0, 1)

    @staticmethod
    def _nan_guard(name: str, tensor: torch.Tensor) -> None:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"{name} contains NaN/Inf after compression")

    def _build_stats(
        self,
        original_tokens: int,
        compressed_tokens: int,
        original_hw: Tuple[int, int],
        compressed_hw: Tuple[int, int],
        skipped: bool = False,
        skip_reason: str = "",
    ) -> MemoryCompressStats:
        return MemoryCompressStats(
            original_num_tokens=original_tokens,
            compressed_num_tokens=compressed_tokens,
            compression_ratio=float(compressed_tokens / max(1, original_tokens)),
            original_hw=original_hw,
            compressed_hw=compressed_hw,
            compression_mode=self.mode,
            compression_location=self.compression_location,
            skipped=bool(skipped),
            skip_reason=str(skip_reason),
        )

    def _pool2d(self, feat_bchw: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
        if kernel_size <= 1 and stride <= 1:
            return feat_bchw
        return F.avg_pool2d(feat_bchw, kernel_size=kernel_size, stride=stride, ceil_mode=False)

    def compress(
        self,
        memory_lbc: torch.Tensor,
        memory_pos_lbc: torch.Tensor,
        hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, Tuple[int, int], MemoryCompressStats]:
        l0 = int(memory_lbc.shape[0])
        hw0 = (int(hw[0]), int(hw[1]))
        if self.mode == "none":
            return memory_lbc, memory_pos_lbc, hw0, self._build_stats(l0, l0, hw0, hw0, skipped=True, skip_reason="mode=none")
        if l0 <= self.min_tokens:
            return memory_lbc, memory_pos_lbc, hw0, self._build_stats(l0, l0, hw0, hw0, skipped=True, skip_reason="below_min_tokens")

        mem_map = self._tokens_to_map(memory_lbc, hw0)
        pos_map = self._tokens_to_map(memory_pos_lbc, hw0)
        if self.mode == "pool":
            mem_local = self._pool2d(mem_map, self.kernel_size, self.stride)
            pos_local = self._pool2d(pos_map, self.kernel_size, self.stride)
            mem_out = self._map_to_tokens(mem_local)
            pos_out = self._map_to_tokens(pos_local)
            hw2 = (int(mem_local.shape[-2]), int(mem_local.shape[-1]))
        elif self.mode == "spatial":
            if self.target_size is not None:
                target_h, target_w = int(self.target_size[0]), int(self.target_size[1])
                mem_local = F.adaptive_avg_pool2d(mem_map, (target_h, target_w))
                pos_local = F.adaptive_avg_pool2d(pos_map, (target_h, target_w))
            else:
                mem_local = self._pool2d(mem_map, self.kernel_size, self.stride)
                pos_local = self._pool2d(pos_map, self.kernel_size, self.stride)
            mem_out = self._map_to_tokens(mem_local)
            pos_out = self._map_to_tokens(pos_local)
            hw2 = (int(mem_local.shape[-2]), int(mem_local.shape[-1]))
        elif self.mode == "hybrid":
            mem_local = self._pool2d(mem_map, self.kernel_size, self.stride)
            pos_local = self._pool2d(pos_map, self.kernel_size, self.stride)
            local_tokens = self._map_to_tokens(mem_local)
            local_pos_tokens = self._map_to_tokens(pos_local)
            if self.keep_global_tokens > 0:
                global_mem_map = F.adaptive_avg_pool2d(mem_map, (1, self.keep_global_tokens))
                global_mem_tokens = self._map_to_tokens(global_mem_map)[: self.keep_global_tokens]
                global_pos_map = F.adaptive_avg_pool2d(pos_map, (1, self.keep_global_tokens))
                global_pos_tokens = self._map_to_tokens(global_pos_map)[: self.keep_global_tokens]
                mem_out = torch.cat([local_tokens, global_mem_tokens], dim=0)
                pos_out = torch.cat([local_pos_tokens, global_pos_tokens], dim=0)
            else:
                mem_out, pos_out = local_tokens, local_pos_tokens
            hw2 = (int(mem_local.shape[-2]), int(mem_local.shape[-1]))
        else:
            return memory_lbc, memory_pos_lbc, hw0, self._build_stats(l0, l0, hw0, hw0, skipped=True, skip_reason="unknown_mode")

        self._nan_guard("compressed_memory", mem_out)
        self._nan_guard("compressed_pos", pos_out)
        l1 = int(mem_out.shape[0])
        return mem_out, pos_out, hw2, self._build_stats(l0, l1, hw0, hw2)

