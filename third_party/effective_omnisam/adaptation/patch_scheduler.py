from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class PatchScheduleResult:
    selected_patch_indices: List[int]
    selected_patch_mask: List[bool]
    patch_score_map: List[float]
    patch_reduction_ratio: float
    selected_starts: List[int]
    selected_strides: List[int]
    complexity_threshold: float


class AdaptivePatchScheduler:
    """
    Lightweight patch scheduler for pseudo-label generation path.
    Baseline-compatible:
    - mode=none: keep all patches
    - mode=fixed_stride: deterministic subsampling by stride
    - mode=topk/two_stage: score-based top-k selection (minimal implementation)
    """

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "none",
        stride_scale: float = 1.0,
        topk: int = 0,
        score_source: str = "hybrid",
        baseline_stride: int = 384,
        high_complexity_stride: int = 384,
        low_complexity_stride: int = 768,
        complexity_threshold: float = 0.0,
        apply_stage: str = "pseudo_label",
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = (mode or "none").lower()
        self.stride_scale = max(1.0, float(stride_scale))
        self.topk = int(topk)
        self.score_source = (score_source or "hybrid").lower()
        self.baseline_stride = int(max(1, baseline_stride))
        self.high_complexity_stride = int(max(1, high_complexity_stride))
        self.low_complexity_stride = int(max(1, low_complexity_stride))
        self.complexity_threshold = float(complexity_threshold)
        self.apply_stage = (apply_stage or "pseudo_label").lower()

    def _all_selected(self, frame_cnt: int) -> PatchScheduleResult:
        idx = list(range(frame_cnt))
        return PatchScheduleResult(
            selected_patch_indices=idx,
            selected_patch_mask=[True] * frame_cnt,
            patch_score_map=[0.0] * frame_cnt,
            patch_reduction_ratio=0.0,
            selected_starts=[],
            selected_strides=[],
            complexity_threshold=float(self.complexity_threshold),
        )

    @staticmethod
    def _boundary_score(frames_bchw: torch.Tensor) -> torch.Tensor:
        """
        frames_bchw: [B, C, H, W]
        Returns score per frame [B], larger means structurally complex.
        """
        if frames_bchw.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        # grayscale proxy
        gray = frames_bchw.mean(dim=1, keepdim=False)
        dx = (gray[:, :, 1:] - gray[:, :, :-1]).abs().mean(dim=(1, 2))
        dy = (gray[:, 1:, :] - gray[:, :-1, :]).abs().mean(dim=(1, 2))
        return dx + dy

    @staticmethod
    def _entropy_score(frames_bchw: torch.Tensor) -> torch.Tensor:
        """
        Cheap local entropy proxy from intensity histogram per frame.
        frames_bchw: [B, C, H, W] in normalized tensor space.
        """
        if frames_bchw.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        gray = frames_bchw.mean(dim=1)  # [B,H,W]
        # normalize to [0,1]
        g = gray
        gmin = g.amin(dim=(1, 2), keepdim=True)
        gmax = g.amax(dim=(1, 2), keepdim=True)
        g = (g - gmin) / (gmax - gmin + 1e-6)
        bins = 16
        idx = torch.clamp((g * (bins - 1)).long(), 0, bins - 1)  # [B,H,W]
        scores = []
        for b in range(idx.shape[0]):
            hist = torch.bincount(idx[b].reshape(-1), minlength=bins).float()
            p = hist / (hist.sum() + 1e-6)
            ent = -(p * (p + 1e-8).log()).sum()
            scores.append(ent)
        return torch.stack(scores, dim=0)

    def _frame_complexity_score(self, frame_bchw: torch.Tensor) -> float:
        if self.score_source == "entropy":
            return float(self._entropy_score(frame_bchw).mean().item())
        if self.score_source == "laplacian":
            return float(self._boundary_score(frame_bchw).mean().item())
        # hybrid default
        e = float(self._entropy_score(frame_bchw).mean().item())
        b = float(self._boundary_score(frame_bchw).mean().item())
        return 0.5 * e + 0.5 * b

    def _score_frames(self, images_bfchw: torch.Tensor, frame_cnt: int) -> List[float]:
        # Use batch-mean complexity score per patch-frame.
        scores: List[float] = []
        for frame_idx in range(frame_cnt):
            frame = images_bfchw[:, frame_idx]  # [B, C, H, W]
            s = self._frame_complexity_score(frame)
            scores.append(float(s))
        return scores

    def _adaptive_stride_select(
        self,
        scores: List[float],
        frame_cnt: int,
        patch_w: int,
        full_w: int,
    ):
        """
        Build selected indices using high/low complexity strides over baseline patch positions.
        Keeps deterministic left-to-right order.
        """
        if frame_cnt <= 0:
            return [], [], [], float(self.complexity_threshold)
        starts = [i * self.baseline_stride for i in range(frame_cnt)]
        # threshold: explicit value or median score
        thr = self.complexity_threshold if self.complexity_threshold > 0 else float(torch.tensor(scores).median().item())
        selected_idx = []
        selected_starts = []
        selected_strides = []
        cur = 0
        selected_idx.append(0)
        selected_starts.append(starts[0])
        while cur < frame_cnt - 1:
            s = scores[cur]
            stride_px = self.high_complexity_stride if s >= thr else self.low_complexity_stride
            step = max(1, int(round(stride_px / max(1, self.baseline_stride))))
            cur = min(frame_cnt - 1, cur + step)
            selected_idx.append(cur)
            selected_starts.append(starts[cur])
            selected_strides.append(stride_px)
            if cur == frame_cnt - 1:
                break
        if selected_starts[-1] != starts[-1]:
            selected_idx.append(frame_cnt - 1)
            selected_starts.append(starts[-1])
            selected_strides.append(self.high_complexity_stride)
        # de-dup while preserving order
        dedup_idx, dedup_starts = [], []
        seen = set()
        for i, st in zip(selected_idx, selected_starts):
            if i in seen:
                continue
            dedup_idx.append(i)
            dedup_starts.append(st)
            seen.add(i)
        return dedup_idx, dedup_starts, selected_strides, float(thr)

    def select(self, images_bfchw: torch.Tensor, frame_cnt: int, patch_w: int = 0, full_w: int = 0) -> PatchScheduleResult:
        if frame_cnt <= 0:
            return PatchScheduleResult([], [], [], 0.0, [], [], float(self.complexity_threshold))
        if (not self.enabled) or self.mode == "none":
            return self._all_selected(frame_cnt)

        if self.mode == "fixed_stride":
            stride = max(1, int(round(self.stride_scale)))
            idx = list(range(0, frame_cnt, stride))
            if not idx:
                idx = [0]
            mask = [i in set(idx) for i in range(frame_cnt)]
            reduction = 1.0 - (len(idx) / float(frame_cnt))
            return PatchScheduleResult(
                selected_patch_indices=idx,
                selected_patch_mask=mask,
                patch_score_map=[0.0] * frame_cnt,
                patch_reduction_ratio=float(reduction),
                selected_starts=[],
                selected_strides=[],
                complexity_threshold=float(self.complexity_threshold),
            )

        if self.mode in {"topk", "two_stage"}:
            scores = self._score_frames(images_bfchw, frame_cnt)
            k = self.topk if self.topk > 0 else max(1, frame_cnt // int(round(self.stride_scale)))
            k = max(1, min(frame_cnt, k))
            ranked = sorted(range(frame_cnt), key=lambda i: scores[i], reverse=True)
            idx = sorted(ranked[:k])
            mask = [i in set(idx) for i in range(frame_cnt)]
            reduction = 1.0 - (len(idx) / float(frame_cnt))
            return PatchScheduleResult(
                selected_patch_indices=idx,
                selected_patch_mask=mask,
                patch_score_map=scores,
                patch_reduction_ratio=float(reduction),
                selected_starts=[],
                selected_strides=[],
                complexity_threshold=float(self.complexity_threshold),
            )

        if self.mode == "adaptive_stride":
            scores = self._score_frames(images_bfchw, frame_cnt)
            idx, starts, strides, thr = self._adaptive_stride_select(scores, frame_cnt, patch_w, full_w)
            mask = [i in set(idx) for i in range(frame_cnt)]
            reduction = 1.0 - (len(idx) / float(frame_cnt))
            return PatchScheduleResult(
                selected_patch_indices=idx,
                selected_patch_mask=mask,
                patch_score_map=scores,
                patch_reduction_ratio=float(reduction),
                selected_starts=starts,
                selected_strides=strides,
                complexity_threshold=float(thr),
            )

        # Unknown mode -> fallback baseline-safe
        return self._all_selected(frame_cnt)
