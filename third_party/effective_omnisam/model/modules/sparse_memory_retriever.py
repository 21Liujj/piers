from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time

import torch


@dataclass
class SparseRetrieveStats:
    original_num_tokens: int
    selected_num_tokens: int
    keep_ratio: float
    selection_mode: str
    cache_hit: bool
    memory_id: str
    original_hw: Tuple[int, int]
    selected_token_shape: Tuple[int, int, int]
    fallback_used: bool = False
    skip_reason: str = ""
    selection_time: float = 0.0
    mask_shape_mismatch: bool = False


class SparseMemoryRetriever:
    """Efficient-SAM2-style sparse memory retrieval with safe fallback paths."""

    def __init__(self) -> None:
        self._cached_indices: Dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        self._cached_indices.clear()

    def pop(self, memory_id: str) -> None:
        if memory_id in self._cached_indices:
            self._cached_indices.pop(memory_id, None)

    @staticmethod
    def _safe_topk_count(
        num_tokens: int,
        keep_ratio: float,
        min_tokens: int,
        max_tokens: Optional[int],
    ) -> int:
        k = int(round(float(num_tokens) * float(keep_ratio)))
        k = max(k, int(min_tokens), 1)
        if max_tokens is not None:
            k = min(k, int(max_tokens))
        return min(k, int(num_tokens))

    @staticmethod
    def feature_norm_score(tokens_lbc: torch.Tensor) -> torch.Tensor:
        # Existing SMR score is batch-averaged, so Stage 1 FoV quotas are batch-shared.
        score = torch.linalg.norm(tokens_lbc.float(), dim=-1).mean(dim=1)
        return score

    _feature_norm_score = feature_norm_score

    @staticmethod
    def _random_score(tokens_lbc: torch.Tensor) -> torch.Tensor:
        return torch.rand(tokens_lbc.shape[0], device=tokens_lbc.device, dtype=torch.float32)

    @staticmethod
    def _cams_score(
        memory_tokens: torch.Tensor,
        query_features: Optional[torch.Tensor] = None,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
    ) -> torch.Tensor:
        """
        CAMS: Class-Aware Memory Selection via query-affinity + scale-adaptive weighting.

        Args:
            memory_tokens: [L_m, B, C] historical memory tokens (matches feature_norm_score's tokens_lbc convention)
                           2D [L_m, C] is also accepted (B implicitly 1).
            query_features: [L_q, C] current frame vision features (from encoder), single batch slice.
            alpha: weight for query-affinity component
            beta: weight for feature-norm component
            gamma: weight for scale-adaptive component

        Returns:
            score: [L_m] importance score for each memory token
        """
        import torch.nn.functional as F

        if memory_tokens.dim() == 2:
            memory_tokens_lbc = memory_tokens.unsqueeze(1)  # [L_m, 1, C]
        else:
            memory_tokens_lbc = memory_tokens  # [L_m, B, C]

        L_m = memory_tokens_lbc.shape[0]
        # Use batch-mean 2D representation [L_m, C] for affinity (batch-shared score, same convention as feature_norm_score)
        memory_2d = memory_tokens_lbc.float().mean(dim=1)  # [L_m, C]

        # Component 1: Feature norm (baseline importance), batch-averaged like feature_norm_score
        feat_norm = torch.linalg.norm(memory_tokens_lbc.float(), dim=-1).mean(dim=1)  # [L_m]

        # Component 2: Query-affinity (query-aware selection)
        if query_features is not None and query_features.numel() > 0:
            query_2d = query_features.float()
            if query_2d.dim() > 2:
                query_2d = query_2d.reshape(-1, query_2d.shape[-1])
            query_norm = F.normalize(query_2d, dim=-1)  # [L_q, C]
            memory_norm = F.normalize(memory_2d, dim=-1)  # [L_m, C]
            # Cross-attention scores: best match across all queries
            affinity_matrix = torch.matmul(query_norm, memory_norm.T)  # [L_q, L_m]
            affinity = affinity_matrix.max(dim=0).values  # [L_m]
        else:
            affinity = torch.zeros_like(feat_norm)

        # Component 3: Scale-adaptive weight (boost small-norm tokens to prevent small-object under-representation)
        tau = feat_norm.median()  # threshold = median norm
        sigma = feat_norm.std() + 1e-6  # softness
        scale_weight = torch.sigmoid((tau - feat_norm) / sigma)  # [L_m], inverse scaling

        # Weighted combination
        score = alpha * affinity + beta * feat_norm + gamma * scale_weight

        return score

    def _boundary_guarded_selection(
        self,
        memory_tokens: torch.Tensor,
        boundary_mask_hw: torch.Tensor,
        quota: int,
        boundary_reserve_ratio: float = 0.125,
        base_score: Optional[torch.Tensor] = None,  # NEW: override feature_norm (e.g. uncertainty)
    ) -> Tuple[torch.Tensor, bool]:
        """
        Route 1 (Phase 1): Boundary-guarded memory selection

        Stage 1: Reserve boundary_reserve_ratio of quota for boundary tokens (by score)
        Stage 2: Fill remaining quota with score-based Top-K (excluding selected)

        Args:
            memory_tokens: [L_m, B, C] memory token features (batch-averaged scoring)
            boundary_mask_hw: [H, W] or [L_m] bool/float boundary mask
            quota: total number of tokens to select
            boundary_reserve_ratio: fraction of quota reserved for boundary (default 0.125 = 12.5%)
            base_score: optional [L_m] score overriding feature_norm (e.g. uncertainty/entropy)

        Returns:
            (indices, mismatch): [quota] selected token indices (sorted), and whether an
            internal shape mismatch forced a fallback to pure feature-norm Top-K.
        """
        L_m = memory_tokens.shape[0]
        device = memory_tokens.device

        # Flatten boundary mask if needed
        if boundary_mask_hw.ndim == 2:
            boundary_mask = boundary_mask_hw.flatten().to(device)
        else:
            boundary_mask = boundary_mask_hw.to(device)

        if boundary_mask.shape[0] != L_m:
            # Shape mismatch, fallback to pure norm
            feat_norm = torch.linalg.norm(memory_tokens.float(), dim=-1).mean(dim=1)  # [L_m], batch-averaged
            return torch.topk(feat_norm, k=quota).indices, True

        if base_score is not None and base_score.shape[0] == L_m:
            feat_norm = base_score.to(device).float()  # NEW: use uncertainty (or other) score
        else:
            feat_norm = torch.linalg.norm(memory_tokens.float(), dim=-1).mean(dim=1)  # [L_m], batch-averaged
        selected = torch.zeros(L_m, dtype=torch.bool, device=device)

        # === Stage 1: Boundary insurance ===
        boundary_quota = int(quota * boundary_reserve_ratio)
        boundary_tokens = torch.where(boundary_mask > 0)[0]

        if len(boundary_tokens) > 0 and boundary_quota > 0:
            boundary_scores = feat_norm[boundary_tokens]
            k_boundary = min(boundary_quota, len(boundary_tokens))
            topk_in_boundary = torch.topk(boundary_scores, k=k_boundary).indices
            selected[boundary_tokens[topk_in_boundary]] = True

        # === Stage 2: Norm-based fill ===
        remaining_quota = quota - selected.sum().item()
        if remaining_quota > 0:
            not_selected = torch.where(~selected)[0]
            if len(not_selected) > 0:
                remaining_scores = feat_norm[not_selected]
                k_fill = min(remaining_quota, len(not_selected))
                topk_fill = torch.topk(remaining_scores, k=k_fill).indices
                selected[not_selected[topk_fill]] = True

        indices = torch.where(selected)[0]
        return torch.sort(indices).values, False  # sorted for positional encoding consistency

    def _class_aware_selection(
        self,
        memory_tokens: torch.Tensor,
        rare_class_mask_hw: torch.Tensor,
        quota: int,
        rare_reserve_ratio: float = 0.15,
        base_score: Optional[torch.Tensor] = None,  # NEW: override feature_norm (e.g. uncertainty)
    ) -> Tuple[torch.Tensor, bool]:
        """
        Route 3 (Class-Aware): Rare-class-aware memory selection

        Stage 1: Reserve rare_reserve_ratio of quota for rare-class tokens (by score)
        Stage 2: Fill remaining quota with score-based Top-K (excluding selected)

        Args:
            memory_tokens: [L_m, B, C] memory token features (batch-averaged scoring)
            rare_class_mask_hw: [H, W] or [L_m] bool/float rare class mask
            quota: total number of tokens to select
            rare_reserve_ratio: fraction of quota reserved for rare classes (default 0.15 = 15%)
            base_score: optional [L_m] score overriding feature_norm (e.g. uncertainty/entropy)

        Returns:
            (indices, mismatch): [quota] selected token indices (sorted), and whether an
            internal shape mismatch forced a fallback to pure feature-norm Top-K.
        """
        L_m = memory_tokens.shape[0]
        device = memory_tokens.device

        # Flatten rare class mask if needed
        if rare_class_mask_hw.ndim == 2:
            rare_mask = rare_class_mask_hw.flatten().to(device)
        else:
            rare_mask = rare_class_mask_hw.to(device)

        if rare_mask.shape[0] != L_m:
            # Shape mismatch, fallback to pure norm
            feat_norm = torch.linalg.norm(memory_tokens.float(), dim=-1).mean(dim=1)  # [L_m], batch-averaged
            return torch.topk(feat_norm, k=quota).indices, True

        if base_score is not None and base_score.shape[0] == L_m:
            feat_norm = base_score.to(device).float()  # NEW: use uncertainty (or other) score
        else:
            feat_norm = torch.linalg.norm(memory_tokens.float(), dim=-1).mean(dim=1)  # [L_m], batch-averaged
        selected = torch.zeros(L_m, dtype=torch.bool, device=device)

        # === Stage 1: Rare class insurance ===
        rare_quota = int(quota * rare_reserve_ratio)
        rare_tokens = torch.where(rare_mask > 0)[0]

        if len(rare_tokens) > 0 and rare_quota > 0:
            rare_scores = feat_norm[rare_tokens]
            k_rare = min(rare_quota, len(rare_tokens))
            topk_in_rare = torch.topk(rare_scores, k=k_rare).indices
            selected[rare_tokens[topk_in_rare]] = True

        # === Stage 2: Norm-based fill ===
        remaining_quota = quota - selected.sum().item()
        if remaining_quota > 0:
            not_selected = torch.where(~selected)[0]
            if len(not_selected) > 0:
                remaining_scores = feat_norm[not_selected]
                k_fill = min(remaining_quota, len(not_selected))
                topk_fill = torch.topk(remaining_scores, k=k_fill).indices
                selected[not_selected[topk_fill]] = True

        indices = torch.where(selected)[0]
        return torch.sort(indices).values, False  # sorted for positional encoding consistency

    def _resolve_score(
        self,
        mode: str,
        tokens_lbc: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        confidence_map: Optional[torch.Tensor],
        overlap_mask: Optional[torch.Tensor],
        boundary_mask: Optional[torch.Tensor],
        query_features: Optional[torch.Tensor],  # NEW: for CAMS mode
        fallback: str,
    ) -> tuple[torch.Tensor, bool, str]:
        mode = (mode or "feature_norm").lower()
        fallback_mode = (fallback or "feature_norm").lower()
        fallback_used = False
        used_mode = mode

        def _fallback() -> torch.Tensor:
            nonlocal fallback_used, used_mode
            fallback_used = True
            used_mode = fallback_mode
            if fallback_mode == "random":
                return self._random_score(tokens_lbc)
            return self._feature_norm_score(tokens_lbc)

        if mode == "feature_norm":
            return self._feature_norm_score(tokens_lbc), fallback_used, used_mode
        if mode == "random":
            return self._random_score(tokens_lbc), fallback_used, used_mode
        if mode == "none":
            return torch.zeros(tokens_lbc.shape[0], device=tokens_lbc.device), fallback_used, used_mode

        if mode == "attn_topk":
            if attn_scores is None:
                return _fallback(), fallback_used, used_mode
            score = attn_scores.detach()
            if score.ndim > 1:
                score = score.reshape(score.shape[0], -1).float().mean(dim=1)
            else:
                score = score.float()
            if score.numel() != tokens_lbc.shape[0]:
                return _fallback(), fallback_used, used_mode
            return score.to(tokens_lbc.device), fallback_used, used_mode

        if mode == "confidence_topk":
            if confidence_map is None:
                return _fallback(), fallback_used, used_mode
            conf = confidence_map.detach().float()
            if conf.ndim >= 3:
                # Support [B, H, W] / [B,1,H,W]
                if conf.ndim == 4 and conf.shape[1] == 1:
                    conf = conf[:, 0]
                conf = conf.reshape(conf.shape[0], -1).mean(dim=0)
            else:
                conf = conf.flatten()
            if conf.numel() != tokens_lbc.shape[0]:
                return _fallback(), fallback_used, used_mode
            return conf.to(tokens_lbc.device), fallback_used, used_mode

        if mode == "hybrid":
            score = self._feature_norm_score(tokens_lbc)
            added = False
            if attn_scores is not None:
                attn = attn_scores.detach().float()
                if attn.ndim > 1:
                    attn = attn.reshape(attn.shape[0], -1).mean(dim=1)
                if attn.numel() == score.numel():
                    score = 0.7 * score + 0.3 * attn.to(score.device)
                    added = True
            if confidence_map is not None:
                conf = confidence_map.detach().float().flatten()
                if conf.numel() == score.numel():
                    score = 0.8 * score + 0.2 * conf.to(score.device)
                    added = True
            if overlap_mask is not None:
                ov = overlap_mask.detach().float().flatten()
                if ov.numel() == score.numel():
                    score = score + 0.1 * ov.to(score.device)
                    added = True
            if boundary_mask is not None:
                bd = boundary_mask.detach().float().flatten()
                if bd.numel() == score.numel():
                    score = score + 0.1 * bd.to(score.device)
                    added = True
            if not added:
                fallback_used = True
                used_mode = "feature_norm"
            return score, fallback_used, used_mode

        if mode == "cams":
            if query_features is None or query_features.numel() == 0:
                return _fallback(), fallback_used, used_mode
            try:
                score = self._cams_score(
                    memory_tokens=tokens_lbc,
                    query_features=query_features,
                    alpha=0.4,
                    beta=0.4,
                    gamma=0.2,
                )
                return score, fallback_used, used_mode
            except Exception:
                return _fallback(), fallback_used, used_mode

        return _fallback(), fallback_used, used_mode

    def retrieve(
        self,
        memory_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
        spatial_shape: Tuple[int, int],
        memory_id: Optional[str] = None,
        attn_scores: Optional[torch.Tensor] = None,
        confidence_map: Optional[torch.Tensor] = None,
        overlap_mask: Optional[torch.Tensor] = None,
        boundary_mask: Optional[torch.Tensor] = None,
        query_features: Optional[torch.Tensor] = None,  # NEW: current-frame vision feats for CAMS
        rare_class_mask: Optional[torch.Tensor] = None,  # NEW: Route 3 class_aware mode
        uncertainty_map: Optional[torch.Tensor] = None,  # NEW: [L_m] entropy score for uncertainty-guided base score
        mode: str = "attn_topk",
        keep_ratio: float = 0.5,
        min_tokens: int = 64,
        max_tokens: Optional[int] = None,
        cache_enabled: bool = True,
        fallback: str = "feature_norm",
        target_tokens: Optional[int] = None,
        boundary_reserve_ratio: float = 0.125,  # NEW: Route 1 boundary_guarded mode
        rare_reserve_ratio: float = 0.15,  # NEW: Route 3 class_aware mode
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], SparseRetrieveStats]:
        t0 = time.perf_counter()
        l0 = int(memory_tokens.shape[0])
        h0, w0 = int(spatial_shape[0]), int(spatial_shape[1])
        sid = str(memory_id or "none")
        mode_l = (mode or "none").lower()
        if mode_l == "none":
            stats = SparseRetrieveStats(
                original_num_tokens=l0,
                selected_num_tokens=l0,
                keep_ratio=1.0,
                selection_mode="none",
                cache_hit=False,
                memory_id=sid,
                original_hw=(h0, w0),
                selected_token_shape=tuple(int(v) for v in memory_tokens.shape),
                skip_reason="mode=none",
                selection_time=float(time.perf_counter() - t0),
            )
            return memory_tokens, pos_tokens, None, stats

        if l0 <= int(max(0, min_tokens)):
            stats = SparseRetrieveStats(
                original_num_tokens=l0,
                selected_num_tokens=l0,
                keep_ratio=1.0,
                selection_mode=mode_l,
                cache_hit=False,
                memory_id=sid,
                original_hw=(h0, w0),
                selected_token_shape=tuple(int(v) for v in memory_tokens.shape),
                skip_reason="below_min_tokens",
                selection_time=float(time.perf_counter() - t0),
            )
            return memory_tokens, pos_tokens, None, stats

        if target_tokens is None:
            k = self._safe_topk_count(l0, keep_ratio, min_tokens, max_tokens)
        else:
            k = min(l0, max(1, int(target_tokens)))
        cache_hit = False
        cached_idx = None
        if cache_enabled and memory_id is not None:
            cached_idx = self._cached_indices.get(sid, None)
            if cached_idx is not None and int(cached_idx.numel()) == k:
                cache_hit = True

        if cache_hit:
            idx = cached_idx.to(memory_tokens.device, non_blocking=True).long()
            sel_mode = f"{mode_l}_cache"
            fallback_used = False
            mask_shape_mismatch = False
        elif mode_l == "boundary_guarded":
            # Route 1 Phase 1: indices computed directly (not via score+topk)
            if boundary_mask is not None:
                idx, mask_shape_mismatch = self._boundary_guarded_selection(
                    memory_tokens=memory_tokens,
                    boundary_mask_hw=boundary_mask,
                    quota=k,
                    boundary_reserve_ratio=boundary_reserve_ratio,
                    base_score=uncertainty_map,  # NEW: uncertainty-guided base score if provided
                )
                fallback_used = mask_shape_mismatch
                sel_mode = "boundary_guarded" if not mask_shape_mismatch else "feature_norm"
                if mask_shape_mismatch:
                    print(f"[SMR] boundary_guarded SHAPE MISMATCH at {sid}: mask.numel={boundary_mask.numel()}, L_m={l0}, fallback to feature_norm", flush=True)
                else:
                    baseline_idx = torch.topk(self._feature_norm_score(memory_tokens), k=k, dim=0, largest=True, sorted=False).indices
                    set_g, set_b = set(idx.tolist()), set(baseline_idx.tolist())
                    inter, union = len(set_g & set_b), len(set_g | set_b)
                    jaccard = inter / union if union > 0 else 1.0
                    print(f"[SMR][overlap] boundary_guarded mem_id={sid} jaccard={jaccard:.4f} inter={inter} union={union} quota={k}", flush=True)
            else:
                score = self._feature_norm_score(memory_tokens)
                idx = torch.topk(score, k=k, dim=0, largest=True, sorted=False).indices.long()
                idx = torch.sort(idx).values
                fallback_used = True
                mask_shape_mismatch = False
                sel_mode = "feature_norm"
            if cache_enabled and memory_id is not None:
                self._cached_indices[sid] = idx.detach().cpu()
        elif mode_l == "class_aware":
            # Route 3: indices computed directly (not via score+topk)
            if rare_class_mask is not None:
                idx, mask_shape_mismatch = self._class_aware_selection(
                    memory_tokens=memory_tokens,
                    rare_class_mask_hw=rare_class_mask,
                    quota=k,
                    rare_reserve_ratio=rare_reserve_ratio,
                    base_score=uncertainty_map,  # NEW: uncertainty-guided base score if provided
                )
                fallback_used = mask_shape_mismatch
                sel_mode = "class_aware" if not mask_shape_mismatch else "feature_norm"
                if mask_shape_mismatch:
                    print(f"[SMR] class_aware SHAPE MISMATCH at {sid}: mask.numel={rare_class_mask.numel()}, L_m={l0}, fallback to feature_norm", flush=True)
                else:
                    baseline_idx = torch.topk(self._feature_norm_score(memory_tokens), k=k, dim=0, largest=True, sorted=False).indices
                    set_g, set_b = set(idx.tolist()), set(baseline_idx.tolist())
                    inter, union = len(set_g & set_b), len(set_g | set_b)
                    jaccard = inter / union if union > 0 else 1.0
                    rare_count = int((rare_class_mask.flatten() > 0).sum().item())
                    print(f"[SMR][overlap] class_aware mem_id={sid} jaccard={jaccard:.4f} inter={inter} union={union} quota={k} rare_pixels={rare_count}", flush=True)
            else:
                score = self._feature_norm_score(memory_tokens)
                idx = torch.topk(score, k=k, dim=0, largest=True, sorted=False).indices.long()
                idx = torch.sort(idx).values
                fallback_used = True
                mask_shape_mismatch = False
                sel_mode = "feature_norm"
            if cache_enabled and memory_id is not None:
                self._cached_indices[sid] = idx.detach().cpu()
        else:
            mask_shape_mismatch = False
            score, fallback_used, sel_mode = self._resolve_score(
                mode=mode_l,
                tokens_lbc=memory_tokens,
                attn_scores=attn_scores,
                confidence_map=confidence_map,
                overlap_mask=overlap_mask,
                boundary_mask=boundary_mask,
                query_features=query_features,  # NEW: pass through for CAMS
                fallback=fallback,
            )
            if score.numel() != l0:
                # shape-protect fallback
                score = self._feature_norm_score(memory_tokens)
                fallback_used = True
                sel_mode = "feature_norm"
            idx = torch.topk(score, k=k, dim=0, largest=True, sorted=False).indices.long()
            idx = torch.sort(idx).values
            if cache_enabled and memory_id is not None:
                self._cached_indices[sid] = idx.detach().cpu()

        if idx.numel() == 0:
            stats = SparseRetrieveStats(
                original_num_tokens=l0,
                selected_num_tokens=l0,
                keep_ratio=1.0,
                selection_mode=mode_l,
                cache_hit=cache_hit,
                memory_id=sid,
                original_hw=(h0, w0),
                selected_token_shape=tuple(int(v) for v in memory_tokens.shape),
                fallback_used=True,
                skip_reason="empty_selection_fallback",
                selection_time=float(time.perf_counter() - t0),
            )
            return memory_tokens, pos_tokens, None, stats

        selected_memory = torch.index_select(memory_tokens, dim=0, index=idx)
        selected_pos = torch.index_select(pos_tokens, dim=0, index=idx)
        if torch.isnan(selected_memory).any() or torch.isnan(selected_pos).any():
            stats = SparseRetrieveStats(
                original_num_tokens=l0,
                selected_num_tokens=l0,
                keep_ratio=1.0,
                selection_mode=mode_l,
                cache_hit=cache_hit,
                memory_id=sid,
                original_hw=(h0, w0),
                selected_token_shape=tuple(int(v) for v in memory_tokens.shape),
                fallback_used=True,
                skip_reason="nan_fallback",
                selection_time=float(time.perf_counter() - t0),
            )
            return memory_tokens, pos_tokens, None, stats

        stats = SparseRetrieveStats(
            original_num_tokens=l0,
            selected_num_tokens=int(selected_memory.shape[0]),
            keep_ratio=float(selected_memory.shape[0] / max(1, l0)),
            selection_mode=sel_mode,
            cache_hit=cache_hit,
            memory_id=sid,
            original_hw=(h0, w0),
            selected_token_shape=tuple(int(v) for v in selected_memory.shape),
            fallback_used=fallback_used,
            selection_time=float(time.perf_counter() - t0),
            mask_shape_mismatch=mask_shape_mismatch,
        )
        return selected_memory, selected_pos, idx, stats
