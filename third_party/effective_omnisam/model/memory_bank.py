import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import time
from torch.nn.init import trunc_normal_
from model.modules.memory_compressor import SpatialMemoryCompressor
from model.modules.sparse_memory_retriever import SparseMemoryRetriever
from model.modules.pano_smb_allocator import PanoSMBAllocator


def extract_boundary_mask_from_labels(
    pseudo_labels_bhw: torch.Tensor,
    kernel_size: int = 3,
    num_classes: int = 13,
) -> torch.Tensor:
    """
    Extract boundary mask from pseudo label tensor using morphological dilation-erosion.

    Args:
        pseudo_labels_bhw: [B, H, W] int tensor of class IDs (may contain ignore_index like 255)
        kernel_size: boundary thickness (odd number)
        num_classes: number of classes for one-hot conversion

    Returns:
        boundary_mask: [B, H, W] float tensor (0/1)
    """
    B, H, W = pseudo_labels_bhw.shape
    device = pseudo_labels_bhw.device

    # Clamp to valid range [0, num_classes-1] to handle ignore_index=255
    labels_clamped = pseudo_labels_bhw.long().clamp(0, num_classes - 1)

    # Convert to one-hot
    one_hot = F.one_hot(labels_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B,C,H,W]

    # Dilate each class channel independently
    pad = kernel_size // 2
    dilated_list = []
    for c in range(num_classes):
        channel = one_hot[:, c:c+1]  # [B,1,H,W]
        dilated = F.max_pool2d(
            F.pad(channel, (pad, pad, pad, pad), mode='constant', value=0),
            kernel_size=kernel_size,
            stride=1
        )
        dilated_list.append(dilated)
    dilated = torch.cat(dilated_list, dim=1)  # [B,C,H,W]

    # Boundary = dilated - original (anywhere class boundary crosses)
    boundary = (dilated - one_hot).sum(dim=1, keepdim=False)  # [B,H,W]
    return (boundary > 0).float()


def extract_rare_class_mask_from_labels(
    pseudo_labels_bhw: torch.Tensor,
    rare_class_ids,
) -> torch.Tensor:
    """
    Extract a binary mask marking pixels belonging to any of rare_class_ids.

    Args:
        pseudo_labels_bhw: [B, H, W] int tensor of class IDs (may contain ignore_index like 255)
        rare_class_ids: iterable of int class IDs considered rare

    Returns:
        rare_mask: [B, H, W] float tensor (0/1)
    """
    if not rare_class_ids:
        return torch.zeros_like(pseudo_labels_bhw, dtype=torch.float32)
    mask = torch.zeros_like(pseudo_labels_bhw, dtype=torch.bool)
    for cid in rare_class_ids:
        mask = mask | (pseudo_labels_bhw == int(cid))
    return mask.float()


class MemoryBank(nn.Module):
    def __init__(self, sam_model, num_maskmem=6):
        super().__init__()
        self.sam_model = sam_model
        self.mask_encoder = sam_model.memory_encoder
        self.num_maskmem = num_maskmem
        self.hidden_dim = sam_model.image_encoder.neck.d_model
        self.mem_dim = sam_model.memory_encoder.out_proj.weight.shape[0]
        self.output_dict_source = {"frame_outputs": {}}
        self.output_dict_target = {"frame_outputs": {}}
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.select_mode = "none"
        self.recent_keep = 0
        self.score_keep = 0
        self.last_effective_memory_length = 0
        self.memory_compress_mode = "none"
        self.memory_compress_location = "before_attention"
        self.memory_compress_enabled = False
        self.memory_compress_verbose = False
        self.memory_compress_log_interval = 200
        self.memory_compress_min_tokens = 64
        self.memory_compress_warmup_iters = 0
        self.memory_enable_in_train = True
        self.memory_enable_in_pseudo_label = True
        self.memory_train_kernel_size = 2
        self.memory_pl_kernel_size = 2
        self.memory_stride = 2
        self.memory_keep_global_tokens = 0
        self.memory_apply_to_pos_enc = True
        self.runtime_stage = "train"
        self.runtime_iter = 0
        self.last_memory_tokens_before = 0
        self.last_memory_tokens_after = 0
        self.last_memory_token_ratio = 1.0
        self.last_attention_time = 0.0
        self.attention_time_accum = 0.0
        self.attention_time_count = 0
        self.last_compression_stats = {}
        self.memory_compressor = SpatialMemoryCompressor(embed_dim=self.mem_dim, mode="none")
        # Efficient-SAM2-style sparse memory retrieval (token selection), optional.
        self.sparse_enabled = False
        self.sparse_mode = "none"
        self.sparse_keep_ratio = 0.5
        self.sparse_min_tokens = 64
        self.sparse_cache_enabled = True
        self.sparse_fallback = "feature_norm"
        self.sparse_fallback_full_when_uncertain = True
        self.sparse_enable_in_train = True
        self.sparse_enable_in_pseudo_label = True
        self.boundary_reserve_ratio = 0.125  # NEW: Route 1 boundary_guarded mode
        self.rare_class_ids = []  # NEW: Route 3 class_aware mode
        self.rare_reserve_ratio = 0.15  # NEW: Route 3 class_aware mode
        self.rare_selection_strategy = "confidence"  # NEW: Route 3 (reserved for future use)
        self.boundary_score_source = "feature_norm"  # NEW: 'feature_norm' | 'uncertainty'
        self.sparse_retriever = SparseMemoryRetriever()
        self.last_sparse_stats = {}
        self.last_sparse_tokens_before = 0
        self.last_sparse_tokens_after = 0
        self.last_sparse_keep_ratio = 1.0
        # Stage 1 uses one batch-shared FoV quota vector, not per-sample dynamic quotas.
        self.pano_smb_enabled = False
        self.pano_smb_min_tokens = 256
        self.pano_smb_verbose = False
        self.pano_smb_log_interval = 200
        self.pano_smb_allocator = PanoSMBAllocator(min_tokens=self.pano_smb_min_tokens)
        self.last_pano_smb_stats = {}
        self.pano_smb_allocation_time_accum = 0.0
        self.pano_smb_allocation_count = 0
        self.pano_smb_aggregate_stats = self._new_pano_smb_aggregate_stats()
        self.runtime_geo_context = {}
        self.last_encoded_memory_meta = {}
        self.last_geo_memory_diag = {}

    def _init_output_dict_source(self):
        self.output_dict_source = {"frame_outputs": {}}

    def _init_output_dict_target(self):
        self.output_dict_target = {"frame_outputs": {}}

    def _encode_new_memory(self, frame_idx, current_vision_feats, mask_for_mem, mem_type, uncertainty_map=None):
        maskmem_out = self.sam_model.memory_encoder(current_vision_feats, mask_for_mem)
        current_memory_feats = maskmem_out['vision_features']
        current_memory_pos_embeds = maskmem_out['vision_pos_enc']
        mem_hw = tuple(current_memory_feats.shape[-2:])
        current_memory_feats = current_memory_feats.flatten(2).permute(2, 0, 1)
        current_memory_pos_embeds = current_memory_pos_embeds[-1].flatten(2).permute(2, 0, 1)
        # NEW: Process uncertainty_map if provided
        uncertainty_flat = None
        if uncertainty_map is not None:
            # uncertainty_map is [B, H, W], resize to mem_hw then flatten
            uncertainty_resized = F.interpolate(
                uncertainty_map.unsqueeze(1), size=mem_hw, mode='bilinear', align_corners=False
            ).squeeze(1)  # [B, mem_H, mem_W]
            uncertainty_flat = uncertainty_resized.flatten(1).permute(1, 0)  # [L, B]
        if self._compression_enabled_now() and self.memory_compress_location == "before_bank":
            # Optional experiment path: compress before writing to memory bank.
            eff_kernel = self._effective_kernel_size()
            self.memory_compressor.configure(
                kernel_size=eff_kernel,
                stride=self.memory_stride,
                keep_global_tokens=self.memory_keep_global_tokens,
                apply_to_pos_enc=self.memory_apply_to_pos_enc,
                min_tokens=self.memory_compress_min_tokens,
                compression_location=self.memory_compress_location,
            )
            try:
                current_memory_feats, current_memory_pos_embeds, mem_hw, cstats = self.memory_compressor.compress(
                    current_memory_feats, current_memory_pos_embeds, mem_hw
                )
                # NEW: Apply same compression to uncertainty_flat if present
                if uncertainty_flat is not None:
                    # Average pooling over spatial dimension (2x2 -> 1 token)
                    # uncertainty_flat is [L_original, B], reshape to [H_orig, W_orig, B]
                    H_orig, W_orig = mem_hw[0] * 2, mem_hw[1] * 2  # mem_hw already updated to compressed
                    B = uncertainty_flat.shape[1]
                    uncertainty_2d = uncertainty_flat[:H_orig*W_orig].reshape(H_orig, W_orig, B)
                    # 2x2 avg pool
                    uncertainty_pooled = F.avg_pool2d(
                        uncertainty_2d.permute(2, 0, 1).unsqueeze(1),  # [B, 1, H, W]
                        kernel_size=2, stride=2
                    ).squeeze(1)  # [B, H_new, W_new]
                    uncertainty_flat = uncertainty_pooled.flatten(1).permute(1, 0)  # [L_compressed, B]
                self.last_compression_stats = cstats.__dict__.copy()
            except Exception as e:
                self.last_compression_stats = {
                    "original_num_tokens": int(current_memory_feats.shape[0]),
                    "compressed_num_tokens": int(current_memory_feats.shape[0]),
                    "compression_ratio": 1.0,
                    "original_hw": mem_hw,
                    "compressed_hw": mem_hw,
                    "compression_mode": "none",
                    "compression_location": self.memory_compress_location,
                    "skipped": True,
                    "skip_reason": f"before_bank_error:{str(e)}",
                }
        importance = float(mask_for_mem.float().mean().item())
        if mem_type == "source":
            self.output_dict_source["frame_outputs"][frame_idx] = {
                "maskmem_features": current_memory_feats.detach(),
                "maskmem_pos_enc": current_memory_pos_embeds.detach(),
                "importance": importance,
                "mem_hw": mem_hw,
                "uncertainty": uncertainty_flat.detach() if uncertainty_flat is not None else None,
            }
        elif mem_type == "target":
            self.output_dict_target["frame_outputs"][frame_idx] = {
                "maskmem_features": current_memory_feats.detach(),
                "maskmem_pos_enc": current_memory_pos_embeds.detach(),
                "importance": importance,
                "mem_hw": mem_hw,
                "uncertainty": uncertainty_flat.detach() if uncertainty_flat is not None else None,
            }
        else:
            raise ValueError("Wrong memory bank type.")
        geometry_meta = dict(self.runtime_geo_context or {})
        self.last_encoded_memory_meta = {
            "frame_idx": int(frame_idx), "mem_type": str(mem_type),
            "mem_hw": [int(mem_hw[0]), int(mem_hw[1])],
            "num_tokens": int(current_memory_feats.shape[0]),
            "geometry_meta": geometry_meta, "importance": importance,
        }
        self.last_geo_memory_diag = {
            "frame_idx": int(frame_idx), "mem_type": str(mem_type),
            "mem_hw": [int(mem_hw[0]), int(mem_hw[1])],
            "num_tokens": int(current_memory_feats.shape[0]),
            "geometry_meta": geometry_meta, "importance": importance,
            "compression_stats": dict(self.last_compression_stats or {}),
            "sparse_stats": dict(self.last_sparse_stats or {}),
            "last_memory_tokens_before": int(self.last_memory_tokens_before),
            "last_memory_tokens_after": int(self.last_memory_tokens_after),
            "last_memory_token_ratio": float(self.last_memory_token_ratio),
            "last_sparse_tokens_before": int(self.last_sparse_tokens_before),
            "last_sparse_tokens_after": int(self.last_sparse_tokens_after),
            "last_sparse_keep_ratio": float(self.last_sparse_keep_ratio),
            "last_effective_memory_length": int(self.last_effective_memory_length),
            "memory_compress_enabled": bool(self.memory_compress_enabled),
            "memory_compress_mode": str(self.memory_compress_mode),
            "memory_compress_location": str(self.memory_compress_location),
            "sparse_enabled": bool(self.sparse_enabled),
            "sparse_mode": str(self.sparse_mode),
            "select_mode": str(self.select_mode),
            "selection_effect": "diagnosis_only",
        }
        self.runtime_geo_context = {}

    def configure_compression(
        self,
        enabled: bool = False,
        select_mode: str = "none",
        recent_keep: int = 0,
        score_keep: int = 0,
        compress_mode: str = "none",
        compress_location: str = "before_attention",
        downsample_ratio: int = 2,  # backward compatibility
        spatial_window_size: int = 2,  # backward compatibility
        global_summary_tokens: int = 0,  # backward compatibility
        kernel_size: int = 2,
        stride: int = 2,
        keep_global_tokens: int = 0,
        apply_to_pos_enc: bool = True,
        min_tokens: int = 64,
        warmup_iters: int = 0,
        enable_in_train: bool = True,
        enable_in_pseudo_label: bool = True,
        train_kernel_size: int = 2,
        pl_kernel_size: int = 2,
        verbose: bool = False,
        log_interval: int = 200,
    ):
        self.memory_compress_enabled = bool(enabled)
        self.select_mode = (select_mode or "none").lower()
        self.recent_keep = max(0, int(recent_keep))
        self.score_keep = max(0, int(score_keep))
        self.memory_compress_mode = (compress_mode or "none").lower()
        self.memory_compress_location = (compress_location or "before_attention").lower()
        self.memory_compress_verbose = bool(verbose)
        self.memory_compress_log_interval = max(1, int(log_interval))
        self.memory_compress_min_tokens = max(0, int(min_tokens))
        self.memory_compress_warmup_iters = max(0, int(warmup_iters))
        self.memory_enable_in_train = bool(enable_in_train)
        self.memory_enable_in_pseudo_label = bool(enable_in_pseudo_label)
        self.memory_train_kernel_size = max(1, int(train_kernel_size))
        self.memory_pl_kernel_size = max(1, int(pl_kernel_size))
        self.memory_stride = max(1, int(stride))
        self.memory_keep_global_tokens = max(0, int(keep_global_tokens if keep_global_tokens is not None else global_summary_tokens))
        self.memory_apply_to_pos_enc = bool(apply_to_pos_enc)
        self.memory_compressor.configure(
            mode=self.memory_compress_mode,
            kernel_size=max(1, int(kernel_size if kernel_size is not None else downsample_ratio)),
            stride=max(1, int(stride if stride is not None else downsample_ratio)),
            downsample_ratio=int(downsample_ratio),
            window_size=int(spatial_window_size),
            keep_global_tokens=self.memory_keep_global_tokens,
            global_tokens=int(global_summary_tokens),
            apply_to_pos_enc=self.memory_apply_to_pos_enc,
            min_tokens=self.memory_compress_min_tokens,
            compression_location=self.memory_compress_location,
        )

    def set_runtime_context(self, stage: str = "train", iter_idx: int = 0, geo_context=None):
        self.runtime_stage = str(stage or "train").lower()
        self.runtime_iter = int(max(0, iter_idx))
        # Geometry metadata is diagnosis-only and must not leak across stages or frames.
        self.runtime_geo_context = dict(geo_context) if geo_context is not None else {}

    def get_last_geo_memory_diag(self):
        return dict(self.last_geo_memory_diag or {})

    def configure_sparse_retrieval(
        self,
        enabled: bool = False,
        mode: str = "attn_topk",
        keep_ratio: float = 0.5,
        min_tokens: int = 64,
        cache_enabled: bool = True,
        fallback: str = "feature_norm",
        fallback_full_when_uncertain: bool = True,
        enable_in_train: bool = True,
        enable_in_pseudo_label: bool = True,
        boundary_reserve_ratio: float = 0.125,  # NEW: for boundary_guarded mode
        rare_class_ids: list = None,  # NEW: Route 3 class_aware mode
        rare_reserve_ratio: float = 0.15,  # NEW: Route 3 class_aware mode
        rare_selection_strategy: str = "confidence",  # NEW: Route 3 (reserved)
        boundary_score_source: str = "feature_norm",  # NEW: 'feature_norm' | 'uncertainty'
    ) -> None:
        self.sparse_enabled = bool(enabled)
        self.sparse_mode = str(mode or "none").lower()
        self.sparse_keep_ratio = float(keep_ratio)
        self.sparse_min_tokens = int(min_tokens)
        self.sparse_cache_enabled = bool(cache_enabled)
        self.sparse_fallback = str(fallback or "feature_norm").lower()
        self.sparse_fallback_full_when_uncertain = bool(fallback_full_when_uncertain)
        self.sparse_enable_in_train = bool(enable_in_train)
        self.sparse_enable_in_pseudo_label = bool(enable_in_pseudo_label)
        self.boundary_reserve_ratio = float(boundary_reserve_ratio)  # NEW
        self.rare_class_ids = list(rare_class_ids or [])  # NEW
        self.rare_reserve_ratio = float(rare_reserve_ratio)  # NEW
        self.rare_selection_strategy = str(rare_selection_strategy or "confidence").lower()  # NEW
        self.boundary_score_source = str(boundary_score_source or "feature_norm").lower()  # NEW
        if not self.sparse_enabled:
            try:
                self.sparse_retriever.clear()
            except Exception:
                pass

    def _sparse_enabled_now(self) -> bool:
        if not self.sparse_enabled:
            return False
        if self.sparse_mode == "none":
            return False
        if self.runtime_stage == "pseudo_label":
            return bool(self.sparse_enable_in_pseudo_label)
        if self.runtime_stage == "train":
            return bool(self.sparse_enable_in_train)
        return True

    def configure_pano_smb(
        self,
        enabled: bool = False,
        min_tokens: int = 256,
        verbose: bool = False,
        log_interval: int = 200,
    ) -> None:
        self.pano_smb_enabled = bool(enabled)
        self.pano_smb_min_tokens = int(min_tokens)
        self.pano_smb_verbose = bool(verbose)
        self.pano_smb_log_interval = max(1, int(log_interval))
        self.pano_smb_allocator = PanoSMBAllocator(min_tokens=self.pano_smb_min_tokens)
        self.last_pano_smb_stats = {}
        self.pano_smb_allocation_time_accum = 0.0
        self.pano_smb_allocation_count = 0
        self.pano_smb_aggregate_stats = self._new_pano_smb_aggregate_stats()

    @staticmethod
    def _new_pano_smb_aggregate_stats():
        return {
            "allocation_calls": 0,
            "history_fov_counts": {},
            "quota_count": 0,
            "quota_sum": 0.0,
            "quota_sum_sq": 0.0,
            "quota_min": None,
            "quota_max": None,
            "importance_count": 0,
            "importance_sum": 0.0,
            "importance_sum_sq": 0.0,
            "fallback_count": 0,
            "budget_error_max": 0,
            "representative": [],
        }

    def _update_pano_smb_aggregate_stats(self, stats):
        agg = self.pano_smb_aggregate_stats
        quotas = [int(v) for v in stats.get("quotas", [])]
        importance = [float(v) for v in stats.get("importance", [])]
        history_fovs = int(stats.get("history_fovs", len(quotas)))
        agg["allocation_calls"] += 1
        key = str(history_fovs)
        agg["history_fov_counts"][key] = int(agg["history_fov_counts"].get(key, 0)) + 1
        if quotas:
            agg["quota_count"] += len(quotas)
            agg["quota_sum"] += float(sum(quotas))
            agg["quota_sum_sq"] += float(sum(v * v for v in quotas))
            agg["quota_min"] = min(quotas) if agg["quota_min"] is None else min(agg["quota_min"], min(quotas))
            agg["quota_max"] = max(quotas) if agg["quota_max"] is None else max(agg["quota_max"], max(quotas))
        if importance:
            agg["importance_count"] += len(importance)
            agg["importance_sum"] += float(sum(importance))
            agg["importance_sum_sq"] += float(sum(v * v for v in importance))
        agg["fallback_count"] += int(bool(stats.get("fallback_used", False)))
        agg["budget_error_max"] = max(
            int(agg["budget_error_max"]), abs(int(stats.get("budget_error", 0)))
        )
        if len(agg["representative"]) < 8:
            agg["representative"].append(
                {"history_fovs": history_fovs, "importance": importance, "quotas": quotas}
            )

    def get_pano_smb_aggregate_stats(self):
        agg = dict(self.pano_smb_aggregate_stats)
        quota_count = int(agg.pop("quota_count"))
        quota_sum = float(agg.pop("quota_sum"))
        quota_sum_sq = float(agg.pop("quota_sum_sq"))
        importance_count = int(agg.pop("importance_count"))
        importance_sum = float(agg.pop("importance_sum"))
        importance_sum_sq = float(agg.pop("importance_sum_sq"))

        def mean_std(count, total, total_sq):
            if count <= 0:
                return None, None
            mean = total / count
            variance = max(0.0, total_sq / count - mean * mean)
            return mean, variance ** 0.5

        quota_mean, quota_std = mean_std(quota_count, quota_sum, quota_sum_sq)
        importance_mean, importance_std = mean_std(
            importance_count, importance_sum, importance_sum_sq
        )
        calls = int(agg.get("allocation_calls", 0))
        agg.update(
            {
                "quota_mean": quota_mean,
                "quota_std": quota_std,
                "importance_mean": importance_mean,
                "importance_std": importance_std,
                "fallback_rate": (float(agg["fallback_count"]) / calls if calls else None),
                "allocation_time_total": float(self.pano_smb_allocation_time_accum),
                "allocation_time_per_call": (
                    float(self.pano_smb_allocation_time_accum) / calls if calls else None
                ),
            }
        )
        return agg

    def _pano_smb_enabled_now(self) -> bool:
        if not self.pano_smb_enabled:
            return False
        if self.select_mode != "none":
            raise ValueError("pano_smb requires memory select_mode=none to conserve the FoV budget")
        if not self._sparse_enabled_now():
            raise ValueError("pano_smb requires sparse retrieval to be active for the current runtime stage")
        return True

    def _maybe_log_pano_smb(self) -> None:
        if not self.pano_smb_verbose or not self.last_pano_smb_stats:
            return
        if (self.runtime_stage or "").lower() == "eval":
            return
        if self.runtime_iter not in (0, 1) and self.runtime_iter % self.pano_smb_log_interval != 0:
            return
        stats = self.last_pano_smb_stats
        pairs = ", ".join(
            f"FoV{i}:importance={importance:.6g},quota={quota}"
            for i, (importance, quota) in enumerate(zip(stats.get("importance", []), stats.get("quotas", [])))
        )
        print(
            "[PanoSMB] "
            f"history_fovs={stats.get('history_fovs', 0)} "
            f"legacy_budget={stats.get('legacy_total_budget', 0)} "
            f"allocated_total={stats.get('allocated_total', 0)} "
            f"budget_error={stats.get('budget_error', 0)} "
            f"min={stats.get('min_quota', 0)} max={stats.get('max_quota', 0)} "
            f"mean={stats.get('mean_quota', 0.0):.3f} std={stats.get('quota_std', 0.0):.3f} "
            f"{pairs}",
            flush=True,
        )

    def _sync_cuda(self, device):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _compression_enabled_now(self) -> bool:
        if not self.memory_compress_enabled:
            return False
        if self.memory_compress_mode == "none":
            return False
        if self.runtime_iter < self.memory_compress_warmup_iters:
            return False
        stage = self.runtime_stage
        if stage == "pseudo_label":
            return self.memory_enable_in_pseudo_label
        if stage == "train":
            return self.memory_enable_in_train
        return True

    def _effective_kernel_size(self) -> int:
        if self.runtime_stage == "pseudo_label":
            return self.memory_pl_kernel_size
        return self.memory_train_kernel_size

    def _maybe_log_compression(self, stats: dict):
        if not self.memory_compress_verbose:
            return
        # Only log when compression is truly active for current runtime stage.
        # This avoids misleading pseudo_label logs when PL compression is disabled.
        if not self._compression_enabled_now():
            return
        # 验证 / 独立 eval：不打印压缩调试（iter 恒为 0 时会每条 memory 都打满屏）
        if (self.runtime_stage or "").lower() == "eval":
            return
        if self.runtime_iter not in (0, 1) and (self.runtime_iter % self.memory_compress_log_interval != 0):
            return
        print(
            "[mem-compress] "
            f"stage={self.runtime_stage} iter={self.runtime_iter} "
            f"enabled={self.memory_compress_enabled} mode={self.memory_compress_mode} "
            f"location={self.memory_compress_location} "
            f"L={stats.get('original_num_tokens', 0)} -> L'={stats.get('compressed_num_tokens', 0)} "
            f"ratio={stats.get('compression_ratio', 1.0):.4f} "
            f"hw={stats.get('original_hw', None)}->{stats.get('compressed_hw', None)} "
            f"attn_time_ms={self.last_attention_time*1000.0:.3f}",
            flush=True,
        )

    def forward(
        self,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        mem_type,
        num_obj_ptr_tokens=0,
        historical_pseudo_labels=None,  # NEW: Optional[dict[int, torch.Tensor]] prev_frame_idx -> [B,H,W] pseudo labels
    ):
        B, C, H, W = current_vision_feats.shape
        device = current_vision_feats.device
        self.last_compression_stats = {}
        self.last_sparse_stats = {}
        self.last_sparse_tokens_before = 0
        self.last_sparse_tokens_after = 0
        self.last_sparse_keep_ratio = 1.0
        self.last_pano_smb_stats = {}
        self.last_encoded_memory_meta = {}
        self.last_geo_memory_diag = {}
        vision_feats = current_vision_feats.flatten(2).permute(2, 0, 1)
        vision_pos_embeds = current_vision_pos_embeds.flatten(2).permute(2, 0, 1)
        
        if frame_idx != 0:
            to_cat_memory, to_cat_memory_pos_embed, t_pos_and_prevs = [], [], []
            to_cat_importance = []
            prepared_chunks = []
            for t_pos in range(self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                prev_frame_idx = frame_idx - t_rel

                if mem_type == "source":
                    out = self.output_dict_source["frame_outputs"].get(prev_frame_idx, None)
                elif mem_type == "target":
                    out = self.output_dict_target["frame_outputs"].get(prev_frame_idx, None)
                else:
                    raise ValueError("Wrong memory bank type.")
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                maskmem_enc = prev["maskmem_pos_enc"].to(device)
                # maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - frame_idx - 1].to(device))
                maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1].to(device))
                mem_hw = tuple(prev.get("mem_hw", (H, W)))
                if self._compression_enabled_now() and self.memory_compress_location == "before_attention":
                    eff_kernel = self._effective_kernel_size()
                    self.memory_compressor.configure(
                        kernel_size=eff_kernel,
                        stride=self.memory_stride,
                        keep_global_tokens=self.memory_keep_global_tokens,
                        apply_to_pos_enc=self.memory_apply_to_pos_enc,
                        min_tokens=self.memory_compress_min_tokens,
                        compression_location=self.memory_compress_location,
                    )
                    try:
                        feats, maskmem_enc, mem_hw, cstats = self.memory_compressor.compress(feats, maskmem_enc, mem_hw)
                        self.last_compression_stats = cstats.__dict__.copy()
                        # BUGFIX: compress uncertainty to match compressed token count
                        if prev.get("uncertainty") is not None:
                            orig_h, orig_w = prev.get("mem_hw", (H, W))
                            B = feats.shape[1]
                            unc_2d = prev["uncertainty"].permute(1, 0).reshape(B, orig_h, orig_w).unsqueeze(1)  # [B, 1, H, W]
                            unc_pool = F.avg_pool2d(unc_2d, kernel_size=eff_kernel, stride=self.memory_stride, ceil_mode=False).squeeze(1)  # [B, H', W']
                            prev["uncertainty"] = unc_pool.flatten(1).permute(1, 0)  # [L', B]
                    except Exception as e:
                        # Safe fallback to no-op for robustness during long training runs.
                        self.last_compression_stats = {
                            "original_num_tokens": int(feats.shape[0]),
                            "compressed_num_tokens": int(feats.shape[0]),
                            "compression_ratio": 1.0,
                            "original_hw": mem_hw,
                            "compressed_hw": mem_hw,
                            "compression_mode": "none",
                            "compression_location": self.memory_compress_location,
                            "skipped": True,
                            "skip_reason": f"compress_error:{str(e)}",
                        }
                prepared_chunks.append(
                    {
                        "feats": feats,
                        "pos": maskmem_enc,
                        "mem_hw": mem_hw,
                        "t_pos": t_pos,
                        "importance": float(prev.get("importance", 0.0)),
                        "uncertainty": prev.get("uncertainty", None),  # NEW: forward uncertainty
                    }
                )

            pano_quotas = None
            if self._pano_smb_enabled_now() and prepared_chunks:
                try:
                    scores = [
                        self.sparse_retriever.feature_norm_score(chunk["feats"])
                        for chunk in prepared_chunks
                    ]
                    importance = self.pano_smb_allocator.importance_from_scores(scores)
                    capacities = [int(chunk["feats"].shape[0]) for chunk in prepared_chunks]
                    legacy_quotas = [
                        self.sparse_retriever._safe_topk_count(
                            num_tokens=capacity,
                            keep_ratio=float(self.sparse_keep_ratio),
                            min_tokens=int(self.sparse_min_tokens),
                            max_tokens=None,
                        )
                        for capacity in capacities
                    ]
                    allocation = self.pano_smb_allocator.allocate(importance, legacy_quotas, capacities)
                    pano_quotas = allocation.quotas
                    self.last_pano_smb_stats = allocation.as_dict()
                    self.last_pano_smb_stats["history_fovs"] = len(prepared_chunks)
                    self.pano_smb_allocation_time_accum += allocation.allocation_time
                    self.pano_smb_allocation_count += 1
                    self._update_pano_smb_aggregate_stats(self.last_pano_smb_stats)
                except Exception as e:
                    raise RuntimeError(f"Pano-SMB allocation failed: {str(e)}") from e

            for chunk_index, chunk in enumerate(prepared_chunks):
                feats = chunk["feats"]
                maskmem_enc = chunk["pos"]

                # NEW: Extract boundary mask for boundary_guarded mode
                boundary_mask_for_this_chunk = None
                if self.sparse_mode == "boundary_guarded" and historical_pseudo_labels is not None:
                    t_rel = self.num_maskmem - chunk["t_pos"]
                    prev_frame_idx = frame_idx - t_rel
                    pseudo_label_bhw = historical_pseudo_labels.get(prev_frame_idx, None)
                    if pseudo_label_bhw is not None:
                        try:
                            # pseudo_label_bhw: [B, H_orig, W_orig], need to match chunk["mem_hw"]
                            H_mem, W_mem = chunk["mem_hw"]
                            B_pl = pseudo_label_bhw.shape[0]
                            if pseudo_label_bhw.shape[1:] != (H_mem, W_mem):
                                # Resize via nearest neighbor (integer labels)
                                pseudo_label_resized = F.interpolate(
                                    pseudo_label_bhw.float().unsqueeze(1),  # [B,1,H,W]
                                    size=(H_mem, W_mem),
                                    mode='nearest'
                                ).squeeze(1).long()  # [B,H_mem,W_mem]
                            else:
                                pseudo_label_resized = pseudo_label_bhw.long()
                            # Extract boundary
                            boundary_mask_bhw = extract_boundary_mask_from_labels(
                                pseudo_label_resized,
                                kernel_size=3,
                                num_classes=13,  # hardcoded for Cityscapes→DensePASS
                            )  # [B,H_mem,W_mem]
                            # sparse_retriever.retrieve expects boundary_mask per sample b=0 (batch-shared)
                            boundary_mask_for_this_chunk = boundary_mask_bhw[0]  # [H_mem,W_mem]
                        except Exception as e:
                            # Fallback to None if boundary extraction fails
                            boundary_mask_for_this_chunk = None

                # NEW: Extract rare-class mask for class_aware mode
                rare_class_mask_for_this_chunk = None
                if self.sparse_mode == "class_aware" and historical_pseudo_labels is not None and self.rare_class_ids:
                    t_rel = self.num_maskmem - chunk["t_pos"]
                    prev_frame_idx = frame_idx - t_rel
                    pseudo_label_bhw = historical_pseudo_labels.get(prev_frame_idx, None)
                    if pseudo_label_bhw is not None:
                        try:
                            H_mem, W_mem = chunk["mem_hw"]
                            if pseudo_label_bhw.shape[1:] != (H_mem, W_mem):
                                pseudo_label_resized = F.interpolate(
                                    pseudo_label_bhw.float().unsqueeze(1),  # [B,1,H,W]
                                    size=(H_mem, W_mem),
                                    mode='nearest'
                                ).squeeze(1).long()  # [B,H_mem,W_mem]
                            else:
                                pseudo_label_resized = pseudo_label_bhw.long()
                            rare_mask_bhw = extract_rare_class_mask_from_labels(
                                pseudo_label_resized,
                                rare_class_ids=self.rare_class_ids,
                            )  # [B,H_mem,W_mem]
                            rare_class_mask_for_this_chunk = rare_mask_bhw[0]  # [H_mem,W_mem]
                        except Exception as e:
                            rare_class_mask_for_this_chunk = None

                # NEW: Extract uncertainty for this chunk (batch-averaged to [L_m])
                uncertainty_for_this_chunk = None
                chunk_uncertainty = chunk.get("uncertainty", None)
                if chunk_uncertainty is not None and self.boundary_score_source == "uncertainty":
                    u = chunk_uncertainty.to(device)  # [L, B]
                    if u.shape[0] == int(feats.shape[0]):
                        uncertainty_for_this_chunk = u.mean(dim=1)  # [L]

                if self._sparse_enabled_now():
                    try:
                        mem_id = f"{mem_type}_f{frame_idx}_p{chunk['t_pos']}_iter{int(self.runtime_iter)}"
                        self.last_sparse_tokens_before += int(feats.shape[0])
                        feats2, pos2, idx2, sstats = self.sparse_retriever.retrieve(
                            memory_tokens=feats,
                            pos_tokens=maskmem_enc,
                            spatial_shape=chunk["mem_hw"],
                            memory_id=mem_id,
                            attn_scores=None,
                            confidence_map=None,
                            boundary_mask=boundary_mask_for_this_chunk,  # NEW: pass boundary mask
                            query_features=vision_feats[:, 0, :] if vision_feats.shape[1] > 0 else None,  # NEW: [L,C] for CAMS
                            rare_class_mask=rare_class_mask_for_this_chunk,  # NEW: Route 3 class_aware mode
                            uncertainty_map=uncertainty_for_this_chunk,  # NEW: uncertainty-guided base score
                            mode=self.sparse_mode,
                            keep_ratio=float(self.sparse_keep_ratio),
                            min_tokens=int(self.sparse_min_tokens),
                            cache_enabled=bool(self.sparse_cache_enabled),
                            fallback=str(self.sparse_fallback),
                            target_tokens=(pano_quotas[chunk_index] if pano_quotas is not None else None),
                            boundary_reserve_ratio=float(self.boundary_reserve_ratio),
                            rare_reserve_ratio=float(self.rare_reserve_ratio),
                        )
                        if pano_quotas is not None and int(feats2.shape[0]) != int(pano_quotas[chunk_index]):
                            raise RuntimeError(
                                "Pano-SMB retrieval did not return the allocated FoV quota"
                            )
                        feats, maskmem_enc = feats2, pos2
                        self.last_sparse_tokens_after += int(feats.shape[0])
                        self.last_sparse_keep_ratio = float(
                            self.last_sparse_tokens_after / max(1, self.last_sparse_tokens_before)
                        )
                        self.last_sparse_stats = getattr(sstats, "__dict__", {}) if sstats is not None else {}
                    except Exception as e:
                        if pano_quotas is not None:
                            raise RuntimeError(f"Pano-SMB sparse retrieval failed: {str(e)}") from e
                        self.last_sparse_stats = {"skipped": True, "skip_reason": f"sparse_error:{str(e)}"}
                to_cat_memory.append(feats)
                to_cat_memory_pos_embed.append(maskmem_enc)
                to_cat_importance.append(chunk["importance"])
            self._maybe_log_pano_smb()
            if self.select_mode in {"recent_topk", "score_topk", "hybrid"} and len(to_cat_memory) > 0:
                n = len(to_cat_memory)
                indices = list(range(n))
                keep_idx = set()
                if self.select_mode in {"recent_topk", "hybrid"}:
                    rk = self.recent_keep if self.recent_keep > 0 else min(n, self.num_maskmem)
                    keep_idx.update(indices[-rk:])
                if self.select_mode in {"score_topk", "hybrid"}:
                    sk = self.score_keep if self.score_keep > 0 else min(n, self.num_maskmem)
                    scored = [(to_cat_importance[i], i) for i in range(n)]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    keep_idx.update([i for _, i in scored[:sk]])
                keep_list = sorted(list(keep_idx))
                if keep_list:
                    to_cat_memory = [to_cat_memory[i] for i in keep_list]
                    to_cat_memory_pos_embed = [to_cat_memory_pos_embed[i] for i in keep_list]
        
        else:
            pix_feat_with_mem = vision_feats + self.no_mem_embed.to(device)
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            self.last_effective_memory_length = 0
            return pix_feat_with_mem

        if len(to_cat_memory) == 0:
            pix_feat_with_mem = vision_feats + self.no_mem_embed.to(device)
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            self.last_effective_memory_length = 0
            self.last_memory_tokens_before = 0
            self.last_memory_tokens_after = 0
            self.last_memory_token_ratio = 1.0
            self.last_attention_time = 0.0
            return pix_feat_with_mem
        memory = torch.cat(to_cat_memory, dim=0).to(device)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0).to(device)
        self.last_memory_tokens_before = int(sum(int(m.shape[0]) for m in to_cat_memory)) if to_cat_memory else 0
        self.last_memory_tokens_after = int(memory.shape[0])
        self.last_memory_token_ratio = float(self.last_memory_tokens_after / max(1, self.last_memory_tokens_before))
        self.last_effective_memory_length = int(memory.shape[0])
        num_obj_ptr_tokens_eff = int(num_obj_ptr_tokens)
        self._sync_cuda(device)
        t0 = time.perf_counter()
        pix_feat_with_mem = self.sam_model.memory_attention(
            curr=vision_feats,
            curr_pos=vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens_eff,
        )
        self._sync_cuda(device)
        self.last_attention_time = float(time.perf_counter() - t0)
        self.attention_time_accum += float(self.last_attention_time)
        self.attention_time_count += 1
        if not self.last_compression_stats:
            self.last_compression_stats = {
                "original_num_tokens": self.last_memory_tokens_before,
                "compressed_num_tokens": self.last_memory_tokens_after,
                "compression_ratio": self.last_memory_token_ratio,
                "original_hw": (H, W),
                "compressed_hw": (H, W),
                "compression_mode": self.memory_compress_mode,
                "compression_location": self.memory_compress_location,
                "skipped": True,
                "skip_reason": "no_stats",
            }
        self._maybe_log_compression(self.last_compression_stats)
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem
