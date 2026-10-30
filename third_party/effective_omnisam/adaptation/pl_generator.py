import os
import time
import torch
import numpy as np
from tqdm import tqdm

from utils.io import ensure_dir
from utils.vis import mask_vis
from utils.tools import merge_width_scatter_np_uncertainty
from utils.pl_merge import PlMergeConfig, merge_patch_sequence
from utils.pl_sidecar import save_sidecar


class PseudoLabelGenerator:
    def __init__(self, model, device, class_names, num_classes):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = num_classes

    @staticmethod
    def _label_rel_path(name: str) -> str:
        rel_name = name.lstrip("/")
        return rel_name.replace(".jpg", "_labelTrainIds.png")

    def _label_abs_path(self, save_dir: str, name: str) -> str:
        return os.path.join(save_dir, "gtFine", self._label_rel_path(name))

    def _sync_cuda(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def run(
        self,
        dataloader,
        save_dir: str,
        max_num: int,
        dataset_name: str,
        uc_threshold: float,
        amp: bool = False,
        patch_scheduler=None,
        fixed_subset: set | None = None,
        timing_warmup_batches: int = 0,
        psr_pld_cfg: dict | None = None,
    ):
        self.model.eval()
        ensure_dir(save_dir)
        ensure_dir(os.path.join(save_dir, "gtFine"))
        ensure_dir(os.path.join(save_dir, "gtFine_overlap"))
        t0 = time.perf_counter()

        completed_names = []
        completed_set = set()
        target_total = max_num
        total_patch_before = 0
        total_patch_after = 0
        complexity_sum = 0.0
        complexity_cnt = 0
        dense_patch_cnt = 0
        sparse_patch_cnt = 0
        t_patch_prepare = 0.0
        t_forward = 0.0
        t_backward = 0.0
        t_merge = 0.0
        t_conf_filter = 0.0
        t_sidecar = 0.0
        high_conf_px = 0
        # PSR-PLD merge config (None → legacy behaviour)
        _psr_cfg: PlMergeConfig | None = None
        if isinstance(psr_pld_cfg, dict):
            pl_merge_d = psr_pld_cfg.get('pl_merge', {})
            if isinstance(pl_merge_d, dict) and pl_merge_d.get('enabled', True):
                _psr_cfg = PlMergeConfig.from_dict(pl_merge_d)
        _debug_saved = 0
        _debug_max = int((psr_pld_cfg or {}).get('pl_merge', {}).get('debug_max_images', 10))
        _save_sidecar = isinstance(psr_pld_cfg, dict) and bool(
            psr_pld_cfg.get('saving', {}).get('save_confidence', False))
        _save_debug = (_psr_cfg is not None and _psr_cfg.save_debug_maps)
        uncertain_px = 0
        valid_px = 0
        total_px = 0
        raw_mem_tokens = 0.0
        compressed_mem_tokens = 0.0
        mem_token_ratio_acc = 0.0
        mem_token_obs = 0
        mem_attn_time = 0.0
        target_subset_names = []
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        progress_bar = tqdm(total=target_total, desc="Pseudolabel Generation")

        use_amp = bool(amp) and self.device.type == "cuda"

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                depth_patches = None
                if len(batch) >= 4 and torch.is_tensor(batch[-1]) and batch[-1].ndim == 5:
                    images, _, names, depth_patches = batch
                else:
                    images, _, names = batch
                batch_names = list(names)

                pending_indices = []
                for idx, name in enumerate(batch_names):
                    if fixed_subset is not None and name not in fixed_subset:
                        continue
                    label_path = self._label_abs_path(save_dir, name)
                    if os.path.isfile(label_path):
                        if name not in completed_set and len(completed_names) < target_total:
                            completed_names.append(name)
                            completed_set.add(name)
                            progress_bar.update(1)
                    elif len(completed_names) + len(pending_indices) < target_total:
                        pending_indices.append(idx)

                if len(completed_names) >= target_total:
                    break

                if not pending_indices:
                    continue

                images = images[pending_indices].to(self.device)
                if depth_patches is not None:
                    depth_patches = depth_patches[pending_indices].to(self.device)
                pending_names = [batch_names[i] for i in pending_indices]
                target_subset_names.extend(pending_names)

                B = images.shape[0]
                frame_cnt = images.shape[1] // 2
                H, W = images.shape[-2:]
                logits_maps = np.zeros((B, 2 * frame_cnt, len(self.class_names), H, W))
                total_patch_before += int(B * frame_cnt)

                do_measure = batch_idx >= int(max(0, timing_warmup_batches))
                if do_measure:
                    self._sync_cuda()
                    t_pp0 = time.perf_counter()
                if (
                    patch_scheduler is not None
                    and getattr(patch_scheduler, "enabled", False)
                    and str(getattr(patch_scheduler, "apply_stage", "pseudo_label")).lower() == "pseudo_label"
                ):
                    full_w = int(frame_cnt * int(getattr(patch_scheduler, "baseline_stride", 384)) + (W - int(getattr(patch_scheduler, "baseline_stride", 384))))
                    schedule = patch_scheduler.select(
                        images[:, :frame_cnt],
                        frame_cnt,
                        patch_w=int(W),
                        full_w=full_w,
                    )
                    selected_idx = schedule.selected_patch_indices
                    if schedule.patch_score_map:
                        complexity_sum += float(np.sum(schedule.patch_score_map))
                        complexity_cnt += int(len(schedule.patch_score_map))
                    if schedule.patch_score_map:
                        thr = float(schedule.complexity_threshold)
                        dense_patch_cnt += int(sum(1 for s in schedule.patch_score_map if s >= thr))
                        sparse_patch_cnt += int(sum(1 for s in schedule.patch_score_map if s < thr))
                else:
                    selected_idx = list(range(frame_cnt))
                if not selected_idx:
                    selected_idx = [0]
                selected_set = set(selected_idx)
                total_patch_after += int(B * len(selected_idx))
                if do_measure:
                    self._sync_cuda()
                    t_patch_prepare += (time.perf_counter() - t_pp0)

                for frame_idx in selected_idx:
                    frame_img = images[:, frame_idx]
                    frame_depth = None
                    if depth_patches is not None and depth_patches.shape[1] >= (frame_idx + 1):
                        frame_depth = depth_patches[:, frame_idx]
                    if hasattr(self.model, "memory_bank"):
                        self.model.memory_bank.set_runtime_context(stage="pseudo_label", iter_idx=int(batch_idx))
                    if do_measure:
                        self._sync_cuda()
                        t_f0 = time.perf_counter()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output, _, _ = self.model(
                            frame_idx if dataset_name != 'Stanford2D3D' else 0,
                            frame_img,
                            depth=frame_depth,
                        )
                    if do_measure:
                        self._sync_cuda()
                        t_forward += (time.perf_counter() - t_f0)
                    logits_maps[:, frame_idx] = output.detach().float().cpu().numpy()
                    raw_mem_tokens += float(getattr(self.model.memory_bank, "last_memory_tokens_before", 0.0))
                    compressed_mem_tokens += float(getattr(self.model.memory_bank, "last_memory_tokens_after", 0.0))
                    mem_token_ratio_acc += float(getattr(self.model.memory_bank, "last_memory_token_ratio", 1.0))
                    mem_attn_time += float(getattr(self.model.memory_bank, "last_attention_time", 0.0))
                    mem_token_obs += 1
                # Fill skipped patches with nearest selected patch logits (baseline-safe fallback).
                for frame_idx in range(frame_cnt):
                    if frame_idx in selected_set:
                        continue
                    nearest = min(selected_idx, key=lambda j: abs(j - frame_idx))
                    logits_maps[:, frame_idx] = logits_maps[:, nearest]

                self.model.memory_bank._init_output_dict_source()

                for frame_idx in selected_idx:
                    frame_img = images[:, frame_cnt + frame_idx]
                    frame_depth = None
                    if depth_patches is not None and depth_patches.shape[1] >= (frame_cnt + frame_idx + 1):
                        frame_depth = depth_patches[:, frame_cnt + frame_idx]
                    if hasattr(self.model, "memory_bank"):
                        self.model.memory_bank.set_runtime_context(stage="pseudo_label", iter_idx=int(batch_idx))
                    if do_measure:
                        self._sync_cuda()
                        t_b0 = time.perf_counter()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output, _, _ = self.model(
                            frame_idx if dataset_name != 'Stanford2D3D' else 0,
                            frame_img,
                            depth=frame_depth,
                        )
                    if do_measure:
                        self._sync_cuda()
                        t_backward += (time.perf_counter() - t_b0)
                    logits_maps[:, frame_cnt + frame_idx] = output.detach().float().cpu().numpy()
                    raw_mem_tokens += float(getattr(self.model.memory_bank, "last_memory_tokens_before", 0.0))
                    compressed_mem_tokens += float(getattr(self.model.memory_bank, "last_memory_tokens_after", 0.0))
                    mem_token_ratio_acc += float(getattr(self.model.memory_bank, "last_memory_token_ratio", 1.0))
                    mem_attn_time += float(getattr(self.model.memory_bank, "last_attention_time", 0.0))
                    mem_token_obs += 1
                for frame_idx in range(frame_cnt):
                    if frame_idx in selected_set:
                        continue
                    nearest = min(selected_idx, key=lambda j: abs(j - frame_idx))
                    logits_maps[:, frame_cnt + frame_idx] = logits_maps[:, frame_cnt + nearest]

                self.model.memory_bank._init_output_dict_source()

                if do_measure:
                    self._sync_cuda()
                    t_m0 = time.perf_counter()
                _stride = int(getattr(patch_scheduler, "baseline_stride", 384)) if patch_scheduler is not None else 384
                _orig_size = (1024, 4096) if dataset_name != 'Stanford2D3D' else (1024, 4096)
                if _psr_cfg is not None:
                    # PSR-PLD merge path (supports normalized_mean etc.)
                    _merge_cfg = PlMergeConfig(
                        mode=_psr_cfg.mode,
                        threshold=uc_threshold,
                        ignore_index=255,
                        save_debug_maps=_psr_cfg.save_debug_maps,
                        debug_max_images=_psr_cfg.debug_max_images,
                    )
                    _result = merge_patch_sequence(
                        logits_np=logits_maps,
                        orig_size=_orig_size,
                        W_patch=W,
                        stride_w=_stride,
                        passes=2,
                        device='cuda',
                        cfg=_merge_cfg,
                    )
                    merged_pred = _result.pred
                    _confidence = _result.confidence  # float32 (N,H,W)
                    _coverage   = _result.coverage    # uint8  (N,H,W)
                else:
                    # Legacy path – bit-exact with old behaviour
                    merged_pred = merge_width_scatter_np_uncertainty(
                        logits_np=logits_maps,
                        orig_size=_orig_size,
                        W_patch=W,
                        stride_w=_stride,
                        passes=2,
                        device='cuda',
                        return_probs=False,
                        threshold=uc_threshold,
                    )
                    _confidence = None
                    _coverage   = None
                if do_measure:
                    self._sync_cuda()
                    t_merge += (time.perf_counter() - t_m0)
                    t_cf0 = time.perf_counter()
                ignore_idx = 255
                valid = (merged_pred != ignore_idx)
                valid_px += int(valid.sum())
                uncertain = (~valid)
                uncertain_px += int(uncertain.sum())
                total_px += int(merged_pred.size)
                # In current implementation, valid pixels are exactly confidence-kept pixels.
                high_conf_px += int(valid.sum())
                if do_measure:
                    t_conf_filter += (time.perf_counter() - t_cf0)
                mask_vis(merged_pred, pending_names, dataset_name, save_dir, None)
                # ── sidecar save ──────────────────────────────────────────────────
                if _confidence is not None and _save_sidecar:
                    ts0 = time.perf_counter()
                    _B_cur = merged_pred.shape[0]
                    for _bi, _nm in enumerate(pending_names[:_B_cur]):
                        _save_debug_now = _save_debug and _debug_saved < _debug_max
                        save_sidecar(
                            pl_root=save_dir,
                            img_name=_nm,
                            confidence=_confidence[_bi],
                            coverage=_coverage[_bi] if _coverage is not None else None,
                            save_coverage=_save_debug_now,
                        )
                        if _save_debug_now:
                            _debug_saved += 1
                    if do_measure:
                        t_sidecar += (time.perf_counter() - ts0)

                for name in pending_names:
                    if name not in completed_set and len(completed_names) < target_total:
                        completed_names.append(name)
                        completed_set.add(name)
                        progress_bar.update(1)

                if len(completed_names) >= target_total:
                    break

        progress_bar.close()
        elapsed = time.perf_counter() - t0
        before = max(1, int(total_patch_before))
        after = max(1, int(total_patch_after))
        stats = {
            "pseudo_label_generation_time": float(elapsed),
            "avg_patch_count_before": float(total_patch_before / max(1, len(completed_names))),
            "avg_patch_count_after": float(total_patch_after / max(1, len(completed_names))),
            "patch_reduction_ratio": float(max(0.0, 1.0 - (after / float(before)))),
            "avg_patch_complexity": float(complexity_sum / max(1, complexity_cnt)),
            "dense_patch_count": int(dense_patch_cnt),
            "sparse_patch_count": int(sparse_patch_cnt),
            "raw_patch_count_total": int(total_patch_before),
            "effective_patch_count_total": int(total_patch_after),
            "avg_raw_patches_per_image": float(total_patch_before / max(1, len(completed_names))),
            "avg_effective_patches_per_image": float(total_patch_after / max(1, len(completed_names))),
            "pl_patch_prepare_time": float(t_patch_prepare),
            "pl_forward_time": float(t_forward),
            "pl_backward_time": float(t_backward),
            "pl_merge_time": float(t_merge),
            "pl_sidecar_time": float(t_sidecar),
            "pl_confidence_filter_time": float(t_conf_filter),
            "pl_total_time": float(elapsed),
            "pl_time_per_image": float(elapsed / max(1, len(completed_names))),
            "pl_images_per_sec": float(len(completed_names) / max(1e-8, elapsed)),
            "high_conf_ratio": float(high_conf_px / max(1, total_px)),
            "uncertain_ratio": float(uncertain_px / max(1, total_px)),
            "valid_pseudo_ratio": float(valid_px / max(1, total_px)),
            "raw_memory_tokens": float(raw_mem_tokens / max(1, mem_token_obs)),
            "compressed_memory_tokens": float(compressed_mem_tokens / max(1, mem_token_obs)),
            "memory_compression_ratio": float(mem_token_ratio_acc / max(1, mem_token_obs)),
            "memory_attention_time": float(mem_attn_time),
            "memory_attention_time_per_call": float(mem_attn_time / max(1, mem_token_obs)),
            "compressor_mode": str(getattr(self.model.memory_bank, "memory_compress_mode", "none")),
            "compress_location": str(getattr(self.model.memory_bank, "memory_compress_location", "before_attention")),
            "memory_bank_size": int(getattr(self.model.memory_bank, "num_maskmem", 0)),
            "target_subset_size": int(len(completed_names)),
            "target_subset_names_sample": target_subset_names[:10],
        }
        if torch.cuda.is_available():
            try:
                stats["peak_gpu_mem_pl"] = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                stats["peak_gpu_mem_pl"] = float("nan")
        return completed_names[:target_total], stats
