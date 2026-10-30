import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm
import time

from utils.metrics import fast_hist, per_class_iu, per_class_dice, boundary_f_score, scores_from_hist
from adaptation.strategies import BaseMemStrategy
from adaptation.methods import PrototypicalAdaptation, PrototypicalAdaptationEMA


class Trainer:
    def __init__(self, args, model_tgt, model_src, optimizer, criteria: Dict[str, torch.nn.Module],
                 adaptation_kwargs: Dict, mem_strategy: BaseMemStrategy,
                 callbacks: List, device):
        self.args = args
        self.model_tgt = model_tgt
        self.model_src = model_src
        self.optimizer = optimizer
        self.criteria = criteria
        self.device = device
        self.callbacks = callbacks
        self.mem_strategy = mem_strategy

        self.proto_bank = PrototypicalAdaptationEMA(**adaptation_kwargs)
        self.alpha = self.args.alpha
        self.scaler = None
        if getattr(self.args, "amp", False) and device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        self.last_epoch_stats: Dict[str, float] = {}
        self._epoch_metric_history: Dict[str, list] = {}
        self.last_iter_metric_records: List[Dict[str, float]] = []
        self.plq_diag_records: List[Dict] = []  # per-step PLQ diagnostics (mode=plq_basic)
        self.proto_sim_diag_records: List[Dict] = []  # per-step R4 prototype similarity diagnostics
        self.geo_memory_diag_records: List[Dict] = []  # per-frame Geo-TeCMA geometry diagnostics
        self.geo_memory_token_summary_records: List[Dict] = []  # diagnosis-only token summary records
        self.geo_memory_token_sample_records: List[Dict] = []  # capped diagnosis-only token sample records

    @staticmethod
    def _safe_float(v):
        try:
            return float(v)
        except Exception:
            return None

    @staticmethod
    def _mean_std(values):
        if not values:
            return 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    def _init_epoch_metric_history(self):
        self._epoch_metric_history = {
            "fusion_alpha": [],
            "text_scale": [],
            "text_logits_mean": [],
            "text_logits_std": [],
            "visual_logits_mean": [],
            "visual_logits_std": [],
            "final_logits_mean": [],
            "final_logits_std": [],
            "fusion_alpha_grad_norm": [],
            "text_scale_grad_norm": [],
            "text_adapter_grad_norm": [],
            "semantic_decoder_grad_norm": [],
            "prediction_change_ratio": [],
            "prediction_change_ratio_cls_1": [],  # sidewalk
            "prediction_change_ratio_cls_3": [],  # wall
            "prediction_change_ratio_cls_4": [],  # fence
            "prediction_change_ratio_cls_5": [],  # pole
            "prediction_change_ratio_cls_6": [],  # traffic light
            "prediction_change_ratio_cls_7": [],  # traffic sign
            "prediction_change_ratio_cls_9": [],  # terrain
            "cross_beta": [],
            "cross_attn_mean": [],
            "cross_attn_std": [],
            "cross_attn_entropy": [],
            "cross_attn_max_mean": [],
            "visual_feature_mean": [],
            "visual_feature_std": [],
            "fused_feature_mean": [],
            "fused_feature_std": [],
            "feature_delta_mean": [],
            "feature_delta_norm": [],
            "feature_delta_ratio": [],
            "logits_delta_mean": [],
            "logits_delta_norm": [],
            "cross_attn_grad_norm": [],
            "cross_beta_grad_norm": [],
            "grounding_gamma": [],
            "grounding_logit_scale": [],
            "grounding_gamma_grad_norm": [],
            "grounding_logit_scale_grad_norm": [],
            "query_self_attn_grad_norm": [],
            "query_cross_attn_grad_norm": [],
            "query_ffn_grad_norm": [],
            "text_to_image_attn_mean": [],
            "text_to_image_attn_std": [],
            "text_to_image_attn_entropy": [],
            "text_to_image_attn_max": [],
            "text_grounding_logits_mean": [],
            "text_grounding_logits_std": [],
            "text_logits_delta_norm": [],
            "final_visual_logits_delta_norm": [],
            "text_only_prediction_change_ratio": [],
            "loss_text_grounding_ce": [],
        }
        self.last_iter_metric_records = []
        self.plq_diag_records = []
        self.proto_sim_diag_records = []
        self.geo_memory_diag_records = []
        self.geo_memory_token_summary_records = []
        self.geo_memory_token_sample_records = []

    def _record_iter_metrics(self, metrics: Dict):
        if not isinstance(metrics, dict):
            return
        for k in self._epoch_metric_history.keys():
            if k in metrics:
                fv = self._safe_float(metrics.get(k))
                if fv is not None:
                    self._epoch_metric_history[k].append(fv)

    def _sync_cuda(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def train_one_epoch(self, epoch: int, global_step, source_loader, target_loader, is_warmup, start_iter: int = 0):
        self.model_tgt.train()
        self.model_src.eval()
        self._init_epoch_metric_history()
        t_epoch = time.perf_counter()
        iter_durations = []
        data_durations = []
        compute_durations = []
        images_seen = 0
        patches_seen = 0

        total_iters = len(target_loader)
        if total_iters == 0:
            self.last_epoch_stats = {"step_time": 0.0, "epoch_time": 0.0}
            return global_step

        src_it = iter(source_loader)
        tgt_it = iter(target_loader)

        for _ in range(start_iter):
            next(src_it)
            next(tgt_it)

        max_iters_cfg = int(max(0, getattr(self.args, "train_max_iters", 0)))
        end_iter = total_iters if max_iters_cfg <= 0 else min(total_iters, start_iter + max_iters_cfg)
        for it in tqdm(range(start_iter, end_iter), desc=f"Epoch {epoch+1} [Train]"):
            t_iter = time.perf_counter()
            self._sync_cuda()
            t_data0 = time.perf_counter()
            batch_s = next(src_it)
            batch_t = next(tgt_it)
            self._sync_cuda()
            do_measure = (it - start_iter) >= int(max(0, getattr(self.args, "train_timing_warmup_iters", 0)))
            if do_measure:
                data_durations.append(time.perf_counter() - t_data0)

            self.optimizer.zero_grad()

            self._sync_cuda()
            t_comp0 = time.perf_counter()
            global_step = self.mem_strategy.forward(
                global_step,
                self.model_tgt,
                self.model_src,
                batch_s,
                batch_t,
                self.criteria,
                self.proto_bank,
                self.alpha,
                self.device,
                self.optimizer,
                self.callbacks,
                is_warmup,
                self.args,
                total_iters,
                self.scaler,
            )
            # Capture per-iter dynamics from strategy callback payload.
            if hasattr(self.mem_strategy, "last_iter_metrics"):
                iter_metrics = getattr(self.mem_strategy, "last_iter_metrics")
                self._record_iter_metrics(iter_metrics)
                # --- PLQ per-step diagnostics (no overhead when mode != plq_basic) ---
                _geo_recs = list(getattr(self.mem_strategy, "geo_diag_records", []) or [])
                if _geo_recs:
                    for _grec in _geo_recs:
                        _out = {"epoch": int(epoch + 1), "iter": int(it + 1)}
                        _out.update(dict(_grec))
                        self.geo_memory_diag_records.append(_out)
                _geo_cfg = self.args.plugins_cfg.get("geo_memory", {}) if isinstance(getattr(self.args, "plugins_cfg", None), dict) else {}
                _writer_cfg = _geo_cfg.get("token_writer", {}) if isinstance(_geo_cfg, dict) else {}
                _first_n_batches = int(max(0, _writer_cfg.get("first_n_batches", 1)))
                if _first_n_batches <= 0 or (it - start_iter) < _first_n_batches:
                    _max_summary = int(max(0, _writer_cfg.get("max_summary_records", 200)))
                    _summary_recs = list(getattr(self.mem_strategy, "geo_token_summary_records", []) or [])
                    if _summary_recs and (_max_summary <= 0 or len(self.geo_memory_token_summary_records) < _max_summary):
                        _room = len(_summary_recs) if _max_summary <= 0 else max(0, _max_summary - len(self.geo_memory_token_summary_records))
                        self.geo_memory_token_summary_records.extend(dict(r) for r in _summary_recs[:_room])
                    _max_token = int(max(0, _writer_cfg.get("max_records", 5)))
                    _token_recs = list(getattr(self.mem_strategy, "geo_token_sample_records", []) or [])
                    if _token_recs and (_max_token <= 0 or len(self.geo_memory_token_sample_records) < _max_token):
                        _room = len(_token_recs) if _max_token <= 0 else max(0, _max_token - len(self.geo_memory_token_sample_records))
                        for _trec in _token_recs[:_room]:
                            _tout = dict(_trec)
                            _tout["epoch"] = int(epoch + 1)
                            self.geo_memory_token_sample_records.append(_tout)
                if isinstance(iter_metrics, dict) and str(iter_metrics.get("refiner_mode", "")) == "plq_basic":
                    _prec = {
                        "epoch": int(epoch + 1),
                        "iter": int(it + 1),
                        "global_step": int(global_step),
                    }
                    _plq_keys = (
                        "plq_mean", "plq_min", "plq_max", "plq_std",
                        "confidence_mean", "entropy_reliability_mean", "local_agreement_mean",
                        "plq_nan_count", "plq_zero_ratio",
                        "plq_common_mean", "plq_rare_mean", "plq_extreme_rare_mean",
                        "refiner_total_valid_ratio", "refiner_rare_valid_ratio",
                        "refiner_common_valid_ratio", "refiner_ext_rare_valid_ratio",
                        "loss_sup_s", "loss_unsup", "gpu_mem_peak_mb",
                    )
                    for _k in _plq_keys:
                        if _k in iter_metrics:
                            try:
                                _prec[_k] = float(iter_metrics[_k])
                            except (TypeError, ValueError):
                                _prec[_k] = iter_metrics[_k]
                    # plq_by_class / plq_histogram / valid_ratio_by_plq_bin
                    for _prefix in ("plq_cls", "plq_hist_", "plq_vr_bin_"):
                        for _ik, _iv in iter_metrics.items():
                            if str(_ik).startswith(_prefix):
                                try:
                                    _prec[_ik] = float(_iv)
                                except (TypeError, ValueError):
                                    _prec[_ik] = _iv
                    self.plq_diag_records.append(_prec)
                # ----------------------------------------------------------------
                # --- R4 prototype similarity diagnostics (diagnostic-only; no filtering) ---
                if isinstance(iter_metrics, dict) and "proto_sim_mean" in iter_metrics:
                    _prec = {
                        "epoch": int(epoch + 1),
                        "iter": int(it + 1),
                        "global_step": int(global_step),
                        "refiner_mode": str(iter_metrics.get("refiner_mode", "unknown")),
                    }
                    _proto_keys = (
                        "proto_sim_mean", "proto_sim_std", "proto_sim_min", "proto_sim_max",
                        "proto_sim_valid_ratio", "low_sim_ratio_at_0.2", "low_sim_ratio_at_0.4",
                        "low_sim_ratio_at_0.6", "refiner_total_valid_ratio", "refiner_rare_valid_ratio",
                        "refiner_ext_rare_valid_ratio", "loss_sup_s", "loss_unsup", "proto_sim_nan_count",
                        "proto_sim_q01", "proto_sim_q05", "proto_sim_q10", "proto_sim_q25",
                        "proto_sim_q50", "proto_sim_q75", "proto_sim_q90", "proto_sim_q95", "proto_sim_q99",
                        "low_sim_ratio_at_0.65", "low_sim_ratio_at_0.70", "low_sim_ratio_at_0.75",
                        "low_sim_ratio_at_0.80", "low_sim_ratio_at_0.85", "low_sim_ratio_at_0.90",
                        "proto_sim_mean_common", "proto_sim_mean_rare", "proto_sim_mean_extreme_rare",
                        "low_sim_ratio_at_0.75_common", "low_sim_ratio_at_0.75_rare",
                        "low_sim_ratio_at_0.80_common", "low_sim_ratio_at_0.80_rare",
                        "low_sim_ratio_at_0.85_common", "low_sim_ratio_at_0.85_rare",
                        "simulated_filter_ratio_total", "simulated_filter_ratio_common",
                        "simulated_filter_ratio_rare", "simulated_filter_ratio_extreme_rare",
                        "simulated_keep_ratio_total", "simulated_keep_ratio_rare",
                        "r2_total_valid_ratio", "r2_common_valid_ratio", "r2_rare_valid_ratio", "r2_extreme_rare_valid_ratio",
                        "proto_gate_filter_ratio_total", "proto_gate_keep_ratio_total",
                        "proto_gate_filter_ratio_common", "proto_gate_filter_ratio_rare", "proto_gate_filter_ratio_extreme_rare",
                        "proto_gate_final_valid_ratio", "proto_gate_final_common_valid_ratio",
                        "proto_gate_final_rare_valid_ratio", "proto_gate_final_extreme_rare_valid_ratio",
                    )
                    for _k in _proto_keys:
                        if _k in iter_metrics:
                            try:
                                _prec[_k] = float(iter_metrics[_k])
                            except (TypeError, ValueError):
                                _prec[_k] = iter_metrics[_k]
                    _prec["total_valid_ratio_after_R2"] = _prec.get("refiner_total_valid_ratio")
                    _prec["rare_valid_ratio_after_R2"] = _prec.get("refiner_rare_valid_ratio")
                    _prec["extreme_rare_valid_ratio_after_R2"] = _prec.get("refiner_ext_rare_valid_ratio")
                    _prec["loss_sup"] = _prec.get("loss_sup_s")
                    _prec["nan_count"] = _prec.get("proto_sim_nan_count", 0.0)
                    for _k in (
                        "proto_valid_classes", "proto_missing_classes", "proto_counts_by_class", "proto_sim_mean_by_class",
                        "low_sim_ratio_at_0.75_by_class", "low_sim_ratio_at_0.80_by_class",
                        "low_sim_ratio_at_0.85_by_class", "proto_sim_q10_by_class",
                        "proto_sim_q25_by_class", "proto_sim_q50_by_class",
                        "proto_gate_filtered_pixels_by_class", "proto_gate_kept_pixels_by_class",
                        "proto_gate_total_pixels_by_class", "proto_gate_threshold_by_class",
                        "proto_gate_fallback_classes", "proto_gate_fallback_reasons",
                    ):
                        if _k in iter_metrics:
                            _prec[_k] = iter_metrics[_k]
                    for _k, _v in iter_metrics.items():
                        if str(_k).startswith("simulated_abs_") or str(_k).startswith("simulated_global_") or str(_k).startswith("simulated_classwise_percentile_"):
                            _prec[_k] = _v
                    self.proto_sim_diag_records.append(_prec)
                # ----------------------------------------------------------------
                if isinstance(iter_metrics, dict):
                    rec = {"epoch": int(epoch + 1), "iter": int(it + 1), "global_step": int(global_step)}
                    keys = (
                        "fusion_alpha",
                        "text_scale",
                        "text_logits_mean",
                        "text_logits_std",
                        "visual_logits_mean",
                        "visual_logits_std",
                        "final_logits_mean",
                        "final_logits_std",
                        "fusion_alpha_grad_norm",
                        "text_scale_grad_norm",
                        "text_adapter_grad_norm",
                        "semantic_decoder_grad_norm",
                        "prediction_change_ratio",
                        "prediction_change_ratio_cls_1",
                        "prediction_change_ratio_cls_3",
                        "prediction_change_ratio_cls_4",
                        "prediction_change_ratio_cls_5",
                        "prediction_change_ratio_cls_6",
                        "prediction_change_ratio_cls_7",
                        "prediction_change_ratio_cls_9",
                        "cross_beta",
                        "cross_attn_mean",
                        "cross_attn_std",
                        "cross_attn_entropy",
                        "cross_attn_max_mean",
                        "visual_feature_mean",
                        "visual_feature_std",
                        "fused_feature_mean",
                        "fused_feature_std",
                        "feature_delta_mean",
                        "feature_delta_norm",
                        "feature_delta_ratio",
                        "logits_delta_mean",
                        "logits_delta_norm",
                        "cross_attn_grad_norm",
                        "cross_beta_grad_norm",
                        "grounding_gamma",
                        "grounding_logit_scale",
                        "grounding_gamma_grad_norm",
                        "grounding_logit_scale_grad_norm",
                        "query_self_attn_grad_norm",
                        "query_cross_attn_grad_norm",
                        "query_ffn_grad_norm",
                        "text_to_image_attn_mean",
                        "text_to_image_attn_std",
                        "text_to_image_attn_entropy",
                        "text_to_image_attn_max",
                        "text_grounding_logits_mean",
                        "text_grounding_logits_std",
                        "text_logits_delta_norm",
                        "final_visual_logits_delta_norm",
                        "text_only_prediction_change_ratio",
                        "loss_text_grounding_ce",
                    )
                    has_any = False
                    for k in keys:
                        fv = self._safe_float(iter_metrics.get(k))
                        if fv is not None:
                            rec[k] = fv
                            has_any = True
                    if has_any:
                        self.last_iter_metric_records.append(rec)
            self._sync_cuda()
            comp_t = time.perf_counter() - t_comp0
            if do_measure:
                compute_durations.append(comp_t)
            try:
                images_t = batch_t[0]
                bsz = int(images_t.shape[0])
                frame_cnt = int(images_t.shape[1]) if images_t.ndim >= 5 else 1
                if do_measure:
                    images_seen += bsz
                    patches_seen += (bsz * frame_cnt)
            except Exception:
                pass
            if hasattr(self.args, "on_train_iter_end") and callable(self.args.on_train_iter_end):
                self.args.on_train_iter_end(epoch, it + 1, global_step)
            if do_measure:
                iter_durations.append(time.perf_counter() - t_iter)

        epoch_time = time.perf_counter() - t_epoch
        step_time = float(sum(iter_durations) / max(1, len(iter_durations)))
        self.last_epoch_stats = {
            "step_time": step_time,
            "epoch_time": float(epoch_time),
            "data_time": float(sum(data_durations) / max(1, len(data_durations))),
            "iter_compute_time": float(sum(compute_durations) / max(1, len(compute_durations))),
            "iter_total_time": float(step_time),
            "train_images_per_sec": float(images_seen / max(1e-8, sum(compute_durations))),
            "train_patches_per_sec": float(patches_seen / max(1e-8, sum(compute_durations))),
            "avg_effective_patches_per_iter": float(patches_seen / max(1, len(iter_durations))),
            "epoch_train_time": float(sum(compute_durations)),
            "epoch_total_time": float(epoch_time),
            "memory_tokens_before": float(getattr(self.model_tgt.memory_bank, "last_memory_tokens_before", 0.0)),
            "memory_tokens_after": float(getattr(self.model_tgt.memory_bank, "last_memory_tokens_after", 0.0)),
            "memory_token_ratio": float(getattr(self.model_tgt.memory_bank, "last_memory_token_ratio", 1.0)),
            "memory_attention_time": float(getattr(self.model_tgt.memory_bank, "attention_time_accum", 0.0)),
            "memory_attention_time_per_call": float(
                getattr(self.model_tgt.memory_bank, "attention_time_accum", 0.0)
                / max(1, int(getattr(self.model_tgt.memory_bank, "attention_time_count", 0)))
            ),
        }
        # Export text-fusion dynamics (start/end + epoch mean/std).
        for k, vals in self._epoch_metric_history.items():
            mean_v, std_v = self._mean_std(vals)
            self.last_epoch_stats[f"{k}_mean"] = mean_v
            self.last_epoch_stats[f"{k}_std"] = std_v
            self.last_epoch_stats[f"{k}_start"] = float(vals[0]) if vals else 0.0
            self.last_epoch_stats[f"{k}_end"] = float(vals[-1]) if vals else 0.0
        return global_step

    def validate(self, epoch: int, val_loader, class_names: List[str]):
        self.model_tgt.eval()
        hist = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
        ignore_index = int(getattr(self.args, "ignore_index", 255))
        bf_sum = 0.0
        bf_cnt = 0

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for idx, batch in enumerate(progress_bar):
                images, labels, *_ = batch
                images, labels = images.to(self.device), labels.to(self.device)
                frame_cnt = images.shape[1]
                # Build historical_pseudo_labels for boundary_guarded mode during eval
                eval_pseudo_labels = {}
                for frame_idx in range(frame_cnt):
                    # Pass accumulated historical predictions (not ground truth) to enable boundary detection
                    hist_pl = eval_pseudo_labels if frame_idx > 0 else None
                    preds = self.mem_strategy.forward_eval(self.model_tgt, images, frame_idx, historical_pseudo_labels=hist_pl)
                    pred = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
                    # Store prediction for use as historical pseudo-label in next frame
                    pred_tensor = torch.argmax(preds, dim=1)  # [B, H, W]
                    eval_pseudo_labels[frame_idx] = pred_tensor
                    label_np = labels[:, frame_idx].squeeze(0).cpu().numpy()
                    hist += fast_hist(label_np.flatten(), pred.flatten(), len(class_names))
                    try:
                        bf_sum += float(boundary_f_score(pred, label_np, ignore_index=ignore_index, tolerance=2))
                        bf_cnt += 1
                    except Exception:
                        pass
                self.mem_strategy.reset_memory(self.model_tgt)

        mIoUs = per_class_iu(hist)
        print("========== per-class IoU (0-100) ==========")
        for i, cname in enumerate(class_names):
            print(f"{cname:<15}: {round(mIoUs[i]*100, 2)}")
        print("==========================================")
        metrics = scores_from_hist(hist)
        cur_mIoU = round(float(metrics.get("mIoU", float("nan"))), 2)
        metrics["mIoU"] = cur_mIoU
        # Extra, richer semantic metrics
        try:
            pc_iou = per_class_iu(hist)
            pc_dice = per_class_dice(hist)
            metrics["per_class_IoU"] = [float(x * 100.0) if x == x else None for x in pc_iou]
            metrics["per_class_Dice"] = [float(x * 100.0) if x == x else None for x in pc_dice]
            print("========== per-class Dice/F1 (0-100) ==========")
            for i, cname in enumerate(class_names):
                v = metrics["per_class_Dice"][i]
                if v is None:
                    print(f"{cname:<15}: NA")
                else:
                    print(f"{cname:<15}: {v:.2f}")
            print("===============================================")
        except Exception:
            pass
        if bf_cnt > 0:
            metrics["BFScore"] = float(bf_sum / float(bf_cnt))
        # Summary print
        try:
            fwdice = metrics.get("FWDice", None)
            bfs = metrics.get("BFScore", None)
            if fwdice is not None:
                print(f"[ValSummary] FWDice={float(fwdice):.2f}")
            if bfs is not None:
                print(f"[ValSummary] BFScore={float(bfs):.2f}")
        except Exception:
            pass

        for cb in self.callbacks:
            if hasattr(cb, 'on_validation_end'):
                cb.on_validation_end(epoch, cur_mIoU, self.model_tgt)

        return metrics

    def reset_optimizer(self, lr: float):
        self.optimizer.state.clear()
