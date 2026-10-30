import os
from typing import Protocol, Tuple
import torch
import wandb
import torch.nn.functional as F
from model.text_align import kl_align_loss
from model.text_semantic import TextAwarePseudoLabelFilter, weighted_unsup_loss
from adaptation.plugins.manager import PluginManager
from adaptation.plugins.pseudo_conf_mask import masked_unsup_ce
from adaptation.plugins.depth_boundary_prior import build_depth_boundary_weight
from adaptation.plugins.depth_confidence import compute_depth_confidence_weight
from utils.io import ensure_dir
from experiments.geo_memory.token_diag_writer import build_geo_token_records

def linear_ramp(cur, start_step, end_step, max_w):
    if cur < start_step: return 0.0
    if cur > end_step:   return max_w
    return max_w * (cur - start_step) / (end_step - start_step)

class BaseMemStrategy(Protocol):
    def forward(self, model, model_src, batch_s, batch_t, criteria, proto_bank, alpha, device) -> Tuple[torch.Tensor,...]:
        ...
    def forward_eval(self, model, images, frame_idx): ...
    def reset_memory(self, model): ...

class WithMemStrategy:
    @staticmethod
    def _param_grad_norm(params):
        total = 0.0
        found = False
        for p in params:
            if p is None or p.grad is None:
                continue
            g = p.grad.detach()
            total += float((g * g).sum().item())
            found = True
        if not found:
            return 0.0
        return float(total ** 0.5)

    @staticmethod
    def _to_uint8_img(img: torch.Tensor) -> "Image.Image":
        from PIL import Image

        x = img.detach().float().cpu()
        if x.ndim == 3 and x.shape[0] in {1, 3}:
            x = x.permute(1, 2, 0)
        if x.ndim == 2:
            arr = x.clamp(0, 255).to(torch.uint8).numpy()
            return Image.fromarray(arr)
        x = x - x.min()
        x = x / x.max().clamp_min(1e-8)
        arr = (x * 255.0).clamp(0, 255).to(torch.uint8).numpy()
        return Image.fromarray(arr)

    @classmethod
    def _save_grounding_debug_vis(cls, args, global_step, frame_idx, frame_img_t, aux_t, pred_t, valid_mask):
        if not bool(getattr(args, "text_fusion_save_debug_vis", False)):
            return
        l_ground = getattr(aux_t, "l_ground", None)
        if l_ground is None:
            return
        root = os.path.join(str(getattr(args, "pd_root", ".")), "vis_text_grounding_debug")
        ensure_dir(root)
        vis_limit = int(getattr(args, "text_fusion_save_vis_num", 20))
        sample_id = int(global_step)
        if sample_id >= vis_limit:
            return
        from PIL import Image

        img = frame_img_t[0]
        visual_pred = torch.argmax(aux_t.l_vis[0], dim=0).to(torch.uint8)
        final_pred = torch.argmax(pred_t[0], dim=0).to(torch.uint8)
        text_pred = torch.argmax(l_ground[0], dim=0).to(torch.uint8)
        changed = (visual_pred != final_pred).to(torch.uint8) * 255
        cls._to_uint8_img((img + 1.0) * 127.5).save(os.path.join(root, f"iter{sample_id:06d}_f{frame_idx}_rgb.png"))
        Image.fromarray(visual_pred.cpu().numpy(), mode="L").save(os.path.join(root, f"iter{sample_id:06d}_f{frame_idx}_visual_pred.png"))
        Image.fromarray(final_pred.cpu().numpy(), mode="L").save(os.path.join(root, f"iter{sample_id:06d}_f{frame_idx}_final_pred.png"))
        Image.fromarray(text_pred.cpu().numpy(), mode="L").save(os.path.join(root, f"iter{sample_id:06d}_f{frame_idx}_text_pred.png"))
        Image.fromarray(changed.cpu().numpy(), mode="L").save(os.path.join(root, f"iter{sample_id:06d}_f{frame_idx}_changed.png"))
        attn = getattr(aux_t, "grounding_aux", {}).get("attn_weights", None)
        debug_classes = getattr(args, "text_fusion_debug_classes", [0, 1, 3, 4, 5, 6, 7, 9, 10])
        if attn is not None:
            attn_map = attn[0]
            q_per_class = max(1, int(attn_map.shape[0] // max(1, l_ground.shape[1])))
            attn_map = attn_map.view(l_ground.shape[1], q_per_class, -1).mean(dim=1)
            hw = aux_t.grounding_aux.get("kv_hw", [l_ground.shape[-2], l_ground.shape[-1]])
            for cls_idx in debug_classes:
                cls_int = int(cls_idx)
                if cls_int < 0 or cls_int >= attn_map.shape[0]:
                    continue
                amap = attn_map[cls_int].view(int(hw[0]), int(hw[1]))
                amap = F.interpolate(amap[None, None], size=l_ground.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                cls._to_uint8_img(amap).save(os.path.join(root, f"iter{sample_id:06d}_f{frame_idx}_attn_cls{cls_int}.png"))


    @staticmethod
    def _build_geo_context(args, tensor_order, frame_idx, frame_cnt, mem_type, stage="train"):
        plugins = getattr(args, "plugins_cfg", {}) or {}
        geo_cfg = plugins.get("geo_memory", {}) if isinstance(plugins, dict) else {}
        erp_w = int(geo_cfg.get("erp_width", getattr(args, "crop_size_w", 4096) or 4096))
        erp_h = int(geo_cfg.get("erp_height", getattr(args, "crop_size_h", 1024) or 1024))
        window = int(geo_cfg.get("window_size", getattr(args, "image_size", 1024) or 1024))
        stride = int(geo_cfg.get("stride", getattr(args, "patch_stride_tgt", 384) or 384))
        inferred_x0 = int(frame_idx) * stride
        inferred_x1 = min(erp_w, inferred_x0 + window)
        random_reverse_possible = str(getattr(args, "dataset", "")).lower() in {"wildpass", "wildpass13", "wildpass19"}
        allow_tensor_order = bool(geo_cfg.get("allow_tensor_order_inference", False))
        true_x_range = [inferred_x0, inferred_x1] if allow_tensor_order and not random_reverse_possible else None
        true_y_range = [0, erp_h]
        seam_margin = int(geo_cfg.get("seam_margin", stride))
        if true_x_range is None:
            seam_region = "unknown"
        elif inferred_x0 <= seam_margin:
            seam_region = "left_seam"
        elif inferred_x1 >= erp_w - seam_margin:
            seam_region = "right_seam"
        else:
            seam_region = "non_seam"
        return {
            "stage": str(stage),
            "mem_type": str(mem_type),
            "tensor_order": int(tensor_order),
            "frame_idx": int(frame_idx),
            "frame_cnt": int(frame_cnt),
            "true_erp_x_range": true_x_range,
            "true_erp_y_range": true_y_range,
            "inferred_erp_x_range_from_tensor_order": [inferred_x0, inferred_x1],
            "tensor_order_inference_allowed": bool(allow_tensor_order),
            "random_reverse_possible": bool(random_reverse_possible),
            "true_erp_x_range_reliable": bool(true_x_range is not None),
            "seam_region": seam_region,
            "latitude_source": str(geo_cfg.get("latitude_source", "mem_hw_token_row")),
            "seam_selection_enabled": bool(geo_cfg.get("seam_selection_enabled", False)),
            "selection_effect": "diagnosis_only",
        }

    @staticmethod
    def _geo_diag_record(args, global_step, context, memory_diag):
        mem_hw = memory_diag.get("mem_hw") if isinstance(memory_diag, dict) else None
        if isinstance(mem_hw, tuple):
            mem_hw = list(mem_hw)
        mem_h = int(mem_hw[0]) if isinstance(mem_hw, (list, tuple)) and len(mem_hw) >= 2 else 0
        mem_w = int(mem_hw[1]) if isinstance(mem_hw, (list, tuple)) and len(mem_hw) >= 2 else 0
        return {
            "global_step": int(global_step),
            "stage": str(context.get("stage", "train")),
            "mem_type": str(context.get("mem_type", "target")),
            "tensor_order": int(context.get("tensor_order", -1)),
            "frame_idx": int(context.get("frame_idx", -1)),
            "frame_cnt": int(context.get("frame_cnt", 0)),
            "true_erp_x_range": context.get("true_erp_x_range"),
            "true_erp_y_range": context.get("true_erp_y_range"),
            "inferred_erp_x_range_from_tensor_order": context.get("inferred_erp_x_range_from_tensor_order"),
            "true_erp_x_range_reliable": bool(context.get("true_erp_x_range_reliable", False)),
            "seam_region": str(context.get("seam_region", "unknown")),
            "random_reverse_possible": bool(context.get("random_reverse_possible", False)),
            "tensor_order_inference_allowed": bool(context.get("tensor_order_inference_allowed", False)),
            "mem_hw": [mem_h, mem_w],
            "memory_num_tokens": int(memory_diag.get("num_tokens", 0)) if isinstance(memory_diag, dict) else 0,
            "token_row_mapping": "token_row = token_idx // mem_w",
            "token_col_mapping": "token_col = token_idx % mem_w",
            "latitude_source": str(context.get("latitude_source", "mem_hw_token_row")),
            "seam_selection_enabled": bool(context.get("seam_selection_enabled", False)),
            "selection_effect": "diagnosis_only",
        }

    def forward(self,
                global_step,
                model,
                model_src,
                batch_s,
                batch_t,
                criteria,
                proto_bank,
                alpha,
                device,
                optimizer,
                callbacks,
                is_warmup,
                args=None,
                total_iters: int = 0,
                scaler=None,
                ):
        images_s, labels_s, *_ = batch_s
        # target batch may optionally contain depth boundary patches at the end:
        #   (images, pseudo_labels, size, name, depth_boundary_patches)
        images_t, labels_t, *t_rest = batch_t
        images_s, labels_s = images_s.to(device), labels_s.to(device)
        images_t, labels_t = images_t.to(device), labels_t.to(device)

        depth_patches = None
        # heuristics: if last element is a Tensor with shape (B,frame_cnt,1,H,W), treat as pseudo-depth patches
        if t_rest:
            last = t_rest[-1]
            if torch.is_tensor(last) and last.ndim == 5:
                depth_patches = last.to(device)

        frame_cnt = images_t.shape[1]
        # Use real per-epoch iteration count to keep ramp schedule consistent
        # when pseudo-label subset size (and thus len(target_loader)) changes.
        total_steps = int(total_iters) if total_iters and int(total_iters) > 0 else 1000
        conf_thr = float(getattr(args, "conf_thr", 0.85))
        ignore_index = int(getattr(args, "ignore_index", 255))
        use_text_proto = bool(getattr(args, "use_text_proto", True))
        use_txt = use_text_proto and bool(getattr(args, "text_proto_path", "") or "")
        text_fusion_enabled = bool(getattr(args, "text_fusion_enabled", False))
        text_fusion_type = str(getattr(args, "text_fusion_type", "logit_add")).lower()
        text_fusion_active = text_fusion_enabled or (use_txt and text_fusion_type in {"logit_add", "hybrid"} and float(getattr(args, "lambda_txt", 0.0)) != 0.0)
        w_txt_align = float(getattr(args, "weight_txt_align", 0.0))
        txt_align_tau = float(getattr(args, "txt_align_tau", 1.0))
        use_text_gate_ssl = bool(getattr(args, "use_text_gate_ssl", False))
        ssl_gate_mode = str(getattr(args, "ssl_gate_mode", "hard"))
        tau_v = float(getattr(args, "tau_v", conf_thr))
        tau_t = float(getattr(args, "tau_t", 0.10))
        lambda_ssl = float(getattr(args, "lambda_ssl", 1.0))
        lambda_fpa = float(getattr(args, "lambda_fpa", 1.0))
        loss_text_grounding_enabled = bool(getattr(args, "loss_text_grounding_enabled", False))
        loss_text_grounding_weight = float(getattr(args, "loss_text_grounding_weight", 0.05))
        loss_text_grounding_apply_source = bool(getattr(args, "loss_text_grounding_apply_source", True))
        loss_text_grounding_apply_target = bool(getattr(args, "loss_text_grounding_apply_target", False))
        text_gate = TextAwarePseudoLabelFilter(tau_v=tau_v, tau_t=tau_t, mode=ssl_gate_mode)
        plugins = PluginManager.from_args(args=args, num_classes=0, device=device)
        geo_enabled = plugins.enabled("geo_memory")
        geo_records = []
        geo_token_summary_records = []
        geo_token_sample_records = []
        geo_cfg = plugins.cfg.get("geo_memory", {}) if isinstance(getattr(plugins, "cfg", None), dict) else {}
        geo_token_writer_enabled = geo_enabled and bool((geo_cfg.get("token_writer", {}) or {}).get("enabled", False))
        if geo_enabled and bool(geo_cfg.get("seam_selection_enabled", False)):
            print("[geo_memory][warn] seam_selection_enabled requested but Task 0-3 is diagnosis-only; selection is unchanged.", flush=True)

        for frame_idx in range(frame_cnt):
            geo_ctx_t = self._build_geo_context(args, frame_idx, frame_idx, frame_cnt, "target", stage="train") if geo_enabled else None
            geo_ctx_s = self._build_geo_context(args, frame_idx, frame_idx, frame_cnt, "source", stage="train") if geo_enabled else None
            if hasattr(model, "memory_bank"):
                model.memory_bank.set_runtime_context(stage="train", iter_idx=int(global_step), geo_context=geo_ctx_t)
            if hasattr(model_src, "memory_bank"):
                model_src.memory_bank.set_runtime_context(stage="train", iter_idx=int(global_step), geo_context=geo_ctx_s)
            frame_img_s = images_s[:, frame_idx]
            frame_lbl_s = labels_s[:, frame_idx]
            frame_img_t = images_t[:, frame_idx]
            frame_pl_t = labels_t[:, frame_idx]
            frame_depth_t = None
            if depth_patches is not None and depth_patches.shape[1] == frame_cnt:
                frame_depth_t = depth_patches[:, frame_idx]

            # NEW: Build historical_pseudo_labels for boundary_guarded mode
            # Only for target domain (source has no pseudo labels)
            historical_pseudo_labels_t = None
            if frame_idx > 0:
                # Collect all previous frames' pseudo labels for this iteration
                historical_pseudo_labels_t = {}
                for prev_idx in range(frame_idx):
                    historical_pseudo_labels_t[prev_idx] = labels_t[:, prev_idx]  # [B,H,W]

            use_amp = scaler is not None and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                need_txt_aux = use_txt and (w_txt_align > 0 or use_text_gate_ssl or (text_fusion_active and text_fusion_type != "none"))
                if need_txt_aux:
                    pred_s, feat_s, feat_s_de, aux_s = model(
                        frame_idx, frame_img_s, mem_type='source', return_aux=True, depth=None
                    )
                else:
                    pred_s, feat_s, feat_s_de = model(frame_idx, frame_img_s, mem_type='source', depth=None)
                _, feat_s_ori, feat_s_de_ori = model_src(frame_idx, frame_img_s, mem_type='source')
                if need_txt_aux:
                    pred_t, feat_t, feat_t_de, aux_t = model(
                        frame_idx, frame_img_t, mem_type='target', return_aux=True, depth=frame_depth_t,
                        historical_pseudo_labels=historical_pseudo_labels_t,
                    )
                else:
                    pred_t, feat_t, feat_t_de = model(
                        frame_idx, frame_img_t, mem_type='target', depth=frame_depth_t,
                        historical_pseudo_labels=historical_pseudo_labels_t,
                    )
            with torch.no_grad():
                probs = torch.softmax(pred_t, 1)
                conf_t, pseudo_t = probs.max(1)
                valid_mask = (frame_pl_t != ignore_index)
                visual_mask = (conf_t > conf_thr) & valid_mask
                gate_out = None
                if need_txt_aux and use_text_gate_ssl:
                    gate_out = text_gate(
                        l_txt=aux_t.l_txt,
                        pseudo_t=frame_pl_t,
                        conf_t=conf_t,
                        valid_mask=valid_mask,
                        txt_tau=float(getattr(args, "txt_tau", 0.07)),
                        num_classes=int(aux_t.l_txt.shape[1]),
                    )
                    mask_t = gate_out.ssl_mask_text
                else:
                    mask_t = visual_mask

                # ---- PseudoLabelRefiner (online, pluggable) ----------
                # mode='off' (default): _pl_cfg check short-circuits here;
                # no import, no instantiation, zero overhead on baselines.
                # mode='identity': returns same frame_pl_t / mask_t;
                # downstream pseudo_conf_mask plugin still runs if enabled.
                _pl_refiner_diag = {}   # populated if refiner is active
                _pl_cfg = getattr(args, 'pseudo_label_refiner', None) or {}
                _pl_mode_raw = _pl_cfg.get('mode', 'off')
                if isinstance(_pl_mode_raw, bool):
                    _pl_mode_raw = 'off' if not _pl_mode_raw else 'on'
                if str(_pl_mode_raw).lower() != 'off':
                    from adaptation.pseudo_label import PseudoLabelRefiner
                    _refiner = PseudoLabelRefiner.from_args(
                        args,
                        num_classes=int(pred_t.shape[1]),
                        device=device,
                    )
                    _pl_result = _refiner.refine(
                        pseudo_label=frame_pl_t,
                        confidence=conf_t,
                        student_logits=pred_t,
                        features=feat_t_de,
                        valid_mask=valid_mask,
                        ignore_index=ignore_index,
                        step=global_step,
                    )
                    if _pl_result is not None:
                        frame_pl_t = _pl_result.refined_label
                        mask_t     = _pl_result.valid_mask
                        _pl_refiner_diag = _pl_result.diagnostics or {}
                # -----------------------------------------------------

                proto_bank.snapshot()
                proto_bank.update(feat_s_de_ori, frame_lbl_s)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_text_grounding_ce = torch.tensor(0.0, device=device)
                loss_sup_s = criteria['s'](pred_s, frame_lbl_s)
                # Baseline unsup (raw): ignore by mask_t and use existing CE path.
                frame_pl_t_f = frame_pl_t.clone()
                frame_pl_t_f[~mask_t] = ignore_index

                # ---- depth prior plugin (legacy) + depth confidence (D-Conf) ----
                depth_metrics = {}
                depth_weight_map = None
                if plugins.enabled("depth_boundary_prior"):
                    dcfg = plugins.cfg.get("depth_boundary_prior", {}) if isinstance(getattr(plugins, "cfg", None), dict) else {}
                    missing_fb = str(dcfg.get("missing_fallback", "skip")).lower()
                    db_patch = None
                    if depth_boundary_patches is not None and depth_boundary_patches.shape[1] == frame_cnt:
                        # depth_boundary_patches: (B,frame_cnt,H,W) in {0,1}
                        db_patch = depth_boundary_patches[:, frame_idx].to(dtype=torch.uint8)
                    out_db = build_depth_boundary_weight(
                        pseudo_label=frame_pl_t,
                        depth_boundary=db_patch,
                        ignore_index=ignore_index,
                        weight_on_inconsistent=float(dcfg.get("weight_on_inconsistent", 0.2)),
                        semantic_boundary_dilate=int(dcfg.get("semantic_boundary_dilate", 1)),
                    )
                    if (not out_db.used) and out_db.reason == "depth_boundary_missing":
                        if missing_fb == "error":
                            raise RuntimeError("[depth_boundary_prior] depth boundary missing but missing_fallback=error")
                        if missing_fb == "ones":
                            depth_weight_map = torch.ones_like(frame_pl_t, dtype=torch.float32, device=device)
                        else:
                            depth_weight_map = None
                    else:
                        depth_weight_map = out_db.weight_map
                    if depth_weight_map is not None:
                        depth_metrics = dict(
                            depth_inconsistent_ratio=float(out_db.inconsistent_ratio),
                            depth_sem_boundary_ratio=float(out_db.semantic_boundary_ratio),
                            depth_boundary_ratio=float(out_db.depth_boundary_ratio),
                            depth_weight_mean=float(out_db.weight_mean),
                        )
                if plugins.enabled("depth_confidence"):
                    dccfg = plugins.cfg.get("depth_confidence", {}) if isinstance(getattr(plugins, "cfg", None), dict) else {}
                    depth_patch = None
                    if depth_patches is not None and depth_patches.shape[1] == frame_cnt:
                        depth_patch = depth_patches[:, frame_idx]
                    out_dc = compute_depth_confidence_weight(
                        depth=depth_patch,
                        pseudo_label=frame_pl_t,
                        cfg=dccfg,
                    )
                    if depth_weight_map is None:
                        depth_weight_map = out_dc.weight_map
                    else:
                        depth_weight_map = depth_weight_map * out_dc.weight_map.to(dtype=depth_weight_map.dtype)
                    depth_metrics.update(
                        dict(
                            depth_weight_mean=float(out_dc.weight_mean),
                            depth_weight_min=float(out_dc.weight_min),
                            depth_weight_max=float(out_dc.weight_max),
                            depth_edge_conflict_ratio=float(out_dc.edge_conflict_ratio),
                            depth_smooth_noisy_ratio=float(out_dc.smooth_noisy_ratio),
                            depth_confidence_used=float(1.0 if out_dc.used_depth else 0.0),
                        )
                    )

                if gate_out is not None and gate_out.soft_weight is not None:
                    # Combine text gate soft weight with depth prior (multiplicative).
                    if depth_weight_map is not None:
                        comb = gate_out.soft_weight * depth_weight_map.to(dtype=gate_out.soft_weight.dtype)
                    else:
                        comb = gate_out.soft_weight
                    loss_unsup_raw = weighted_unsup_loss(
                        pred_t=pred_t,
                        pseudo_t=frame_pl_t_f,
                        ignore_index=ignore_index,
                        weight_map=comb,
                    )
                else:
                    # If no text soft-weight, still apply depth weight by switching to weighted_unsup_loss.
                    if depth_weight_map is not None:
                        loss_unsup_raw = weighted_unsup_loss(
                            pred_t=pred_t,
                            pseudo_t=frame_pl_t_f,
                            ignore_index=ignore_index,
                            weight_map=depth_weight_map,
                        )
                    else:
                        loss_unsup_raw = criteria['t'](pred_t, frame_pl_t_f)

                loss_unsup = loss_unsup_raw
                pseudo_metrics = {}
                if plugins.enabled("pseudo_conf_mask"):
                    pcfg = plugins.cfg.get("pseudo_conf_mask", {}) if isinstance(getattr(plugins, "cfg", None), dict) else {}
                    out = masked_unsup_ce(
                        pred_logits=pred_t,
                        pseudo_label=frame_pl_t,
                        conf_map=conf_t,
                        base_mask=mask_t,
                        threshold=float(pcfg.get("threshold", conf_thr)),
                        ignore_index=int(pcfg.get("ignore_index", ignore_index)),
                        min_valid_ratio=float(pcfg.get("min_valid_ratio", 0.0)),
                        min_valid_pixels=int(pcfg.get("min_valid_pixels", 0)),
                        fallback=str(pcfg.get("fallback", "skip")),
                        weight_map=(
                            (gate_out.soft_weight * depth_weight_map.to(dtype=gate_out.soft_weight.dtype))
                            if (gate_out is not None and gate_out.soft_weight is not None and depth_weight_map is not None)
                            else (gate_out.soft_weight if (gate_out is not None and gate_out.soft_weight is not None) else depth_weight_map)
                        ),
                    )
                    loss_unsup = out.loss_masked
                    pseudo_metrics = dict(
                        loss_unsup_raw=float(out.loss_raw.item()),
                        loss_unsup_masked=float(out.loss_masked.item()),
                        pseudo_valid_ratio=float(out.valid_ratio),
                        pseudo_mean_confidence=float(out.mean_confidence),
                        pseudo_filtered_ratio=float(out.filtered_ratio),
                        pseudo_valid_pixels=int(out.valid_pixels),
                        pseudo_total_pixels=int(out.total_pixels),
                    )
                    if out.fallback_used is not None:
                        pseudo_metrics["pseudo_conf_mask_fallback"] = str(out.fallback_used)
                        if device.type == "cuda" and int(global_step) % 50 == 0:
                            print(
                                f"[warn][pseudo_conf_mask] fallback={out.fallback_used} "
                                f"valid_ratio={out.valid_ratio:.6f} valid_pixels={out.valid_pixels}",
                                flush=True,
                            )
                else:
                    pseudo_metrics = dict(
                        loss_unsup_raw=float(loss_unsup_raw.item()),
                        loss_unsup_masked=float(loss_unsup_raw.item()),
                        pseudo_valid_ratio=float(mask_t.float().mean().item()),
                        pseudo_mean_confidence=float(conf_t[mask_t].mean().item()) if mask_t.any() else 0.0,
                        pseudo_filtered_ratio=float(1.0 - mask_t.float().mean().item()),
                    )
                text_weight_mean = 1.0
                text_gate_accept_ratio = float(mask_t.float().mean().item())
                text_gate_reject_ratio = float(1.0 - text_gate_accept_ratio)
                if gate_out is not None and gate_out.soft_weight is not None:
                    text_weight_mean = float(gate_out.soft_weight[valid_mask].mean().item()) if valid_mask.any() else 0.0
                if depth_weight_map is not None:
                    valid_mask_f = (frame_pl_t != ignore_index).float()
                    conf_mask_f = mask_t.float()
                    final_weight = valid_mask_f * conf_mask_f * depth_weight_map.float()
                    pseudo_metrics["final_valid_ratio"] = float((final_weight > 0).float().mean().item())
                else:
                    pseudo_metrics["final_valid_ratio"] = float(mask_t.float().mean().item())
                pseudo_metrics["text_gate_accept_ratio"] = float(text_gate_accept_ratio)
                pseudo_metrics["text_gate_reject_ratio"] = float(text_gate_reject_ratio)
                pseudo_metrics["text_weight_mean"] = float(text_weight_mean)
                if gate_out is not None and gate_out.accept_ratio_per_class is not None:
                    for cls_idx, acc in enumerate(gate_out.accept_ratio_per_class.detach().cpu().tolist()):
                        pseudo_metrics[f"text_gate_accept_cls_{cls_idx}"] = float(acc)
                    for cls_idx, rej in enumerate(gate_out.reject_ratio_per_class.detach().cpu().tolist()):
                        pseudo_metrics[f"text_gate_reject_cls_{cls_idx}"] = float(rej)
                visual_logits_std = (
                    float(aux_t.l_vis.std(unbiased=False).item())
                    if need_txt_aux
                    else float(pred_t.std(unbiased=False).item())
                )
                final_logits_std = float(pred_t.std(unbiased=False).item())
                pred_change_ratio = 0.0
                pred_change_ratio_by_class = {}
                if need_txt_aux:
                    visual_pred = torch.argmax(aux_t.l_vis, dim=1)
                    final_pred = torch.argmax(pred_t, dim=1)
                    change_mask = (visual_pred != final_pred)
                    eval_mask = valid_mask
                    if eval_mask.any():
                        pred_change_ratio = float(change_mask[eval_mask].float().mean().item())
                    else:
                        pred_change_ratio = 0.0
                    # Per-class change ratio conditioned on visual prediction class.
                    # Focus classes are mapped in Cityscapes-13 label space:
                    # sidewalk=1, wall=3, fence=4, pole=5, traffic light=6, traffic sign=7, terrain=9
                    focus_cls = (1, 3, 4, 5, 6, 7, 9)
                    for cls_idx in focus_cls:
                        cls_mask = (visual_pred == int(cls_idx)) & eval_mask
                        if cls_mask.any():
                            pred_change_ratio_by_class[cls_idx] = float(change_mask[cls_mask].float().mean().item())
                        else:
                            pred_change_ratio_by_class[cls_idx] = 0.0

                loss_proto  = proto_bank.mse_pull_loss(feat_t_de, frame_pl_t_f)
                loss_txt_align = torch.tensor(0.0, device=device)
                if use_txt and w_txt_align > 0:
                    # source: full supervision area; target: only high-confidence pixels (mask_t)
                    loss_txt_align_s = kl_align_loss(aux_s.l_vis, aux_s.l_txt, mask=None, tau=txt_align_tau)
                    loss_txt_align_t = kl_align_loss(aux_t.l_vis, aux_t.l_txt, mask=mask_t, tau=txt_align_tau)
                    loss_txt_align = 0.5 * (loss_txt_align_s + loss_txt_align_t)
                if need_txt_aux and loss_text_grounding_enabled:
                    losses = []
                    if loss_text_grounding_apply_source and getattr(aux_s, "l_ground", None) is not None:
                        losses.append(criteria["s"](aux_s.l_ground, frame_lbl_s))
                    if loss_text_grounding_apply_target and getattr(aux_t, "l_ground", None) is not None:
                        tgt_lbl = frame_pl_t.clone()
                        tgt_lbl[~mask_t] = ignore_index
                        losses.append(criteria["t"](aux_t.l_ground, tgt_lbl))
                    if losses:
                        loss_text_grounding_ce = sum(losses) / max(1, len(losses))

            beta_scheduled  = linear_ramp(global_step, 0, 0.1*total_steps*frame_cnt, 1.0)
            alpha_scheduled = linear_ramp(global_step, 0.1*total_steps*frame_cnt, 0.2*total_steps*frame_cnt, alpha)
            txt_align_scheduled = linear_ramp(global_step, 0, 0.2*total_steps*frame_cnt, w_txt_align)

            feat_t_de.retain_grad()
            loss_frame = (
                loss_sup_s
                + (lambda_ssl * beta_scheduled) * loss_unsup
                + (lambda_fpa * alpha_scheduled) * loss_proto
                + txt_align_scheduled * loss_txt_align
                + (loss_text_grounding_weight * loss_text_grounding_ce)
            )

            if scaler is not None and device.type == "cuda":
                scaler.scale(loss_frame).backward()
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
            else:
                loss_frame.backward()

            text_scale_grad_norm = 0.0
            fusion_alpha_grad_norm = 0.0
            text_adapter_grad_norm = 0.0
            cross_attn_grad_norm = 0.0
            cross_beta_grad_norm = 0.0
            grounding_gamma_grad_norm = 0.0
            grounding_logit_scale_grad_norm = 0.0
            query_self_attn_grad_norm = 0.0
            query_cross_attn_grad_norm = 0.0
            query_ffn_grad_norm = 0.0
            semantic_decoder_grad_norm = 0.0
            if need_txt_aux:
                txt_head_obj = getattr(model, "txt_head", None)
                txt_fusion_obj = getattr(model, "text_fusion", None)
                txt_cross_obj = getattr(model, "text_cross_fusion", None)
                txt_ground_obj = getattr(model, "text_grounding_fusion", None)
                decoder_obj = getattr(model, "decoder", None)
                if txt_head_obj is not None:
                    if hasattr(txt_head_obj, "text_scale"):
                        text_scale_grad_norm = self._param_grad_norm([txt_head_obj.text_scale])
                    if hasattr(txt_head_obj, "projector"):
                        text_adapter_grad_norm = self._param_grad_norm(list(txt_head_obj.projector.parameters()))
                if txt_fusion_obj is not None and hasattr(txt_fusion_obj, "alpha"):
                    fusion_alpha_grad_norm = self._param_grad_norm([txt_fusion_obj.alpha])
                if txt_cross_obj is not None:
                    if hasattr(txt_cross_obj, "beta"):
                        cross_beta_grad_norm = self._param_grad_norm([txt_cross_obj.beta])
                    cross_params = [p for name, p in txt_cross_obj.named_parameters() if name != "beta"]
                    cross_attn_grad_norm = self._param_grad_norm(cross_params)
                if txt_ground_obj is not None:
                    if hasattr(txt_ground_obj, "gamma"):
                        grounding_gamma_grad_norm = self._param_grad_norm([txt_ground_obj.gamma])
                    if hasattr(txt_ground_obj, "logit_scale"):
                        grounding_logit_scale_grad_norm = self._param_grad_norm([txt_ground_obj.logit_scale])
                    self_params = []
                    cross_params = []
                    ffn_params = []
                    for name, p in txt_ground_obj.named_parameters():
                        if "self_attn" in name:
                            self_params.append(p)
                        elif "cross_attn" in name:
                            cross_params.append(p)
                        elif "ffn" in name:
                            ffn_params.append(p)
                    query_self_attn_grad_norm = self._param_grad_norm(self_params)
                    query_cross_attn_grad_norm = self._param_grad_norm(cross_params)
                    query_ffn_grad_norm = self._param_grad_norm(ffn_params)
                if decoder_obj is not None:
                    semantic_decoder_grad_norm = self._param_grad_norm(list(decoder_obj.parameters()))

            if scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            with torch.no_grad():
                grad_proto = (feat_t_de.grad.norm().item()
                            if feat_t_de.grad is not None else 0.0)
                ratio_proto = (alpha_scheduled * loss_proto.item()) / (loss_frame.item() + 1e-8)
                proto_delta = proto_bank.delta_mean()
                cov_t = mask_t.float().mean().item()
                conf_mean = conf_t[mask_t].mean().item() if mask_t.any() else 0.0
                valid_ratio = valid_mask.float().mean().item()
                unsup_kept_ratio = float(cov_t / max(valid_ratio, 1e-8))

            extra_metrics = dict(
                loss_sup_s=loss_sup_s.item(),
                loss_unsup=loss_unsup.item(),
                loss_proto=loss_proto.item(),
                loss_sup=loss_sup_s.item(),
                loss_ssl=loss_unsup.item(),
                loss_fpa=loss_proto.item(),
                total_loss=loss_frame.item(),
                loss_txt_align=float(loss_txt_align.item()) if torch.is_tensor(loss_txt_align) else float(loss_txt_align),
                beta=beta_scheduled,
                alpha=alpha_scheduled,
                txt_align_w=txt_align_scheduled,
                ratio_proto=ratio_proto,
                grad_proto=grad_proto,
                proto_delta=proto_delta,
                cov_t=cov_t,
                conf_mean=conf_mean,
                visual_mask_ratio=float(visual_mask.float().mean().item()),
                text_mask_ratio=float(gate_out.text_mask_ratio if gate_out is not None else cov_t),
                text_sim_mean=float(gate_out.text_sim_mean if gate_out is not None else 0.0),
                text_sim_std=float(gate_out.text_sim_std if gate_out is not None else 0.0),
                text_logits_mean=float(aux_t.l_txt.mean().item()) if need_txt_aux else 0.0,
                text_logits_std=float(aux_t.l_txt.std(unbiased=False).item()) if need_txt_aux else 0.0,
                visual_logits_mean=float(aux_t.l_vis.mean().item()) if need_txt_aux else float(pred_t.mean().item()),
                visual_logits_std=visual_logits_std,
                final_logits_mean=float(pred_t.mean().item()),
                final_logits_std=final_logits_std,
                logits_delta_mean=float((pred_t - aux_t.l_vis).mean().item()) if need_txt_aux else 0.0,
                logits_delta_norm=float((pred_t - aux_t.l_vis).norm().item()) if need_txt_aux else 0.0,
                text_grounding_logits_mean=float(aux_t.l_ground.mean().item()) if (need_txt_aux and getattr(aux_t, "l_ground", None) is not None) else 0.0,
                text_grounding_logits_std=float(aux_t.l_ground.std(unbiased=False).item()) if (need_txt_aux and getattr(aux_t, "l_ground", None) is not None) else 0.0,
                text_only_prediction_change_ratio=float(
                    ((torch.argmax(aux_t.l_vis, dim=1) != torch.argmax(aux_t.l_ground, dim=1))[valid_mask].float().mean().item())
                ) if (need_txt_aux and getattr(aux_t, "l_ground", None) is not None and valid_mask.any()) else 0.0,
                text_scale=float(aux_t.text_scale) if need_txt_aux else 0.0,
                fusion_alpha=float(aux_t.fusion_alpha) if need_txt_aux else 0.0,
                prediction_change_ratio=float(pred_change_ratio) if need_txt_aux else 0.0,
                text_scale_grad_norm=float(text_scale_grad_norm) if need_txt_aux else 0.0,
                fusion_alpha_grad_norm=float(fusion_alpha_grad_norm) if need_txt_aux else 0.0,
                text_adapter_grad_norm=float(text_adapter_grad_norm) if need_txt_aux else 0.0,
                cross_attn_grad_norm=float(cross_attn_grad_norm) if need_txt_aux else 0.0,
                cross_beta_grad_norm=float(cross_beta_grad_norm) if need_txt_aux else 0.0,
                grounding_gamma=float(aux_t.grounding_gamma) if need_txt_aux else 0.0,
                grounding_logit_scale=float(aux_t.grounding_logit_scale) if need_txt_aux else 0.0,
                grounding_gamma_grad_norm=float(grounding_gamma_grad_norm) if need_txt_aux else 0.0,
                grounding_logit_scale_grad_norm=float(grounding_logit_scale_grad_norm) if need_txt_aux else 0.0,
                query_self_attn_grad_norm=float(query_self_attn_grad_norm) if need_txt_aux else 0.0,
                query_cross_attn_grad_norm=float(query_cross_attn_grad_norm) if need_txt_aux else 0.0,
                query_ffn_grad_norm=float(query_ffn_grad_norm) if need_txt_aux else 0.0,
                cross_beta=float(aux_t.cross_beta) if need_txt_aux else 0.0,
                cross_attn_mean=float(aux_t.cross_attn_mean) if need_txt_aux else 0.0,
                cross_attn_std=float(aux_t.cross_attn_std) if need_txt_aux else 0.0,
                cross_attn_entropy=float(aux_t.cross_attn_entropy) if need_txt_aux else 0.0,
                cross_attn_max_mean=float(aux_t.cross_attn_max_mean) if need_txt_aux else 0.0,
                text_to_image_attn_mean=float(aux_t.grounding_attn_mean) if need_txt_aux else 0.0,
                text_to_image_attn_std=float(aux_t.grounding_attn_std) if need_txt_aux else 0.0,
                text_to_image_attn_entropy=float(aux_t.grounding_attn_entropy) if need_txt_aux else 0.0,
                text_to_image_attn_max=float(aux_t.grounding_attn_max_mean) if need_txt_aux else 0.0,
                text_logits_delta_norm=float(
                    (aux_t.l_ground - aux_t.l_txt).norm().item()
                ) if (need_txt_aux and getattr(aux_t, "l_ground", None) is not None and aux_t.l_txt is not None) else 0.0,
                final_visual_logits_delta_norm=float((pred_t - aux_t.l_vis).norm().item()) if need_txt_aux else 0.0,
                loss_text_grounding_ce=float(loss_text_grounding_ce.item()) if torch.is_tensor(loss_text_grounding_ce) else float(loss_text_grounding_ce),
                visual_feature_mean=float(aux_t.visual_feature_mean) if need_txt_aux else 0.0,
                visual_feature_std=float(aux_t.visual_feature_std) if need_txt_aux else 0.0,
                fused_feature_mean=float(aux_t.fused_feature_mean) if need_txt_aux else 0.0,
                fused_feature_std=float(aux_t.fused_feature_std) if need_txt_aux else 0.0,
                feature_delta_mean=float(aux_t.feature_delta_mean) if need_txt_aux else 0.0,
                feature_delta_norm=float(aux_t.feature_delta_norm) if need_txt_aux else 0.0,
                feature_delta_ratio=float(aux_t.feature_delta_ratio) if need_txt_aux else 0.0,
                semantic_decoder_grad_norm=float(semantic_decoder_grad_norm),
                unsup_kept_ratio=unsup_kept_ratio,
                effective_memory_length=float(getattr(model.memory_bank, "last_effective_memory_length", 0.0)),
                memory_tokens_before=float(getattr(model.memory_bank, "last_memory_tokens_before", 0.0)),
                memory_tokens_after=float(getattr(model.memory_bank, "last_memory_tokens_after", 0.0)),
                memory_token_ratio=float(getattr(model.memory_bank, "last_memory_token_ratio", 1.0)),
                memory_compress_mode=str(getattr(model.memory_bank, "memory_compress_mode", "none")),
                memory_compress_location=str(getattr(model.memory_bank, "memory_compress_location", "before_attention")),
                memory_attention_time=float(getattr(model.memory_bank, "last_attention_time", 0.0)),
                memory_attention_time_ms=float(getattr(model.memory_bank, "last_attention_time", 0.0) * 1000.0),
                memory_attention_time_avg=float(
                    getattr(model.memory_bank, "attention_time_accum", 0.0)
                    / max(1, int(getattr(model.memory_bank, "attention_time_count", 0)))
                ),
                sparse_tokens_before=float(getattr(model.memory_bank, "last_sparse_tokens_before", 0.0)),
                sparse_tokens_after=float(getattr(model.memory_bank, "last_sparse_tokens_after", 0.0)),
                sparse_keep_ratio=float(getattr(model.memory_bank, "last_sparse_keep_ratio", 1.0)),
                depth_prior_enabled=float(1.0 if bool(getattr(args, "depth_prior", {}).get("enable", False)) else 0.0),
                depth_confidence_enabled=float(1.0 if plugins.enabled("depth_confidence") else 0.0),
                depth_adapter_enabled=float(1.0 if bool(getattr(args, "depth_adapter", {}).get("enable", False)) else 0.0),
            )
            depth_adapter_obj = getattr(model, "depth_adapter", None)
            if depth_adapter_obj is None and hasattr(model, "module"):
                depth_adapter_obj = getattr(model.module, "depth_adapter", None)
            if depth_adapter_obj is not None and hasattr(depth_adapter_obj, "alpha"):
                extra_metrics["depth_adapter_alpha"] = float(depth_adapter_obj.alpha.detach().item())
            else:
                extra_metrics["depth_adapter_alpha"] = 0.0
            extra_metrics.update(pseudo_metrics)
            extra_metrics.update(depth_metrics)
            if geo_enabled and hasattr(model, "memory_bank"):
                _memory_diag = model.memory_bank.get_last_geo_memory_diag()
                geo_records.append(self._geo_diag_record(
                    args=args,
                    global_step=global_step,
                    context=geo_ctx_t or {},
                    memory_diag=_memory_diag,
                ))
                if geo_token_writer_enabled:
                    _summary_rec, _token_recs = build_geo_token_records(
                        args=args,
                        global_step=global_step,
                        context=geo_ctx_t or {},
                        memory_diag=_memory_diag,
                        geo_cfg=geo_cfg,
                    )
                    if _summary_rec:
                        geo_token_summary_records.append(_summary_rec)
                    if _token_recs:
                        geo_token_sample_records.extend(_token_recs)
                extra_metrics["geo_memory_diag_enabled"] = 1.0
                extra_metrics["geo_true_erp_x_range_reliable"] = float(bool((geo_ctx_t or {}).get("true_erp_x_range_reliable", False)))
                extra_metrics["geo_random_reverse_possible"] = float(bool((geo_ctx_t or {}).get("random_reverse_possible", False)))
                extra_metrics["geo_seam_selection_enabled"] = 0.0
            # ---- PseudoLabelRefiner sanity logging (L3 diagnosis, no loss change) ----
            if _pl_refiner_diag:
                _rmode = str(_pl_refiner_diag.get("refiner_mode", "unknown"))
                extra_metrics["refiner_mode"]          = _rmode
                extra_metrics["refiner_total_valid_ratio"]   = float(_pl_refiner_diag.get("total_valid_ratio", 0.0))
                extra_metrics["refiner_total_ignore_ratio"]  = float(_pl_refiner_diag.get("total_ignore_ratio", 1.0))
                extra_metrics["refiner_common_valid_ratio"]  = float(_pl_refiner_diag.get("common_class_valid_ratio", 0.0))
                extra_metrics["refiner_rare_valid_ratio"]    = float(_pl_refiner_diag.get("rare_class_valid_ratio", 0.0))
                extra_metrics["refiner_ext_rare_valid_ratio"]= float(_pl_refiner_diag.get("extreme_rare_valid_ratio", 0.0))
                for _k in (
                    "r2_total_valid_ratio", "r2_common_valid_ratio", "r2_rare_valid_ratio", "r2_extreme_rare_valid_ratio",
                    "proto_gate_filter_ratio_total", "proto_gate_keep_ratio_total",
                    "proto_gate_filter_ratio_common", "proto_gate_filter_ratio_rare", "proto_gate_filter_ratio_extreme_rare",
                    "proto_gate_final_valid_ratio", "proto_gate_final_common_valid_ratio",
                    "proto_gate_final_rare_valid_ratio", "proto_gate_final_extreme_rare_valid_ratio",
                ):
                    if _k in _pl_refiner_diag:
                        _v = _pl_refiner_diag.get(_k)
                        extra_metrics[_k] = None if _v is None else float(_v)
                for _k in (
                    "proto_gate_filtered_pixels_by_class", "proto_gate_kept_pixels_by_class",
                    "proto_gate_total_pixels_by_class", "proto_gate_threshold_by_class",
                    "proto_gate_fallback_classes", "proto_gate_fallback_reasons",
                ):
                    if _k in _pl_refiner_diag:
                        extra_metrics[_k] = _pl_refiner_diag.get(_k)
                # ---- PLQ-specific diagnostics (mode=plq_basic only) ----
                if str(_pl_refiner_diag.get("mode", "")) == "plq_basic":
                    _print_refiner = bool(getattr(args, "refiner_verbose_log", True))
                    extra_metrics["plq_mean"]                 = float(_pl_refiner_diag.get("plq_mean", float("nan")))
                    extra_metrics["plq_min"]                  = float(_pl_refiner_diag.get("plq_min",  float("nan")))
                    extra_metrics["plq_max"]                  = float(_pl_refiner_diag.get("plq_max",  float("nan")))
                    extra_metrics["plq_std"]                  = float(_pl_refiner_diag.get("plq_std",  float("nan")))
                    extra_metrics["confidence_mean"]          = float(_pl_refiner_diag.get("confidence_mean",           float("nan")))
                    extra_metrics["entropy_reliability_mean"] = float(_pl_refiner_diag.get("entropy_reliability_mean",  float("nan")))
                    extra_metrics["local_agreement_mean"]     = float(_pl_refiner_diag.get("local_agreement_mean",      float("nan")))
                    extra_metrics["plq_nan_count"]            = int(_pl_refiner_diag.get("plq_nan_count", 0))
                    extra_metrics["plq_zero_ratio"]           = float(_pl_refiner_diag.get("plq_zero_ratio", float("nan")))
                    extra_metrics["plq_common_mean"]          = float(_pl_refiner_diag.get("plq_common_mean",       float("nan")))
                    extra_metrics["plq_rare_mean"]            = float(_pl_refiner_diag.get("plq_rare_mean",         float("nan")))
                    extra_metrics["plq_extreme_rare_mean"]    = float(_pl_refiner_diag.get("plq_extreme_rare_mean", float("nan")))
                    for _c, _v in (_pl_refiner_diag.get("plq_by_class") or {}).items():
                        extra_metrics[f"plq_cls{_c}_mean"] = float(_v)
                    for _b, _cnt in (_pl_refiner_diag.get("plq_histogram") or {}).items():
                        extra_metrics[f"plq_hist_{_b}"] = int(_cnt)
                    for _b, _r in (_pl_refiner_diag.get("valid_ratio_by_plq_bin") or {}).items():
                        extra_metrics[f"plq_vr_bin_{_b}"] = float(_r)
                    if _print_refiner:
                        print(
                            f"[plq_diag] step={int(global_step)-1} "
                            f"plq={extra_metrics['plq_mean']:.4f}"
                            f"({extra_metrics['plq_min']:.3f}-{extra_metrics['plq_max']:.3f}) "
                            f"std={extra_metrics['plq_std']:.4f} "
                            f"conf={extra_metrics['confidence_mean']:.4f} "
                            f"ent_rel={extra_metrics['entropy_reliability_mean']:.4f} "
                            f"local_agr={extra_metrics['local_agreement_mean']:.4f} "
                            f"nan={extra_metrics['plq_nan_count']} "
                            f"zero_r={extra_metrics['plq_zero_ratio']:.4f}",
                            flush=True,
                        )
                # ---- end PLQ-specific diagnostics ----
                if _pl_refiner_diag.get("proto_diag_enabled", False):
                    extra_metrics["proto_sim_mean"] = float(_pl_refiner_diag.get("proto_sim_mean", float("nan")))
                    extra_metrics["proto_sim_std"] = float(_pl_refiner_diag.get("proto_sim_std", float("nan")))
                    extra_metrics["proto_sim_min"] = float(_pl_refiner_diag.get("proto_sim_min", float("nan")))
                    extra_metrics["proto_sim_max"] = float(_pl_refiner_diag.get("proto_sim_max", float("nan")))
                    extra_metrics["proto_sim_valid_ratio"] = float(_pl_refiner_diag.get("proto_sim_valid_ratio", 0.0))
                    extra_metrics["low_sim_ratio_at_0.2"] = float(_pl_refiner_diag.get("low_sim_ratio_at_0.2", 0.0))
                    extra_metrics["low_sim_ratio_at_0.4"] = float(_pl_refiner_diag.get("low_sim_ratio_at_0.4", 0.0))
                    extra_metrics["low_sim_ratio_at_0.6"] = float(_pl_refiner_diag.get("low_sim_ratio_at_0.6", 0.0))
                    extra_metrics["proto_sim_nan_count"] = int(_pl_refiner_diag.get("proto_sim_nan_count", 0))
                    _proto_valid_classes = _pl_refiner_diag.get("proto_valid_classes", []) or []
                    _proto_missing_classes = _pl_refiner_diag.get("proto_missing_classes", []) or []
                    extra_metrics["proto_valid_classes"] = list(_proto_valid_classes)
                    extra_metrics["proto_missing_classes"] = list(_proto_missing_classes)
                    extra_metrics["proto_counts_by_class"] = dict(_pl_refiner_diag.get("proto_counts_by_class", {}) or {})
                    extra_metrics["proto_sim_mean_by_class"] = dict(_pl_refiner_diag.get("proto_sim_mean_by_class", {}) or {})
                    extra_metrics["proto_valid_class_count"] = int(len(_proto_valid_classes))
                    extra_metrics["proto_missing_class_count"] = int(len(_proto_missing_classes))
                    for _k in (
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
                    ):
                        if _k in _pl_refiner_diag:
                            _v = _pl_refiner_diag.get(_k)
                            extra_metrics[_k] = None if _v is None else float(_v)
                    for _k in (
                        "low_sim_ratio_at_0.75_by_class", "low_sim_ratio_at_0.80_by_class",
                        "low_sim_ratio_at_0.85_by_class", "proto_sim_q10_by_class",
                        "proto_sim_q25_by_class", "proto_sim_q50_by_class",
                    ):
                        if _k in _pl_refiner_diag:
                            extra_metrics[_k] = dict(_pl_refiner_diag.get(_k, {}) or {})
                    for _k, _v in (_pl_refiner_diag or {}).items():
                        if str(_k).startswith("simulated_abs_") or str(_k).startswith("simulated_global_") or str(_k).startswith("simulated_classwise_percentile_"):
                            extra_metrics[_k] = None if _v is None else float(_v)
                    if bool(getattr(args, "refiner_verbose_log", True)):
                        print(
                            f"[proto_sim_diag] step={int(global_step)-1} "
                            f"mean={extra_metrics['proto_sim_mean']:.4f} "
                            f"std={extra_metrics['proto_sim_std']:.4f} "
                            f"range=({extra_metrics['proto_sim_min']:.4f},{extra_metrics['proto_sim_max']:.4f}) "
                            f"valid_ratio={extra_metrics['proto_sim_valid_ratio']:.4f} "
                            f"valid_cls={_proto_valid_classes} missing_cls={_proto_missing_classes} "
                            f"low@0.2={extra_metrics['low_sim_ratio_at_0.2']:.4f} "
                            f"low@0.4={extra_metrics['low_sim_ratio_at_0.4']:.4f} "
                            f"low@0.6={extra_metrics['low_sim_ratio_at_0.6']:.4f} "
                            f"nan={extra_metrics['proto_sim_nan_count']}",
                            flush=True,
                        )
                # Per-class valid ratios (flattened, safe cast)
                for c, vr in (_pl_refiner_diag.get("class_valid_ratio") or {}).items():
                    extra_metrics[f"refiner_cls{c}_valid_ratio"] = float(vr)
                for c, sc in (_pl_refiner_diag.get("selected_pixel_count_by_class") or {}).items():
                    extra_metrics[f"refiner_cls{c}_selected"] = int(sc)
                for c, cc in (_pl_refiner_diag.get("candidate_pixel_count_by_class") or {}).items():
                    extra_metrics[f"refiner_cls{c}_candidate"] = int(cc)
                # gpu_mem_peak — captured here, after cuda ops
                if device.type == "cuda":
                    extra_metrics["gpu_mem_peak_mb"] = float(
                        torch.cuda.max_memory_allocated(device) / 1024 / 1024
                    )
                # Per-step print for log scanning (every iter during short-run)
                _print_refiner = bool(getattr(args, "refiner_verbose_log", True))
                if _print_refiner:
                    _cls_vr  = _pl_refiner_diag.get("class_valid_ratio", {})
                    _skipped = _pl_refiner_diag.get("skipped_class_by_reason", {})
                    print(
                        f"[refiner_diag] step={int(global_step)-1} mode={_rmode} "
                        f"total_vr={extra_metrics['refiner_total_valid_ratio']:.4f} "
                        f"common_vr={extra_metrics['refiner_common_valid_ratio']:.4f} "
                        f"rare_vr={extra_metrics['refiner_rare_valid_ratio']:.4f} "
                        f"ext_rare_vr={extra_metrics['refiner_ext_rare_valid_ratio']:.4f} "
                        f"loss_sup={float(extra_metrics.get('loss_sup_s', float('nan'))):.4f} "
                        f"loss_unsup={float(extra_metrics.get('loss_unsup', float('nan'))):.4f} "
                        f"pseudo_vr={float(extra_metrics.get('cov_t', float('nan'))):.4f} "
                        f"ignore_r={float(extra_metrics.get('refiner_total_ignore_ratio', float('nan'))):.4f}",
                        flush=True,
                    )
                    # Compact per-class summary (just ratios)
                    _cls_str = " ".join(
                        f"c{c}={float(_cls_vr.get(c, 0)):.2f}"
                        for c in range(min(13, 1 + max((int(k) for k in _cls_vr), default=-1)))
                    )
                    if _cls_str:
                        print(f"[refiner_diag]   per_cls_vr: {_cls_str}", flush=True)
                    if _skipped:
                        print(f"[refiner_diag]   skipped: {_skipped}", flush=True)
            # ---- end refiner sanity logging ----------------------------------
            if need_txt_aux:
                for cls_idx, cls_ratio in pred_change_ratio_by_class.items():
                    extra_metrics[f"prediction_change_ratio_cls_{int(cls_idx)}"] = float(cls_ratio)
            self.geo_diag_records = list(geo_records)
            self.geo_token_summary_records = list(geo_token_summary_records)
            self.geo_token_sample_records = list(geo_token_sample_records)
            self.last_iter_metrics = dict(extra_metrics)
            log_interval = int(getattr(args, "text_log_interval", 50))
            if need_txt_aux and log_interval > 0 and (int(global_step) % log_interval == 0):
                print(
                    "[text-logits] "
                    f"iter={int(global_step)} "
                    f"visual_mean={extra_metrics['visual_logits_mean']:.6f} "
                    f"visual_std={extra_metrics['visual_logits_std']:.6f} "
                    f"text_mean={extra_metrics['text_logits_mean']:.6f} "
                    f"text_std={extra_metrics['text_logits_std']:.6f} "
                    f"final_mean={extra_metrics['final_logits_mean']:.6f} "
                    f"final_std={extra_metrics['final_logits_std']:.6f} "
                    f"alpha={extra_metrics['fusion_alpha']:.6f} "
                    f"text_scale={extra_metrics['text_scale']:.6f} "
                    f"pred_change={extra_metrics['prediction_change_ratio']:.6f}",
                    flush=True,
                )
                print(
                    "[text-grad] "
                    f"iter={int(global_step)} "
                    f"alpha_grad={extra_metrics['fusion_alpha_grad_norm']:.6e} "
                    f"cross_beta_grad={extra_metrics['cross_beta_grad_norm']:.6e} "
                    f"cross_attn_grad={extra_metrics['cross_attn_grad_norm']:.6e} "
                    f"text_scale_grad={extra_metrics['text_scale_grad_norm']:.6e} "
                    f"text_adapter_grad={extra_metrics['text_adapter_grad_norm']:.6e} "
                    f"semantic_decoder_grad={extra_metrics['semantic_decoder_grad_norm']:.6e}",
                    flush=True,
                )
                print(
                    "[text-grounding] "
                    f"iter={int(global_step)} "
                    f"gamma={extra_metrics['grounding_gamma']:.6f} "
                    f"logit_scale={extra_metrics['grounding_logit_scale']:.6f} "
                    f"attn_entropy={extra_metrics['text_to_image_attn_entropy']:.6f} "
                    f"ground_mean={extra_metrics['text_grounding_logits_mean']:.6f} "
                    f"ground_std={extra_metrics['text_grounding_logits_std']:.6f} "
                    f"text_only_change={extra_metrics['text_only_prediction_change_ratio']:.6f} "
                    f"loss_ground={extra_metrics['loss_text_grounding_ce']:.6f}",
                    flush=True,
                )
                print(
                    "[text-grounding-grad] "
                    f"iter={int(global_step)} "
                    f"gamma_grad={extra_metrics['grounding_gamma_grad_norm']:.6e} "
                    f"logit_scale_grad={extra_metrics['grounding_logit_scale_grad_norm']:.6e} "
                    f"self_attn_grad={extra_metrics['query_self_attn_grad_norm']:.6e} "
                    f"cross_attn_grad={extra_metrics['query_cross_attn_grad_norm']:.6e} "
                    f"ffn_grad={extra_metrics['query_ffn_grad_norm']:.6e}",
                    flush=True,
                )
                self._save_grounding_debug_vis(args, global_step, frame_idx, frame_img_t, aux_t, pred_t, valid_mask)

            for cb in callbacks:
                if hasattr(cb, 'on_iter_end'):
                    cb.on_iter_end(global_step=global_step,
                                loss=loss_frame.item(),
                                lr=optimizer.param_groups[0]['lr'],
                                metrics=extra_metrics)

            global_step += 1

        # reset mem
        self.reset_memory(model)
        self.reset_memory(model_src)

        return global_step


    def forward_eval(self, model, images, frame_idx, historical_pseudo_labels=None):
        return model(
            frame_idx,
            images[:, frame_idx],
            mem_type='source',
            historical_pseudo_labels=historical_pseudo_labels,
        )[0]

    def reset_memory(self, model):
        model.memory_bank._init_output_dict_source()
        model.memory_bank._init_output_dict_target()


class NoMemStrategy(WithMemStrategy):

    def forward(self,
                global_step,
                model,
                model_src,
                batch_s,
                batch_t,
                criteria,
                proto_bank,
                alpha,
                device,
                callbacks,
                is_warmup
                ):
        images_t, labels_t, *_ = batch_t
        images_t, labels_t = images_t.to(device), labels_t.to(device)
        frame_cnt = images_t.shapep[1]

        for frame_idx in range(frame_cnt):
            images_s, labels_s, *_ = batch_s
            images_s, labels_s = images_s.to(device), labels_s.to(device)

            frame_img_s = images_s
            frame_lbl_s = labels_s
            frame_img_t = images_t[:, frame_idx]
            frame_pl_t = labels_t[:, frame_idx]

            pred_s, _, feat_s_de = model(0, frame_img_s, mem_type='source')
            _, feat_s_ori, feat_s_de_ori = model_src(0, frame_img_s, mem_type='source')
            pred_t, _, feat_t_de = model(0, frame_img_t, mem_type='target')

            s_pt = proto_bank.calculate_batch_prototypes(feat_s_de_ori, frame_lbl_s)
            proto_bank.update_global_prototypes(global_step+1, frame_idx, s_pt)
            t_pt = proto_bank.calculate_batch_prototypes(feat_t_de, frame_pl_t)
            loss_pt = proto_bank.prototype_loss(frame_idx, t_pt)

            loss_unsup = criteria['t'](pred_t, frame_pl_t)
            loss_sup_s = criteria['s'](pred_s, frame_lbl_s)

            loss_frame = loss_sup_s + loss_unsup + alpha * loss_pt

            for cb in callbacks:
                if hasattr(cb, 'on_iter_end'):
                    cb.on_iter_end(global_step=global_step,
                                   loss=loss_frame.item(), lr=optimizer.param_groups[0]['lr'])

            loss_frame.backward()
            optimizer.step()

        self.reset_memory(model)
        self.reset_memory(model_src)

        return global_step
