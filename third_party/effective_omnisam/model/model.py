import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from sam.sam_lora_image_encoder import LoRA_Sam
except ModuleNotFoundError:
    from sam_lora_image_encoder import LoRA_Sam
from model.modules.decoder_sam import SegFormerHead, SegFormerHead_adaption
from model import MemoryBank
from model.text_align import (
    ImageToTextCrossAttentionFusion,
    TextAlignAux,
    TextToImageGroundingFusion,
    TextPrototypeLogitHead,
    VisualTextLogitFusion,
)
from model.text_semantic import LightweightLanguageAdapter
from model.depth_adapter import LightDepthAdapter
# NEW: Import SG-DRF modules
from model.modules.semantic_density_estimator import SemanticDensityEstimator
from model.modules.dynamic_fsmc import DynamicFSMC
from model.modules.dynamic_scm import DynamicSCM

class OmniSAM(nn.Module):
    def __init__(self, sam_model, num_classes, num_maskmem):
        super().__init__()
        self.sam_model = sam_model
        self.encoder = LoRA_Sam(sam_model, 4)
        self.decoder = SegFormerHead(num_classes=num_classes)
        # self.decoder = DMLPHead(num_classes=num_classes)
        self.num_maskmem = num_maskmem
        self.memory_bank = MemoryBank(sam_model, num_maskmem=num_maskmem)

    def forward(self, frame_idx, x, mem_type="source", historical_pseudo_labels=None):
        _, _, height, width = x.shape
        backbone_out = self.encoder(x)
        current_vision_feats = backbone_out['vision_features']
        current_vision_pos_embeds = backbone_out['vision_pos_enc'][-1]
        x = backbone_out['backbone_fpn']
        x[-1] = self.memory_bank(
                    frame_idx,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    num_obj_ptr_tokens=0,
                    mem_type=mem_type,
                    historical_pseudo_labels=historical_pseudo_labels,
                )
        x = self.decoder(x)
        pred = F.interpolate(x, size=(height,width), mode='bilinear', align_corners=False)
        # Encode new memory
        if self.num_maskmem > 0:
            output = torch.argmax(pred, 1)
            mask_for_mem = output.unsqueeze(1)
            with torch.no_grad():
                probs = torch.softmax(pred, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            self.memory_bank._encode_new_memory(frame_idx, current_vision_feats, mask_for_mem, mem_type, uncertainty_map=entropy)

        return pred

class OmniSAM_adapt(nn.Module):
    def __init__(
        self,
        sam_model,
        num_classes: int = 13,
        num_maskmem: int = 9,
        text_prototypes: torch.Tensor | None = None,
        d_txt: int | None = None,
        lambda_txt: float = 0.0,
        txt_tau: float = 0.07,
        use_language_adapter: bool = False,
        adapter_num_heads: int = 4,
        adapter_scale: float = 0.1,
        use_lite_decoder: bool = False,
        decoder_channel_ratio: float = 1.0,
        decoder_conv_mode: str = "standard",
        decoder_upsample_mode: str = "bilinear",
        text_adapter_hidden_dim: int = 512,
        text_adapter_dropout: float = 0.0,
        text_adapter_normalize_output: bool = True,
        text_adapter_zero_init_last: bool = False,
        text_adapter_projector_type: str = "mlp",
        text_adapter_freeze: bool = False,
        text_logit_normalize_visual: bool = True,
        text_logit_normalize_text: bool = True,
        text_logit_learnable_scale: bool = True,
        text_logit_init_scale: float = 10.0,
        text_fusion_alpha_init: float = 0.0,
        text_fusion_alpha_learnable: bool = True,
        text_fusion_alpha_max: float | None = None,
        text_fusion_enabled: bool = False,
        text_fusion_type: str = "logit_add",
        text_cross_num_heads: int = 4,
        text_cross_dropout: float = 0.0,
        text_cross_beta_init: float = 0.05,
        text_cross_beta_learnable: bool = True,
        text_cross_beta_max: float | None = None,
        text_cross_use_ffn: bool = True,
        text_cross_ffn_ratio: float = 4.0,
        text_cross_norm_visual: bool = True,
        text_cross_norm_text: bool = True,
        text_cross_return_attn: bool = False,
        text_cross_attn_temperature: float = 1.0,
        text_ground_num_heads: int = 4,
        text_ground_num_layers: int = 2,
        text_ground_dropout: float = 0.0,
        text_ground_gamma_init: float = 0.05,
        text_ground_gamma_learnable: bool = True,
        text_ground_logit_scale_init: float = 10.0,
        text_ground_logit_scale_learnable: bool = True,
        text_ground_use_query_self_attn: bool = True,
        text_ground_use_ffn: bool = True,
        text_ground_ffn_ratio: float = 4.0,
        text_ground_num_queries_per_class: int = 1,
        text_ground_use_tie_breaker: bool = False,
        text_ground_query_delta_init: float = 0.0,
        text_ground_use_learnable_query_delta: bool = False,
        text_ground_kv_stride: int = 1,
        text_ground_return_attn: bool = True,
        text_ground_attn_temperature: float = 1.0,
        depth_adapter_cfg: dict | None = None,
        # NEW: SG-DRF parameters
        enable_sgdrf: bool = False,
        sgdrf_backbone_channels: int = 256,
        sgdrf_enable_dynamic_fsmc: bool = True,
        sgdrf_enable_dynamic_scm: bool = True,
        sgdrf_fsmc_min_ratio: float = 4.0,
        sgdrf_fsmc_max_ratio: float = 16.0,
        sgdrf_scm_base_pad: int = 1024,
        sgdrf_scm_min_pad: int = 512,
        sgdrf_scm_max_pad: int = 1536,
        sgdrf_training_stage: str = "joint",  # 'cold_start' or 'joint'
    ):
        super().__init__()
        self.sam_model = sam_model
        self.encoder = LoRA_Sam(sam_model, 4)
        ratio = float(decoder_channel_ratio) if bool(use_lite_decoder) else 1.0
        conv_mode = str(decoder_conv_mode) if bool(use_lite_decoder) else "standard"
        upsample_mode = str(decoder_upsample_mode) if bool(use_lite_decoder) else "bilinear"
        self.decoder = SegFormerHead_adaption(
            num_classes=num_classes,
            channel_ratio=ratio,
            conv_mode=conv_mode,
            upsample_mode=upsample_mode,
        )
        # self.decoder = DMLPHead_adaption(num_classes=num_classes)
        self.num_maskmem = num_maskmem
        self.memory_bank = MemoryBank(sam_model, num_maskmem=num_maskmem)
        self.lambda_txt = float(lambda_txt)
        self.txt_tau = float(txt_tau)
        self.use_language_adapter = bool(use_language_adapter)
        self.depth_adapter: LightDepthAdapter | None = None

        self.txt_head: TextPrototypeLogitHead | None = None
        self.lang_adapter: LightweightLanguageAdapter | None = None
        self.text_fusion: VisualTextLogitFusion | None = None
        self.text_cross_fusion: ImageToTextCrossAttentionFusion | None = None
        self.text_grounding_fusion: TextToImageGroundingFusion | None = None
        self.text_fusion_enabled = bool(text_fusion_enabled)
        self.text_fusion_type = str(text_fusion_type or "logit_add").lower()
        d_decoder = int(getattr(self.decoder, "out_channels", 256))
        dcfg = depth_adapter_cfg if isinstance(depth_adapter_cfg, dict) else {}
        if bool(dcfg.get("enable", False)):
            self.depth_adapter = LightDepthAdapter(
                out_channels=d_decoder,
                mid_channels=int(dcfg.get("mid_channels", 32)),
                gate_init=float(dcfg.get("gate_init", 0.0)),
                norm=str(dcfg.get("norm", "gn")),
            )
        if text_prototypes is not None:
            if d_txt is None:
                d_txt = int(text_prototypes.shape[-1])
            self.txt_head = TextPrototypeLogitHead(
                d_model=d_decoder,
                d_text=int(d_txt),
                num_classes=num_classes,
                adapter_hidden_dim=int(text_adapter_hidden_dim),
                adapter_dropout=float(text_adapter_dropout),
                adapter_normalize_output=bool(text_adapter_normalize_output),
                adapter_zero_init_last=bool(text_adapter_zero_init_last),
                adapter_projector_type=str(text_adapter_projector_type),
                normalize_visual=bool(text_logit_normalize_visual),
                normalize_text=bool(text_logit_normalize_text),
                learnable_scale=bool(text_logit_learnable_scale),
                init_scale=float(text_logit_init_scale),
            )
            self.txt_head.set_text_prototypes(text_prototypes)
            self.text_fusion = VisualTextLogitFusion(
                alpha_init=float(text_fusion_alpha_init),
                alpha_learnable=bool(text_fusion_alpha_learnable),
                alpha_max=text_fusion_alpha_max,
            )
            self.text_cross_fusion = ImageToTextCrossAttentionFusion(
                dim=d_decoder,
                num_heads=int(text_cross_num_heads),
                dropout=float(text_cross_dropout),
                beta_init=float(text_cross_beta_init),
                beta_learnable=bool(text_cross_beta_learnable),
                beta_max=text_cross_beta_max,
                use_ffn=bool(text_cross_use_ffn),
                ffn_ratio=float(text_cross_ffn_ratio),
                norm_visual=bool(text_cross_norm_visual),
                norm_text=bool(text_cross_norm_text),
                return_attn=bool(text_cross_return_attn),
                attn_temperature=float(text_cross_attn_temperature),
            )
            self.text_grounding_fusion = TextToImageGroundingFusion(
                dim=d_decoder,
                num_classes=int(num_classes),
                num_heads=int(text_ground_num_heads),
                num_layers=int(text_ground_num_layers),
                dropout=float(text_ground_dropout),
                gamma_init=float(text_ground_gamma_init),
                gamma_learnable=bool(text_ground_gamma_learnable),
                use_query_self_attn=bool(text_ground_use_query_self_attn),
                use_ffn=bool(text_ground_use_ffn),
                ffn_ratio=float(text_ground_ffn_ratio),
                logit_scale_init=float(text_ground_logit_scale_init),
                logit_scale_learnable=bool(text_ground_logit_scale_learnable),
                num_queries_per_class=int(text_ground_num_queries_per_class),
                use_tie_breaker=bool(text_ground_use_tie_breaker),
                query_delta_init=float(text_ground_query_delta_init),
                use_learnable_query_delta=bool(text_ground_use_learnable_query_delta),
                kv_stride=int(text_ground_kv_stride),
                return_attn=bool(text_ground_return_attn),
                attn_temperature=float(text_ground_attn_temperature),
            )
            if bool(text_adapter_freeze):
                for p in self.txt_head.projector.parameters():
                    p.requires_grad = False
            if self.use_language_adapter:
                self.lang_adapter = LightweightLanguageAdapter(
                    d_in=d_decoder,
                    d_txt=int(d_txt),
                    num_heads=int(adapter_num_heads),
                    adapter_scale=float(adapter_scale),
                )

        # ========== NEW: Initialize SG-DRF modules ==========
        self.enable_sgdrf = bool(enable_sgdrf)
        self.sgdrf_training_stage = str(sgdrf_training_stage)
        self.density_estimator: SemanticDensityEstimator | None = None
        self.dynamic_fsmc: DynamicFSMC | None = None
        self.dynamic_scm: DynamicSCM | None = None

        if self.enable_sgdrf:
            # Density Estimator (always needed for SG-DRF)
            self.density_estimator = SemanticDensityEstimator(
                in_channels=int(sgdrf_backbone_channels)
            )

            # Dynamic FSMC (optional)
            if bool(sgdrf_enable_dynamic_fsmc):
                self.dynamic_fsmc = DynamicFSMC(
                    in_channels=int(sgdrf_backbone_channels),
                    min_ratio=float(sgdrf_fsmc_min_ratio),
                    max_ratio=float(sgdrf_fsmc_max_ratio)
                )

            # Dynamic SCM (optional)
            if bool(sgdrf_enable_dynamic_scm):
                self.dynamic_scm = DynamicSCM(
                    base_pad_width=int(sgdrf_scm_base_pad),
                    min_pad_width=int(sgdrf_scm_min_pad),
                    max_pad_width=int(sgdrf_scm_max_pad)
                )

            # Apply cold-start freezing if needed
            self._apply_sgdrf_cold_start_freeze()

            print(f"[OmniSAM] SG-DRF enabled - Stage: {self.sgdrf_training_stage}, "
                  f"Dynamic FSMC: {sgdrf_enable_dynamic_fsmc}, "
                  f"Dynamic SCM: {sgdrf_enable_dynamic_scm}")

    def _apply_sgdrf_cold_start_freeze(self):
        """
        Apply parameter freezing for SG-DRF cold-start stage
        """
        if self.sgdrf_training_stage == 'cold_start' and self.density_estimator is not None:
            # Keep baseline segmentation trainable; defer random routing branches.
            for param in self.parameters():
                param.requires_grad = True
            for module in (self.dynamic_fsmc, self.dynamic_scm):
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False
                    module.eval()
            print("[SG-DRF] Cold-start mode: baseline path + density estimator; FSMC/SCM bypassed")

    def set_sgdrf_training_stage(self, stage: str):
        """
        Switch SG-DRF training stage

        Args:
            stage: 'cold_start' or 'joint'
        """
        if not self.enable_sgdrf:
            print("[SG-DRF] Warning: SG-DRF is not enabled")
            return

        assert stage in ['cold_start', 'joint'], \
            f"Invalid stage: {stage}. Must be 'cold_start' or 'joint'"

        self.sgdrf_training_stage = stage

        if stage == 'cold_start':
            # Keep baseline segmentation trainable and defer dynamic branches.
            for param in self.parameters():
                param.requires_grad = True
            for module in (self.dynamic_fsmc, self.dynamic_scm):
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False
                    module.eval()
            print("[SG-DRF] Switched to cold-start mode (baseline-safe forward)")

        elif stage == 'joint':
            # Unfreeze all parameters and restore training mode for dynamic branches.
            for param in self.parameters():
                param.requires_grad = True
            for module in (self.dynamic_fsmc, self.dynamic_scm):
                if module is not None:
                    module.train()
            # Keep density estimator trainable to adapt to evolving features
            if self.density_estimator is not None:
                self.density_estimator.train()
            print("[SG-DRF] Switched to joint training mode")

    def forward(self, frame_idx, x, mem_type="source", return_aux: bool = False, depth=None, historical_pseudo_labels=None, return_sgdrf_aux: bool = False):
        _, _, height, width = x.shape
        backbone_out = self.encoder(x)
        current_vision_feats = backbone_out['vision_features']
        current_vision_pos_embeds = backbone_out['vision_pos_enc'][-1]
        feat = backbone_out['backbone_fpn']

        # ========== NEW: SG-DRF Processing ==========
        density_map = None
        sgdrf_stats = {}

        if self.enable_sgdrf and self.density_estimator is not None:
            # Step 1: Estimate semantic density from current_vision_feats
            density_input = (current_vision_feats.detach()
                             if self.sgdrf_training_stage == 'cold_start'
                             else current_vision_feats)
            density_map = self.density_estimator(density_input)

            sgdrf_stats['density_mean'] = density_map.mean().item()
            sgdrf_stats['density_std'] = density_map.std().item()
            sgdrf_stats['density_min'] = density_map.min().item()
            sgdrf_stats['density_max'] = density_map.max().item()
            sgdrf_stats['initial_feats_shape'] = list(current_vision_feats.shape)
            sgdrf_stats['initial_pos_shape'] = list(current_vision_pos_embeds.shape)

            # Step 2: Apply Dynamic FSMC compression (optional)
            if self.sgdrf_training_stage != 'cold_start' and self.dynamic_fsmc is not None:
                feat_to_compress = current_vision_feats
                compressed_feat, fsmc_stats = self.dynamic_fsmc(feat_to_compress, density_map)
                
                # SHAPE FIX: Verify FSMC output shape matches input
                if compressed_feat.shape != feat_to_compress.shape:
                    import warnings
                    warnings.warn(f"FSMC shape mismatch! Input: {feat_to_compress.shape}, Output: {compressed_feat.shape}")
                    if compressed_feat.shape[-2:] != feat_to_compress.shape[-2:]:
                        compressed_feat = F.interpolate(compressed_feat, size=feat_to_compress.shape[-2:], mode='bilinear', align_corners=False)
                
                current_vision_feats = compressed_feat
                sgdrf_stats['fsmc'] = fsmc_stats
                sgdrf_stats['after_fsmc_feats_shape'] = list(current_vision_feats.shape)

            # SHAPE FIX: Validate and align position embeddings
            if current_vision_feats.shape[-2:] != current_vision_pos_embeds.shape[-2:]:
                print(f"[SG-DRF] Aligning pos_embeds: feats {current_vision_feats.shape}, pos {current_vision_pos_embeds.shape}")
                current_vision_pos_embeds = F.interpolate(current_vision_pos_embeds, size=current_vision_feats.shape[-2:], mode='bilinear', align_corners=False)
                sgdrf_stats['pos_embeds_realigned'] = True

            # Step 3: Apply Dynamic SCM padding (optional)
            if self.sgdrf_training_stage != 'cold_start' and self.dynamic_scm is not None:
                # SHAPE FIX: Assert alignment before padding
                assert current_vision_feats.shape[-2:] == current_vision_pos_embeds.shape[-2:], f"Shape mismatch before SCM: feats {current_vision_feats.shape}, pos {current_vision_pos_embeds.shape}"
                
                padded_feats, pad_width = self.dynamic_scm(current_vision_feats, density_map, return_pad_width=True)
                padded_pos_embeds = self.dynamic_scm.pad_tensor(current_vision_pos_embeds, pad_width)
                
                # SHAPE FIX: Final verification before memory_bank
                assert padded_feats.shape[-2:] == padded_pos_embeds.shape[-2:], f"Shape mismatch after SCM padding: feats {padded_feats.shape}, pos {padded_pos_embeds.shape}"
                
                sgdrf_stats['padded_feats_shape'] = list(padded_feats.shape)
                sgdrf_stats['padded_pos_shape'] = list(padded_pos_embeds.shape)
                
                feat[-1] = self.memory_bank(frame_idx, padded_feats, padded_pos_embeds, mem_type=mem_type, num_obj_ptr_tokens=0, historical_pseudo_labels=historical_pseudo_labels)
                feat[-1] = self.dynamic_scm.remove_circular_padding(feat[-1], pad_width)
                sgdrf_stats['scm_pad_width'] = pad_width
            else:
                # SHAPE FIX: Ensure alignment even without SCM
                if current_vision_feats.shape[-2:] != current_vision_pos_embeds.shape[-2:]:
                    print(f"[SG-DRF] Final alignment: feats {current_vision_feats.shape}, pos {current_vision_pos_embeds.shape}")
                    current_vision_pos_embeds = F.interpolate(current_vision_pos_embeds, size=current_vision_feats.shape[-2:], mode='bilinear', align_corners=False)
                
                feat[-1] = self.memory_bank(frame_idx, current_vision_feats, current_vision_pos_embeds, mem_type=mem_type, num_obj_ptr_tokens=0, historical_pseudo_labels=historical_pseudo_labels)
        else:
            # Original flow without SG-DRF
            feat[-1] = self.memory_bank(
                        frame_idx,
                        current_vision_feats,
                        current_vision_pos_embeds,
                        mem_type=mem_type,
                        num_obj_ptr_tokens=0,
                        historical_pseudo_labels=historical_pseudo_labels,
                    )

        # ========== Continue with original decoder flow ==========
        x, feat_de = self.decoder(feat)
        feat_de_visual = feat_de
        if self.depth_adapter is not None:
            feat_de = self.depth_adapter(feat_de, depth=depth)
            feat_de_visual = feat_de
            x = self.decoder.linear_pred(self.decoder.dropout(feat_de))
        l_vis = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)

        if self.lang_adapter is not None and self.txt_head is not None:
            text_proto = self.txt_head.text_prototypes  # (K, D_txt)
            feat_de, _ = self.lang_adapter(feat_de, text_proto)

        l_txt = None
        fusion_alpha = 0.0
        text_scale = 0.0
        cross_aux = {}
        ground_aux = {}
        l_ground = None
        projected_text_proto = None
        need_txt_aux = bool(return_aux)
        if self.text_fusion_enabled:
            fusion_type = self.text_fusion_type
        elif self.txt_head is not None and self.lambda_txt != 0.0:
            fusion_type = "logit_add"
        else:
            fusion_type = "none"
        if self.txt_head is not None and fusion_type in {"image_to_text_cross_attn", "hybrid"}:
            projected_text_proto = self.txt_head.projector(self.txt_head.text_prototypes)
            feat_de_fused, cross_aux = self.text_cross_fusion(feat_de, projected_text_proto)
            feat_de = feat_de_fused
            x = self.decoder.linear_pred(self.decoder.dropout(feat_de))
        if self.txt_head is not None and fusion_type == "text_to_image_grounding":
            projected_text_proto = self.txt_head.projector(self.txt_head.text_prototypes)
            l_ground_low, ground_aux = self.text_grounding_fusion(feat_de_visual, projected_text_proto, visual_logits=x)
            l_ground = F.interpolate(l_ground_low, size=(height, width), mode='bilinear', align_corners=False)
            pred = l_vis + self.text_grounding_fusion.gamma.to(dtype=l_vis.dtype) * l_ground
        if self.txt_head is not None and (self.lambda_txt != 0.0 or need_txt_aux):
            # feat_de is the fused decoder feature map (B,256,H',W')
            txt_feat = feat_de if fusion_type in {"image_to_text_cross_attn", "hybrid"} else feat_de_visual
            l_txt_low = self.txt_head(txt_feat)
            l_txt = F.interpolate(l_txt_low, size=(height, width), mode='bilinear', align_corners=False)
            if fusion_type == "text_to_image_grounding":
                text_scale = float(self.txt_head.text_scale.detach().item())
            elif fusion_type == "image_to_text_cross_attn":
                pred = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
            elif fusion_type == "hybrid" and self.lambda_txt != 0.0:
                l_vis_fused = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
                if self.text_fusion is not None:
                    pred = self.text_fusion(l_vis_fused, self.lambda_txt * l_txt)
                    fusion_alpha = float(self.text_fusion.current_alpha().detach().item())
                else:
                    pred = l_vis_fused + (self.lambda_txt * l_txt)
            elif self.lambda_txt != 0.0:
                if self.text_fusion is not None and fusion_type != "none":
                    pred = self.text_fusion(l_vis, self.lambda_txt * l_txt)
                    fusion_alpha = float(self.text_fusion.current_alpha().detach().item())
                else:
                    pred = l_vis + (self.lambda_txt * l_txt)
            else:
                pred = l_vis
            text_scale = float(self.txt_head.text_scale.detach().item())
        else:
            pred = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False) if fusion_type in {"image_to_text_cross_attn", "hybrid"} else l_vis

        # Encode new memory
        if self.num_maskmem > 0:
            output = torch.argmax(pred, 1)
            mask_for_mem = output.unsqueeze(1)
            # NEW: Compute uncertainty (entropy) for uncertainty-guided token selection
            with torch.no_grad():
                probs = torch.softmax(pred, dim=1)  # [B, K, H, W]
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B, H, W]
            self.memory_bank._encode_new_memory(frame_idx, current_vision_feats, mask_for_mem, mem_type, uncertainty_map=entropy)

        # ========== Return results with optional SG-DRF auxiliary outputs ==========
        if return_sgdrf_aux and self.enable_sgdrf:
            # Upsample density map to original resolution for visualization/loss computation
            if density_map is not None:
                density_map_full = F.interpolate(
                    density_map, size=(height, width),
                    mode='bilinear', align_corners=False
                )
            else:
                density_map_full = None

            if return_aux and l_txt is not None:
                return pred, feat, feat_de, TextAlignAux(
                    l_vis=l_vis,
                    l_txt=l_txt,
                    l_ground=l_ground,
                    fusion_alpha=fusion_alpha,
                    text_scale=text_scale,
                    cross_beta=float(cross_aux.get("beta", 0.0)),
                    cross_attn_mean=float(cross_aux.get("attn_mean", 0.0)),
                    cross_attn_std=float(cross_aux.get("attn_std", 0.0)),
                    cross_attn_entropy=float(cross_aux.get("attn_entropy", 0.0)),
                    cross_attn_max_mean=float(cross_aux.get("text_attn_max", 0.0)),
                    feature_delta_mean=float(cross_aux.get("feature_delta_mean", 0.0)),
                    feature_delta_norm=float(cross_aux.get("feature_delta_norm", 0.0)),
                    feature_delta_ratio=float(cross_aux.get("feature_delta_ratio", 0.0)),
                    fused_feature_mean=float(cross_aux.get("fused_feature_mean", 0.0)),
                    fused_feature_std=float(cross_aux.get("fused_feature_std", 0.0)),
                    visual_feature_mean=float(cross_aux.get("visual_feature_mean", 0.0)),
                    visual_feature_std=float(cross_aux.get("visual_feature_std", 0.0)),
                    cross_attn_aux=dict(cross_aux),
                    grounding_gamma=float(ground_aux.get("gamma", 0.0)),
                    grounding_logit_scale=float(ground_aux.get("logit_scale", 0.0)),
                    grounding_attn_mean=float(ground_aux.get("attn_mean", 0.0)),
                    grounding_attn_std=float(ground_aux.get("attn_std", 0.0)),
                    grounding_attn_entropy=float(ground_aux.get("attn_entropy", 0.0)),
                    grounding_attn_max_mean=float(ground_aux.get("attn_max_mean", 0.0)),
                    grounding_logits_mean=float(ground_aux.get("grounding_logits_mean", 0.0)),
                    grounding_logits_std=float(ground_aux.get("grounding_logits_std", 0.0)),
                    grounding_feature_delta_mean=float(ground_aux.get("query_delta_mean", 0.0)),
                    grounding_feature_delta_norm=float(ground_aux.get("query_delta_norm", 0.0)),
                    grounding_feature_delta_ratio=float(ground_aux.get("query_delta_ratio", 0.0)),
                    grounding_topk_area_ratio=float(ground_aux.get("topk_area_ratio", 0.0)),
                    grounding_aux=dict(ground_aux),
                ), density_map_full, sgdrf_stats
            else:
                return pred, feat, feat_de, density_map_full, sgdrf_stats

        if return_aux and l_txt is not None:
            return pred, feat, feat_de, TextAlignAux(
                l_vis=l_vis,
                l_txt=l_txt,
                l_ground=l_ground,
                fusion_alpha=fusion_alpha,
                text_scale=text_scale,
                cross_beta=float(cross_aux.get("beta", 0.0)),
                cross_attn_mean=float(cross_aux.get("attn_mean", 0.0)),
                cross_attn_std=float(cross_aux.get("attn_std", 0.0)),
                cross_attn_entropy=float(cross_aux.get("attn_entropy", 0.0)),
                cross_attn_max_mean=float(cross_aux.get("text_attn_max", 0.0)),
                feature_delta_mean=float(cross_aux.get("feature_delta_mean", 0.0)),
                feature_delta_norm=float(cross_aux.get("feature_delta_norm", 0.0)),
                feature_delta_ratio=float(cross_aux.get("feature_delta_ratio", 0.0)),
                fused_feature_mean=float(cross_aux.get("fused_feature_mean", 0.0)),
                fused_feature_std=float(cross_aux.get("fused_feature_std", 0.0)),
                visual_feature_mean=float(cross_aux.get("visual_feature_mean", 0.0)),
                visual_feature_std=float(cross_aux.get("visual_feature_std", 0.0)),
                cross_attn_aux=dict(cross_aux),
                grounding_gamma=float(ground_aux.get("gamma", 0.0)),
                grounding_logit_scale=float(ground_aux.get("logit_scale", 0.0)),
                grounding_attn_mean=float(ground_aux.get("attn_mean", 0.0)),
                grounding_attn_std=float(ground_aux.get("attn_std", 0.0)),
                grounding_attn_entropy=float(ground_aux.get("attn_entropy", 0.0)),
                grounding_attn_max_mean=float(ground_aux.get("attn_max_mean", 0.0)),
                grounding_logits_mean=float(ground_aux.get("grounding_logits_mean", 0.0)),
                grounding_logits_std=float(ground_aux.get("grounding_logits_std", 0.0)),
                grounding_feature_delta_mean=float(ground_aux.get("query_delta_mean", 0.0)),
                grounding_feature_delta_norm=float(ground_aux.get("query_delta_norm", 0.0)),
                grounding_feature_delta_ratio=float(ground_aux.get("query_delta_ratio", 0.0)),
                grounding_topk_area_ratio=float(ground_aux.get("topk_area_ratio", 0.0)),
                grounding_aux=dict(ground_aux),
            )
        return pred, feat, feat_de

    def compute_sgdrf_consistency_loss(self, density_map, seg_logits):
        """
        Compute consistency loss between density map and segmentation uncertainty

        This ensures the density estimator's predictions align with the
        segmentation model's uncertainty (entropy).

        Args:
            density_map: (B, 1, H, W) - predicted semantic density [0, 1]
            seg_logits: (B, num_classes, H, W) - segmentation logits

        Returns:
            consistency_loss: scalar tensor
        """
        if not self.enable_sgdrf or self.density_estimator is None:
            return torch.tensor(0.0, device=seg_logits.device)

        # Compute segmentation entropy (uncertainty)
        seg_probs = F.softmax(seg_logits, dim=1)  # (B, C, H, W)
        entropy = -(seg_probs * torch.log(seg_probs + 1e-8)).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Normalize entropy to [0, 1]
        num_classes = seg_logits.size(1)
        max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32, device=seg_logits.device))
        entropy_normalized = entropy / max_entropy

        # Downsample to density_map resolution if needed
        if entropy_normalized.shape != density_map.shape:
            entropy_normalized = F.interpolate(
                entropy_normalized,
                size=density_map.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # MSE loss
        consistency_loss = F.mse_loss(density_map, entropy_normalized)

        return consistency_loss
