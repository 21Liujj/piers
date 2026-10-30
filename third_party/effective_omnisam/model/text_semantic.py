from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextGateOutput:
    ssl_mask_text: torch.Tensor
    text_similarity_map: torch.Tensor
    effective_kept_ratio: float
    visual_mask_ratio: float
    text_mask_ratio: float
    text_sim_mean: float
    text_sim_std: float
    soft_weight: Optional[torch.Tensor] = None
    accept_ratio_per_class: Optional[torch.Tensor] = None
    reject_ratio_per_class: Optional[torch.Tensor] = None


class TextPrototypeBank(nn.Module):
    """
    Lightweight text prototype bank.

    - Builds class-wise text prototypes from prompt templates.
    - Supports:
      * offline cache (initialized once)
      * refresh per epoch (manual refresh via refresh(...))
    - Optional projector from text feature space to visual feature space.
    """

    def __init__(
        self,
        class_names: List[str],
        templates: Optional[List[str]] = None,
        text_model: str = "ViT-B-32",
        text_pretrained: str = "openai",
        device: str = "cpu",
        d_v: Optional[int] = None,
        use_projector: bool = False,
    ) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("class_names must be non-empty")
        self.class_names = list(class_names)
        self.templates = templates or [
            "a {cls} region in a panoramic street scene",
            "a {cls} region in an urban driving panorama",
            "a street-view panorama containing {cls}",
        ]
        self.text_model = text_model
        self.text_pretrained = text_pretrained
        self.device_name = device
        self.use_projector = bool(use_projector and d_v is not None)

        self.register_buffer("text_prototypes", torch.empty(0), persistent=True)
        self.projector: Optional[nn.Module] = None
        if self.use_projector:
            # Initialized after knowing text dim.
            self.projector = nn.Identity()
            self._projector_out_dim = int(d_v)
        else:
            self._projector_out_dim = None

    @torch.no_grad()
    def refresh(self) -> torch.Tensor:
        try:
            import open_clip
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "open_clip_torch is required for TextPrototypeBank refresh."
            ) from e

        device = torch.device(self.device_name)
        model, _, _ = open_clip.create_model_and_transforms(
            self.text_model,
            pretrained=self.text_pretrained,
        )
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(self.text_model)

        protos = []
        for cls in self.class_names:
            prompts = [tpl.format(cls=cls) for tpl in self.templates]
            tokens = tokenizer(prompts).to(device)
            txt = model.encode_text(tokens)
            txt = F.normalize(txt, dim=-1)
            proto = F.normalize(txt.mean(dim=0), dim=-1)
            protos.append(proto)
        text_prototypes = torch.stack(protos, dim=0).detach().cpu()

        if self.text_prototypes.numel() == 0:
            self.register_buffer(
                "text_prototypes",
                text_prototypes.clone(),
                persistent=True,
            )
        else:
            self.text_prototypes = text_prototypes.clone()

        if self.use_projector:
            d_t = int(text_prototypes.shape[-1])
            if isinstance(self.projector, nn.Identity):
                self.projector = nn.Sequential(
                    nn.Linear(d_t, self._projector_out_dim, bias=False),
                    nn.LayerNorm(self._projector_out_dim),
                )
        return self.text_prototypes

    def set_cached(self, text_prototypes: torch.Tensor) -> None:
        if text_prototypes.dim() != 2:
            raise ValueError(f"text_prototypes must be 2D [K,D], got {tuple(text_prototypes.shape)}")
        text_prototypes = F.normalize(text_prototypes.detach().cpu(), dim=-1)
        if self.text_prototypes.numel() == 0:
            self.register_buffer("text_prototypes", text_prototypes.clone(), persistent=True)
        else:
            self.text_prototypes = text_prototypes.clone()

    def get(self, device: torch.device, project_to_visual: bool = False) -> torch.Tensor:
        if self.text_prototypes.numel() == 0:
            raise RuntimeError("TextPrototypeBank is empty. Call set_cached(...) or refresh().")
        t = self.text_prototypes.to(device)
        if project_to_visual and self.use_projector and self.projector is not None:
            t = self.projector(t)
            t = F.normalize(t, dim=-1)
        return t


class TextAwarePseudoLabelFilter(nn.Module):
    """
    Text-aware pseudo-label gating.

    Inputs:
      l_txt: (B, K, H, W) text-branch logits from pixel-text similarity
      pseudo_t: (B, H, W) pseudo label ids
      conf_t: (B, H, W) visual confidence
      valid_mask: (B, H, W) baseline validity (e.g., pseudo!=ignore)
    """

    def __init__(self, tau_v: float = 0.85, tau_t: float = 0.10, mode: str = "hard") -> None:
        super().__init__()
        self.tau_v = float(tau_v)
        self.tau_t = float(tau_t)
        self.mode = str(mode)

    @staticmethod
    def _gather_class_similarity(l_txt: torch.Tensor, pseudo_t: torch.Tensor, txt_tau: float) -> torch.Tensor:
        # l_txt is cosine/tau in current implementation, convert back to cosine.
        cosine_map = l_txt * max(float(txt_tau), 1e-6)
        k = int(cosine_map.shape[1])
        idx_raw = pseudo_t.long()
        # NOTE: pseudo_t may contain ignore_index (e.g. 255) or other invalid ids.
        # torch.gather requires all indices in [0, K-1]; otherwise CUDA will assert.
        invalid = (idx_raw < 0) | (idx_raw >= k)
        idx = idx_raw.clamp(0, max(k - 1, 0)).unsqueeze(1)  # (B,1,H,W)
        sim = torch.gather(cosine_map, dim=1, index=idx).squeeze(1)  # (B,H,W)
        # For invalid pixels, force low similarity so they won't pass text threshold.
        if invalid.any():
            sim = sim.masked_fill(invalid, -1.0)
        return sim

    def forward(
        self,
        l_txt: torch.Tensor,
        pseudo_t: torch.Tensor,
        conf_t: torch.Tensor,
        valid_mask: torch.Tensor,
        txt_tau: float = 0.07,
        num_classes: Optional[int] = None,
    ) -> TextGateOutput:
        if self.mode not in {"hard", "soft"}:
            raise ValueError(f"ssl_gate_mode must be hard/soft, got {self.mode}")

        text_sim = self._gather_class_similarity(l_txt=l_txt, pseudo_t=pseudo_t, txt_tau=txt_tau)
        visual_mask = (conf_t > self.tau_v) & valid_mask
        text_mask = (text_sim > self.tau_t) & valid_mask
        joint_mask = visual_mask & text_mask

        visual_ratio = float(visual_mask.float().mean().item())
        text_ratio = float(joint_mask.float().mean().item())
        valid_sim = text_sim[valid_mask]
        if valid_sim.numel() == 0:
            sim_mean, sim_std = 0.0, 0.0
        else:
            sim_mean = float(valid_sim.mean().item())
            sim_std = float(valid_sim.std(unbiased=False).item())

        soft_weight = None
        if self.mode == "soft":
            conf_n = torch.clamp((conf_t - self.tau_v) / max(1.0 - self.tau_v, 1e-6), 0.0, 1.0)
            sim_n = torch.clamp((text_sim - self.tau_t) / max(1.0 - self.tau_t, 1e-6), 0.0, 1.0)
            soft_weight = conf_n * sim_n * valid_mask.float()

        accept_ratio_per_class = None
        reject_ratio_per_class = None
        k = int(num_classes if num_classes is not None else l_txt.shape[1])
        if k > 0:
            accept = torch.zeros(k, device=pseudo_t.device, dtype=torch.float32)
            reject = torch.zeros(k, device=pseudo_t.device, dtype=torch.float32)
            for cls_idx in range(k):
                cls_mask = (pseudo_t == cls_idx) & valid_mask
                denom = cls_mask.float().sum()
                if denom.item() > 0:
                    cls_accept = (joint_mask & cls_mask).float().sum() / denom
                    accept[cls_idx] = cls_accept
                    reject[cls_idx] = 1.0 - cls_accept
            accept_ratio_per_class = accept
            reject_ratio_per_class = reject

        return TextGateOutput(
            ssl_mask_text=joint_mask,
            text_similarity_map=text_sim,
            effective_kept_ratio=text_ratio,
            visual_mask_ratio=visual_ratio,
            text_mask_ratio=text_ratio,
            text_sim_mean=sim_mean,
            text_sim_std=sim_std,
            soft_weight=soft_weight,
            accept_ratio_per_class=accept_ratio_per_class,
            reject_ratio_per_class=reject_ratio_per_class,
        )


class LightweightLanguageAdapter(nn.Module):
    """
    Lightweight language adapter (stage-2):
      text_proto --MLP--> visual space
      visual tokens x text tokens (1-layer cross-attn)
      residual enhancement on one feature level.

    Input:
      visual_feat: (B, C, H, W)
      text_proto:  (K, D_t)
    Output:
      enhanced_feat: (B, C, H, W)
      attn_map: optional (B, HW, K)
    """

    def __init__(
        self,
        d_in: int,
        d_txt: int,
        num_heads: int = 4,
        adapter_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(d_txt, d_in, bias=False),
            nn.LayerNorm(d_in),
            nn.GELU(),
            nn.Linear(d_in, d_in, bias=False),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_in,
            num_heads=num_heads,
            batch_first=True,
        )
        self.adapter_scale = float(adapter_scale)

    def forward(self, visual_feat: torch.Tensor, text_proto: torch.Tensor):
        b, c, h, w = visual_feat.shape
        v = visual_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        t = self.text_proj(text_proto)  # (K, C)
        t = t.unsqueeze(0).expand(b, -1, -1).contiguous()  # (B, K, C)
        delta, attn = self.cross_attn(query=v, key=t, value=t, need_weights=True, average_attn_weights=True)
        v_enh = v + self.adapter_scale * delta
        feat_enh = v_enh.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return feat_enh, attn


def weighted_unsup_loss(
    pred_t: torch.Tensor,
    pseudo_t: torch.Tensor,
    ignore_index: int,
    weight_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-pixel CE with optional dense weighting.
    """
    ce = F.cross_entropy(pred_t, pseudo_t, ignore_index=ignore_index, reduction="none")
    valid = (pseudo_t != int(ignore_index))
    if weight_map is not None:
        ce = ce * weight_map
    ce = ce[valid]
    if ce.numel() == 0:
        return pred_t.new_tensor(0.0)
    return ce.mean()
