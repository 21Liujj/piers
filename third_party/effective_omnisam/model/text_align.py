from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextAlignAux:
    l_vis: torch.Tensor
    l_txt: torch.Tensor
    l_ground: Optional[torch.Tensor] = None
    fusion_alpha: float = 0.0
    text_scale: float = 0.0
    cross_beta: float = 0.0
    cross_attn_mean: float = 0.0
    cross_attn_std: float = 0.0
    cross_attn_entropy: float = 0.0
    cross_attn_max_mean: float = 0.0
    feature_delta_mean: float = 0.0
    feature_delta_norm: float = 0.0
    feature_delta_ratio: float = 0.0
    fused_feature_mean: float = 0.0
    fused_feature_std: float = 0.0
    visual_feature_mean: float = 0.0
    visual_feature_std: float = 0.0
    cross_attn_aux: Dict[str, Any] = field(default_factory=dict)
    grounding_gamma: float = 0.0
    grounding_logit_scale: float = 0.0
    grounding_attn_mean: float = 0.0
    grounding_attn_std: float = 0.0
    grounding_attn_entropy: float = 0.0
    grounding_attn_max_mean: float = 0.0
    grounding_logits_mean: float = 0.0
    grounding_logits_std: float = 0.0
    grounding_feature_delta_mean: float = 0.0
    grounding_feature_delta_norm: float = 0.0
    grounding_feature_delta_ratio: float = 0.0
    grounding_topk_area_ratio: float = 0.0
    grounding_aux: Dict[str, Any] = field(default_factory=dict)


def load_text_prototypes(
    path: str,
    num_classes: int,
    mode: str = "real",
    normalize: bool = True,
    shuffle_seed: int = 0,
    random_seed: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    payload = torch.load(path, map_location="cpu")
    meta: Dict = {
        "text_proto_path": path,
        "text_proto_mode": mode,
    }
    if isinstance(payload, dict) and "text_prototypes" in payload:
        text_prototypes = payload["text_prototypes"].float()
        for k in ("text_model", "text_pretrained", "source_mode", "class_names", "templates"):
            if k in payload:
                meta[k] = payload[k]
    elif torch.is_tensor(payload):
        text_prototypes = payload.float()
    else:
        raise ValueError(f"Unsupported text prototype payload at {path}")

    if text_prototypes.dim() != 2:
        raise ValueError(f"text prototypes must be 2D [C,D], got {tuple(text_prototypes.shape)}")
    if int(text_prototypes.shape[0]) != int(num_classes):
        raise ValueError(
            f"text prototypes class mismatch: C={int(text_prototypes.shape[0])}, num_classes={int(num_classes)}"
        )

    mode = str(mode or "real").strip().lower()
    if mode == "real":
        pass
    elif mode == "shuffle":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(shuffle_seed))
        idx = torch.randperm(text_prototypes.shape[0], generator=g)
        text_prototypes = text_prototypes[idx]
        meta["shuffle_index"] = idx.tolist()
        meta["shuffle_seed"] = int(shuffle_seed)
    elif mode == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(random_seed))
        text_prototypes = torch.randn_like(text_prototypes, generator=g)
        meta["random_seed"] = int(random_seed)
    else:
        raise ValueError(f"Unsupported text prototype mode: {mode}. expected real|shuffle|random")

    if normalize:
        text_prototypes = F.normalize(text_prototypes, dim=-1)
    meta["text_proto_shape"] = list(text_prototypes.shape)
    norms = torch.norm(text_prototypes, dim=-1)
    meta["text_proto_norm_mean"] = float(norms.mean().item())
    return text_prototypes, meta


class PixelTextAlignHead(nn.Module):
    """
    Project pixel features to text space and compute cosine-sim logits to class text prototypes.

    - pixel_feats: (B, D_in, H, W)
    - text_prototypes: (C, D_txt) normalized or will be normalized internally
    """

    def __init__(
        self,
        d_in: int,
        d_txt: int,
        num_classes: int,
        tau: float = 0.07,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(d_in, d_txt, kernel_size=1, bias=False)
        # Store prototypes as a buffer so they move with .to(device)
        self.register_buffer("text_prototypes", torch.empty(num_classes, d_txt), persistent=True)
        self.tau = float(tau)

    @torch.no_grad()
    def set_text_prototypes(self, protos: torch.Tensor) -> None:
        if protos.dim() != 2:
            raise ValueError(f"text prototypes must be 2D (C,D), got {tuple(protos.shape)}")
        self.text_prototypes.copy_(F.normalize(protos, dim=-1))

    def forward(self, pixel_feats: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
        if pixel_feats.dim() != 4:
            raise ValueError(f"pixel_feats must be 4D (B,D,H,W), got {tuple(pixel_feats.shape)}")
        feat = self.proj(pixel_feats)
        feat = F.normalize(feat, dim=1)  # normalize across channel dimension
        protos = F.normalize(self.text_prototypes, dim=-1)
        # cosine sim: (B,C,H,W)
        logits = torch.einsum("bdhw,cd->bchw", feat, protos)
        t = float(self.tau if tau is None else tau)
        return logits / max(t, 1e-6)


def kl_align_loss(
    l_vis: torch.Tensor,
    l_txt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Pixel-wise KL( p_vis || p_txt ) with optional boolean mask over pixels.
    l_vis/l_txt: (B,C,H,W)
    mask: (B,H,W) boolean, True means included
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    lv = l_vis / tau
    lt = l_txt / tau
    log_p = F.log_softmax(lv, dim=1)
    q = F.softmax(lt, dim=1)
    # F.kl_div expects input=log-probs, target=probs
    kl = F.kl_div(log_p, q, reduction="none")  # (B,C,H,W)
    kl = kl.sum(dim=1)  # (B,H,W)
    if mask is not None:
        kl = kl[mask]
        if kl.numel() == 0:
            return torch.tensor(0.0, device=l_vis.device, dtype=l_vis.dtype)
    return kl.mean()


class TextPrototypeProjector(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_model: int,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        normalize_output: bool = True,
        zero_init_last: bool = False,
        projector_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.normalize_output = bool(normalize_output)
        self.projector_type = str(projector_type or "mlp").strip().lower()
        if self.projector_type == "linear":
            self.net = nn.Linear(int(d_text), int(d_model), bias=True)
            if zero_init_last:
                nn.init.zeros_(self.net.weight)
                nn.init.zeros_(self.net.bias)
        elif self.projector_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(d_text, int(hidden_dim), bias=True),
                nn.LayerNorm(int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(d_model), bias=True),
            )
            if zero_init_last:
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(f"Unsupported projector_type={projector_type}. expected mlp|linear")

    def forward(self, text_proto: torch.Tensor) -> torch.Tensor:
        out = self.net(text_proto)
        if self.normalize_output:
            out = F.normalize(out, dim=-1)
        return out


class ImageToTextCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        beta_init: float = 0.05,
        beta_learnable: bool = True,
        beta_max: Optional[float] = None,
        use_ffn: bool = True,
        ffn_ratio: float = 4.0,
        norm_visual: bool = True,
        norm_text: bool = True,
        return_attn: bool = False,
        attn_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(max(1, num_heads))
        if dim % num_heads != 0:
            for cand in range(num_heads, 0, -1):
                if dim % cand == 0:
                    num_heads = cand
                    break
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} is not divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.norm_visual = bool(norm_visual)
        self.norm_text = bool(norm_text)
        self.return_attn = bool(return_attn)
        self.use_ffn = bool(use_ffn)
        self.attn_temperature = float(attn_temperature)
        if self.attn_temperature <= 0:
            raise ValueError("attn_temperature must be > 0")
        self.beta = nn.Parameter(torch.tensor(float(beta_init)), requires_grad=bool(beta_learnable))
        self.beta_max = beta_max
        self.visual_ln = nn.LayerNorm(dim)
        self.text_ln = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        if self.use_ffn:
            hidden = max(dim, int(round(dim * float(ffn_ratio))))
            self.ffn_ln = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, dim),
            )
        else:
            self.ffn_ln = None
            self.ffn = None

    def current_beta(self) -> torch.Tensor:
        if self.beta_max is None:
            return self.beta
        return torch.clamp(self.beta, min=-float(self.beta_max), max=float(self.beta_max))

    def forward(
        self,
        visual_feature: torch.Tensor,
        text_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if visual_feature.dim() != 4:
            raise ValueError(f"visual_feature must be 4D [B,C,H,W], got {tuple(visual_feature.shape)}")
        bsz, channels, height, width = visual_feature.shape
        if channels != self.dim:
            raise ValueError(f"visual_feature channels={channels} != fusion dim={self.dim}")
        if text_feature.dim() == 2:
            text_tokens = text_feature.unsqueeze(0).expand(bsz, -1, -1)
        elif text_feature.dim() == 3:
            if int(text_feature.shape[0]) != bsz:
                raise ValueError(f"text_feature batch={int(text_feature.shape[0])} != visual batch={bsz}")
            text_tokens = text_feature
        else:
            raise ValueError(f"text_feature must be [K,C] or [B,K,C], got {tuple(text_feature.shape)}")
        text_tokens = text_tokens.to(device=visual_feature.device, dtype=visual_feature.dtype)
        if int(text_tokens.shape[-1]) != channels:
            raise ValueError(f"text_feature dim={int(text_tokens.shape[-1])} != visual channels={channels}")

        image_tokens = visual_feature.flatten(2).transpose(1, 2)  # [B, HW, C]
        q = self.visual_ln(image_tokens) if self.norm_visual else image_tokens
        kv = self.text_ln(text_tokens) if self.norm_text else text_tokens
        temp = max(self.attn_temperature, 1e-6)
        if abs(temp - 1.0) > 1e-8:
            q = q / temp
        attn_out, attn_weights = self.cross_attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        beta = self.current_beta().to(dtype=image_tokens.dtype)
        fused = image_tokens + beta * attn_out
        if self.use_ffn and self.ffn is not None and self.ffn_ln is not None:
            fused = fused + beta * self.ffn(self.ffn_ln(fused))

        fused_feature = fused.transpose(1, 2).reshape(bsz, channels, height, width)
        delta = fused - image_tokens
        delta_abs = delta.abs()
        base_abs = image_tokens.abs().mean().clamp_min(1e-8)

        if attn_weights.dim() == 4:
            attn_prob = attn_weights.mean(dim=1)  # [B, HW, K]
        else:
            attn_prob = attn_weights
        attn_prob = attn_prob.clamp_min(1e-8)
        attn_entropy = -(attn_prob * attn_prob.log()).sum(dim=-1)
        attn_max = attn_prob.max(dim=-1).values

        aux: Dict[str, Any] = {
            "beta": float(beta.detach().item()),
            "attn_mean": float(attn_prob.mean().detach().item()),
            "attn_std": float(attn_prob.std(unbiased=False).detach().item()),
            "attn_entropy": float(attn_entropy.mean().detach().item()),
            "text_attn_max": float(attn_max.mean().detach().item()),
            "feature_delta_mean": float(delta_abs.mean().detach().item()),
            "feature_delta_norm": float(delta.norm().detach().item()),
            "feature_delta_ratio": float((delta_abs.mean() / base_abs).detach().item()),
            "visual_feature_mean": float(visual_feature.mean().detach().item()),
            "visual_feature_std": float(visual_feature.std(unbiased=False).detach().item()),
            "fused_feature_mean": float(fused_feature.mean().detach().item()),
            "fused_feature_std": float(fused_feature.std(unbiased=False).detach().item()),
            "attn_temperature": float(self.attn_temperature),
        }
        if self.return_attn:
            aux["attn_weights"] = attn_prob.detach()
        return fused_feature, aux


class SoftPromptingTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_query_self_attn: bool = True,
        use_ffn: bool = True,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.use_query_self_attn = bool(use_query_self_attn)
        self.use_ffn = bool(use_ffn)
        self.query_ln = nn.LayerNorm(dim)
        self.image_ln = nn.LayerNorm(dim)
        if self.use_query_self_attn:
            self.self_attn_ln = nn.LayerNorm(dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=float(dropout),
                batch_first=True,
            )
        else:
            self.self_attn_ln = None
            self.self_attn = None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        if self.use_ffn:
            hidden = max(dim, int(round(dim * float(ffn_ratio))))
            self.ffn_ln = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, dim),
            )
        else:
            self.ffn_ln = None
            self.ffn = None

    def forward(
        self,
        queries: torch.Tensor,
        image_tokens: torch.Tensor,
        attn_temperature: float = 1.0,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        aux: Dict[str, Any] = {}
        if self.use_query_self_attn and self.self_attn is not None and self.self_attn_ln is not None:
            q_norm = self.self_attn_ln(queries)
            self_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
            queries = queries + self_out
        q = self.query_ln(queries)
        kv = self.image_ln(image_tokens)
        temp = max(float(attn_temperature), 1e-6)
        if abs(temp - 1.0) > 1e-8:
            q = q / temp
        cross_out, cross_weights = self.cross_attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        queries = queries + cross_out
        if self.use_ffn and self.ffn is not None and self.ffn_ln is not None:
            queries = queries + self.ffn(self.ffn_ln(queries))

        if cross_weights.dim() == 4:
            attn_prob = cross_weights.mean(dim=1)  # [B, Q, HW]
        else:
            attn_prob = cross_weights
        attn_prob = attn_prob.clamp_min(1e-8)
        aux.update(
            attn_prob=attn_prob.detach() if return_attn else None,
            attn_mean=float(attn_prob.mean().detach().item()),
            attn_std=float(attn_prob.std(unbiased=False).detach().item()),
            attn_entropy=float((-(attn_prob * attn_prob.log()).sum(dim=-1)).mean().detach().item()),
            attn_max_mean=float(attn_prob.max(dim=-1).values.mean().detach().item()),
        )
        return queries, aux


class TextToImageGroundingFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        gamma_init: float = 0.05,
        gamma_learnable: bool = True,
        use_query_self_attn: bool = True,
        use_ffn: bool = True,
        ffn_ratio: float = 4.0,
        normalize_image: bool = True,
        normalize_text: bool = True,
        logit_scale_init: float = 10.0,
        logit_scale_learnable: bool = True,
        num_queries_per_class: int = 1,
        use_tie_breaker: bool = False,
        tie_breaker_init_std: float = 0.02,
        query_delta_init: float = 0.0,
        use_learnable_query_delta: bool = False,
        kv_stride: int = 1,
        return_attn: bool = True,
        attn_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(max(1, num_heads))
        if dim % num_heads != 0:
            for cand in range(num_heads, 0, -1):
                if dim % cand == 0:
                    num_heads = cand
                    break
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} is not divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_classes = int(num_classes)
        self.num_layers = int(max(1, num_layers))
        self.normalize_image = bool(normalize_image)
        self.normalize_text = bool(normalize_text)
        self.num_queries_per_class = int(max(1, num_queries_per_class))
        self.use_tie_breaker = bool(use_tie_breaker) and self.num_queries_per_class > 1
        self.kv_stride = int(max(1, kv_stride))
        self.return_attn = bool(return_attn)
        self.attn_temperature = float(attn_temperature)
        if self.attn_temperature <= 0:
            raise ValueError("attn_temperature must be > 0")
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)), requires_grad=bool(gamma_learnable))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)), requires_grad=bool(logit_scale_learnable))
        if self.use_tie_breaker:
            self.tie_breaker = nn.Parameter(
                torch.randn(self.num_queries_per_class, dim) * float(tie_breaker_init_std)
            )
        else:
            self.register_parameter("tie_breaker", None)
        self.use_learnable_query_delta = bool(use_learnable_query_delta)
        if self.use_learnable_query_delta:
            qn = self.num_classes * self.num_queries_per_class
            self.query_delta = nn.Parameter(torch.full((qn, dim), float(query_delta_init)))
        else:
            self.register_parameter("query_delta", None)
        self.layers = nn.ModuleList(
            [
                SoftPromptingTransformerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=float(dropout),
                    use_query_self_attn=bool(use_query_self_attn),
                    use_ffn=bool(use_ffn),
                    ffn_ratio=float(ffn_ratio),
                )
                for _ in range(self.num_layers)
            ]
        )

    def _build_queries(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if self.num_queries_per_class == 1:
            queries = text_tokens
        else:
            queries = text_tokens[:, :, None, :].expand(-1, -1, self.num_queries_per_class, -1)
            if self.tie_breaker is not None:
                queries = queries + self.tie_breaker[None, None, :, :]
            queries = queries.reshape(text_tokens.shape[0], self.num_classes * self.num_queries_per_class, self.dim)
        if self.query_delta is not None:
            queries = queries + self.query_delta.unsqueeze(0).to(dtype=queries.dtype, device=queries.device)
        return queries

    def _pool_kv(self, visual_feature: torch.Tensor) -> torch.Tensor:
        if self.kv_stride <= 1:
            return visual_feature
        return F.avg_pool2d(visual_feature, kernel_size=self.kv_stride, stride=self.kv_stride, ceil_mode=False)

    def forward(
        self,
        visual_feature: torch.Tensor,
        text_feature: torch.Tensor,
        visual_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if visual_feature.dim() != 4:
            raise ValueError(f"visual_feature must be 4D [B,C,H,W], got {tuple(visual_feature.shape)}")
        bsz, channels, height, width = visual_feature.shape
        if channels != self.dim:
            raise ValueError(f"visual_feature channels={channels} != grounding dim={self.dim}")
        if text_feature.dim() == 2:
            text_tokens = text_feature.unsqueeze(0).expand(bsz, -1, -1)
        elif text_feature.dim() == 3:
            if int(text_feature.shape[0]) != bsz:
                raise ValueError(f"text_feature batch={int(text_feature.shape[0])} != visual batch={bsz}")
            text_tokens = text_feature
        else:
            raise ValueError(f"text_feature must be [K,C] or [B,K,C], got {tuple(text_feature.shape)}")
        if int(text_tokens.shape[1]) != self.num_classes:
            raise ValueError(f"text_feature classes={int(text_tokens.shape[1])} != num_classes={self.num_classes}")
        if int(text_tokens.shape[-1]) != self.dim:
            raise ValueError(f"text_feature dim={int(text_tokens.shape[-1])} != grounding dim={self.dim}")
        text_tokens = text_tokens.to(device=visual_feature.device, dtype=visual_feature.dtype)

        image_tokens_full = visual_feature.flatten(2).transpose(1, 2)  # [B, HW, C]
        visual_feature_kv = self._pool_kv(visual_feature)
        kv_height, kv_width = visual_feature_kv.shape[-2:]
        image_tokens_kv = visual_feature_kv.flatten(2).transpose(1, 2)
        queries = self._build_queries(text_tokens)
        query_in = F.normalize(queries, dim=-1) if self.normalize_text else queries

        layer_aux: List[Dict[str, Any]] = []
        grounding_queries = query_in
        for layer in self.layers:
            grounding_queries, aux = layer(
                grounding_queries,
                image_tokens_kv,
                attn_temperature=self.attn_temperature,
                return_attn=self.return_attn,
            )
            layer_aux.append(aux)

        img_norm = F.normalize(image_tokens_full, dim=-1) if self.normalize_image else image_tokens_full
        query_norm = F.normalize(grounding_queries, dim=-1) if self.normalize_text else grounding_queries
        logits_all = torch.einsum("bnc,bqc->bnq", img_norm, query_norm)  # [B, HW, Q]
        if self.num_queries_per_class > 1:
            logits_all = logits_all.view(bsz, height * width, self.num_classes, self.num_queries_per_class).max(dim=-1).values
        text_grounding_logits = logits_all.transpose(1, 2).reshape(bsz, self.num_classes, height, width)
        text_grounding_logits = self.logit_scale.to(dtype=text_grounding_logits.dtype) * text_grounding_logits

        query_delta = grounding_queries - query_in
        q_delta_abs = query_delta.abs()
        q_base_abs = query_in.abs().mean().clamp_min(1e-8)
        attn_entropy_vals = [aux["attn_entropy"] for aux in layer_aux]
        attn_mean_vals = [aux["attn_mean"] for aux in layer_aux]
        attn_std_vals = [aux["attn_std"] for aux in layer_aux]
        attn_max_vals = [aux["attn_max_mean"] for aux in layer_aux]
        last_attn = layer_aux[-1].get("attn_prob", None) if layer_aux else None
        topk_area_ratio = 0.0
        if last_attn is not None:
            k = max(1, int(round(last_attn.shape[-1] * 0.05)))
            topk_vals = torch.topk(last_attn, k=k, dim=-1).values
            topk_area_ratio = float(topk_vals.sum(dim=-1).mean().detach().item())

        aux: Dict[str, Any] = {
            "gamma": float(self.gamma.detach().item()),
            "logit_scale": float(self.logit_scale.detach().item()),
            "attn_mean": float(sum(attn_mean_vals) / max(1, len(attn_mean_vals))),
            "attn_std": float(sum(attn_std_vals) / max(1, len(attn_std_vals))),
            "attn_entropy": float(sum(attn_entropy_vals) / max(1, len(attn_entropy_vals))),
            "attn_max_mean": float(sum(attn_max_vals) / max(1, len(attn_max_vals))),
            "topk_area_ratio": float(topk_area_ratio),
            "grounding_logits_mean": float(text_grounding_logits.mean().detach().item()),
            "grounding_logits_std": float(text_grounding_logits.std(unbiased=False).detach().item()),
            "query_delta_mean": float(q_delta_abs.mean().detach().item()),
            "query_delta_norm": float(query_delta.norm().detach().item()),
            "query_delta_ratio": float((q_delta_abs.mean() / q_base_abs).detach().item()),
            "kv_stride": float(self.kv_stride),
            "kv_hw": [int(kv_height), int(kv_width)],
            "attn_temperature": float(self.attn_temperature),
        }
        if visual_logits is not None:
            if visual_logits.shape != text_grounding_logits.shape:
                raise ValueError(
                    f"visual_logits shape={tuple(visual_logits.shape)} != grounding logits shape={tuple(text_grounding_logits.shape)}"
                )
            logits_delta = text_grounding_logits - visual_logits.to(dtype=text_grounding_logits.dtype)
            aux["grounding_visual_logits_delta_mean"] = float(logits_delta.mean().detach().item())
            aux["grounding_visual_logits_delta_norm"] = float(logits_delta.norm().detach().item())
        if self.return_attn and last_attn is not None:
            aux["attn_weights"] = last_attn
        return text_grounding_logits, aux


class TextPrototypeLogitHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_text: int,
        num_classes: int,
        adapter_hidden_dim: int = 512,
        adapter_dropout: float = 0.0,
        adapter_normalize_output: bool = True,
        adapter_zero_init_last: bool = False,
        adapter_projector_type: str = "mlp",
        normalize_visual: bool = True,
        normalize_text: bool = True,
        learnable_scale: bool = True,
        init_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.normalize_visual = bool(normalize_visual)
        self.normalize_text = bool(normalize_text)
        self.projector = TextPrototypeProjector(
            d_text=d_text,
            d_model=d_model,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
            normalize_output=adapter_normalize_output,
            zero_init_last=adapter_zero_init_last,
            projector_type=adapter_projector_type,
        )
        self.register_buffer("text_prototypes", torch.empty(num_classes, d_text), persistent=True)
        self.text_scale = nn.Parameter(torch.tensor(float(init_scale)), requires_grad=bool(learnable_scale))

    @torch.no_grad()
    def set_text_prototypes(self, protos: torch.Tensor) -> None:
        if protos.dim() != 2:
            raise ValueError(f"text prototypes must be 2D (C,D), got {tuple(protos.shape)}")
        self.text_prototypes.copy_(protos)

    def forward(self, pixel_feats: torch.Tensor) -> torch.Tensor:
        if pixel_feats.dim() != 4:
            raise ValueError(f"pixel_feats must be 4D (B,D,H,W), got {tuple(pixel_feats.shape)}")
        vis = pixel_feats
        txt = self.projector(self.text_prototypes)
        if self.normalize_visual:
            vis = F.normalize(vis, dim=1)
        if self.normalize_text:
            txt = F.normalize(txt, dim=-1)
        logits = torch.einsum("bdhw,cd->bchw", vis, txt)
        return self.text_scale * logits


class VisualTextLogitFusion(nn.Module):
    def __init__(self, alpha_init: float = 0.0, alpha_learnable: bool = True, alpha_max: Optional[float] = None) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)), requires_grad=bool(alpha_learnable))
        self.alpha_max = alpha_max

    def current_alpha(self) -> torch.Tensor:
        if self.alpha_max is None:
            return self.alpha
        return torch.clamp(self.alpha, min=-float(self.alpha_max), max=float(self.alpha_max))

    def forward(self, visual_logits: torch.Tensor, text_logits: torch.Tensor) -> torch.Tensor:
        return visual_logits + self.current_alpha() * text_logits

