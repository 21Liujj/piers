import utils.sam2_bootstrap  # noqa: F401

import os
import sys


def _sanitize_thread_env(name: str, default: str = "1") -> None:
    """libgomp rejects empty/invalid OMP_NUM_THREADS; force a positive int."""
    raw = os.environ.get(name, "")
    try:
        n = int(str(raw).strip())
        if n < 1:
            raise ValueError
    except (ValueError, TypeError):
        os.environ[name] = default


_sanitize_thread_env("OMP_NUM_THREADS")
_sanitize_thread_env("MKL_NUM_THREADS")

import argparse
import random
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import datetime
from typing import Any, Dict, List, Tuple, Optional

# Repo root contains a ./wandb/ run output directory; Python may treat it as a namespace
# package and shadow the real `wandb` (AttributeError: module 'wandb' has no attribute 'init').
# Prepend site-packages so the pip `wandb` package wins. Do not strip sys.path: some
# environments only have '' + repo, and removing both leaves no import path.
import site

for _d in reversed(getattr(site, "getsitepackages", lambda: [])()):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)
_usite = getattr(site, "getusersitepackages", lambda: None)()
if _usite and _usite not in sys.path:
    sys.path.insert(0, _usite)
import wandb

from adaptation.engine import Engine
from utils.wandb_console import attach_wandb_console_log

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    seed: int = field(
        default=1234,
        metadata={
            "help":"Random seed used for trainig"
        }
    )
    batch_size: int = field(
        default=2,
        metadata={
            "help":"Batchsize for training"
        }
    )
    amp: bool = field(
        default=False,
        metadata={
            "help": "Enable AMP (mixed precision) training when CUDA is available.",
        },
    )
    tf32: bool = field(
        default=True,
        metadata={
            "help": "Enable TF32 matmul on Ampere+ GPUs for faster training (CUDA only).",
        },
    )
    num_epochs: int = field(
        default=1,
        metadata={
            "help":"Number of training epochs"
        }
    )
    lr: float = field(
        default=2e-6,
        metadata={
            "help":"Learning rate"
        }
    )
    alpha: float = field(
        default=0.1,
        metadata={
            "help":"Weight for prototype pull loss"
        }
    )
    backbone: str = field(
        default="sam2_s",
        metadata={
            "help":"Backbone to use for training",
            "choices": ["sam2_t", "sam2_s", "sam2_b+", "sam2_l"],
        }
    )
    dataset: str = field(
        default="Cityscapes",
        metadata={
            "help":"Dataset to use for training",
            "choices": ["Cityscapes", "Stanford2D3D", "SynPASS"],
        }
    )
    class_set: int = field(
        default=13,
        metadata={
            "help": "Label space: 13 (legacy mapping) or 19 (Cityscapes TrainIds). 19 is only supported for Cityscapes.",
            "choices": [13, 19],
        },
    )
    num_maskmem: int = field(
        default=9,
        metadata={
            "help":"Capacity of memory bank (must match ckpt; omnisam_s.pth uses 9)"
        }
    )
    memory_bank_size: int = field(
        default=0,
        metadata={
            "help": "Override memory bank size (e.g. 9/6/4/2). 0 means use num_maskmem.",
        },
    )
    image_size: int = field(
        default=1024,
        metadata={
            "help":"Width (height) of the image patches"
        }
    )
    patch_stride_src: int = field(
        default=128,
        metadata={
            "help":"Stride for sliding window operation in source domain"
        }
    )
    patch_stride_tgt: int = field(
        default=384,
        metadata={
            "help":"Stride for sliding window operation in target domain"
        }
    )
    warmup_pl: int = field(
        default=2000,
        metadata={
            "help":"Number of pseudolabel generated for warmup epoch"
        }
    )
    pl_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size for pseudo-label generation dataloader (target domain).",
        },
    )
    pl_num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of dataloader workers for pseudo-label generation (0 for main process).",
        },
    )
    pl_shuffle: bool = field(
        default=False,
        metadata={"help": "Shuffle pseudo-label dataloader. Keep False for deterministic time ablation."},
    )
    pl_subset_file: str = field(
        default="",
        metadata={"help": "Optional fixed target subset file (one image path per line) for pseudo-label generation."},
    )
    pl_timing_warmup_batches: int = field(
        default=0,
        metadata={"help": "Warmup batches excluded from pseudo-label timing stats."},
    )
    train_timing_warmup_iters: int = field(
        default=0,
        metadata={"help": "Warmup iters excluded from training timing stats."},
    )
    train_max_iters: int = field(
        default=0,
        metadata={"help": "Max train iterations per epoch for time-ablation quick runs. 0 means full epoch."},
    )
    iters_pl: int = field(
        default=200,
        metadata={
            "help":"Number of pseudolabel generated for each epoch"
        }
    )
    uc_threshold: float = field(
        default=0.85,
        metadata={
            "help":"Threshold for uncertainty mask generation"
        }
    )
    conf_thr: float = field(
        default=0.85,
        metadata={
            "help": "Max-softmax confidence threshold for filtering pseudo-label pixels during training (loss_unsup).",
        },
    )
    ignore_index: int = field(
        default=255,
        metadata={
            "help": "Ignore index used in loss; low-confidence pixels will be set to this value.",
        },
    )
    data_root: str = field(
        default="data",
        metadata={
            "help":"Root of training data"
        }
    )
    stanford_processed_root: str = field(
        default="",
        metadata={
            "help": "Processed Stanford360_SPAN8 root (lists/images/labels_trainid). Overrides .../Stanford2d3d_Seg when set.",
        },
    )
    stanford_flat_layout: bool = field(
        default=False,
        metadata={
            "help": "Use flat Stanford360 layout with stanford_processed_root for PL/adapt/val builders.",
        },
    )
    stanford_legacy_area_crop: bool = field(
        default=False,
        metadata={
            "help": "Legacy Stanford panorama vertical crop (0,320,W,1728) before resize.",
        },
    )
    stanford_spin8_root: str = field(
        default="",
        metadata={
            "help": "SPin8 source root for StanfordPin8DataSet (default: data_root/Stanford2d3d_Seg).",
        },
    )
    pd_root: str = field(
        default="data_pd_cg_src_2000_01_ori",
        metadata={
            "help":"Root of pseudolabel data"
        }
    )
    depth_prior_root: str = field(
        default="",
        metadata={
            "help": "Optional offline depth prior root (contains depth_boundary_npy/...). "
                    "Used by depth_boundary_prior plugin during training.",
        },
    )
    tgt_ckpt: str = field(
        default="exp2/Cityscapes/best_sam2_s_10.pth",
        metadata={
            "help": "target model checkpoint path"
        }
    )
    src_ckpt: str = field(
        default="exp2/Cityscapes/best_sam2_s_10.pth",
        metadata={
            "help": "source model checkpoint path"
        }
    )
    reuse_pl_epoch: int = field(
        default=-1,
        metadata={
            "help": "Reuse pseudo-labels from this 1-based epoch index and skip generation for that epoch"
        }
    )
    resume_ckpt: str = field(
        default="",
        metadata={
            "help": "Resume adaptation from saved training state checkpoint"
        }
    )
    auto_rebuild_pseudo_labels: bool = field(
        default=True,
        metadata={
            "help": "Automatically rebuild pseudo labels when integrity check fails"
        }
    )
    validate_pseudo_labels: bool = field(
        default=True,
        metadata={
            "help": "Validate pseudo label completeness before training"
        }
    )
    reuse_pl_dir: str = field(
        default="",
        metadata={
            "help": "If set, always reuse pseudo-labels under this epoch directory (e.g. .../epoch1) for all epochs and skip generation."
        },
    )
    generate_only: bool = field(
        default=False,
        metadata={
            "help": "If set, only generate pseudo-labels for epoch1 and exit (no train/val).",
        },
    )
    regenerate_pl_each_epoch: bool = field(
        default=False,
        metadata={
            "help": "If True, always regenerate pseudo-labels for each epoch (no reuse of existing epoch dirs).",
        },
    )
    # -------------------------------------------------------------------------
    # Structural efficiency controls
    # -------------------------------------------------------------------------
    use_patch_scheduler: bool = field(
        default=False,
        metadata={"help": "Enable adaptive patch scheduler in pseudo-label generation."},
    )
    use_adaptive_patch_scheduler: bool = field(
        default=False,
        metadata={"help": "Alias of use_patch_scheduler (preferred by docs)."},
    )
    patch_scheduler_mode: str = field(
        default="none",
        metadata={"help": "Patch scheduler mode: none|fixed_stride|topk|two_stage|adaptive_stride."},
    )
    adaptive_score_type: str = field(
        default="hybrid",
        metadata={"help": "Alias of patch_score_source: entropy|laplacian|hybrid."},
    )
    adaptive_threshold: float = field(
        default=0.0,
        metadata={"help": "Complexity threshold for adaptive_stride mode (0 auto/median)."},
    )
    adaptive_stride_high_complexity: int = field(
        default=384,
        metadata={"help": "Stride (pixels) for high-complexity region."},
    )
    adaptive_stride_low_complexity: int = field(
        default=768,
        metadata={"help": "Stride (pixels) for low-complexity region."},
    )
    adaptive_apply_stage: str = field(
        default="pseudo_label",
        metadata={"help": "Apply stage: pseudo_label|train|adapt. First version uses pseudo_label."},
    )
    patch_stride_scale: float = field(
        default=1.0,
        metadata={"help": "Stride scale for fixed_stride mode (>=1)."},
    )
    patch_topk: int = field(
        default=0,
        metadata={"help": "Top-k patches in topk/two_stage mode; 0 means auto."},
    )
    patch_score_source: str = field(
        default="hybrid",
        metadata={"help": "Patch score source: entropy|boundary|hybrid."},
    )
    use_pl_update_controller: bool = field(
        default=False,
        metadata={"help": "Enable pseudo-label update controller."},
    )
    pl_update_mode: str = field(
        default="every_epoch",
        metadata={"help": "Pseudo-label update mode: every_epoch|every_n_epoch|score_based."},
    )
    pl_update_interval: int = field(
        default=1,
        metadata={"help": "Update interval when pl_update_mode=every_n_epoch."},
    )
    pl_score_threshold: float = field(
        default=0.5,
        metadata={"help": "Reserved threshold for score_based pseudo-label update mode."},
    )
    use_memory_compression: bool = field(
        default=False,
        metadata={"help": "Enable memory bank compression/selective retention."},
    )
    memory_select_mode: str = field(
        default="none",
        metadata={"help": "Memory select mode: none|recent_topk|score_topk|hybrid."},
    )
    memory_reduce_dim: int = field(
        default=0,
        metadata={"help": "Reserved memory feature reduce dim (0 disables)."},
    )
    memory_recent_keep: int = field(
        default=0,
        metadata={"help": "Recent memories kept when selection is enabled (0 auto)."},
    )
    memory_score_keep: int = field(
        default=0,
        metadata={"help": "Important memories kept when score selection is enabled (0 auto)."},
    )
    memory_compress_mode: str = field(
        default="none",
        metadata={"help": "Dense memory compression mode: none|pool|spatial|hybrid."},
    )
    memory_compress_location: str = field(
        default="before_attention",
        metadata={"help": "Compression insertion point: before_attention|before_bank."},
    )
    memory_downsample_ratio: int = field(
        default=2,
        metadata={"help": "Downsample ratio for pool mode."},
    )
    memory_spatial_window_size: int = field(
        default=2,
        metadata={"help": "Local window size for spatial/hybrid mode."},
    )
    memory_global_summary_tokens: int = field(
        default=0,
        metadata={"help": "Global summary token count for hybrid mode."},
    )
    mem_compression_enabled: bool = field(
        default=False,
        metadata={"help": "Enable EfficientTAM-style non-learnable memory compression."},
    )
    mem_compression_mode: str = field(
        default="pool",
        metadata={"help": "Compression mode: none|pool|spatial|hybrid."},
    )
    mem_compression_location: str = field(
        default="before_attention",
        metadata={"help": "Compression location: before_attention|before_bank."},
    )
    mem_compression_kernel_size: int = field(
        default=2,
        metadata={"help": "Pooling kernel size for memory compression."},
    )
    mem_compression_stride: int = field(
        default=2,
        metadata={"help": "Pooling stride for memory compression."},
    )
    mem_compression_keep_global_tokens: int = field(
        default=0,
        metadata={"help": "Hybrid mode global summary token count."},
    )
    mem_compression_apply_to_pos_enc: bool = field(
        default=True,
        metadata={"help": "Apply the same spatial pooling to memory position embeddings."},
    )
    mem_compression_min_tokens: int = field(
        default=64,
        metadata={"help": "Skip compression when memory token count is below this value."},
    )
    mem_compression_warmup_iters: int = field(
        default=0,
        metadata={"help": "Warmup iterations before enabling compression."},
    )
    mem_compression_enable_in_train: bool = field(
        default=True,
        metadata={"help": "Enable memory compression during UDA training iterations."},
    )
    mem_compression_enable_in_pseudo_label: bool = field(
        default=True,
        metadata={"help": "Enable memory compression during pseudo-label generation."},
    )
    mem_compression_train_kernel_size: int = field(
        default=2,
        metadata={"help": "Kernel size used during train stage."},
    )
    mem_compression_pl_kernel_size: int = field(
        default=2,
        metadata={"help": "Kernel size used during pseudo-label stage."},
    )
    mem_compression_verbose: bool = field(
        default=False,
        metadata={"help": "Print memory compression debug logs (throttled; eval 阶段从不打印)."},
    )
    mem_compression_log_interval: int = field(
        default=200,
        metadata={"help": "Log interval when mem_compression_verbose=True."},
    )
    use_lite_decoder: bool = field(
        default=False,
        metadata={"help": "Enable lightweight decoder variants."},
    )
    # ========== SCM (Seamless Circular Memory) Parameters ==========
    enable_scm: bool = field(
        default=False,
        metadata={"help": "Enable Seamless Circular Memory (circular padding for 360° panorama)."},
    )
    scm_pad_width: int = field(
        default=256,
        metadata={"help": "Circular padding width in pixels for SCM (recommended: 256-1024)."},
    )
    decoder_channel_ratio: float = field(
        default=1.0,
        metadata={"help": "Decoder internal channel ratio, e.g. 0.75/0.5."},
    )
    decoder_conv_mode: str = field(
        default="standard",
        metadata={"help": "Decoder conv mode: standard|depthwise."},
    )
    decoder_upsample_mode: str = field(
        default="bilinear",
        metadata={"help": "Decoder upsample mode: bilinear|nearest."},
    )

    # -------------------------------------------------------------------------
    # Multi-modal (RGB + Text) options
    # -------------------------------------------------------------------------
    text_proto_path: str = field(
        default="",
        metadata={
            "help": "Path to saved text prototypes (.pt) built by tools/build_text_prototypes.py. Empty disables text branch.",
        },
    )
    lambda_txt: float = field(
        default=0.0,
        metadata={
            "help": "Fusion weight: L_fuse = L_vis + lambda_txt * L_txt. Set >0 to enable fusion when text_proto_path is set.",
        },
    )
    txt_tau: float = field(
        default=0.07,
        metadata={
            "help": "Temperature used inside PixelTextAlignHead when computing cosine logits.",
        },
    )
    weight_txt_align: float = field(
        default=0.0,
        metadata={
            "help": "Weight for KL alignment loss between visual logits and text logits (L_txt_align).",
        },
    )
    txt_align_tau: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for KL alignment loss (softmax temperature).",
        },
    )
    text_proto_ablation: str = field(
        default="real",
        metadata={
            "help": "Deprecated alias for text_proto_mode: real|shuffle|random.",
        },
    )
    text_proto_mode: str = field(
        default="real",
        metadata={
            "help": "Text prototype mode: real|shuffle|random.",
        },
    )
    text_proto_shuffle_seed: int = field(
        default=0,
        metadata={"help": "Shuffle seed for text_proto_mode=shuffle."},
    )
    text_proto_random_seed: int = field(
        default=0,
        metadata={"help": "Random seed for text_proto_mode=random."},
    )
    text_proto_normalize: bool = field(
        default=True,
        metadata={"help": "L2-normalize loaded text prototypes."},
    )
    use_text_proto: bool = field(
        default=True,
        metadata={"help": "Enable text prototype branch globally."},
    )
    text_fusion_enabled: bool = field(
        default=False,
        metadata={"help": "Enable text fusion module."},
    )
    text_fusion_type: str = field(
        default="logit_add",
        metadata={"help": "Text fusion type: none|logit_add|image_to_text_cross_attn|text_to_image_grounding|hybrid."},
    )
    use_language_adapter: bool = field(
        default=False,
        metadata={"help": "Enable lightweight language adapter (stage-2+)."},
    )
    use_text_gate_ssl: bool = field(
        default=False,
        metadata={"help": "Enable text-aware pseudo-label gating for unsupervised loss."},
    )
    use_text_proto_align: bool = field(
        default=False,
        metadata={"help": "Enable text-assisted prototype alignment loss (stage-3)."},
    )
    use_proto_memory: bool = field(
        default=False,
        metadata={"help": "Enable prototype memory for stage-3 alignment."},
    )
    ssl_gate_mode: str = field(
        default="hard",
        metadata={"help": "Pseudo-label gate mode: hard|soft."},
    )
    tau_v: float = field(
        default=0.85,
        metadata={"help": "Visual confidence threshold for SSL gate."},
    )
    tau_t: float = field(
        default=0.10,
        metadata={"help": "Text similarity threshold for SSL gate."},
    )
    text_proto_refresh: str = field(
        default="cache",
        metadata={"help": "Text prototype refresh mode: cache|epoch."},
    )
    adapter_num_layers: int = field(
        default=1,
        metadata={"help": "Language adapter layer count (stage-2+)."},
    )
    adapter_num_heads: int = field(
        default=4,
        metadata={"help": "Language adapter attention heads (stage-2+)."},
    )
    adapter_scale: float = field(
        default=0.1,
        metadata={"help": "Language adapter residual scale."},
    )
    text_adapter_hidden_dim: int = field(
        default=512,
        metadata={"help": "Hidden dim for trainable text adapter/projector."},
    )
    text_adapter_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout in text adapter/projector."},
    )
    text_adapter_normalize_output: bool = field(
        default=True,
        metadata={"help": "L2-normalize projected text prototypes."},
    )
    text_adapter_zero_init_last: bool = field(
        default=False,
        metadata={"help": "Zero init last linear in text adapter."},
    )
    text_adapter_projector_type: str = field(
        default="mlp",
        metadata={"help": "Text projector type: mlp|linear."},
    )
    text_adapter_freeze: bool = field(
        default=False,
        metadata={"help": "Freeze text projector parameters for diagnosis."},
    )
    text_logit_normalize_visual: bool = field(
        default=True,
        metadata={"help": "Normalize visual feature before cosine text logits."},
    )
    text_logit_normalize_text: bool = field(
        default=True,
        metadata={"help": "Normalize projected text prototype before cosine logits."},
    )
    text_logit_learnable_scale: bool = field(
        default=True,
        metadata={"help": "Use learnable text logit scale parameter."},
    )
    text_logit_init_scale: float = field(
        default=10.0,
        metadata={"help": "Initial scale for text logits."},
    )
    text_fusion_alpha_init: float = field(
        default=0.0,
        metadata={"help": "Initial alpha for visual-text logit fusion."},
    )
    text_fusion_alpha_learnable: bool = field(
        default=True,
        metadata={"help": "Whether fusion alpha is learnable."},
    )
    text_fusion_alpha_max: str = field(
        default="",
        metadata={"help": "Optional clamp absolute max value for fusion alpha. Empty disables clamp."},
    )
    text_cross_num_heads: int = field(
        default=4,
        metadata={"help": "Number of heads for image-to-text cross attention."},
    )
    text_cross_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout for image-to-text cross attention."},
    )
    text_cross_beta_init: float = field(
        default=0.05,
        metadata={"help": "Initial beta for image-to-text feature fusion."},
    )
    text_cross_beta_learnable: bool = field(
        default=True,
        metadata={"help": "Whether cross-attention beta is learnable."},
    )
    text_cross_beta_max: str = field(
        default="",
        metadata={"help": "Optional clamp absolute max value for cross-attention beta."},
    )
    text_cross_use_ffn: bool = field(
        default=True,
        metadata={"help": "Use FFN after image-to-text cross attention."},
    )
    text_cross_ffn_ratio: float = field(
        default=4.0,
        metadata={"help": "FFN ratio for image-to-text cross attention."},
    )
    text_cross_norm_visual: bool = field(
        default=True,
        metadata={"help": "LayerNorm image tokens before cross attention."},
    )
    text_cross_norm_text: bool = field(
        default=True,
        metadata={"help": "LayerNorm text tokens before cross attention."},
    )
    text_cross_return_attn: bool = field(
        default=False,
        metadata={"help": "Return attention weights from cross attention for debug."},
    )
    text_cross_attn_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature applied to image queries before text cross attention (>0). Lower is sharper."},
    )
    text_ground_num_heads: int = field(
        default=4,
        metadata={"help": "Number of heads for text-to-image grounding attention."},
    )
    text_ground_num_layers: int = field(
        default=2,
        metadata={"help": "Number of transformer layers in text-to-image grounding."},
    )
    text_ground_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout used in text-to-image grounding."},
    )
    text_ground_gamma_init: float = field(
        default=0.05,
        metadata={"help": "Initial gamma for visual + grounding logits fusion."},
    )
    text_ground_gamma_learnable: bool = field(
        default=True,
        metadata={"help": "Whether grounding gamma is learnable."},
    )
    text_ground_logit_scale_init: float = field(
        default=10.0,
        metadata={"help": "Initial logit scale for text grounding logits."},
    )
    text_ground_logit_scale_learnable: bool = field(
        default=True,
        metadata={"help": "Whether grounding logit scale is learnable."},
    )
    text_ground_use_query_self_attn: bool = field(
        default=True,
        metadata={"help": "Use self-attention on text queries before cross-attention."},
    )
    text_ground_use_ffn: bool = field(
        default=True,
        metadata={"help": "Use FFN in grounding transformer blocks."},
    )
    text_ground_ffn_ratio: float = field(
        default=4.0,
        metadata={"help": "FFN ratio in grounding transformer blocks."},
    )
    text_ground_num_queries_per_class: int = field(
        default=1,
        metadata={"help": "Reserved number of queries per class for grounding."},
    )
    text_ground_use_tie_breaker: bool = field(
        default=False,
        metadata={"help": "Use learnable tie-breaker offsets when num_queries_per_class > 1."},
    )
    text_ground_query_delta_init: float = field(
        default=0.0,
        metadata={"help": "Initial value of learnable query delta."},
    )
    text_ground_use_learnable_query_delta: bool = field(
        default=False,
        metadata={"help": "Enable learnable query delta parameters."},
    )
    text_ground_kv_stride: int = field(
        default=1,
        metadata={"help": "Spatial pooling stride applied to image K/V tokens in grounding."},
    )
    text_ground_return_attn: bool = field(
        default=True,
        metadata={"help": "Return grounding attention maps for logging/debug."},
    )
    text_ground_attn_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature applied to text queries before image cross-attention (>0)."},
    )
    loss_text_grounding_enabled: bool = field(
        default=False,
        metadata={"help": "Enable CE loss on text grounding logits."},
    )
    loss_text_grounding_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for grounding CE loss."},
    )
    loss_text_grounding_apply_source: bool = field(
        default=True,
        metadata={"help": "Apply grounding CE loss on source labels."},
    )
    loss_text_grounding_apply_target: bool = field(
        default=False,
        metadata={"help": "Apply grounding CE loss on target pseudo labels."},
    )
    text_fusion_log_every: int = field(
        default=20,
        metadata={"help": "Logging interval hint for text fusion diagnostics."},
    )
    text_fusion_save_debug_vis: bool = field(
        default=False,
        metadata={"help": "Save text cross-attention debug visualizations."},
    )
    text_fusion_save_vis_num: int = field(
        default=20,
        metadata={"help": "Max number of text cross-attention debug visualizations."},
    )
    text_adapter_lr_mult: float = field(
        default=1.0,
        metadata={"help": "LR multiplier for text adapter/projector parameters."},
    )
    text_fusion_alpha_lr_mult: float = field(
        default=1.0,
        metadata={"help": "LR multiplier for fusion alpha parameter."},
    )
    text_scale_lr_mult: float = field(
        default=1.0,
        metadata={"help": "LR multiplier for text scale parameter."},
    )
    text_log_interval: int = field(
        default=50,
        metadata={"help": "Iteration interval for [text-logits]/[text-grad] prints and JSONL records."},
    )
    text_train_stage: str = field(
        default="stage2",
        metadata={"help": "Trainable parameter stage: stage1|stage2|stage3|all."},
    )
    lambda_ssl: float = field(
        default=1.0,
        metadata={"help": "Weight for unsupervised SSL term."},
    )
    lambda_fpa: float = field(
        default=1.0,
        metadata={"help": "Weight for visual prototype (FPA) term."},
    )
    lambda_pt: float = field(
        default=0.0,
        metadata={"help": "Weight for text-prototype alignment term (stage-3)."},
    )
    lambda_mem: float = field(
        default=0.0,
        metadata={"help": "Weight for prototype memory consistency term (stage-3)."},
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Enable Weights & Biases logging."},
    )
    use_tensorboard: bool = field(
        default=True,
        metadata={"help": "Enable TensorBoard logging."},
    )
    tb_log_dir: str = field(
        default="runs",
        metadata={"help": "TensorBoard log directory root."},
    )

    # -------------------------------------------------------------------------
    # Experiment config (YAML) overlay
    # -------------------------------------------------------------------------
    config: str = field(
        default="",
        metadata={"help": "Optional YAML config file to overlay args (OmegaConf)."},
    )
    opts: str = field(
        default="",
        metadata={"help": "Optional overrides: comma-separated key=value pairs, supports dotted keys (e.g. plugins.pseudo_conf_mask.enabled=true)."},
    )
    # -------------------------------------------------------------------------
    # SG-DRF (Spatial-Guided Dynamic Receptive Field) parameters
    # -------------------------------------------------------------------------
    enable_sgdrf: bool = field(
        default=False,
        metadata={"help": "Enable SG-DRF framework"},
    )
    enable_dynamic_fsmc: bool = field(
        default=True,
        metadata={"help": "Enable Dynamic FSMC module"},
    )
    enable_dynamic_scm: bool = field(
        default=True,
        metadata={"help": "Enable Dynamic SCM module"},
    )
    sgdrf_cold_start_epochs: int = field(
        default=2,
        metadata={"help": "Epochs for density-estimator cold start"},
    )
    sgdrf_density_weight: float = field(
        default=0.1,
        metadata={"help": "Density loss weight"},
    )
    sgdrf_density_lr: float = field(
        default=5e-5,
        metadata={"help": "Density estimator learning rate"},
    )
    sgdrf_grad_clip_norm: float = field(
        default=1.0,
        metadata={"help": "SGDRF gradient clipping max norm"},
    )
    sgdrf_fsmc_min_ratio: float = field(
        default=4.0,
        metadata={"help": "FSMC minimum compression ratio"},
    )
    sgdrf_fsmc_max_ratio: float = field(
        default=16.0,
        metadata={"help": "FSMC maximum compression ratio"},
    )
    sgdrf_scm_base_pad: int = field(
        default=1024,
        metadata={"help": "SCM base padding in source pixels"},
    )
    sgdrf_scm_min_pad: int = field(
        default=512,
        metadata={"help": "SCM minimum padding in source pixels"},
    )
    sgdrf_scm_max_pad: int = field(
        default=1536,
        metadata={"help": "SCM maximum padding in source pixels"},
    )


def _parse_scalar(v: str) -> Any:
    s = str(v).strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"none", "null"}:
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _set_by_dotted_key(root: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in str(dotted).split(".") if p]
    cur: Dict[str, Any] = root
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    if not parts:
        return
    cur[parts[-1]] = value


def _load_experiment_overrides(args: TrainConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load YAML config and CLI overrides.
    Returns: (merged_cfg_dict, plugins_cfg_dict)
    """
    merged: Dict[str, Any] = {}
    plugins_cfg: Dict[str, Any] = {}

    cfg_path = str(getattr(args, "config", "") or "").strip()
    if cfg_path:
        try:
            from omegaconf import OmegaConf  # dependency already used by sam/training
            cfg = OmegaConf.load(cfg_path)
            merged = OmegaConf.to_container(cfg, resolve=True) or {}
        except Exception as e:
            raise RuntimeError(f"Failed to load config YAML: {cfg_path} ({e})")

    # Apply --opts k=v,k2=v2 (dotted keys supported)
    raw_opts = str(getattr(args, "opts", "") or "").strip()
    if raw_opts:
        for chunk in [c.strip() for c in raw_opts.split(",") if c.strip()]:
            if "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)
            _set_by_dotted_key(merged, k.strip(), _parse_scalar(v))

    # Map top-level depth_confidence to plugin config if provided.
    if isinstance(merged, dict) and isinstance(merged.get("depth_confidence", None), dict):
        if not isinstance(merged.get("plugins", None), dict):
            merged["plugins"] = {}
        merged["plugins"]["depth_confidence"] = merged.get("depth_confidence", {})

    # Extract plugins subtree (optional)
    if isinstance(merged, dict) and isinstance(merged.get("plugins", None), dict):
        plugins_cfg = merged.get("plugins", {}) or {}

    return merged, plugins_cfg


def _flatten_text_multimodal_cfg(merged_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(merged_cfg, dict):
        return {}
    flat: Dict[str, Any] = {}
    multimodal = merged_cfg.get("multimodal", {})
    if isinstance(multimodal, dict) and "use_text_proto" in multimodal:
        flat["use_text_proto"] = multimodal.get("use_text_proto")

    text_proto = merged_cfg.get("text_proto", {})
    if isinstance(text_proto, dict):
        mapping = {
            "path": "text_proto_path",
            "mode": "text_proto_mode",
            "normalize": "text_proto_normalize",
            "shuffle_seed": "text_proto_shuffle_seed",
            "random_seed": "text_proto_random_seed",
        }
        for src, dst in mapping.items():
            if src in text_proto:
                flat[dst] = text_proto.get(src)

    text_adapter = merged_cfg.get("text_adapter", {})
    if isinstance(text_adapter, dict):
        mapping = {
            "hidden_dim": "text_adapter_hidden_dim",
            "dropout": "text_adapter_dropout",
            "normalize_output": "text_adapter_normalize_output",
            "zero_init_last": "text_adapter_zero_init_last",
            "projector_type": "text_adapter_projector_type",
            "freeze": "text_adapter_freeze",
        }
        for src, dst in mapping.items():
            if src in text_adapter:
                flat[dst] = text_adapter.get(src)

    text_fusion = merged_cfg.get("text_fusion", {})
    if isinstance(text_fusion, dict):
        mapping = {
            "enabled": "text_fusion_enabled",
            "type": "text_fusion_type",
            "log_every": "text_log_interval",
            "save_debug_vis": "text_fusion_save_debug_vis",
            "save_vis_num": "text_fusion_save_vis_num",
        }
        for src, dst in mapping.items():
            if src in text_fusion:
                flat[dst] = text_fusion.get(src)
        if "alpha_init" in text_fusion:
            flat["text_fusion_alpha_init"] = text_fusion.get("alpha_init")
        if "beta_init" in text_fusion:
            flat["text_cross_beta_init"] = text_fusion.get("beta_init")
        if "beta_learnable" in text_fusion:
            flat["text_cross_beta_learnable"] = text_fusion.get("beta_learnable")
        if "beta_max" in text_fusion:
            flat["text_cross_beta_max"] = text_fusion.get("beta_max")
        if "num_heads" in text_fusion:
            flat["text_cross_num_heads"] = text_fusion.get("num_heads")
        if "dropout" in text_fusion:
            flat["text_cross_dropout"] = text_fusion.get("dropout")
        if "use_ffn" in text_fusion:
            flat["text_cross_use_ffn"] = text_fusion.get("use_ffn")
        if "ffn_ratio" in text_fusion:
            flat["text_cross_ffn_ratio"] = text_fusion.get("ffn_ratio")
        if "norm_visual" in text_fusion:
            flat["text_cross_norm_visual"] = text_fusion.get("norm_visual")
        if "norm_text" in text_fusion:
            flat["text_cross_norm_text"] = text_fusion.get("norm_text")
        if "return_attn" in text_fusion:
            flat["text_cross_return_attn"] = text_fusion.get("return_attn")
        if "attn_temperature" in text_fusion:
            flat["text_cross_attn_temperature"] = text_fusion.get("attn_temperature")
        grounding_mapping = {
            "num_heads": "text_ground_num_heads",
            "num_layers": "text_ground_num_layers",
            "dropout": "text_ground_dropout",
            "gamma_init": "text_ground_gamma_init",
            "gamma_learnable": "text_ground_gamma_learnable",
            "logit_scale_init": "text_ground_logit_scale_init",
            "logit_scale_learnable": "text_ground_logit_scale_learnable",
            "use_query_self_attn": "text_ground_use_query_self_attn",
            "use_ffn": "text_ground_use_ffn",
            "ffn_ratio": "text_ground_ffn_ratio",
            "num_queries_per_class": "text_ground_num_queries_per_class",
            "use_tie_breaker": "text_ground_use_tie_breaker",
            "query_delta_init": "text_ground_query_delta_init",
            "use_learnable_query_delta": "text_ground_use_learnable_query_delta",
            "kv_stride": "text_ground_kv_stride",
            "return_attn": "text_ground_return_attn",
            "ground_attn_temperature": "text_ground_attn_temperature",
        }
        if str(text_fusion.get("type", "")).lower() == "text_to_image_grounding":
            for src, dst in grounding_mapping.items():
                if src in text_fusion:
                    flat[dst] = text_fusion.get(src)

    text_gate = merged_cfg.get("text_gate", {})
    if isinstance(text_gate, dict) and "enabled" in text_gate:
        flat["use_text_gate_ssl"] = bool(text_gate.get("enabled"))
    loss_text_grounding = merged_cfg.get("loss_text_grounding", {})
    if isinstance(loss_text_grounding, dict):
        mapping = {
            "enabled": "loss_text_grounding_enabled",
            "weight": "loss_text_grounding_weight",
            "apply_source": "loss_text_grounding_apply_source",
            "apply_target": "loss_text_grounding_apply_target",
        }
        for src, dst in mapping.items():
            if src in loss_text_grounding:
                flat[dst] = loss_text_grounding.get(src)
    return flat

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_arg() -> TrainConfig:
    parser = argparse.ArgumentParser(description="OmniSAM Training (Refactored)")
    for f in TrainConfig.__dataclass_fields__.values():
        if f.type == bool:
            parser.add_argument(
                f"--{f.name}",
                action=argparse.BooleanOptionalAction,
                default=f.default,
            )
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = parser.parse_args()
    return TrainConfig(**vars(args))

def main():
    args = parse_arg()
    merged_cfg, plugins_cfg = _load_experiment_overrides(args)
    # Overlay top-level keys from YAML onto args (only if attribute exists in args).
    # Unknown keys are ignored for backward compatibility.
    if isinstance(merged_cfg, dict):
        flat_text_cfg = _flatten_text_multimodal_cfg(merged_cfg)
        for k, v in flat_text_cfg.items():
            if hasattr(args, k):
                try:
                    setattr(args, k, v)
                except Exception:
                    pass
        if isinstance(merged_cfg.get("depth_prior", None), dict):
            setattr(args, "depth_prior", merged_cfg.get("depth_prior", {}))
        if isinstance(merged_cfg.get("depth_adapter", None), dict):
            setattr(args, "depth_adapter", merged_cfg.get("depth_adapter", {}))
        if isinstance(merged_cfg.get("pseudo_label_reliability", None), dict):
            setattr(args, "pseudo_label_reliability",
                    merged_cfg.get("pseudo_label_reliability", {}))
        if isinstance(merged_cfg.get("pseudo_label_refiner", None), dict):
            setattr(args, "pseudo_label_refiner",
                    merged_cfg.get("pseudo_label_refiner", {}))
        for k, v in merged_cfg.items():
            if k == "plugins":
                continue
            if hasattr(args, k):
                try:
                    setattr(args, k, v)
                except Exception:
                    pass
    # Attach plugin config as a dynamic attribute (dataclass has no dict field).
    setattr(args, "plugins_cfg", plugins_cfg)

    # aliases / compatibility knobs
    if bool(getattr(args, "use_adaptive_patch_scheduler", False)):
        args.use_patch_scheduler = True
    if getattr(args, "adaptive_score_type", ""):
        args.patch_score_source = str(args.adaptive_score_type)
    if int(getattr(args, "memory_bank_size", 0)) > 0:
        args.num_maskmem = int(args.memory_bank_size)
    # Backward/forward compatibility for memory compression options.
    if bool(getattr(args, "use_memory_compression", False)):
        args.mem_compression_enabled = True
    if str(getattr(args, "memory_compress_mode", "none")).lower() != "none":
        args.mem_compression_mode = str(getattr(args, "memory_compress_mode"))
    if str(getattr(args, "memory_compress_location", "before_attention")).lower() != "before_attention":
        args.mem_compression_location = str(getattr(args, "memory_compress_location"))
    if int(getattr(args, "memory_downsample_ratio", 2)) != 2:
        args.mem_compression_kernel_size = int(getattr(args, "memory_downsample_ratio", 2))
        args.mem_compression_stride = int(getattr(args, "memory_downsample_ratio", 2))
    if int(getattr(args, "memory_spatial_window_size", 2)) != 2 and args.mem_compression_mode in {"spatial", "hybrid"}:
        args.mem_compression_kernel_size = int(getattr(args, "memory_spatial_window_size", 2))
        args.mem_compression_stride = int(getattr(args, "memory_spatial_window_size", 2))
    if int(getattr(args, "memory_global_summary_tokens", 0)) > 0:
        args.mem_compression_keep_global_tokens = int(getattr(args, "memory_global_summary_tokens", 0))
    # compatibility: old flag text_proto_ablation maps to text_proto_mode
    if str(getattr(args, "text_proto_mode", "") or "").strip() == "":
        args.text_proto_mode = args.text_proto_ablation
    if str(getattr(args, "text_proto_ablation", "") or "").strip() in {"none", "shuffle", "random"}:
        args.text_proto_mode = {"none": "real", "shuffle": "shuffle", "random": "random"}[
            str(getattr(args, "text_proto_ablation")).strip()
        ]
    valid_text_proto_mode = {"real", "shuffle", "random"}
    if args.text_proto_mode not in valid_text_proto_mode:
        raise ValueError(
            f"text_proto_mode must be one of {sorted(valid_text_proto_mode)}, "
            f"got: {args.text_proto_mode}"
        )
    if args.text_train_stage not in {"stage1", "stage2", "stage3", "all"}:
        raise ValueError(f"text_train_stage must be stage1|stage2|stage3|all, got {args.text_train_stage}")
    if args.text_adapter_projector_type not in {"mlp", "linear"}:
        raise ValueError(f"text_adapter_projector_type must be mlp|linear, got {args.text_adapter_projector_type}")
    if args.text_fusion_type not in {"none", "logit_add", "image_to_text_cross_attn", "text_to_image_grounding", "hybrid"}:
        raise ValueError(
            f"text_fusion_type must be none|logit_add|image_to_text_cross_attn|text_to_image_grounding|hybrid, got {args.text_fusion_type}"
        )
    if float(getattr(args, "text_cross_attn_temperature", 1.0)) <= 0:
        raise ValueError("text_cross_attn_temperature must be > 0")
    if float(getattr(args, "text_ground_attn_temperature", 1.0)) <= 0:
        raise ValueError("text_ground_attn_temperature must be > 0")
    if args.text_fusion_type == "image_to_text_cross_attn" and float(getattr(args, "text_fusion_alpha_init", 0.0)) != 0.0:
        print(
            "[text_fusion] text_fusion_alpha_init is for logit_add; image_to_text_cross_attn uses text_cross_beta_init.",
            flush=True,
        )
    if args.ssl_gate_mode not in {"hard", "soft"}:
        raise ValueError(f"ssl_gate_mode must be hard|soft, got {args.ssl_gate_mode}")
    if args.text_proto_refresh not in {"cache", "epoch"}:
        raise ValueError(f"text_proto_refresh must be cache|epoch, got {args.text_proto_refresh}")
    if args.patch_scheduler_mode not in {"none", "fixed_stride", "topk", "two_stage", "adaptive_stride"}:
        raise ValueError(f"patch_scheduler_mode must be none|fixed_stride|topk|two_stage|adaptive_stride, got {args.patch_scheduler_mode}")
    if args.adaptive_apply_stage not in {"pseudo_label", "train", "adapt"}:
        raise ValueError(f"adaptive_apply_stage must be pseudo_label|train|adapt, got {args.adaptive_apply_stage}")
    if args.pl_update_mode not in {"every_epoch", "every_n_epoch", "score_based"}:
        raise ValueError(f"pl_update_mode must be every_epoch|every_n_epoch|score_based, got {args.pl_update_mode}")
    if args.memory_select_mode not in {"none", "recent_topk", "score_topk", "hybrid"}:
        raise ValueError(f"memory_select_mode must be none|recent_topk|score_topk|hybrid, got {args.memory_select_mode}")
    if args.memory_compress_mode not in {"none", "pool", "spatial", "hybrid"}:
        raise ValueError(f"memory_compress_mode must be none|pool|spatial|hybrid, got {args.memory_compress_mode}")
    if args.memory_compress_location not in {"before_attention", "before_bank"}:
        raise ValueError(f"memory_compress_location must be before_attention|before_bank, got {args.memory_compress_location}")
    if args.mem_compression_mode not in {"none", "pool", "spatial", "hybrid"}:
        raise ValueError(f"mem_compression_mode must be none|pool|spatial|hybrid, got {args.mem_compression_mode}")
    if args.mem_compression_location not in {"before_attention", "before_bank"}:
        raise ValueError(
            f"mem_compression_location must be before_attention|before_bank, got {args.mem_compression_location}"
        )
    if args.decoder_conv_mode not in {"standard", "depthwise"}:
        raise ValueError(f"decoder_conv_mode must be standard|depthwise, got {args.decoder_conv_mode}")
    if args.decoder_upsample_mode not in {"bilinear", "nearest"}:
        raise ValueError(f"decoder_upsample_mode must be bilinear|nearest, got {args.decoder_upsample_mode}")
    set_seed(args.seed)
    if torch.cuda.is_available() and getattr(args, "tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if args.class_set == 19 and args.dataset != "Cityscapes":
        raise ValueError("class_set=19 is only supported for dataset=Cityscapes in this repo.")
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if bool(getattr(args, "use_wandb", False)):
        wandb.init(
            project="OmniSAM",
            name=dt,
            group=args.backbone + "_" + args.dataset + f"_{args.class_set}cls",
            config=vars(args)
        )
        _clog = attach_wandb_console_log()
        if _clog is not None:
            print(f"[wandb] Console log (stdout/stderr mirror): {_clog}", flush=True)
    else:
        print("[logger] wandb disabled, using TensorBoard only.", flush=True)

    # Print plugin enablement early (before Engine.setup) for ablations.
    try:
        from adaptation.plugins.manager import PluginManager

        pm = PluginManager.from_args(args=args, num_classes=0, device=torch.device("cpu"))
        print(pm.format_enabled_plugins(prefix="[plugins] "), flush=True)
    except Exception:
        # Do not block training if plugins package is missing/partially installed.
        pass

    engine = Engine(args)
    engine.setup()

    if getattr(args, "generate_only", False):
        # Generate epoch1 pseudo-labels under pd_root and exit.
        engine.generate_pseudo_labels(epoch=0)
        print("Pseudo-label generation finished (generate_only=True).")
        if bool(getattr(args, "use_wandb", False)):
            wandb.finish()
        if getattr(engine, "tb_writer", None) is not None:
            engine.tb_writer.close()
        return

    for epoch in range(engine.start_epoch, args.num_epochs):
        epoch_idx = epoch + 1
        if getattr(args, "reuse_pl_dir", ""):
            engine.save_dir = args.reuse_pl_dir
            engine.is_warmup = True if epoch == 0 else False
            print(f"Reuse fixed pseudo-label dir for epoch {epoch_idx}: {engine.save_dir}")
            engine.train(epoch)
            engine.validate(epoch)
            continue

        pl_dir = engine._pl_epoch_dir(epoch)
        should_reuse = False
        # Structural pseudo-label update controller (algorithm-level reduction)
        pl_decision = engine.get_pl_update_decision(epoch)
        if not pl_decision.should_update:
            reuse_dir = engine._pl_epoch_dir(pl_decision.reuse_epoch)
            engine.save_dir = reuse_dir
            engine.is_warmup = True if epoch == 0 else False
            print(
                f"Skip pseudo-label update for epoch {epoch_idx}; "
                f"reuse epoch {pl_decision.reuse_epoch + 1}: {reuse_dir} "
                f"(reason={pl_decision.reason})"
            )
            engine.train(epoch)
            engine.validate(epoch)
            continue

        if epoch < engine.start_epoch:
            should_reuse = True
        elif args.reuse_pl_epoch == epoch_idx:
            should_reuse = True
        elif getattr(args, "regenerate_pl_each_epoch", False):
            should_reuse = False
        elif engine.has_valid_pseudo_labels(epoch):
            should_reuse = True

        if should_reuse:
            engine.save_dir = pl_dir
            engine.is_warmup = True if epoch == 0 else False
            print(f"Reuse existing pseudo-labels for epoch {epoch_idx}: {pl_dir}")
        else:
            engine.generate_pseudo_labels(epoch)

        engine.train(epoch)
        engine.validate(epoch)
    
    if bool(getattr(args, "use_wandb", False)):
        wandb.finish()
    if getattr(engine, "tb_writer", None) is not None:
        engine.tb_writer.close()
    print('Training Finished.')

if __name__ == "__main__":
    main()