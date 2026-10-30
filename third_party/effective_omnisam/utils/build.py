import os
import random
import time
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader

from sam.build_sam import build_sam2
from model.model import OmniSAM, OmniSAM_adapt
from model.text_align import load_text_prototypes

from dataset.Cityscapes import Cityscapes13, Cityscapes19
from dataset.WildPASS import WildPASSforPL, WildPASSforAdaptation, WildPASSforAdaptation19
from dataset.DensePASS import DensePASS13, DensePASS19
from dataset.DensePASS_circular import DensePASS13Circular, DensePASS19Circular
from dataset.SynPASS import SynPASS13
from dataset.StanfordPin import StanfordPin8DataSet
from dataset.StanfordPan import StanfordPan8forPL, StanfordPan8forAdaptation, StanfordPan8forVal
from dataset.Stanford360Span8 import (
    Stanford360FlatAdaptation,
    Stanford360PanoramaPL,
    Stanford360PanoramaVal,
)

from utils.label_spaces import CITYSCAPES13_CLASSES, CITYSCAPES19_CLASSES
from utils.checkpoint_adapt import smart_load_checkpoint

def _seed_worker(_):
    seed = int(time.time() * 1000) % (2**32)
    random.seed(seed)


def _resolve_data_root(data_root: str, repo_root: str) -> str:
    return data_root if os.path.isabs(data_root) else os.path.join(repo_root, data_root)


def _num_workers(default_workers: int) -> int:
    cpu_count = os.cpu_count() or default_workers
    return max(1, min(default_workers, cpu_count))


DATASET_NAME2CLASSES_13 = {
    'Stanford2D3D': ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"],
    'Cityscapes': CITYSCAPES13_CLASSES,
    'SynPASS': CITYSCAPES13_CLASSES,
    'WildPASS': CITYSCAPES13_CLASSES,
    'DensePASS': CITYSCAPES13_CLASSES,
}

DATASET_NAME2CLASSES_19 = {
    'Cityscapes': CITYSCAPES19_CLASSES,
    'DensePASS': CITYSCAPES19_CLASSES,
    'WildPASS': CITYSCAPES19_CLASSES,
}


def resolve_class_names(dataset_name: str, class_set: int):
    if class_set == 13:
        return DATASET_NAME2CLASSES_13[dataset_name]
    if class_set == 19:
        if dataset_name not in DATASET_NAME2CLASSES_19:
            raise ValueError(f"dataset={dataset_name} does not support class_set=19 in this repo")
        return DATASET_NAME2CLASSES_19[dataset_name]
    raise ValueError(f"Unsupported class_set={class_set}")

# backbone registry: backbone -> (checkpoint_path, yaml_cfg_path)
SAM_CFGS: Dict[str, Tuple[str, str]] = {
    "sam2_l": (
        "sam/checkpoints/sam2.1_hiera_large.pt",
        "/configs/sam2.1/sam2.1_hiera_l.yaml",
    ),
    "sam2_b+": (
        "sam/checkpoints/sam2.1_hiera_base_plus.pt",
        "/configs/sam2.1/sam2.1_hiera_b+.yaml",
    ),
    "sam2_s": (
        "sam/checkpoints/sam2.1_hiera_small.pt",
        "/configs/sam2.1/sam2.1_hiera_s.yaml",
    ),
    "sam2_t": (
        "sam/checkpoints/sam2.1_hiera_tiny.pt",
        "/configs/sam2.1/sam2.1_hiera_t.yaml",
    ),
}


def build_model(args, num_classes, num_maskmem, device):
    sam2_checkpoint, model_cfg = SAM_CFGS[args.backbone]
    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    text_prototypes = None
    d_txt = None
    use_text_proto = bool(getattr(args, "use_text_proto", True))
    proto_path = (getattr(args, "text_proto_path", "") or "") if use_text_proto else ""
    if proto_path:
        mode = str(getattr(args, "text_proto_mode", getattr(args, "text_proto_ablation", "real")))
        text_prototypes, text_meta = load_text_prototypes(
            path=proto_path,
            num_classes=int(num_classes),
            mode=mode,
            normalize=bool(getattr(args, "text_proto_normalize", True)),
            shuffle_seed=int(getattr(args, "text_proto_shuffle_seed", getattr(args, "seed", 0))),
            random_seed=int(getattr(args, "text_proto_random_seed", getattr(args, "seed", 0))),
        )
        print(
            f"[text_proto] path={proto_path} mode={mode} shape={tuple(text_prototypes.shape)} "
            f"norm_mean={text_meta.get('text_proto_norm_mean', 0.0):.6f}",
            flush=True,
        )
        if "shuffle_index" in text_meta:
            print(f"[text_proto] shuffle_index={text_meta['shuffle_index']}", flush=True)
        if "random_seed" in text_meta:
            print(f"[text_proto] random_seed={text_meta['random_seed']}", flush=True)
        if "source_mode" in text_meta:
            print(
                f"[text_proto] source_mode={text_meta.get('source_mode')} "
                f"text_model={text_meta.get('text_model')} pretrained={text_meta.get('text_pretrained')}",
                flush=True,
            )
        d_txt = int(text_prototypes.shape[-1])

    model = OmniSAM_adapt(
        sam,
        num_classes=num_classes,
        num_maskmem=num_maskmem,
        text_prototypes=text_prototypes,
        d_txt=d_txt,
        lambda_txt=float(getattr(args, "lambda_txt", 0.0)),
        txt_tau=float(getattr(args, "txt_tau", 0.07)),
        use_language_adapter=bool(getattr(args, "use_language_adapter", False)),
        adapter_num_heads=int(getattr(args, "adapter_num_heads", 4)),
        adapter_scale=float(getattr(args, "adapter_scale", 0.1)),
        use_lite_decoder=bool(getattr(args, "use_lite_decoder", False)),
        decoder_channel_ratio=float(getattr(args, "decoder_channel_ratio", 1.0)),
        decoder_conv_mode=str(getattr(args, "decoder_conv_mode", "standard")),
        decoder_upsample_mode=str(getattr(args, "decoder_upsample_mode", "bilinear")),
        text_adapter_hidden_dim=int(getattr(args, "text_adapter_hidden_dim", 512)),
        text_adapter_dropout=float(getattr(args, "text_adapter_dropout", 0.0)),
        text_adapter_normalize_output=bool(getattr(args, "text_adapter_normalize_output", True)),
        text_adapter_zero_init_last=bool(getattr(args, "text_adapter_zero_init_last", False)),
        text_adapter_projector_type=str(getattr(args, "text_adapter_projector_type", "mlp")),
        text_adapter_freeze=bool(getattr(args, "text_adapter_freeze", False)),
        text_logit_normalize_visual=bool(getattr(args, "text_logit_normalize_visual", True)),
        text_logit_normalize_text=bool(getattr(args, "text_logit_normalize_text", True)),
        text_logit_learnable_scale=bool(getattr(args, "text_logit_learnable_scale", True)),
        text_logit_init_scale=float(getattr(args, "text_logit_init_scale", 10.0)),
        text_fusion_alpha_init=float(getattr(args, "text_fusion_alpha_init", 0.0)),
        text_fusion_alpha_learnable=bool(getattr(args, "text_fusion_alpha_learnable", True)),
        text_fusion_alpha_max=(
            None if getattr(args, "text_fusion_alpha_max", None) in (None, "", "none", "null") else float(getattr(args, "text_fusion_alpha_max"))
        ),
        text_fusion_enabled=bool(getattr(args, "text_fusion_enabled", False)),
        text_fusion_type=str(getattr(args, "text_fusion_type", "logit_add")),
        text_cross_num_heads=int(getattr(args, "text_cross_num_heads", 4)),
        text_cross_dropout=float(getattr(args, "text_cross_dropout", 0.0)),
        text_cross_beta_init=float(getattr(args, "text_cross_beta_init", 0.05)),
        text_cross_beta_learnable=bool(getattr(args, "text_cross_beta_learnable", True)),
        text_cross_beta_max=(
            None if getattr(args, "text_cross_beta_max", None) in (None, "", "none", "null") else float(getattr(args, "text_cross_beta_max"))
        ),
        text_cross_use_ffn=bool(getattr(args, "text_cross_use_ffn", True)),
        text_cross_ffn_ratio=float(getattr(args, "text_cross_ffn_ratio", 4.0)),
        text_cross_norm_visual=bool(getattr(args, "text_cross_norm_visual", True)),
        text_cross_norm_text=bool(getattr(args, "text_cross_norm_text", True)),
        text_cross_return_attn=bool(getattr(args, "text_cross_return_attn", False)),
        text_cross_attn_temperature=float(getattr(args, "text_cross_attn_temperature", 1.0)),
        text_ground_num_heads=int(getattr(args, "text_ground_num_heads", 4)),
        text_ground_num_layers=int(getattr(args, "text_ground_num_layers", 2)),
        text_ground_dropout=float(getattr(args, "text_ground_dropout", 0.0)),
        text_ground_gamma_init=float(getattr(args, "text_ground_gamma_init", 0.05)),
        text_ground_gamma_learnable=bool(getattr(args, "text_ground_gamma_learnable", True)),
        text_ground_logit_scale_init=float(getattr(args, "text_ground_logit_scale_init", 10.0)),
        text_ground_logit_scale_learnable=bool(getattr(args, "text_ground_logit_scale_learnable", True)),
        text_ground_use_query_self_attn=bool(getattr(args, "text_ground_use_query_self_attn", True)),
        text_ground_use_ffn=bool(getattr(args, "text_ground_use_ffn", True)),
        text_ground_ffn_ratio=float(getattr(args, "text_ground_ffn_ratio", 4.0)),
        text_ground_num_queries_per_class=int(getattr(args, "text_ground_num_queries_per_class", 1)),
        text_ground_use_tie_breaker=bool(getattr(args, "text_ground_use_tie_breaker", False)),
        text_ground_query_delta_init=float(getattr(args, "text_ground_query_delta_init", 0.0)),
        text_ground_use_learnable_query_delta=bool(getattr(args, "text_ground_use_learnable_query_delta", False)),
        text_ground_kv_stride=int(getattr(args, "text_ground_kv_stride", 1)),
        text_ground_return_attn=bool(getattr(args, "text_ground_return_attn", True)),
        text_ground_attn_temperature=float(getattr(args, "text_ground_attn_temperature", 1.0)),
        enable_sgdrf=bool(getattr(args, "enable_sgdrf", False)),
        sgdrf_backbone_channels=int(getattr(args, "sgdrf_backbone_channels", 256)),
        sgdrf_enable_dynamic_fsmc=bool(getattr(args, "enable_dynamic_fsmc", True)),
        sgdrf_enable_dynamic_scm=bool(getattr(args, "enable_dynamic_scm", True)),
        sgdrf_fsmc_min_ratio=float(getattr(args, "sgdrf_fsmc_min_ratio", 4.0)),
        sgdrf_fsmc_max_ratio=float(getattr(args, "sgdrf_fsmc_max_ratio", 16.0)),
        sgdrf_scm_base_pad=int(getattr(args, "sgdrf_scm_base_pad", 1024)),
        sgdrf_scm_min_pad=int(getattr(args, "sgdrf_scm_min_pad", 512)),
        sgdrf_scm_max_pad=int(getattr(args, "sgdrf_scm_max_pad", 1536)),
        sgdrf_training_stage=str(getattr(args, "sgdrf_training_stage", "joint")),
        depth_adapter_cfg=(
            getattr(args, "depth_adapter", {})
            if isinstance(getattr(args, "depth_adapter", {}), dict)
            else {}
        ),
    ).to(device)
    mem_enabled = bool(getattr(args, "mem_compression_enabled", False)) or bool(
        getattr(args, "use_memory_compression", False)
    )
    if mem_enabled:
        model.memory_bank.configure_compression(
            enabled=mem_enabled,
            select_mode=str(getattr(args, "memory_select_mode", "none")),
            recent_keep=int(getattr(args, "memory_recent_keep", 0)),
            score_keep=int(getattr(args, "memory_score_keep", 0)),
            compress_mode=str(getattr(args, "mem_compression_mode", getattr(args, "memory_compress_mode", "none"))),
            compress_location=str(
                getattr(args, "mem_compression_location", getattr(args, "memory_compress_location", "before_attention"))
            ),
            downsample_ratio=int(getattr(args, "memory_downsample_ratio", 2)),
            spatial_window_size=int(getattr(args, "memory_spatial_window_size", 2)),
            global_summary_tokens=int(getattr(args, "memory_global_summary_tokens", 0)),
            kernel_size=int(getattr(args, "mem_compression_kernel_size", 2)),
            stride=int(getattr(args, "mem_compression_stride", 2)),
            keep_global_tokens=int(getattr(args, "mem_compression_keep_global_tokens", 0)),
            apply_to_pos_enc=bool(getattr(args, "mem_compression_apply_to_pos_enc", True)),
            min_tokens=int(getattr(args, "mem_compression_min_tokens", 64)),
            warmup_iters=int(getattr(args, "mem_compression_warmup_iters", 0)),
            enable_in_train=bool(getattr(args, "mem_compression_enable_in_train", True)),
            enable_in_pseudo_label=bool(getattr(args, "mem_compression_enable_in_pseudo_label", True)),
            train_kernel_size=int(getattr(args, "mem_compression_train_kernel_size", 2)),
            pl_kernel_size=int(getattr(args, "mem_compression_pl_kernel_size", 2)),
            verbose=bool(getattr(args, "mem_compression_verbose", False)),
            log_interval=int(getattr(args, "mem_compression_log_interval", 200)),
        )
    return model

def load_checkpoint(model, ckpt_path):
    if ckpt_path is None or ckpt_path == '':
        return
    smart_load_checkpoint(model, ckpt_path)

class DatasetBuilder:
    def __init__(self, args):
        self.args = args
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_root = _resolve_data_root(args.data_root, self.repo_root)

    def _stanford_seg_root(self) -> str:
        r = str(getattr(self.args, "stanford_processed_root", "") or "").strip()
        if r:
            return r if os.path.isabs(r) else os.path.join(self.repo_root, r)
        return os.path.join(self.data_root, "Stanford2d3d_Seg")

    def _stanford_spin8_root(self) -> str:
        r = str(getattr(self.args, "stanford_spin8_root", "") or "").strip()
        if r:
            return r if os.path.isabs(r) else os.path.join(self.repo_root, r)
        return os.path.join(self.data_root, "Stanford2d3d_Seg")

    def _stanford_flat(self) -> bool:
        return bool(getattr(self.args, "stanford_flat_layout", False)) and bool(
            str(getattr(self.args, "stanford_processed_root", "") or "").strip()
        )

    # ------------------------------
    def build_tgt_for_pl(self):
        if self.args.dataset in ["SynPASS", "Cityscapes"]:
            depth_prior_cfg = getattr(self.args, "depth_prior", {}) if isinstance(getattr(self.args, "depth_prior", {}), dict) else {}
            depth_prior_enabled = bool(depth_prior_cfg.get("enable", False))
            depth_prior_root = str(depth_prior_cfg.get("depth_dir", "") or getattr(self.args, "depth_prior_root", "") or "").strip()
            depth_file_exts = depth_prior_cfg.get("file_exts", [".png", ".jpg", ".jpeg", ".npy", ".pt"])
            depth_normalize = str(depth_prior_cfg.get("normalize", "minmax"))
            if not depth_prior_enabled:
                depth_prior_root = ""
            dataset_name = 'WildPASS'
            class_set = getattr(self.args, "class_set", 13)
            class_names = resolve_class_names(dataset_name, class_set)
            num_classes = len(class_names)
            root_syn = os.path.join(self.data_root, 'WildPASS')
            val_list = 'dataset/densepass_list/train.txt'
            tgt_train_dataset = WildPASSforPL(
                root_syn,
                val_list,
                crop_size=(4096, 1024),
                sliding_window=(384, 1024),
                depth_prior_root=depth_prior_root,
                depth_file_exts=depth_file_exts,
                depth_normalize=depth_normalize,
            )
            pl_bs = int(getattr(self.args, "pl_batch_size", 1))
            pl_nw = int(getattr(self.args, "pl_num_workers", 4))
            tgt_train_loader = DataLoader(
                tgt_train_dataset,
                batch_size=pl_bs,
                shuffle=bool(getattr(self.args, "pl_shuffle", False)),
                num_workers=_num_workers(pl_nw),
                pin_memory=True,
            )
        elif self.args.dataset == 'Stanford2D3D':
            class_names = DATASET_NAME2CLASSES_13['Stanford2D3D']
            num_classes = len(class_names)
            root_syn = self._stanford_seg_root()
            if self._stanford_flat():
                val_list = os.path.join(root_syn, 'lists', 'train.txt')
                tgt_train_dataset = Stanford360PanoramaPL(
                    root_syn,
                    val_list,
                    crop_size=(3072, 1024),
                    sliding_window=(256, 1024),
                    legacy_area_crop=bool(getattr(self.args, "stanford_legacy_area_crop", False)),
                )
            else:
                val_list = 'dataset/s2d3d_pan_list/train.txt'
                tgt_train_dataset = StanfordPan8forPL(
                    root_syn, val_list, crop_size=(3072, 1024), sliding_window=(256, 1024), set='train'
                )
            pl_bs = int(getattr(self.args, "pl_batch_size", 1))
            pl_nw = int(getattr(self.args, "pl_num_workers", 4))
            tgt_train_loader = DataLoader(
                tgt_train_dataset,
                batch_size=pl_bs,
                shuffle=bool(getattr(self.args, "pl_shuffle", False)),
                num_workers=_num_workers(pl_nw),
                pin_memory=True,
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        frame_cnt = tgt_train_dataset._get_frame_cnt()
        return tgt_train_loader, class_names, num_classes, frame_cnt

    def build_tgt_from_pd(self, pd_dir: str):
        class_set = getattr(self.args, "class_set", 13)
        depth_prior_cfg = getattr(self.args, "depth_prior", {}) if isinstance(getattr(self.args, "depth_prior", {}), dict) else {}
        depth_prior_enabled = bool(depth_prior_cfg.get("enable", False))
        depth_prior_root = str(depth_prior_cfg.get("depth_dir", "") or getattr(self.args, "depth_prior_root", "") or "").strip()
        depth_file_exts = depth_prior_cfg.get("file_exts", [".png", ".jpg", ".jpeg", ".npy", ".pt"])
        depth_normalize = str(depth_prior_cfg.get("normalize", "minmax"))
        if not depth_prior_enabled:
            depth_prior_root = ""
        if self.args.dataset == 'Stanford2D3D':
            if self._stanford_flat():
                root_val = self._stanford_seg_root()
                val_list = os.path.join(root_val, 'lists', 'val.txt')
                used_path = os.path.join(os.path.dirname(pd_dir), 'used_path.txt')
                target_dataset = Stanford360FlatAdaptation(
                    pd_dir,
                    used_path,
                    image_root=root_val,
                    crop_size=(3072, 1024),
                    sliding_window=(256, 1024),
                    legacy_area_crop=bool(getattr(self.args, "stanford_legacy_area_crop", False)),
                )
                val_dataset = Stanford360PanoramaVal(
                    root_val,
                    val_list,
                    crop_size=(3072, 1024),
                    sliding_window=(256, 1024),
                    passes=2,
                    legacy_area_crop=bool(getattr(self.args, "stanford_legacy_area_crop", False)),
                )
            else:
                root_val = self._stanford_seg_root()
                val_list = 'dataset/s2d3d_pin_list/val.txt'
                target_dataset = StanfordPan8forAdaptation(
                    pd_dir,
                    os.path.join(os.path.dirname(pd_dir), 'used_path.txt'),
                    crop_size=(3072, 1024),
                    sliding_window=(256, 1024),
                )
                val_dataset = StanfordPan8forVal(
                    root_val, val_list, crop_size=(3072, 1024), sliding_window=(256, 1024), set='val'
                )

        elif self.args.dataset in ["SynPASS", "Cityscapes"]:
            root_val = os.path.join(self.data_root, 'DensePASS')
            val_list = 'dataset/densepass_list/val.txt'
            if class_set == 19:
                target_dataset = WildPASSforAdaptation19(
                    pd_dir,
                    os.path.join(os.path.dirname(pd_dir), 'used_path.txt'),
                    crop_size=(4096, 1024),
                    sliding_window=(384, 1024),
                    data_root=self.data_root,
                    depth_prior_root=depth_prior_root,
                    depth_file_exts=depth_file_exts,
                    depth_normalize=depth_normalize,
                )
                # SCM support: DensePASS19
                if getattr(self.args, 'enable_scm', False):
                    val_dataset = DensePASS19Circular(root_val, val_list, crop_size=(4096, 1024), sliding_window=(384, 1024), enable_circular_padding=True, circular_pad_width=getattr(self.args, 'scm_pad_width', 256))
                else:
                    val_dataset = DensePASS19(root_val, val_list, crop_size=(4096, 1024), sliding_window=(384, 1024))
            else:
                target_dataset = WildPASSforAdaptation(
                    pd_dir,
                    os.path.join(os.path.dirname(pd_dir), 'used_path.txt'),
                    crop_size=(4096, 1024),
                    sliding_window=(384, 1024),
                    data_root=self.data_root,
                    depth_prior_root=depth_prior_root,
                    depth_file_exts=depth_file_exts,
                    depth_normalize=depth_normalize,
                )
                # SCM support: DensePASS13
                if getattr(self.args, 'enable_scm', False):
                    val_dataset = DensePASS13Circular(root_val, val_list, crop_size=(4096, 1024), sliding_window=(384, 1024), enable_circular_padding=True, circular_pad_width=getattr(self.args, 'scm_pad_width', 256))
                else:
                    val_dataset = DensePASS13(root_val, val_list, crop_size=(4096, 1024), sliding_window=(384, 1024))

        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=_num_workers(4), pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=self.args.batch_size, num_workers=_num_workers(8),
                                drop_last=True,
                                worker_init_fn=_seed_worker)
                                    
        return target_loader, val_loader
    
    def build_src(self):
        class_set = getattr(self.args, "class_set", 13)
        class_names = resolve_class_names(self.args.dataset, class_set)
        num_classes = len(class_names)

        if self.args.dataset == 'Stanford2D3D':
            root_train = self._stanford_spin8_root()
            train_list = 'dataset/s2d3d_pin_list/train.txt'
            val_list = 'dataset/s2d3d_pin_list/val.txt'
            train_dataset = StanfordPin8DataSet(root_train, train_list, crop_size=(1024, 1024), set='train')
            val_dataset = StanfordPin8DataSet(root_train, val_list, crop_size=(1024, 1024), set='val')
        elif self.args.dataset == 'SynPASS':
            root_train = os.path.join(self.data_root, 'SynPASS')
            train_list = 'dataset/synpass_list/train.txt'
            val_list = 'dataset/synpass_list/val.txt'
            if class_set == 19:
                raise ValueError("SynPASS class_set=19 not supported in this repo (only legacy 13-class mapping).")
            train_dataset = SynPASS13(root_train, train_list, crop_size=(2048, 1024), sliding_window=(128, 1024))
            val_dataset = SynPASS13(root_train, val_list, crop_size=(2048, 1024), sliding_window=(128, 1024))
        elif self.args.dataset == 'Cityscapes':
            root_train = os.path.join(self.data_root, 'cityscapes')
            train_list = 'dataset/cityscapes_list/train.txt'
            val_list = 'dataset/cityscapes_list/val.txt'
            if class_set == 19:
                train_dataset = Cityscapes19(root_train, train_list, crop_size=(2048, 1024), sliding_window=(128, 1024), set='train')
                val_dataset = Cityscapes19(root_train, val_list, crop_size=(2048, 1024), sliding_window=(128, 1024), set='val')
            else:
                train_dataset = Cityscapes13(root_train, train_list, crop_size=(2048, 1024), sliding_window=(128, 1024), set='train')
                val_dataset = Cityscapes13(root_train, val_list, crop_size=(2048, 1024), sliding_window=(128, 1024), set='val')
        else:
            raise ValueError(self.args.dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=_num_workers(8), drop_last=True,
                                  worker_init_fn=_seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, 
                                num_workers=_num_workers(4), pin_memory=True)    
                                
        return train_loader, val_loader

    def build_criteria(self, device, dataset_name):
        if dataset_name == 'Stanford2D3D':
            crit_s = torch.nn.CrossEntropyLoss(ignore_index=255)
        else:
            class_set = getattr(self.args, "class_set", 13)
            if class_set == 13:
                weight = torch.Tensor([2.8149, 6.9850, 3.7890, 9.9428, 9.7702, 9.5111,
                                       10.3114, 10.0265, 4.6323, 9.5608, 7.8698, 9.5169, 10.3737]).to(device)
                crit_s = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weight)
            else:
                crit_s = torch.nn.CrossEntropyLoss(ignore_index=255)
        crit_t = torch.nn.CrossEntropyLoss(ignore_index=255)
        return {'s': crit_s, 't': crit_t}

    def write_splits_after_pl(self, save_dir: str, used_paths: List[str]):
        used_path_file = os.path.join(save_dir, 'used_path.txt')
        with open(used_path_file, 'w') as f:
            for p in used_paths:
                f.write(f"{p}\n")

        if os.path.exists(os.path.join(save_dir, 'unused_path.txt')):
            all_path_file = os.path.join(save_dir, 'unused_path.txt')
        else:
            all_path_file = (
                'dataset/densepass_list/train.txt'
                if self.args.dataset != 'Stanford2D3D'
                else 'dataset/s2d3d_pan_list/train.txt'
            )

        with open(all_path_file, 'r') as f1, open(used_path_file, 'r') as f2:
            paths1 = set(f1.read().splitlines())
            paths2 = set(f2.read().splitlines())
        filtered_paths = paths1 - paths2

        train_filtered_file = os.path.join(save_dir, 'unused_path.txt')
        with open(train_filtered_file, 'w') as f_out:
            f_out.write('\n'.join(filtered_paths))