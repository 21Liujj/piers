import os
import random
import time
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sam.build_sam import build_sam2
from model.model import OmniSAM, OmniSAM_adapt

from dataset.Cityscapes import Cityscapes13, Cityscapes19
from dataset.WildPASS import WildPASSforPL, WildPASSforAdaptation
from dataset.DensePASS import DensePASS13
from dataset.SynPASS import SynPASS13
from dataset.StanfordPin import StanfordPin8DataSet
from utils.label_spaces import CITYSCAPES13_CLASSES, CITYSCAPES19_CLASSES
from utils.checkpoint_adapt import smart_load_checkpoint

DATASET_NAME2CLASSES_13 = {
    'Stanford2D3D': ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"],
    'Cityscapes': CITYSCAPES13_CLASSES,
    'SynPASS': CITYSCAPES13_CLASSES,
    'WildPASS': CITYSCAPES13_CLASSES,
}

DATASET_NAME2CLASSES_19 = {
    'Cityscapes': CITYSCAPES19_CLASSES,
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

def build_model(args, num_classes, num_maskmem, device, local_rank):
    sam2_checkpoint, model_cfg = SAM_CFGS[args.backbone]
    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)

    enable_sgdrf = bool(getattr(args, "enable_sgdrf", False))
    if enable_sgdrf:
        model = OmniSAM_adapt(sam, num_classes=num_classes, num_maskmem=num_maskmem, enable_sgdrf=True, sgdrf_backbone_channels=int(getattr(args, "sgdrf_backbone_channels", 256)), sgdrf_enable_dynamic_fsmc=bool(getattr(args, "enable_dynamic_fsmc", True)), sgdrf_enable_dynamic_scm=bool(getattr(args, "enable_dynamic_scm", True)), sgdrf_fsmc_min_ratio=float(getattr(args, "sgdrf_fsmc_min_ratio", 4.0)), sgdrf_fsmc_max_ratio=float(getattr(args, "sgdrf_fsmc_max_ratio", 16.0)), sgdrf_scm_base_pad=int(getattr(args, "sgdrf_scm_base_pad", 1024)), sgdrf_scm_min_pad=int(getattr(args, "sgdrf_scm_min_pad", 512)), sgdrf_scm_max_pad=int(getattr(args, "sgdrf_scm_max_pad", 1536)), sgdrf_training_stage=str(getattr(args, "sgdrf_training_stage", "cold_start"))).to(device)
        model.sgdrf_density_loss_weight = float(getattr(args, "sgdrf_density_weight", 0.1))
        model.sgdrf_consistency_loss_weight = float(getattr(args, "sgdrf_consistency_weight", 0.05))
    else:
        model = OmniSAM(sam, num_classes=num_classes, num_maskmem=num_maskmem).to(device)


    # Configure MTC+SMR (Memory Token Compression + Sparse Memory Retrieval)
    enable_mtc = bool(getattr(args, "enable_mtc", False))
    enable_smr = bool(getattr(args, "enable_smr", False))

    if enable_mtc or enable_smr:
        # Access memory_bank through the model
        memory_bank = None
        if hasattr(model, "memory_bank"):
            memory_bank = model.memory_bank
        elif hasattr(model, "segmentor") and hasattr(model.segmentor, "memory_bank"):
            memory_bank = model.segmentor.memory_bank

        if memory_bank is not None:
            if enable_mtc:
                # Configure Memory Token Compression
                compress_mode = str(getattr(args, "mtc_mode", "pool"))
                kernel_size = int(getattr(args, "mtc_kernel_size", 2))
                min_tokens = int(getattr(args, "mtc_min_tokens", 64))
                enable_in_train = bool(getattr(args, "mtc_enable_in_train", True))
                enable_in_pl = bool(getattr(args, "mtc_enable_in_pseudo_label", True))

                memory_bank.configure_compression(
                    enabled=True,
                    compress_mode=compress_mode,
                    kernel_size=kernel_size,
                    stride=kernel_size,  # Usually same as kernel_size
                    min_tokens=min_tokens,
                    enable_in_train=enable_in_train,
                    enable_in_pseudo_label=enable_in_pl,
                    verbose=True
                )
                if local_rank == 0:
                    print(f"[MTC] Enabled: mode={compress_mode}, kernel={kernel_size}, min_tokens={min_tokens}")

            if enable_smr:
                # Configure Sparse Memory Retrieval
                sparse_mode = str(getattr(args, "smr_mode", "feature_norm"))
                keep_ratio = float(getattr(args, "smr_keep_ratio", 0.5))
                min_tokens = int(getattr(args, "smr_min_tokens", 64))
                enable_in_train = bool(getattr(args, "smr_enable_in_train", True))
                enable_in_pl = bool(getattr(args, "smr_enable_in_pseudo_label", True))

                memory_bank.configure_sparse_retrieval(
                    enabled=True,
                    mode=sparse_mode,
                    keep_ratio=keep_ratio,
                    min_tokens=min_tokens,
                    enable_in_train=enable_in_train,
                    enable_in_pseudo_label=enable_in_pl
                )
                if local_rank == 0:
                    print(f"[SMR] Enabled: mode={sparse_mode}, keep_ratio={keep_ratio}, min_tokens={min_tokens}")
        else:
            if local_rank == 0:
                print("[WARNING] MTC/SMR enabled but memory_bank not found in model")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    return model

def load_checkpoint(model, ckpt_path):
    if ckpt_path is None or ckpt_path == '':
        return
    smart_load_checkpoint(model, ckpt_path)

class DatasetBuilder:
    def __init__(self, args):
        self.args = args

    def build_src(self):
        class_set = getattr(self.args, "class_set", 13)
        class_names = resolve_class_names(self.args.dataset, class_set)
        num_classes = len(class_names)

        if self.args.dataset == 'Stanford2D3D':
            root_train = os.path.join(self.args.data_root, 'Stanford2d3d_Seg')
            train_list = 'dataset/s2d3d_pin_list/train.txt'
            val_list = 'dataset/s2d3d_pin_list/val.txt'
            train_dataset = StanfordPin8DataSet(root_train, train_list, crop_size=(1024, 1024), set='train')
            val_dataset = StanfordPin8DataSet(root_train, val_list, crop_size=(1024, 1024), set='val')
        elif self.args.dataset == 'SynPASS':
            root_train = os.path.join(self.args.data_root, 'SynPASS')
            train_list = 'dataset/synpass_list/train.txt'
            val_list = 'dataset/synpass_list/val.txt'
            train_dataset = SynPASS13(root_train, train_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size))
            val_dataset = SynPASS13(root_train, val_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size))
        elif self.args.dataset == 'Cityscapes':
            root_train = os.path.join(self.args.data_root, 'cityscapes')
            # Override with SGDRF-specific lists if provided
            if hasattr(self.args, 'sgdrf_train_list') and self.args.sgdrf_train_list:
                train_list = self.args.sgdrf_train_list
            else:
                train_list = 'dataset/cityscapes_list/train.txt'
            if hasattr(self.args, 'sgdrf_val_list') and self.args.sgdrf_val_list:
                val_list = self.args.sgdrf_val_list
            else:
                val_list = 'dataset/cityscapes_list/val.txt'
            
            if class_set == 19:
                train_dataset = Cityscapes19(root_train, train_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size), set='train')
                val_dataset = Cityscapes19(root_train, val_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size), set='val')
            else:
                train_dataset = Cityscapes13(root_train, train_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size), set='train')
                val_dataset = Cityscapes13(root_train, val_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size), set='val')
        else:
            raise ValueError(self.args.dataset)

        # Apply max_train_samples if specified (ablation experiments)
        max_samples = getattr(self.args, 'max_train_samples', -1)
        if max_samples > 0 and hasattr(train_dataset, 'files'):
            orig_len = len(train_dataset.files)
            train_dataset.files = train_dataset.files[:max_samples]
            if hasattr(train_dataset, 'labels'):
                train_dataset.labels = train_dataset.labels[:max_samples]
            print(f"[ABLATION] Truncated training dataset: {orig_len} -> {len(train_dataset.files)} samples")

        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
            sampler=val_sampler,
        )

        return train_loader, val_loader, train_sampler, val_sampler

    def _apply_sample_limit(self, dataset, max_samples):
        """Truncate dataset samples if max_samples is set."""
        if max_samples > 0 and hasattr(dataset, 'files'):
            original_len = len(dataset.files)
            dataset.files = dataset.files[:max_samples]
            if hasattr(dataset, 'labels'):
                dataset.labels = dataset.labels[:max_samples]
            return original_len, len(dataset.files)
        return -1, -1
