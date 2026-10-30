import utils.sam2_bootstrap  # noqa: F401

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import random
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.distributed as dist
import datetime
import wandb

from train.trainer import Trainer
from utils.wandb_console import attach_wandb_console_log

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class TrainConfig:
    seed: int = field(
        default=1234,
        metadata={
            "help":"Random seed used for trainig"
        }
    )
    batch_size: int = field(
        default=8,
        metadata={
            "help":"Batchsize for training"
        }
    )
    num_epochs: int = field(
        default=10,
        metadata={
            "help":"Number of training epochs"
        }
    )
    lr: float = field(
        default=1e-4,
        metadata={
            "help":"Learning rate"
        }
    )
    backbone: str = field(
        default="sam2_t",
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
        default=3,
        metadata={
            "help":"Capacity of memory bank"
        }
    )
    image_size: int = field(
        default=1024,
        metadata={
            "help":"Width (height) of the image patches"
        }
    )
    patch_stride: int = field(
        default=128,
        metadata={
            "help":"Stride for sliding window operation during training"
        }
    )
    data_root: str = field(
        default="data",
        metadata={
            "help":"Root of training data"
        }
    )
    load_ckpt: str = field(
        default=None,
        metadata={
            "help":"Load checkpoints before training"
        }
    )
    save_ckpt_dir: str = field(
        default="checkpoints",
        metadata={
            "help":"Directory for saving checkpoints"
        }
    )
    # SGDRFNet configuration parameters
    enable_sgdrf: bool = field(
        default=False,
        metadata={
            "help":"Enable SG-DRF framework"
        }
    )
    enable_dynamic_fsmc: bool = field(
        default=True,
        metadata={
            "help":"Enable Dynamic FSMC module"
        }
    )
    enable_dynamic_scm: bool = field(
        default=True,
        metadata={
            "help":"Enable Dynamic SCM module"
        }
    )
    sgdrf_training_stage: str = field(
        default="cold_start",
        metadata={
            "help":"SGDRFNet training stage: cold_start or joint",
            "choices": ["cold_start", "joint"],
        }
    )
    sgdrf_consistency_weight: float = field(
        default=0.05,
        metadata={
            "help":"Weight for consistency loss in joint training"
        }
    )
    sgdrf_backbone_channels: int = field(
        default=256,
        metadata={
            "help":"Number of channels in SGDRFNet backbone feature maps"
        }
    )
    sgdrf_cold_start_epochs: int = field(default=2, metadata={"help":"Epochs for density-estimator cold start"})
    sgdrf_density_weight: float = field(default=0.1, metadata={"help":"Density loss weight"})
    sgdrf_density_lr: float = field(default=5e-5, metadata={"help":"Density estimator learning rate"})
    sgdrf_grad_clip_norm: float = field(default=1.0, metadata={"help":"SGDRF gradient clipping max norm"})
    sgdrf_fsmc_min_ratio: float = field(default=4.0, metadata={"help":"FSMC minimum compression ratio"})
    sgdrf_fsmc_max_ratio: float = field(default=16.0, metadata={"help":"FSMC maximum compression ratio"})
    sgdrf_scm_base_pad: int = field(default=1024, metadata={"help":"SCM base padding in source pixels"})
    sgdrf_scm_min_pad: int = field(default=512, metadata={"help":"SCM minimum padding in source pixels"})
    sgdrf_scm_max_pad: int = field(default=1536, metadata={"help":"SCM maximum padding in source pixels"})
    # SGDRFNet ablation experiment parameters
    sgdrf_train_list: str = field(default="", metadata={"help":"Path to SGDRF-specific train list file (overrides default train_list for ablation)"})
    sgdrf_val_list: str = field(default="", metadata={"help":"Path to SGDRF-specific val list file (overrides default val_list for ablation)"})
    max_train_samples: int = field(default=-1, metadata={"help":"Maximum training samples (-1 for no limit)"})
    max_train_iters: int = field(default=-1, metadata={"help":"Maximum training iterations (-1 for no limit)"})
    # MTC (Memory Token Compression) parameters
    enable_mtc: bool = field(default=False, metadata={"help":"Enable Memory Token Compression"})
    mtc_mode: str = field(default="pool", metadata={"help":"MTC compression mode (pool/spatial)", "choices": ["pool", "spatial"]})
    mtc_kernel_size: int = field(default=2, metadata={"help":"MTC pooling kernel size"})
    mtc_min_tokens: int = field(default=64, metadata={"help":"MTC minimum token count"})
    mtc_enable_in_train: bool = field(default=True, metadata={"help":"Enable MTC during training"})
    mtc_enable_in_pseudo_label: bool = field(default=True, metadata={"help":"Enable MTC during pseudo-label generation"})
    # SMR (Sparse Memory Retrieval) parameters
    enable_smr: bool = field(default=False, metadata={"help":"Enable Sparse Memory Retrieval"})
    smr_mode: str = field(default="feature_norm", metadata={"help":"SMR scoring mode", "choices": ["feature_norm"]})
    smr_keep_ratio: float = field(default=0.5, metadata={"help":"SMR token keep ratio (0.0-1.0)"})
    smr_min_tokens: int = field(default=64, metadata={"help":"SMR minimum token count"})
    smr_enable_in_train: bool = field(default=True, metadata={"help":"Enable SMR during training"})
    smr_enable_in_pseudo_label: bool = field(default=True, metadata={"help":"Enable SMR during pseudo-label generation"})
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
            parser.add_argument(f"--{f.name}", action="store_true" if f.default is False else "store_false")
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = parser.parse_args()
    return TrainConfig(**vars(args))

def main():
    args = parse_arg()
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed)
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if rank == 0:
        wandb.init(
            project="OmniSAM_src_train",
            name=dt,
            group=args.backbone + "_" + args.dataset + f"_{args.class_set}cls",
            config=vars(args)
        )
        _clog = attach_wandb_console_log()
        if _clog is not None:
            print(f"[wandb] Console log (stdout/stderr mirror): {_clog}", flush=True)
    trainer = Trainer(args, device, local_rank, world_size)
    trainer.setup()

    for epoch in range(args.num_epochs):
        trainer.train(epoch)
        trainer.validate(epoch)
    
    wandb.finish()
    dist.destroy_process_group()
    print('Training Finished.')

if __name__ == "__main__":
    main()