import utils.sam2_bootstrap  # noqa: F401 — keep same bootstrap order as validation.py

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from sam.build_sam import build_sam2
from model.model import OmniSAM_adapt
from utils.build import load_checkpoint

# Reuse the exact same validation pipeline / metrics / dataset logic
from validation import (
    collect_ckpt_paths,
    resolve_ckpt_names,
    print_metrics_comparison_table,
    setup_logging,
    setup_seed,
    validation,
    build_dataset_and_loader,
)


def build_sam_model_muti(args: argparse.Namespace, device: torch.device, ckpt_path: Optional[str] = None) -> nn.Module:
    """
    Same as validation.py::build_sam_model, but with optional text modality.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sam_vendor = os.path.join(repo_root, "sam")
    backbone_to_hydra_and_ckpt = {
        "sam2_s": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "sam2_b+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "sam2_l": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        "sam2_t": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    }
    if args.backbone not in backbone_to_hydra_and_ckpt:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    model_cfg, ckpt_name = backbone_to_hydra_and_ckpt[args.backbone]
    sam2_checkpoint = os.path.join(sam_vendor, "checkpoints", ckpt_name)
    if not os.path.isfile(sam2_checkpoint):
        raise FileNotFoundError(f"SAM2 backbone weight not found: {sam2_checkpoint}")

    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)

    text_prototypes = None
    proto_path = (getattr(args, "text_proto_path", "") or "").strip()
    if proto_path:
        payload = torch.load(proto_path, map_location="cpu")
        if isinstance(payload, dict) and "text_prototypes" in payload:
            text_prototypes = payload["text_prototypes"]
        elif torch.is_tensor(payload):
            text_prototypes = payload
        else:
            raise ValueError(f"Unsupported text prototype payload: {proto_path}")
        if text_prototypes.shape[0] != args.num_classes:
            raise ValueError(
                f"text prototypes class count mismatch: proto C={text_prototypes.shape[0]} vs val classes={args.num_classes}"
            )
        print(
            f"Loaded text prototypes: {os.path.abspath(proto_path)} "
            f"shape={tuple(text_prototypes.shape)}, lambda_txt={args.lambda_txt}, txt_tau={args.txt_tau}"
        )

    model = OmniSAM_adapt(
        sam,
        num_classes=args.num_classes,
        num_maskmem=args.num_maskmem,
        text_prototypes=text_prototypes,
        d_txt=int(text_prototypes.shape[-1]) if text_prototypes is not None else None,
        lambda_txt=float(args.lambda_txt),
        txt_tau=float(args.txt_tau),
    ).to(device)

    load_path = ckpt_path if ckpt_path is not None else getattr(args, "ckpt", None)
    if load_path:
        load_path = os.path.abspath(load_path)
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        load_checkpoint(model, load_path)
        print(f"Loaded checkpoint from: {load_path}")
    else:
        print("No checkpoint: using SAM2 backbone + randomly initialized decoder / LoRA.")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation with multi-ckpt comparison + optional RGB-Text modality"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backbone", type=str, default="sam2_s")
    parser.add_argument("--dataset", type=str, default="DensePASS")
    parser.add_argument("--class_set", type=int, default=13, choices=[13, 19])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--wildpass_layout",
        type=str,
        default="cityscapes",
        choices=["cityscapes", "flat"],
    )
    parser.add_argument("--wildpass_processed_root", type=str, default="")
    parser.add_argument("--val_list", type=str, default="")
    parser.add_argument("--wildpass_image_dir", type=str, default="images")
    parser.add_argument("--wildpass_label_dir", type=str, default="labels_trainid")
    parser.add_argument("--use_mem_bank", action="store_true")
    parser.add_argument("--num_maskmem", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ckpt", action="append", default=None, metavar="PATH")
    parser.add_argument("--ckpt_list", type=str, default=None, metavar="FILE")
    parser.add_argument("--ckpt_names", type=str, default=None, metavar="NAMES")
    parser.add_argument("--metrics_json", type=str, default=None)
    parser.add_argument("--metrics_json_per_run", action="store_true")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_append", action="store_true")

    # Text modality switches (new)
    parser.add_argument(
        "--text_proto_path",
        type=str,
        default="",
        help="Path to text prototypes (.pt). Empty = visual-only validation.",
    )
    parser.add_argument(
        "--lambda_txt",
        type=float,
        default=0.0,
        help="Fusion weight for L_fuse = L_vis + lambda_txt * L_txt",
    )
    parser.add_argument(
        "--txt_tau",
        type=float,
        default=0.07,
        help="Temperature for pixel-text cosine logits.",
    )
    args = parser.parse_args()
    setup_seed(args.seed)

    log_fh, old_out, old_err = setup_logging(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, class_names, num_classes = build_dataset_and_loader(args)
    args.num_classes = num_classes

    if args.use_mem_bank:
        args.num_maskmem = args.num_maskmem if args.num_maskmem is not None else 9
    else:
        args.num_maskmem = 0
    print(
        f"Using backbone: {args.backbone}, use_mem_bank: {args.use_mem_bank}, num_maskmem: {args.num_maskmem}"
    )

    ckpt_paths = collect_ckpt_paths(args)
    if not ckpt_paths:
        ckpt_paths = [None]
    names = resolve_ckpt_names(ckpt_paths, args.ckpt_names)

    original_metrics_json = args.metrics_json
    n_runs = len(ckpt_paths)
    all_rows = []
    all_runs_payload = []

    for run_idx, (run_name, ckpt_path) in enumerate(zip(names, ckpt_paths)):
        args.ckpt = ckpt_path
        args.run_label = run_name
        args._skip_per_run_json = False
        args.metrics_json = None
        if original_metrics_json:
            if n_runs > 1 and args.metrics_json_per_run:
                root, ext = os.path.splitext(original_metrics_json)
                ext = ext or ".json"
                args.metrics_json = f"{root}_{run_idx}{ext}"
            elif n_runs == 1:
                args.metrics_json = original_metrics_json
            elif n_runs > 1 and not args.metrics_json_per_run:
                args._skip_per_run_json = True

        print(f"\n{'#' * 20} [{run_idx + 1}/{len(ckpt_paths)}] {run_name} {'#' * 20}")
        print(f"ckpt: {os.path.abspath(ckpt_path) if ckpt_path else '(none)'}")
        model = build_sam_model_muti(args, device, ckpt_path=ckpt_path).to(device)
        agg = validation(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            backbone_name=args.backbone,
            num_maskmem=args.num_maskmem,
            dataset_name=args.dataset,
            args=args,
        )
        row = {"name": run_name, "ckpt": os.path.abspath(ckpt_path) if ckpt_path else None, **agg}
        all_rows.append(row)
        all_runs_payload.append(
            {
                "name": run_name,
                "ckpt": row["ckpt"],
                "metrics": {k: round(v, 4) for k, v in agg.items()},
                "text_proto_path": os.path.abspath(args.text_proto_path) if args.text_proto_path else None,
                "lambda_txt": float(args.lambda_txt),
                "txt_tau": float(args.txt_tau),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if original_metrics_json and n_runs > 1 and not args.metrics_json_per_run:
        combined = {
            "dataset": args.dataset,
            "class_set": getattr(args, "class_set", 13),
            "backbone": args.backbone,
            "text_proto_path": os.path.abspath(args.text_proto_path) if args.text_proto_path else None,
            "lambda_txt": float(args.lambda_txt),
            "txt_tau": float(args.txt_tau),
            "runs": all_runs_payload,
        }
        with open(original_metrics_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"Wrote combined metrics JSON ({len(all_runs_payload)} runs): {original_metrics_json}")

    if len(ckpt_paths) > 1:
        print_metrics_comparison_table(all_rows)

    print("Validation (multi-modal) Finished.")
    if log_fh is not None:
        try:
            print(f"log end: {datetime.now().isoformat(timespec='seconds')}")
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            log_fh.close()


if __name__ == "__main__":
    print("File name:", __file__)
    main()

