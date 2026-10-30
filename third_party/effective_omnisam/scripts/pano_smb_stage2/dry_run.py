#!/usr/bin/env python3
"""Print the frozen Stage 2 command without executing adaptation."""

import argparse
import shlex
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python", default="/root/miniconda3/envs/omnisam/bin/python")
    parser.add_argument("--fixed-pl", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--src-ckpt", required=True)
    parser.add_argument("--tgt-ckpt", required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    for path in (args.config, args.fixed_pl, args.subset, args.src_ckpt, args.tgt_ckpt):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    command = [
        "env", "WANDB_MODE=offline", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", f"CUDA_VISIBLE_DEVICES={args.gpu}",
        args.python, "run_adaptation.py", "--config", args.config, "--seed", "1",
        "--dataset", "Cityscapes", "--class_set", "13", "--backbone", "sam2_s",
        "--data_root", "data", "--pd_root", args.run_dir, "--src_ckpt", args.src_ckpt,
        "--tgt_ckpt", args.tgt_ckpt, "--amp", "--tf32", "--num_epochs", "1",
        "--batch_size", "4", "--pl_batch_size", "1", "--pl_num_workers", "2",
        "--no-pl_shuffle", "--pl_subset_file", args.subset, "--warmup_pl", "2000",
        "--iters_pl", "2000", "--no-use_text_proto", "--text_proto_path", "",
        "--lambda_txt", "0.0", "--weight_txt_align", "0.0", "--no-use_text_gate_ssl",
        "--no-use_wandb", "--no-use_tensorboard", "--reuse_pl_dir", args.fixed_pl,
    ]
    print(" ".join(shlex.quote(value) for value in command))
    print("DRY_RUN_ONLY: command was not executed")


if __name__ == "__main__":
    main()
