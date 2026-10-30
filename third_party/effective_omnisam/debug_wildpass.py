import utils.sam2_bootstrap  # noqa: F401

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from validation import (
    build_dataset_and_loader,
    build_sam_model,
    DATASET_NAME2CLASSES,
    setup_seed,
)
from utils.tools import merge_width_scatter_np


def main():
    """
    对 WildPASS 做快速 sanity check：
    - 只跑前 N 张图（默认 5）
    - 打印每张图的 GT / Pred unique classes
    - 保存 GT / Pred 可视化到 debug_wildpass/ 目录
    """
    import argparse

    parser = argparse.ArgumentParser("Debug WildPASS mapping / prediction")
    parser.add_argument("--ckpt", type=str, required=True, help="OmniSAM ckpt (.pth)")
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录（含 WildPASS/）")
    parser.add_argument("--num_samples", type=int, default=5, help="调试样本数")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    setup_seed(args.seed)

    repo_root = Path(__file__).resolve().parent
    if not os.path.isabs(args.data_root):
        args.data_root = str(repo_root / args.data_root)

    # 复用 validation 的参数结构
    class SimpleArgs:
        pass

    v_args = SimpleArgs()
    v_args.dataset = "WildPASS"
    v_args.data_root = args.data_root
    v_args.batch_size = 1
    v_args.backbone = "sam2_s"
    v_args.use_mem_bank = True
    v_args.num_maskmem = 9
    v_args.ckpt = args.ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, class_names, num_classes = build_dataset_and_loader(v_args)
    v_args.num_classes = num_classes

    model = build_sam_model(v_args, device)
    model.eval()

    out_dir = repo_root / "debug_wildpass"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Debug WildPASS: saving to {out_dir}")

    num_frames = loader.dataset._get_frame_cnt()
    print(f"WildPASS frame_cnt (sliding windows per pass): {num_frames}")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.num_samples:
                break
            image, label, _, names, ori_label = batch
            name = names[0]

            image = image.to(device)
            label = label.to(device)
            ori_label_np = ori_label.to(device).squeeze(0).cpu().numpy()

            B = image.shape[0]
            H, W = image.shape[-2:]

            logits_maps = np.zeros((B, 2 * num_frames, len(class_names), H, W))

            # forward passes，和 validation 保持一致
            for frame_idx in range(num_frames):
                frame_img = image[:, frame_idx]
                output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, frame_idx] = output.cpu().numpy()

            model.memory_bank._init_output_dict_source()

            for frame_idx in range(num_frames):
                frame_img = image[:, num_frames + frame_idx]
                output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, num_frames + frame_idx] = output.cpu().numpy()

            model.memory_bank._init_output_dict_source()

            merged_pred = merge_width_scatter_np(
                logits_np=logits_maps,
                orig_size=(1024, 4096),
                W_patch=W,
                stride_w=384,
                passes=2,
                device="cuda" if torch.cuda.is_available() else "cpu",
                return_probs=False,
            )

            # resize 到 2048×400 与 GT 对齐
            pred_resized = np.zeros((B, 400, 2048), dtype=np.int32)
            import cv2

            for i in range(B):
                pred_resized[i] = cv2.resize(
                    merged_pred[i], (2048, 400), interpolation=cv2.INTER_NEAREST
                )

            pred_np = pred_resized[0]

            gt_unique = np.unique(ori_label_np)
            pred_unique = np.unique(pred_np)
            print(f"[{idx}] {name}")
            print("  GT  unique:", gt_unique.tolist())
            print("  Pred unique:", pred_unique.tolist())

            # 保存 GT / Pred 可视化方便肉眼检查
            gt_img = Image.fromarray(ori_label_np.astype(np.uint8), mode="L")
            pred_img = Image.fromarray(pred_np.astype(np.uint8), mode="L")
            gt_img.save(out_dir / f"{idx:03d}_{name}_gt.png")
            pred_img.save(out_dir / f"{idx:03d}_{name}_pred.png")

    print("Done. Inspect images and unique-class logs above.")


if __name__ == "__main__":
    main()

