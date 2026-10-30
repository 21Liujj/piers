import utils.sam2_bootstrap  # noqa: F401 — 必须在 SAM2 / utils.build 之前

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings('ignore')

import argparse
from adaptation.boundary_prompt_inference import apply_bepl_to_logits
import json
import random
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import fast_hist, per_class_iu, per_class_dice, scores_from_hist
from utils.build import load_checkpoint
from dataset.DensePASS import DensePASS13, DensePASS19
from dataset.DensePASS_circular import DensePASS13Circular, DensePASS19Circular
from dataset.Stanford360Span8 import Stanford360PanoramaVal
from dataset.WildPASS import WildPASS13
from sam.build_sam import build_sam2
from model.model import OmniSAM_adapt
from utils.tools import merge_width_scatter_np
from utils.seg_vis import label_ids_to_rgb, overlay_rgb_on_label
from PIL import Image as PILImage
# Mapping dataset names to their classes
DATASET_NAME2CLASSES = {
    "Stanford2D3D": ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"],
    "SynPASS": ["road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain",
                "sky", "person", "car"],
    "Cityscapes": ["road", "sidewalk", "building", "wall", "fence", "pole",
                   "traffic light", "traffic sign", "vegetation", "terrain",
                   "sky", "person", "car"],
    "DensePASS": ["road", "sidewalk", "building", "wall", "fence", "pole",
                  "traffic light", "traffic sign", "vegetation", "terrain",
                  "sky", "person", "car"],
    "WildPASS": ["road", "sidewalk", "building", "wall", "fence", "pole",
                 "traffic light", "traffic sign", "vegetation", "terrain",
                 "sky", "person", "car"],
}

# DensePASS / Cityscapes 19-class names (TrainIds order).
DENSEPASS19_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

def _read_ckpt_list_file(path: str) -> list:
    """One checkpoint path per line; empty lines and # comments ignored."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


def collect_ckpt_paths(args: argparse.Namespace) -> list:
    """Merge --ckpt (repeatable) and --ckpt_list file paths in order."""
    paths = []
    if getattr(args, "ckpt", None):
        paths.extend(args.ckpt)
    list_path = getattr(args, "ckpt_list", None)
    if list_path:
        list_path = os.path.abspath(list_path)
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"--ckpt_list not found: {list_path}")
        paths.extend(_read_ckpt_list_file(list_path))
    return paths


def resolve_ckpt_names(ckpt_paths: list, names_csv: Optional[str]) -> list:
    if names_csv:
        names = [x.strip() for x in names_csv.split(",") if x.strip()]
        if len(names) != len(ckpt_paths):
            raise ValueError(
                f"--ckpt_names 数量 ({len(names)}) 必须与 checkpoint 数量 ({len(ckpt_paths)}) 一致"
            )
        return names
    return [
        os.path.splitext(os.path.basename(p))[0] if p else "no_ckpt"
        for p in ckpt_paths
    ]


def print_metrics_comparison_table(rows: list) -> None:
    """rows: list of dict with keys name, ckpt, and metric floats."""
    if not rows:
        return
    headers = ["name", "mIoU", "mAcc", "PixAcc", "FWIoU", "mDice"]
    print("\n========== 多模型指标对比 ==========")
    col_w = [max(len(h), 12) for h in headers]
    col_w[0] = max(col_w[0], 28)
    head = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(head)
    print("-" * len(head))
    for r in rows:
        line = [
            str(r.get("name", ""))[: col_w[0]],
            f'{r["mIoU"]:.2f}',
            f'{r["mAcc"]:.2f}',
            f'{r["PixAcc"]:.2f}',
            f'{r["FWIoU"]:.2f}',
            f'{r["mDice"]:.2f}',
        ]
        print("  ".join(line[i].ljust(col_w[i]) for i in range(len(headers))))
    print("====================================\n")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def setup_logging(args: argparse.Namespace):
    """
    If --log_file is provided, tee stdout/stderr to that file.
    Returns (fh, old_stdout, old_stderr). Caller must restore and close.
    """
    log_path = getattr(args, "log_file", None)
    if not log_path:
        return None, None, None

    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    mode = "a" if getattr(args, "log_append", False) else "w"
    fh = open(log_path, mode, encoding="utf-8")

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, fh)
    sys.stderr = _Tee(old_err, fh)

    print("==========================================")
    print(f"validation.py log start: {datetime.now().isoformat(timespec='seconds')}")
    print(f"log_file: {log_path}")
    print("==========================================")
    return fh, old_out, old_err


def setup_seed(seed: int = 1234):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set deterministic mode for cuDNN for a trade-off between determinism and performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def _resolve_repo_relative_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(repo_root, path)


def _align_gt_hw_to_pred_hw(ol_hw: np.ndarray, pred_h: int, pred_w: int) -> np.ndarray:
    """将 GT 语义图缩放到与 pred_resized 相同 H×W（与 cv2.resize(..., dsize=(W,H)) 一致）。"""
    if ol_hw.shape[0] == pred_h and ol_hw.shape[1] == pred_w:
        return ol_hw.astype(np.int64)
    return cv2.resize(
        ol_hw.astype(np.uint8),
        (pred_w, pred_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


def _dataloader_sample_rel_path(names, batch_index: int) -> str:
    """从 DataLoader batch 中取出第 batch_index 个样本的文件名（相对 lists 的路径）。"""
    if isinstance(names, (list, tuple)):
        x = names[batch_index]
    elif isinstance(names, torch.Tensor):
        x = names[batch_index]
    elif isinstance(names, np.ndarray):
        x = names.reshape(-1)[batch_index]
    else:
        x = names
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item() if x.numel() == 1 else x
    return str(x)


def _write_pred_gt_histogram_files(
    out_json: str,
    out_md: str,
    dataset_name: str,
    class_names: list,
    num_classes: int,
    gt_hist: np.ndarray,
    pred_hist: np.ndarray,
    pred_oor: int,
) -> None:
    gt_total = int(gt_hist.sum())
    pred_total = int(pred_hist.sum() + pred_oor)
    payload = {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "note": "直方图仅在 GT∈[0,num_classes) 的像素上统计（与 fast_hist 对 GT 的过滤一致；忽略 255）。",
        "valid_gt_pixels": gt_total,
        "pred_histogram_includes_oor_separately": True,
        "pred_out_of_range_on_valid_gt_pixels": int(pred_oor),
        "class_names": list(class_names),
        "gt_histogram_counts": gt_hist.tolist(),
        "pred_histogram_counts": pred_hist.tolist(),
        "gt_histogram_ratio": [round(float(x) / gt_total, 8) for x in gt_hist] if gt_total else [],
        "pred_histogram_ratio_on_valid_gt": [round(float(x) / pred_total, 8) for x in pred_hist] if pred_total else [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    lines = [
        f"# Pred / GT 类直方图（{dataset_name}，valid-GT 像素）",
        "",
        f"- valid GT 像素数: {gt_total}",
        f"- 预测落在类范围外的像素数（在 valid-GT 上）: {pred_oor}",
        "",
        "| class_id | name | GT count | GT ratio | Pred count | Pred ratio |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in range(num_classes):
        g = int(gt_hist[c])
        p = int(pred_hist[c])
        gr = (g / gt_total) if gt_total else 0.0
        pr = (p / pred_total) if pred_total else 0.0
        nm = class_names[c] if c < len(class_names) else str(c)
        lines.append(f"| {c} | {nm} | {g} | {gr:.6f} | {p} | {pr:.6f} |")
    if pred_oor:
        lines.append(f"| (oor) | pred∉[0,{num_classes}) | — | — | {pred_oor} | {pred_oor / pred_total:.6f} |")
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote pred/GT histogram: {out_json} and {out_md}")


def validation(model: nn.Module, data_loader: DataLoader, device: torch.device,
               num_classes: int, class_names: list, backbone_name: str,
               num_maskmem: int, dataset_name: str, args):
    """
    Run validation using sliding window predictions.
    """
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    save_n = int(getattr(args, "save_pred_debug", 0) or 0)
    do_hist = bool(getattr(args, "pred_gt_histogram", False))
    debug_dir = getattr(args, "debug_pred_vis_dir", None) or ""
    debug_dir = _resolve_repo_relative_path(str(debug_dir).strip())
    pred_debug_saved = 0
    gt_hist = np.zeros(num_classes, dtype=np.int64)
    pred_hist = np.zeros(num_classes, dtype=np.int64)
    pred_oor = 0
    if save_n > 0 and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    print('----- Start Validation -----')
    n_batches = len(data_loader)
    num_frames = data_loader.dataset._get_frame_cnt()
    t_val0 = time.perf_counter()
    total_eval_images = 0
    print(
        f'[{dataset_name}] 共 {n_batches} 张验证图；每张约 {2 * num_frames} 次前向（双 pass × 滑窗）。'
        f'首张常需数分钟，请耐心等待。',
        flush=True,
    )

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Val', total=n_batches, mininterval=2.0)):
            image, label, _, names, ori_label = batch
            image = image.to(device)
            label = label.to(device)
            ol_bt = ori_label.detach().cpu().numpy().astype(np.int64)
            if ol_bt.ndim == 2:
                ol_bt = ol_bt[np.newaxis, :, :]
            B = image.shape[0]
            H, W = image.shape[-2:]
            # Initialize logits_maps for both groups of mask memories
            logits_maps = np.zeros((B, 2 * num_frames, len(class_names), H, W))

            # Process first set of frames
            for frame_idx in range(num_frames):
                frame_img = image[:, frame_idx]
                frame_label = label[:, frame_idx]
                if dataset_name == "Stanford2D3D":
                    output, feat, feat_de = model(0, frame_img)
                else:
                    output, feat, feat_de = model(frame_idx, frame_img)
                logits_maps[:, frame_idx] = output.cpu().numpy()


            model.memory_bank._init_output_dict_source()

            # Process second set of frames
            for frame_idx in range(num_frames):
                frame_img = image[:, num_frames + frame_idx]
                frame_label = label[:, num_frames + frame_idx]
                if dataset_name == "Stanford2D3D":
                    output, _, _ = model(0, frame_img)
                else:
                    output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, num_frames + frame_idx] = output.cpu().numpy()

            model.memory_bank._init_output_dict_source()

            # Merge：DensePASS / WildPASS 均先 resize 到 4096×1024 再滑窗，保证每块为 1024×1024（SAM2 骨干要求）
            if dataset_name in ("DensePASS", "WildPASS"):
                # Calculate padded width for SCM
                padded_width = (4096 + 2 * args.scm_pad_width) if args.enable_scm else 4096

                merged_pred = merge_width_scatter_np(
                    logits_np=logits_maps,
                    orig_size=(1024, padded_width),
                    W_patch=W,
                    stride_w=384,
                    passes=2,
                    device=str(device),
                    return_probs=False,
                )

                # SCM: Crop padded prediction back to original width
                if args.enable_scm:
                    pad_w = args.scm_pad_width
                    merged_pred = merged_pred[:, :, pad_w:-pad_w]

                # BEPL: Apply boundary-enhanced prompt
                if args.enable_bepl:
                    merged_pred_tensor = torch.from_numpy(merged_pred)
                    merged_pred_tensor = apply_bepl_to_logits(
                        merged_pred_tensor,
                        seam_width=args.bepl_seam_width,
                        strength=args.bepl_strength
                    )
                    merged_pred = merged_pred_tensor.numpy()



                pred_h, pred_w = 400, 2048
                pred_resized = np.zeros((B, pred_h, pred_w))
                gt_eval = np.zeros((B, pred_h, pred_w), dtype=np.int64)
                for i in range(B):
                    pred_resized[i] = cv2.resize(
                        merged_pred[i], (pred_w, pred_h), interpolation=cv2.INTER_NEAREST
                    )
                    gt_eval[i] = _align_gt_hw_to_pred_hw(ol_bt[i], pred_h, pred_w)
                hist_np = fast_hist(
                    gt_eval.reshape(-1),
                    pred_resized.reshape(-1).astype(np.int64),
                    num_classes,
                )
                hist += hist_np

                if do_hist or (save_n > 0 and pred_debug_saved < save_n):
                    for i in range(B):
                        gt_i = gt_eval[i]
                        pr_i = pred_resized[i].astype(np.int64)
                        valid_gt = (gt_i >= 0) & (gt_i < num_classes)
                        if do_hist and valid_gt.any():
                            gv = gt_i[valid_gt]
                            pv = pr_i[valid_gt]
                            gt_hist += np.bincount(gv, minlength=num_classes)
                            okp = (pv >= 0) & (pv < num_classes)
                            if okp.any():
                                pred_hist += np.bincount(pv[okp], minlength=num_classes)
                            pred_oor += int((~okp).sum())

                        if (
                            save_n > 0
                            and pred_debug_saved < save_n
                            and dataset_name == "WildPASS"
                            and debug_dir
                        ):
                            proc = str(getattr(args, "wildpass_processed_root", "") or "").strip()
                            img_sub = str(getattr(args, "wildpass_image_dir", "images") or "images")
                            if not proc:
                                tqdm.write("[save_pred_debug] wildpass_processed_root 未设置，跳过存图")
                                pred_debug_saved = save_n
                                break
                            rel = _dataloader_sample_rel_path(names, i)
                            img_path = os.path.join(proc, img_sub, rel.lstrip("/"))
                            if not os.path.isfile(img_path):
                                tqdm.write(f"[save_pred_debug] 缺少 RGB: {img_path}")
                            else:
                                rgb = np.array(PILImage.open(img_path).convert("RGB"))
                                rgb = np.array(
                                    PILImage.fromarray(rgb).resize((2048, 400), PILImage.BICUBIC)
                                )
                                pred_c = label_ids_to_rgb(pr_i, num_classes=num_classes)
                                gt_c = label_ids_to_rgb(gt_i.astype(np.int64), num_classes=num_classes)
                                overlay = overlay_rgb_on_label(rgb, pred_c, alpha=0.45)
                                safe = rel.replace("/", "_").replace("\\", "_")
                                base = os.path.join(debug_dir, f"sample_{safe}")
                                cv2.imwrite(base + "_pred_argmax.png", pr_i.astype(np.uint8))
                                PILImage.fromarray(pred_c).save(base + "_pred_color.png")
                                PILImage.fromarray(gt_c).save(base + "_gt_color.png")
                                PILImage.fromarray(rgb).save(base + "_rgb.png")
                                PILImage.fromarray(overlay).save(base + "_overlay.png")
                                pred_debug_saved += 1
            elif dataset_name == "Stanford2D3D":
                ol = ol_bt[0]
                if ol.ndim == 3:
                    ol = ol[0]
                elif ol.ndim != 2:
                    ol = np.squeeze(ol)
                H_full, W_full = int(ol.shape[-2]), int(ol.shape[-1])
                stride_w = int(getattr(args, "stanford_stride_w", 256))
                # Calculate padded width for SCM
                padded_width = (4096 + 2 * args.scm_pad_width) if args.enable_scm else 4096

                merged_pred = merge_width_scatter_np(
                    logits_np=logits_maps,
                    orig_size=(H_full, W_full),
                    W_patch=W,
                    stride_w=stride_w,
                    passes=2,
                    device=str(device),
                    return_probs=False,
                )
                hist_np = fast_hist(
                    ol.flatten(),
                    merged_pred.flatten().astype(np.int64),
                    num_classes,
                )
                hist += hist_np
            total_eval_images += B
            # 避免与 tqdm 同一行拼接（日志里易出现「0/500」旁直接接上「N items」）
            tqdm.write(f'{(index + 1) * B} items processed')

    mIoUs = per_class_iu(hist)
    dices = per_class_dice(hist)
    for idx, cname in enumerate(class_names):
        iou_pct = round(mIoUs[idx] * 100, 2) if not np.isnan(mIoUs[idx]) else float("nan")
        dice_pct = round(dices[idx] * 100, 2) if not np.isnan(dices[idx]) else float("nan")
        print(f'===> {cname:<15}:\t IoU {iou_pct}\t Dice {dice_pct}')
    agg = scores_from_hist(hist)
    cur_mIoU = round(agg["mIoU"], 2)
    if do_hist:
        hj = str(getattr(args, "pred_gt_histogram_json", "") or "").strip()
        hm = str(getattr(args, "pred_gt_histogram_md", "") or "").strip()
        if hj and hm:
            _write_pred_gt_histogram_files(
                _resolve_repo_relative_path(hj),
                _resolve_repo_relative_path(hm),
                dataset_name,
                class_names,
                num_classes,
                gt_hist,
                pred_hist,
                pred_oor,
            )
    print(
        f'\n--- {backbone_name} aggregate ({dataset_name} val set) ---\n'
        f'  mIoU:   {agg["mIoU"]:.2f}\n'
        f'  mAcc:   {agg["mAcc"]:.2f}  (mean per-class recall)\n'
        f'  PixAcc: {agg["PixAcc"]:.2f}\n'
        f'  FWIoU:  {agg["FWIoU"]:.2f}  (freq-weighted IoU)\n'
        f'  mDice:  {agg["mDice"]:.2f}  (mean Dice / F1)\n'
    )
    eval_time_sec = float(time.perf_counter() - t_val0)
    ips = float(total_eval_images / eval_time_sec) if eval_time_sec > 1e-9 else 0.0

    out_path = getattr(args, "metrics_json", None)
    if out_path and not getattr(args, "_skip_per_run_json", False):
        payload = {
            "name": getattr(args, "run_label", None),
            "dataset": dataset_name,
            "backbone": backbone_name,
            "ckpt": getattr(args, "ckpt", None),
            **{k: round(v, 4) for k, v in agg.items()},
            "per_class_iou_pct": [round(float(x) * 100, 4) if not np.isnan(x) else None for x in mIoUs],
            "eval_time_sec": round(eval_time_sec, 4),
            "num_eval_images": int(total_eval_images),
            "images_per_sec": round(ips, 4),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'Wrote metrics JSON: {out_path}')
    timing_path = getattr(args, "timing_json", None)
    if timing_path:
        timing_path = os.path.abspath(timing_path)
        os.makedirs(os.path.dirname(timing_path), exist_ok=True)
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "eval_time_sec": round(eval_time_sec, 6),
                    "num_eval_images": int(total_eval_images),
                    "images_per_sec": round(ips, 6),
                    "dataset": dataset_name,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Wrote timing JSON: {timing_path}")
    return agg

def build_dataset_and_loader(args: argparse.Namespace):
    """
    Build validation dataset and dataloader based on the chosen dataset.
    """
    dataset_name = args.dataset
    if dataset_name not in DATASET_NAME2CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    class_set = getattr(args, "class_set", 13)
    if dataset_name == "DensePASS" and class_set == 19:
        class_names = DENSEPASS19_CLASSES
    else:
        class_names = DATASET_NAME2CLASSES[dataset_name]
    num_classes = len(class_names)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_root = getattr(args, "data_root", "data")
    if not os.path.isabs(data_root):
        data_root = os.path.join(repo_root, data_root)

    if dataset_name == "Stanford2D3D":
        spr = str(getattr(args, "stanford_processed_root", "") or "").strip()
        if not spr:
            raise NotImplementedError(
                "Stanford2D3D validation requires --stanford_processed_root "
                "(processed Stanford360_SPAN8 with lists/val.txt)."
            )
        root_syn = spr if os.path.isabs(spr) else os.path.join(repo_root, spr)
        val_ov = str(getattr(args, "stanford_val_list", "") or "").strip()
        if val_ov:
            val_list = val_ov if os.path.isabs(val_ov) else os.path.join(repo_root, val_ov)
        else:
            val_list = os.path.join(root_syn, "lists", "val.txt")
        syn_w, syn_h = 3072, 1024
        sw_w_stride, sw_size = int(getattr(args, "stanford_stride_w", 256)), 1024
        assert (syn_w - sw_size) % sw_w_stride == 0 and sw_w_stride <= sw_size
        val_dataset = Stanford360PanoramaVal(
            root_syn,
            val_list,
            crop_size=(syn_w, syn_h),
            sliding_window=(sw_w_stride, sw_size),
            passes=2,
            legacy_area_crop=bool(getattr(args, "stanford_legacy_area_crop", False)),
        )
    elif dataset_name == "DensePASS":
        root_syn = os.path.join(data_root, "DensePASS")
        val_list = os.path.join(repo_root, "dataset/densepass_list/val.txt")
        syn_w, syn_h = 4096, 1024
        sw_w_stride, sw_size = 384, 1024
        assert (syn_w - sw_size) % sw_w_stride == 0 and  sw_w_stride <= sw_size

        # Select dataset class based on --enable_scm flag
        if args.enable_scm:
            print(f"[SCM] Using circular padding with pad_width={args.scm_pad_width}")
            DensePASSClass13 = DensePASS13Circular
            DensePASSClass19 = DensePASS19Circular
            extra_kwargs = {
                'enable_circular_padding': True,
                'circular_pad_width': args.scm_pad_width
            }
        else:
            print("[Baseline] Using standard DensePASS no circular padding")
            DensePASSClass13 = DensePASS13
            DensePASSClass19 = DensePASS19
            extra_kwargs = {}

        if class_set == 19:
            val_dataset = DensePASSClass19(
                root_syn,
                val_list,
                crop_size=(syn_w, syn_h),
                sliding_window=(sw_w_stride, sw_size),
                passes=2,
                **extra_kwargs
            )
        else:
            val_dataset = DensePASSClass13(
                root_syn,
                val_list,
                crop_size=(syn_w, syn_h),
                sliding_window=(sw_w_stride, sw_size),
                passes=2,
                **extra_kwargs
            )
    elif dataset_name == "WildPASS":
        custom_root = str(getattr(args, "wildpass_processed_root", "") or "").strip()
        if custom_root:
            root_syn = os.path.abspath(custom_root)
        else:
            root_syn = os.path.join(data_root, "WildPASS")
        val_override = str(getattr(args, "val_list", "") or "").strip()
        if val_override:
            val_list = val_override if os.path.isabs(val_override) else os.path.abspath(
                os.path.join(repo_root, val_override)
            )
        elif custom_root:
            val_list = os.path.join(root_syn, "lists", "val.txt")
        else:
            val_list = os.path.join(repo_root, "dataset/wildpass_list/val.txt")
        # 与 DensePASS 一致：先 resize 到 4096×1024，滑窗 1024×1024 / stride 384，避免 400 高导致 SAM2 Hiera 位置编码报错
        syn_w, syn_h = 4096, 1024
        sw_w_stride, sw_size = 384, 1024
        assert (syn_w - sw_size) % sw_w_stride == 0 and sw_w_stride <= sw_size
        layout = str(getattr(args, "wildpass_layout", "cityscapes") or "cityscapes").strip()
        if layout == "flat":
            val_dataset = WildPASS13(
                root_syn,
                val_list,
                crop_size=(syn_w, syn_h),
                sliding_window=(sw_w_stride, sw_size),
                passes=2,
                set_name="val",
                layout="flat",
                image_dir=str(getattr(args, "wildpass_image_dir", "images") or "images"),
                label_dir=str(getattr(args, "wildpass_label_dir", "labels_trainid") or "labels_trainid"),
            )
        else:
            val_dataset = WildPASS13(
                root_syn,
                val_list,
                crop_size=(syn_w, syn_h),
                sliding_window=(sw_w_stride, sw_size),
                passes=2,
                set_name="val",
            )
    else:
        raise ValueError("Unknown dataset choice.")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    return val_loader, class_names, num_classes

def build_sam_model(args: argparse.Namespace, device: torch.device, ckpt_path: Optional[str] = None) -> nn.Module:
    """
    Build and load the SAM model with the specified backbone and weights.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sam_vendor = os.path.join(repo_root, "sam")

    # Hydra 使用包内配置名（与 sam/sam/build_sam.py 中 HF 映射一致），不是磁盘上的 sam/configs/...
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
    model = OmniSAM_adapt(sam, num_classes=args.num_classes, num_maskmem=args.num_maskmem).to(device)

    load_path = ckpt_path if ckpt_path is not None else getattr(args, "ckpt", None)
    if load_path:
        load_path = os.path.abspath(load_path)
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        load_checkpoint(model, load_path)
        print(f"Loaded checkpoint from: {load_path}")
    else:
        print("No checkpoint: using SAM2 backbone + randomly initialized decoder / LoRA (not paper weights).")

    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch Single-GPU Training Example')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--backbone', type=str, default='sam2_s',
                        help='Backbone model: sam2_l | sam2_b+ | sam2_s | sam2_t')
    parser.add_argument('--dataset', type=str, default='DensePASS',
                        help='Dataset: DensePASS | WildPASS | Stanford2D3D')
    parser.add_argument(
        '--class_set',
        type=int,
        default=13,
        choices=[13, 19],
        help='DensePASS class set: 13 (paper UDA) or 19 (Cityscapes TrainIds). Default: 13',
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='Root folder containing DensePASS/ and WildPASS/ (default: data)',
    )
    parser.add_argument(
        '--wildpass_layout',
        type=str,
        default='cityscapes',
        choices=['cityscapes', 'flat'],
        help='WildPASS folder layout: cityscapes (leftImg8bit/gtFine) or flat (images/labels_trainid).',
    )
    parser.add_argument(
        '--wildpass_processed_root',
        type=str,
        default='',
        help='If set, use this directory as WildPASS root (e.g. data_extra/processed/WildPASS13).',
    )
    parser.add_argument(
        '--val_list',
        type=str,
        default='',
        help='Override validation list file path (default: dataset/wildpass_list/val.txt or processed lists/val.txt).',
    )
    parser.add_argument(
        '--wildpass_image_dir',
        type=str,
        default='images',
        help='When wildpass_layout=flat, subdirectory under processed root for RGB images.',
    )
    parser.add_argument(
        '--wildpass_label_dir',
        type=str,
        default='labels_trainid',
        help='When wildpass_layout=flat, subdirectory for trainId labels.',
    )
    parser.add_argument(
        '--stanford_processed_root',
        type=str,
        default='',
        help='Stanford360_SPAN8 processed root (lists/, images/, labels_trainid/) for Stanford2D3D.',
    )
    parser.add_argument(
        '--stanford_val_list',
        type=str,
        default='',
        help='Optional override path to val list (default: <processed>/lists/val.txt).',
    )
    parser.add_argument(
        '--stanford_legacy_area_crop',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Apply legacy Stanford panorama crop (0,320,W,1728).',
    )
    parser.add_argument(
        '--stanford_stride_w',
        type=int,
        default=256,
        help='Horizontal sliding-window stride for Stanford2D3D (default 256).',
    )
    parser.add_argument(
        '--timing_json',
        type=str,
        default='',
        help='Optional path to write eval timing JSON (eval_time_sec, images_per_sec, ...).',
    )
    parser.add_argument(
        '--save_pred_debug',
        type=int,
        default=0,
        metavar='N',
        help='WildPASS：保存前 N 张验证图的 pred/GT 可视化到 --debug_pred_vis_dir（需 wildpass_processed_root）。0 关闭。',
    )
    parser.add_argument(
        '--pred_gt_histogram',
        action='store_true',
        help='在 valid-GT 像素上累计 pred/GT 类直方图，写入 --pred_gt_histogram_json / --pred_gt_histogram_md。',
    )
    parser.add_argument(
        '--debug_pred_vis_dir',
        type=str,
        default='logs/wildpass_results/debug_pred_vis',
        help='--save_pred_debug 输出目录（相对仓库根目录）。',
    )
    parser.add_argument(
        '--pred_gt_histogram_json',
        type=str,
        default='logs/wildpass_results/reports/wildpass_pred_gt_histogram.json',
        help='pred/GT 直方图 JSON 路径（相对仓库根目录）。',
    )
    parser.add_argument(
        '--pred_gt_histogram_md',
        type=str,
        default='logs/wildpass_results/reports/wildpass_pred_gt_histogram.md',
        help='pred/GT 直方图 Markdown 表路径。',
    )
    parser.add_argument('--use_mem_bank', action='store_true',
                        help='Whether to use memory bank')
    parser.add_argument(
        '--num_maskmem',
        type=int,
        default=None,
        help='Memory bank size (must match training ckpt). With --use_mem_bank, default 9 (OmniSAM release ckpt); use 0 with no mem bank.',
    )
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 1)')
    parser.add_argument(
        '--ckpt',
        action='append',
        default=None,
        metavar='PATH',
        help='OmniSAM / training .pth；可重复传入多个，依次验证。与 --ckpt_list 合并，顺序为先全部 --ckpt 再文件内路径。',
    )
    parser.add_argument(
        '--ckpt_list',
        type=str,
        default=None,
        metavar='FILE',
        help='文本文件：每行一个 .pth 路径，# 后为注释。与 --ckpt 合并。',
    )
    parser.add_argument(
        '--ckpt_names',
        type=str,
        default=None,
        metavar='NAMES',
        help='逗号分隔的短名称，与 checkpoint 数量一致，用于日志与对比表（默认用文件名 stem）。',
    )
    parser.add_argument(
        '--metrics_json',
        type=str,
        default=None,
        help='保存指标 JSON：单个 ckpt 时写一条；多个 ckpt 时写合并列表（含 runs 数组），除非加 --metrics_json_per_run。',
    )
    parser.add_argument(
        '--metrics_json_per_run',
        action='store_true',
        help='与 --metrics_json 同用：多个 ckpt 时为每个模型各写一个文件（路径加 _0,_1 后缀）。',
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='将本次验证的 stdout/stderr 同时写入日志文件（含进度条与指标输出）。',
    )
    parser.add_argument(
        '--log_append',
        action='store_true',
        help='与 --log_file 同用：追加写入而不是覆盖。',
    )

    # SCM (Seamless Circular Memory) arguments
    parser.add_argument(
        '--enable_scm',
        action='store_true',
        help='Enable Seamless Circular Memory with circular padding for 360 degree panorama'
    )
    parser.add_argument(
        '--scm_pad_width',
        type=int,
        default=512,
        help='Circular padding width for SCM (default: 512, half of window_size 1024)'
    )
    parser.add_argument(
        '--enable_bepl',
        action='store_true',
        help='Enable Boundary-Enhanced Prompt Learning (BEPL)'
    )
    parser.add_argument(
        '--bepl_seam_width',
        type=int,
        default=128,
        help='BEPL seam width (default: 128)'
    )
    parser.add_argument(
        '--bepl_strength',
        type=float,
        default=0.2,
        help='BEPL enhancement strength (default: 0.2)'
    )

    args = parser.parse_args()
    setup_seed(args.seed)

    log_fh, old_out, old_err = setup_logging(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, class_names, num_classes = build_dataset_and_loader(args)
    args.num_classes = num_classes  # Make number of classes available for model building

    if args.use_mem_bank:
        # 官方 omnisam_s.pth 等为 num_maskmem=9；若与训练时不一致会报 maskmem_tpos_enc 尺寸错误
        args.num_maskmem = args.num_maskmem if args.num_maskmem is not None else 9
    else:
        args.num_maskmem = 0
    print(f"Using backbone: {args.backbone}, Use memory bank: {args.use_mem_bank}, num_maskmem: {args.num_maskmem}")

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
        if ckpt_path:
            print(f"ckpt: {os.path.abspath(ckpt_path)}")
        else:
            print("ckpt: (none — random decoder / LoRA)")

        model = build_sam_model(args, device, ckpt_path=ckpt_path).to(device)
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

        row = {
            "name": run_name,
            "ckpt": os.path.abspath(ckpt_path) if ckpt_path else None,
            **agg,
        }
        all_rows.append(row)
        all_runs_payload.append(
            {
                "name": run_name,
                "ckpt": row["ckpt"],
                "metrics": {k: round(v, 4) for k, v in agg.items()},
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
            "runs": all_runs_payload,
        }
        with open(original_metrics_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"Wrote combined metrics JSON ({len(all_runs_payload)} runs): {original_metrics_json}")

    if len(ckpt_paths) > 1:
        print_metrics_comparison_table(all_rows)

    print("Validation Finished.")
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
