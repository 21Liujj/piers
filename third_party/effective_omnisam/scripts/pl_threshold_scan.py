#!/usr/bin/env python
"""
scripts/pl_threshold_scan.py

PL-only diagnostic: runs merge_patch_sequence at multiple (mode, threshold)
combinations on a fixed50 subset without entering training.

Usage
-----
  cd /root/autodl-tmp/OmniSAM-main
  /root/miniconda3/envs/omnisam/bin/python scripts/pl_threshold_scan.py \
      --checkpoint /root/autodl-tmp/omnisam_s.pth \
      --dataset Cityscapes \
      --class_set 13 \
      --data_root data \
      --pd_root data_pd_scan \
      --num_images 50 \
      --seed 1 \
      --out_dir logs/pl_threshold_scan

Output
------
  logs/pl_threshold_scan/scan_results.jsonl   – one JSON per (mode,threshold)
  logs/pl_threshold_scan/scan_summary.txt     – human-readable table
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import torch

# ── repo root on path ────────────────────────────────────────────────────────
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_root))
import utils.sam2_bootstrap  # noqa: F401  must follow sys.path setup

from utils.pl_merge import PlMergeConfig, merge_patch_sequence
from utils.tools import merge_width_scatter_np_uncertainty
from utils.label_spaces import CITYSCAPES13_CLASSES


# small classes present in 13-class space
SMALL_CLASS_INDICES = {5: "pole", 6: "traffic_light", 7: "traffic_sign", 11: "person"}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--dataset",     default="Cityscapes")
    p.add_argument("--class_set",   type=int, default=13)
    p.add_argument("--data_root",   default="data")
    p.add_argument("--pd_root",     default="data_pd_scan")
    p.add_argument("--num_images",  type=int, default=50)
    p.add_argument("--seed",        type=int, default=1)
    p.add_argument("--out_dir",     default="logs/pl_threshold_scan")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--amp",         action="store_true")
    p.add_argument("--no_gpu_merge",action="store_true",
                   help="Force merge on CPU (slower, for debugging)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Build model + dataloader  (re-uses build.py helpers)
# ─────────────────────────────────────────────────────────────────────────────
def _build(args_ns):
    from utils.build import DatasetBuilder
    import dataclasses

    @dataclasses.dataclass
    class _Stub:
        dataset: str = "Cityscapes"
        class_set: int = 13
        data_root: str = "data"
        backbone: str = "sam2_s"
        num_maskmem: int = 9
        mem_compression_enabled: bool = False
        mem_compression_mode: str = "none"
        mem_compression_location: str = "before_attention"
        mem_compression_kernel_size: int = 2
        mem_compression_stride: int = 2
        mem_compression_keep_global_tokens: int = 0
        memory_select_mode: str = "none"
        pl_batch_size: int = 1
        pl_num_workers: int = 4
        pl_shuffle: bool = False
        depth_prior: dict = dataclasses.field(default_factory=dict)
        sparse_memory_enabled: bool = False
        sparse_memory_keep_ratio: float = 0.5
        sparse_memory_min_tokens: int = 64
        sparse_memory_saliency: str = "feature_norm"
        sparse_memory_cache_saliency: bool = True
        sparse_memory_fallback: bool = True
        sparse_apply_in_pl: bool = False
        sparse_apply_in_train: bool = True
        plugins_cfg: dict = dataclasses.field(default_factory=dict)
        decoder_conv_mode: str = "standard"
        decoder_upsample_mode: str = "bilinear"
        depth_adapter: dict = dataclasses.field(default_factory=dict)
        text_fusion_enabled: bool = False
        use_text_proto: bool = False
        text_proto_path: str = ""
        text_fusion_type: str = "none"
        lambda_txt: float = 0.0

    stub = _Stub(
        dataset=args_ns.dataset,
        class_set=args_ns.class_set,
        data_root=args_ns.data_root,
    )

    db = DatasetBuilder(stub)
    loader, class_names, num_classes, frame_cnt = db.build_tgt_for_pl()
    return loader, class_names, num_classes, frame_cnt


# ─────────────────────────────────────────────────────────────────────────────
# Forward one batch through the model, collect logits_maps
# ─────────────────────────────────────────────────────────────────────────────
def _forward_batch(model, batch, device, frame_cnt, use_amp):
    depth_patches = None
    if len(batch) >= 4 and torch.is_tensor(batch[-1]) and batch[-1].ndim == 5:
        images, _, names, depth_patches = batch
    else:
        images, _, names = batch
    images = images.to(device)
    B = images.shape[0]
    H, W = images.shape[-2:]
    C = model.decoder.num_classes if hasattr(model, "decoder") else 13
    logits_maps = np.zeros((B, 2 * frame_cnt, C, H, W), dtype=np.float32)

    with torch.no_grad():
        for fwd_rev in range(2):
            if fwd_rev == 1:
                model.memory_bank._init_output_dict_source()
            for fi in range(frame_cnt):
                fi_img = fwd_rev * frame_cnt + fi
                frame_img = images[:, fi_img]
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out, _, _ = model(fi, frame_img)
                logits_maps[:, fi_img] = out.detach().float().cpu().numpy()
        model.memory_bank._init_output_dict_source()

    return logits_maps, list(names)


# ─────────────────────────────────────────────────────────────────────────────
# Stats per (mode, threshold)
# ─────────────────────────────────────────────────────────────────────────────
def _collect_stats(
    all_logits: List[np.ndarray],      # list of (B,2L,C,H,Wp) per batch
    all_names: List[List[str]],
    mode: str,
    threshold: float,
    orig_size: tuple,
    W_patch: int,
    stride_w: int,
    merge_device: str,
    class_names: List[str],
) -> dict:
    t0 = time.perf_counter()
    cfg = PlMergeConfig(mode=mode, threshold=threshold, ignore_index=255)

    total_px = 0
    valid_px = 0
    cov_hist_acc: Counter = Counter()
    vcr_num: Dict[int, int] = {}   # valid per coverage
    vcr_den: Dict[int, int] = {}   # total per coverage
    cls_valid: Dict[int, int] = {i: 0 for i in range(len(class_names))}
    cls_total: Dict[int, int] = {i: 0 for i in range(len(class_names))}
    conf_sum = 0.0
    conf_sq  = 0.0
    conf_cnt = 0

    t_merge_only = 0.0
    num_images = 0

    for logits_batch, names_batch in zip(all_logits, all_names):
        tm0 = time.perf_counter()
        if mode == "legacy_sum":
            pred = merge_width_scatter_np_uncertainty(
                logits_np=logits_batch,
                orig_size=orig_size,
                W_patch=W_patch,
                stride_w=stride_w,
                passes=2,
                device=merge_device,
                return_probs=False,
                threshold=threshold,
            )
            result_conf = None
            result_cov  = None
        else:
            res = merge_patch_sequence(
                logits_np=logits_batch,
                orig_size=orig_size,
                W_patch=W_patch,
                stride_w=stride_w,
                passes=2,
                device=merge_device,
                cfg=cfg,
            )
            pred = res.pred
            result_conf = res.confidence
            result_cov  = res.coverage

        t_merge_only += time.perf_counter() - tm0

        B = pred.shape[0]
        num_images += B

        for bi in range(B):
            p = pred[bi]
            valid_mask = p != 255
            total_px += p.size
            valid_px  += int(valid_mask.sum())

            if result_cov is not None:
                cov_b = result_cov[bi]
                for cv in [1, 2, 3]:
                    mask_cv = cov_b == cv
                    vcr_den[cv] = vcr_den.get(cv, 0) + int(mask_cv.sum())
                    vcr_num[cv] = vcr_num.get(cv, 0) + int((p[mask_cv] != 255).sum())
                cov_hist_acc.update(dict(Counter(cov_b.ravel().tolist())))
            else:
                # legacy: approximate coverage from geometry
                starts = list(range(0, orig_size[1] - W_patch + 1, stride_w))
                cov_col = np.zeros(orig_size[1], dtype=int)
                for s in starts:
                    cov_col[s:s + W_patch] += 1
                cov_img = np.broadcast_to(cov_col[None, :], p.shape)
                for cv in [1, 2, 3]:
                    mask_cv = cov_img == cv
                    vcr_den[cv] = vcr_den.get(cv, 0) + int(mask_cv.sum())
                    vcr_num[cv] = vcr_num.get(cv, 0) + int((p[mask_cv] != 255).sum())
                    cov_hist_acc[cv] += int(mask_cv.sum())

            if result_conf is not None:
                c_v = result_conf[bi][valid_mask]
                conf_sum += float(c_v.sum())
                conf_sq  += float((c_v ** 2).sum())
                conf_cnt += len(c_v)

            for ci in range(len(class_names)):
                cls_total[ci] += int((p != 255).sum() + (p == 255).sum())
                cls_valid[ci] += int((p == ci).sum())

    t_total = time.perf_counter() - t0

    valid_ratio = valid_px / max(1, total_px)
    conf_mean = conf_sum / max(1, conf_cnt) if conf_cnt > 0 else float("nan")
    conf_std  = float(np.sqrt(max(0, conf_sq / max(1, conf_cnt) - conf_mean**2))) if conf_cnt > 0 else float("nan")

    vcr = {cv: vcr_num.get(cv, 0) / max(1, vcr_den.get(cv, 0)) for cv in [1, 2, 3]}
    class_valid_px   = {class_names[i]: int(v) for i, v in cls_valid.items()}
    class_retain     = {class_names[i]: cls_valid[i] / max(1, cls_valid[i] + (total_px // len(class_names) - cls_valid[i])) for i in range(len(class_names))}

    # simpler retained ratio: class pixels / total non-ignore
    class_retained_ratio = {}
    for ci, nm in enumerate(class_names):
        class_retained_ratio[nm] = cls_valid[ci] / max(1, valid_px) if valid_px > 0 else 0.0

    zero_ret = [nm for nm, v in class_valid_px.items() if v == 0]

    return {
        "merge_mode":             mode,
        "threshold":              threshold,
        "num_images":             num_images,
        "pl_merge_time_sec":      float(t_merge_only),
        "total_scan_time_sec":    float(t_total),
        "valid_pixel_ratio":      float(valid_ratio),
        "ignore_ratio":           float(1.0 - valid_ratio),
        "confidence_mean":        float(conf_mean),
        "confidence_std":         float(conf_std),
        "coverage_histogram":     {str(k): int(v) for k, v in sorted(cov_hist_acc.items())},
        "valid_ratio_by_coverage": {str(k): float(v) for k, v in vcr.items()},
        "class_valid_pixels":     class_valid_px,
        "class_retained_pixel_fraction": class_retained_ratio,
        "zero_retained_classes":  zero_ret,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = _parse()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    merge_device = "cpu" if args.no_gpu_merge else str(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[scan] device={device}  merge_device={merge_device}", flush=True)
    print(f"[scan] loading model from {args.checkpoint} ...", flush=True)

    # ── build model ──────────────────────────────────────────────────────────
    from utils.build import build_model
    import dataclasses

    @dataclasses.dataclass
    class _ModelArgs:
        backbone: str = "sam2_s"
        num_maskmem: int = 9
        mem_compression_enabled: bool = False
        mem_compression_mode: str = "none"
        mem_compression_location: str = "before_attention"
        mem_compression_kernel_size: int = 2
        mem_compression_stride: int = 2
        mem_compression_keep_global_tokens: int = 0
        memory_select_mode: str = "none"
        sparse_memory_enabled: bool = False
        sparse_memory_keep_ratio: float = 0.5
        sparse_memory_min_tokens: int = 64
        sparse_memory_saliency: str = "feature_norm"
        sparse_memory_cache_saliency: bool = True
        sparse_memory_fallback: bool = True
        sparse_apply_in_pl: bool = False
        sparse_apply_in_train: bool = True
        plugins_cfg: dict = dataclasses.field(default_factory=dict)
        decoder_conv_mode: str = "standard"
        decoder_upsample_mode: str = "bilinear"
        depth_adapter: dict = dataclasses.field(default_factory=dict)
        text_fusion_enabled: bool = False
        use_text_proto: bool = False
        text_proto_path: str = ""
        text_fusion_type: str = "none"
        lambda_txt: float = 0.0
        class_set: int = 13
        dataset: str = "Cityscapes"

    from utils.build import build_model
    model_args = _ModelArgs(class_set=args.class_set, dataset=args.dataset)
    class_names = CITYSCAPES13_CLASSES[:args.class_set]
    num_classes  = len(class_names)
    model = build_model(model_args, num_classes, model_args.num_maskmem, device)

    from utils.checkpoint_adapt import smart_load_checkpoint
    smart_load_checkpoint(model, args.checkpoint)
    model.eval()
    print("[scan] model loaded.", flush=True)

    # ── dataloader ───────────────────────────────────────────────────────────
    loader, _, _, frame_cnt = _build(args)
    STRIDE   = 384
    W_PATCH  = 1024
    ORIG_SIZE = (1024, 4096)

    # ── collect logits for fixed50 ───────────────────────────────────────────
    print(f"[scan] collecting logits for {args.num_images} images ...", flush=True)
    all_logits: List[np.ndarray] = []
    all_names:  List[List[str]]  = []
    seen = 0
    t_fwd_start = time.perf_counter()
    for batch in loader:
        if seen >= args.num_images:
            break
        logits_batch, names_batch = _forward_batch(
            model, batch, device, frame_cnt, use_amp=args.amp
        )
        # trim to remaining quota
        remaining = args.num_images - seen
        logits_batch = logits_batch[:remaining]
        names_batch  = names_batch[:remaining]
        all_logits.append(logits_batch)
        all_names.append(names_batch)
        seen += len(names_batch)
        print(f"  [{seen}/{args.num_images}]", flush=True)
    t_fwd = time.perf_counter() - t_fwd_start
    print(f"[scan] forward done in {t_fwd:.1f}s  ({seen} images)", flush=True)

    # ── sweep ────────────────────────────────────────────────────────────────
    sweep = [("legacy_sum", 0.85)]
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        sweep.append(("normalized_mean", thr))

    jsonl_path = os.path.join(args.out_dir, "scan_results.jsonl")
    results = []
    with open(jsonl_path, "w") as fout:
        for mode, thr in sweep:
            print(f"[scan] mode={mode}  threshold={thr:.2f} ...", end="  ", flush=True)
            stats = _collect_stats(
                all_logits, all_names, mode, thr,
                ORIG_SIZE, W_PATCH, STRIDE, merge_device, class_names,
            )
            stats["pl_forward_time_sec"] = float(t_fwd)
            fout.write(json.dumps(stats, ensure_ascii=False) + "\n")
            results.append(stats)
            vr = stats["valid_pixel_ratio"]
            cm = stats["confidence_mean"]
            mt = stats["pl_merge_time_sec"]
            print(f"valid={vr:.4f}  conf_mean={cm:.4f}  merge={mt:.2f}s", flush=True)

    # ── summary table ─────────────────────────────────────────────────────────
    legacy_vr = results[0]["valid_pixel_ratio"]
    print("\n" + "="*90, flush=True)
    print(f"{'mode':<20}{'thr':>6}  {'valid_r':>8}  {'delta_vs_legacy':>16}  "
          f"{'cov1_vr':>8}  {'cov2_vr':>8}  {'cov3_vr':>8}  {'merge_s':>8}", flush=True)
    print("-"*90, flush=True)

    match_thr = None
    best_delta = 1e9
    for r in results:
        mode = r["merge_mode"]
        thr  = r["threshold"]
        vr   = r["valid_pixel_ratio"]
        delta = vr - legacy_vr
        vcr  = r["valid_ratio_by_coverage"]
        mt   = r["pl_merge_time_sec"]
        label = f"{mode}@{thr:.2f}"
        print(f"{label:<26}  {vr:>8.4f}  {delta:>+16.4f}  "
              f"{vcr.get('1',0):>8.4f}  {vcr.get('2',0):>8.4f}  "
              f"{vcr.get('3',0):>8.4f}  {mt:>8.2f}s", flush=True)
        if mode == "normalized_mean" and abs(delta) < best_delta:
            best_delta = abs(delta)
            match_thr  = thr

    print("="*90, flush=True)
    print(f"\nA0-norm-match (closest valid_ratio to legacy@0.85): "
          f"normalized_mean @ threshold={match_thr}", flush=True)

    # small classes
    print("\nSmall-class retained pixels:", flush=True)
    print(f"{'class':<20}", end="")
    for r in results:
        label = f"{r['merge_mode'][:3]}@{r['threshold']:.2f}"
        print(f"  {label:>12}", end="")
    print()
    for ci, cname in enumerate(class_names):
        if ci not in SMALL_CLASS_INDICES:
            continue
        print(f"{cname:<20}", end="")
        for r in results:
            px = r["class_valid_pixels"].get(cname, 0)
            print(f"  {px:>12}", end="")
        print()

    # save summary
    summary_path = os.path.join(args.out_dir, "scan_summary.txt")
    print(f"\nResults written to {jsonl_path}", flush=True)
    print(f"Summary written to {summary_path}", flush=True)

    with open(summary_path, "w") as f:
        f.write(f"A0-norm-match: normalized_mean @ threshold={match_thr}\n")
        f.write(f"legacy valid_ratio = {legacy_vr:.4f}\n")
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
