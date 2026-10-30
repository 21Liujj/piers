#!/usr/bin/env python3
"""
WildPASS13 eval：复用 validation.build_sam_model（与 validation.py 主流程一致）。

补充实验约定：在优化后的代码版本上，用「base/paper」与「final ~60.26」两支权重在 WildPASS val 上验证；
标注应与在 WildPASS2K 上生成/转换的 13 类 TrainId 管线一致。

默认 ckpt（可被 ``--ckpt`` 覆盖；批量对比见 ``01_val_ckpts_wildpass.sh``）：

- base/paper: ``<repo>/exp2/Cityscapes/best_sam2_s_10.pth``
- final: ``<repo>/data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth``

集中定义：``scripts/extra_exp/wildpass/wildpass_default_ckpts.inc.sh``。

默认 **不与训练共用** memory/sparse（``--memory-accel`` 默认 off）；训练加速仅在 ``run_adaptation`` +
``configs/extra_exp/wildpass/wildpass_train_final_optimized.yaml``。

- 比较的是「不同 .pth 权重」在同一套 forward / 滑窗 / merge 下的精度与 wall-clock。
- 可选 ``--memory-accel on``：在 load ckpt 后按 ``--memory-config``（训练 YAML）显式配置
  ``memory_bank.configure_compression`` / ``configure_sparse_retrieval``，并 ``set_runtime_context('eval')``，
  使 compression / sparse 在非 train 阶段按 enabled 生效（见 ``tools/wildpass_memory_accel.py``）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf  # noqa: E402

from validation import (  # noqa: E402
    build_dataset_and_loader,
    build_sam_model,
    setup_seed,
    validation,
)
from tools.wildpass_memory_accel import (  # noqa: E402
    disable_memory_accel,
    enable_memory_accel_from_yaml,
    log_memory_status,
    write_accel_meta,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--repo-root", type=str, default="")
    ap.add_argument("--eval-config", type=str, default="configs/extra_exp/wildpass/wildpass13_eval.yaml")
    ap.add_argument("--backbone", type=str, default="sam2_s")
    ap.add_argument("--metrics-json", type=str, default="metrics.json")
    ap.add_argument("--timing-json", type=str, default="timing.json")
    ap.add_argument("--summary-md", type=str, default="summary.md")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-maskmem", type=int, default=9)
    ap.add_argument("--use-mem-bank", action="store_true", default=True)
    ap.add_argument("--no-use-mem-bank", action="store_false", dest="use_mem_bank")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--memory-accel",
        type=str,
        default="off",
        choices=("off", "on"),
        help="off: 显式关闭 memory compression 与 sparse retrieval；on: 从 --memory-config 读取训练 YAML 并配置 MemoryBank。",
    )
    ap.add_argument(
        "--memory-config",
        type=str,
        default="",
        help="训练侧 final YAML（含 plugins.efficient_tam_memory / efficient_sam2_sparse 与 mem_compression_* 等）；memory-accel=on 时必填。",
    )
    ap.add_argument(
        "--save-pred-debug",
        type=int,
        default=0,
        dest="save_pred_debug",
        metavar="N",
        help="传给 validation：保存前 N 张 WildPASS pred/GT 调试图到 --debug-pred-vis-dir。",
    )
    ap.add_argument(
        "--pred-gt-histogram",
        action="store_true",
        dest="pred_gt_histogram",
        help="传给 validation：写出 pred/GT 类直方图 JSON + Markdown。",
    )
    ap.add_argument(
        "--debug-pred-vis-dir",
        type=str,
        default="logs/wildpass_results/debug_pred_vis",
        dest="debug_pred_vis_dir",
    )
    ap.add_argument(
        "--pred-gt-histogram-json",
        type=str,
        default="logs/wildpass_results/reports/wildpass_pred_gt_histogram.json",
        dest="pred_gt_histogram_json",
    )
    ap.add_argument(
        "--pred-gt-histogram-md",
        type=str,
        default="logs/wildpass_results/reports/wildpass_pred_gt_histogram.md",
        dest="pred_gt_histogram_md",
    )
    args = ap.parse_args()
    if args.memory_accel == "on" and not str(args.memory_config).strip():
        ap.error("--memory-config is required when --memory-accel on")

    repo = Path(args.repo_root or REPO_ROOT).resolve()
    cfg_path = Path(args.eval_config)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    cfg = OmegaConf.load(str(cfg_path))

    processed = cfg.get("processed_root") or cfg.get("root")
    proc_path = Path(processed)
    if not proc_path.is_absolute():
        proc_path = (repo / proc_path).resolve()

    class Args:
        pass

    ns = Args()
    ns.seed = args.seed
    ns.backbone = args.backbone
    ns.dataset = "WildPASS"
    ns.class_set = 13
    ns.data_root = str(repo / str(cfg.get("data_root", "data")))
    ns.wildpass_layout = str(cfg.get("wildpass_layout", "flat"))
    ns.wildpass_processed_root = str(proc_path)
    ns.val_list = str(proc_path / str(cfg.get("val_list", "lists/val.txt")))
    ns.wildpass_image_dir = str(cfg.get("image_dir", "images"))
    ns.wildpass_label_dir = str(cfg.get("label_dir", "labels_trainid"))
    ns.batch_size = args.batch_size
    ns.ckpt = args.ckpt
    ns.ckpt_list = None
    ns.ckpt_names = None
    ns.metrics_json = os.path.abspath(args.metrics_json)
    ns.metrics_json_per_run = False
    ns.log_file = None
    ns.log_append = False
    ns.use_mem_bank = bool(args.use_mem_bank)
    ns.num_maskmem = args.num_maskmem if ns.use_mem_bank else 0
    ns.run_label = "wildpass13"
    ns._skip_per_run_json = False
    ns.save_pred_debug = int(args.save_pred_debug)
    ns.pred_gt_histogram = bool(args.pred_gt_histogram)
    ns.debug_pred_vis_dir = str(args.debug_pred_vis_dir)
    ns.pred_gt_histogram_json = str(args.pred_gt_histogram_json)
    ns.pred_gt_histogram_md = str(args.pred_gt_histogram_md)

    setup_seed(ns.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loader, class_names, num_classes = build_dataset_and_loader(ns)
    ns.num_classes = num_classes

    num_images = len(val_loader.dataset)
    num_frames = val_loader.dataset._get_frame_cnt()
    model = build_sam_model(ns, device, ckpt_path=args.ckpt).to(device)

    mem_cfg_path = ""
    if args.memory_accel == "on":
        mem_p = Path(args.memory_config)
        if not mem_p.is_absolute():
            mem_p = (repo / mem_p).resolve()
        if not mem_p.is_file():
            raise FileNotFoundError(f"--memory-config not found: {mem_p}")
        mem_cfg_path = str(mem_p)
        status = enable_memory_accel_from_yaml(model, mem_p, num_classes=num_classes)
        log_memory_status("[memory_accel=on]", status)
    else:
        status = disable_memory_accel(model)
        log_memory_status("[memory_accel=off]", status)
        print(
            "[memory_accel=off] memory compression / sparse retrieval explicitly disabled "
            f"(memory_compress_enabled={status.get('memory_compress_enabled')}, "
            f"sparse_enabled={status.get('sparse_enabled')})",
            flush=True,
        )

    timing_dir = Path(args.timing_json).resolve().parent
    write_accel_meta(
        timing_dir / "memory_accel_meta.json",
        memory_accel=args.memory_accel,
        memory_config=mem_cfg_path or "(n/a)",
        status=status,
    )

    t0 = time.perf_counter()
    agg = validation(
        model=model,
        data_loader=val_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        backbone_name=args.backbone,
        num_maskmem=ns.num_maskmem,
        dataset_name="WildPASS",
        args=ns,
    )
    elapsed = time.perf_counter() - t0
    ips = float(num_images / elapsed) if elapsed > 0 else 0.0

    timing = {
        "dataset": "WildPASS13",
        "eval_time_sec": round(elapsed, 4),
        "images_per_sec": round(ips, 4),
        "num_eval_images": int(num_images),
        "num_frames_per_image": int(num_frames),
        "memory_accel": args.memory_accel,
        "memory_config": mem_cfg_path or None,
        "note": "DensePASS-style sliding-window WildPASS eval",
    }
    timing.update(status)
    Path(args.timing_json).write_text(json.dumps(timing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_lines = [
        "# WildPASS13 evaluation summary",
        "",
        f"- memory_accel: `{args.memory_accel}`",
        f"- memory_config: `{mem_cfg_path or '(n/a)'}`",
        f"- memory_compress_enabled: `{status.get('memory_compress_enabled')}`",
        f"- sparse_enabled: `{status.get('sparse_enabled')}`",
        f"- runtime_stage: `{status.get('runtime_stage')}`",
        f"- memory_enable_in_train: `{status.get('memory_enable_in_train')}`",
        f"- sparse_enable_in_train: `{status.get('sparse_enable_in_train')}`",
        "",
        f"- checkpoint: `{args.ckpt}`",
        f"- mIoU: {agg.get('mIoU', float('nan')):.4f}",
        f"- mAcc: {agg.get('mAcc', float('nan')):.4f}",
        f"- PixAcc: {agg.get('PixAcc', float('nan')):.4f}",
        f"- FWIoU: {agg.get('FWIoU', float('nan')):.4f}",
        f"- mDice: {agg.get('mDice', float('nan')):.4f}",
        f"- eval_time_sec: {timing['eval_time_sec']}",
        f"- images_per_sec: {timing['images_per_sec']}",
        "",
    ]
    Path(args.summary_md).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({"timing": timing, "metrics": {k: round(float(v), 4) for k, v in agg.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
