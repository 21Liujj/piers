import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path when running as a script from tools/
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.sam2_bootstrap  # noqa: F401

from utils.build import DatasetBuilder, build_model, load_checkpoint, resolve_class_names


@dataclass
class Agg:
    px: int = 0
    conf_sum: float = 0.0
    kept_px: int = 0
    kept_conf_sum: float = 0.0

    def add(self, n_px: int, conf_sum: float, n_kept: int, kept_conf_sum: float):
        self.px += int(n_px)
        self.conf_sum += float(conf_sum)
        self.kept_px += int(n_kept)
        self.kept_conf_sum += float(kept_conf_sum)


def _pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _reset_mem(model):
    # Keep consistent with adaptation.strategies.WithMemStrategy.reset_memory
    model.memory_bank._init_output_dict_source()
    model.memory_bank._init_output_dict_target()


@torch.no_grad()
def run_stats(
    model,
    target_loader,
    class_names: List[str],
    conf_thr: float,
    ignore_index: int,
    max_batches: int,
    device: torch.device,
) -> Tuple[Dict[int, Agg], Dict[int, Agg], Dict[str, float]]:
    num_classes = len(class_names)

    # (A) aggregate by model predicted class on target domain
    by_pred: Dict[int, Agg] = defaultdict(Agg)
    # (B) aggregate by pseudo-label class (from gtFine pseudo labels)
    by_pl: Dict[int, Agg] = defaultdict(Agg)

    total_px = 0
    total_kept_px = 0
    pl_ignore_px = 0
    pl_total_px = 0

    for bidx, batch in enumerate(target_loader):
        if max_batches > 0 and bidx >= max_batches:
            break

        images_t, labels_t, *_ = batch
        images_t = images_t.to(device)
        labels_t = labels_t.to(device)

        frame_cnt = images_t.shape[1]
        for frame_idx in range(frame_cnt):
            frame_img_t = images_t[:, frame_idx]
            frame_pl_t = labels_t[:, frame_idx]  # (B,H,W)

            logits, *_ = model(frame_idx, frame_img_t, mem_type="target")
            probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
            conf, pred = probs.max(dim=1)  # (B,H,W)
            kept = conf > conf_thr

            total_px += int(conf.numel())
            total_kept_px += int(kept.sum().item())

            # pseudo-label ignore ratio (from saved pseudo labels)
            pl_total_px += int(frame_pl_t.numel())
            pl_ignore_px += int((frame_pl_t == ignore_index).sum().item())

            # aggregate by predicted class
            for cid in range(num_classes):
                m = pred == cid
                n_px = int(m.sum().item())
                if n_px:
                    csum = float(conf[m].sum().item())
                    mk = m & kept
                    n_kept = int(mk.sum().item())
                    ksum = float(conf[mk].sum().item()) if n_kept else 0.0
                    by_pred[cid].add(n_px, csum, n_kept, ksum)

            # aggregate by pseudo-label class (ignore_index excluded)
            valid = frame_pl_t != ignore_index
            for cid in range(num_classes):
                m = (frame_pl_t == cid) & valid
                n_px = int(m.sum().item())
                if n_px:
                    csum = float(conf[m].sum().item())
                    mk = m & kept
                    n_kept = int(mk.sum().item())
                    ksum = float(conf[mk].sum().item()) if n_kept else 0.0
                    by_pl[cid].add(n_px, csum, n_kept, ksum)

        _reset_mem(model)

    summary = {
        "conf_thr": conf_thr,
        "total_px": float(total_px),
        "kept_px": float(total_kept_px),
        "kept_ratio": _safe_div(total_kept_px, total_px),
        "pl_ignore_ratio": _safe_div(pl_ignore_px, pl_total_px),
    }
    return by_pred, by_pl, summary


def render_aggs(title: str, aggs: Dict[int, Agg], class_names: List[str]) -> str:
    num_classes = len(class_names)
    lines = []
    lines.append(title)
    lines.append(f"{'class':<15} {'px':>12} {'mean_conf':>10} {'kept%':>9} {'kept_mean':>10}")
    for cid in range(num_classes):
        a = aggs.get(cid, Agg())
        mean_conf = _safe_div(a.conf_sum, a.px)
        kept_ratio = _safe_div(a.kept_px, a.px)
        kept_mean = _safe_div(a.kept_conf_sum, a.kept_px)
        lines.append(
            f"{class_names[cid]:<15} {a.px:>12,} {mean_conf:>10.4f} {_pct(kept_ratio):>9} {kept_mean:>10.4f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Analyze per-class confidence/coverage on target domain (no training).")
    ap.add_argument("--backbone", type=str, default="sam2_s", choices=["sam2_t", "sam2_s", "sam2_b+", "sam2_l"])
    ap.add_argument("--dataset", type=str, default="Cityscapes", choices=["Cityscapes", "SynPASS", "Stanford2D3D"])
    ap.add_argument(
        "--class_set",
        type=int,
        default=13,
        choices=[13, 19],
        help="Label space: 13 (legacy mapping) or 19 (Cityscapes TrainIds).",
    )
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--pd_root", type=str, default="pd_uda_densepass")
    ap.add_argument(
        "--pl_dir",
        type=str,
        required=True,
        help="Pseudo-label epoch dir, e.g. pd_uda_densepass/DensePASS/sam2_s/Cityscapes/epoch1",
    )
    ap.add_argument("--tgt_ckpt", type=str, required=True, help="Checkpoint path for model_tgt to evaluate.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_maskmem", type=int, default=9)
    ap.add_argument("--conf_thr", type=float, default=0.85, help="Max-softmax threshold used to define 'kept' pixels.")
    ap.add_argument("--ignore_index", type=int, default=255)
    ap.add_argument("--max_batches", type=int, default=50, help="Limit number of target batches for quick analysis.")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Force device for inference.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    class_names = resolve_class_names("Cityscapes" if args.dataset in ["Cityscapes", "SynPASS"] else "Stanford2D3D", args.class_set)

    # Build target loader from pseudo-label dir (same as adaptation training).
    class Dummy:
        pass

    # Mimic fields used by DatasetBuilder/build_model
    dummy = Dummy()
    dummy.backbone = args.backbone
    dummy.dataset = args.dataset
    dummy.class_set = args.class_set
    dummy.data_root = args.data_root
    dummy.pd_root = args.pd_root
    dummy.batch_size = args.batch_size
    dummy.num_maskmem = args.num_maskmem

    db = DatasetBuilder(dummy)
    target_loader, _ = db.build_tgt_from_pd(args.pl_dir)

    model = build_model(dummy, num_classes=len(class_names), num_maskmem=args.num_maskmem, device=device)
    load_checkpoint(model, args.tgt_ckpt)
    model.eval()

    by_pred, by_pl, summary = run_stats(
        model=model,
        target_loader=target_loader,
        class_names=class_names,
        conf_thr=args.conf_thr,
        ignore_index=args.ignore_index,
        max_batches=args.max_batches,
        device=device,
    )

    print("== Summary ==")
    print(f"conf_thr={summary['conf_thr']}")
    print(f"kept_ratio(model_pred_conf>thr)={_pct(summary['kept_ratio'])}")
    print(f"pseudo_label_ignore_ratio(from PNG)={_pct(summary['pl_ignore_ratio'])}  [ignore_index={args.ignore_index}]")
    print("")
    print(render_aggs("== Aggregated by MODEL predicted class ==", by_pred, class_names))
    print("")
    print(render_aggs("== Aggregated by PSEUDO-LABEL class (from saved PNG) ==", by_pl, class_names))


if __name__ == "__main__":
    main()

