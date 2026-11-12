"""Indoor LoRA fine-tune on the ORIGINAL OmniSAM checkpoint (semantic branch only).

Base: third_party/effective_omnisam `build_sam_model` + `OmniSAM_adapt`.
The frozen 13-class street checkpoint is loaded with strict=False, so the
13-class decoder classifier (decoder.linear_pred) is skipped and the new
8-class SPAN8 head stays randomly initialised -> it is fully trained.

Trainable:
  * image-encoder LoRA (linear_a_/linear_b_) inside encoder.sam.image_encoder
  * decoder SegFormer head (decoder.*), including the new 8-class linear_pred
Frozen:
  * encoder base weights, memory bank, DINO/text branch, all outdoor modules

Geometry: indoor 1024x512 canvas, 512x512 windows, stride 64, 9 starts x 2
passes = 18 windows per image (see indoor_geometry.py).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/root/autodl-tmp/piers-main")
EFF = REPO / "third_party" / "effective_omnisam"
sys.path.insert(0, str(EFF))
sys.path.insert(0, str(REPO / "scripts" / "piers"))

from indoor_geometry import INDOOR_W, INDOOR_H, INDOOR_WIN, INDOOR_STRIDE, semantic_starts  # noqa: E402

SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]
NUM_CLASSES = 8
IGNORE = 255
DATA_ROOT = "/root/autodl-tmp/data_extra/Stanford2d3d_Seg"
CKPT = "/root/autodl-tmp/weights/omnisam_s.pth"
RUNS = REPO / "runs" / "INDOOR-FOLD1-20261109"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_stanford_loader(list_name, crop_size=(INDOOR_W, INDOOR_H),
                          sliding_window=(INDOOR_STRIDE, INDOOR_WIN), passes=2):
    """Reuse the upstream Stanford360Span8 dataset class verbatim."""
    from dataset.Stanford360Span8 import Stanford360PanoramaVal
    ds = Stanford360PanoramaVal(
        DATA_ROOT, str(Path(DATA_ROOT) / "lists" / list_name),
        crop_size=crop_size, sliding_window=sliding_window, passes=passes)
    return ds


def build_model(num_classes=NUM_CLASSES, device="cuda", ckpt=CKPT):
    """Build OmniSAM_adapt with an 8-class head, load the frozen 13-class ckpt."""
    import argparse as _ap
    from sam.build_sam import build_sam2
    from model.model import OmniSAM_adapt
    from utils.build import load_checkpoint

    sam_vendor = EFF / "sam"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_ckpt = sam_vendor / "checkpoints" / "sam2.1_hiera_small.pt"
    if not sam2_ckpt.is_file():
        raise FileNotFoundError(f"SAM2 backbone weight not found: {sam2_ckpt}")
    sam = build_sam2(model_cfg, str(sam2_ckpt), device=torch.device(device))
    model = OmniSAM_adapt(sam, num_classes=num_classes, num_maskmem=9).to(device)
    load_checkpoint(model, ckpt)  # strict=False -> 13-cls head skipped
    return model


def split_trainable(model):
    """Return (trainable_params, lora_params, head_params, frozen_count)."""
    lora, head, frozen = [], [], 0
    for n, p in model.named_parameters():
        if "linear_a_" in n or "linear_b_" in n:
            p.requires_grad_(True); lora.append((n, p))
        elif n.startswith("decoder."):
            p.requires_grad_(True); head.append((n, p))
        else:
            p.requires_grad_(False); frozen += p.numel()
    return lora, head, frozen


def window_loss(logits, labels):
    """Per-window CE with ignore=255. logits: (B,C,H,W), labels: (B,H,W)."""
    return F.cross_entropy(logits, labels, ignore_index=IGNORE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "indoor_lora"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--head-warmup", type=int, default=200)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train-list", default="train_fit.txt")
    ap.add_argument("--val-list", default="train_val.txt")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {"args": vars(args), "device": device}

    # ---------- CPU-only audit of what will be trainable ----------
    raw = torch.load(CKPT, map_location="cpu")
    ks = list(raw.keys())
    rep["ckpt_audit"] = {
        "total_tensors": len(ks),
        "lora_tensors": len([k for k in ks if "linear_a_" in k or "linear_b_" in k]),
        "decoder_tensors": len([k for k in ks if k.startswith("decoder.")]),
        "memory_tensors": len([k for k in ks if "memory" in k.lower()]),
        "head_shape_13cls": list(raw["decoder.linear_pred.weight"].shape),
        "note": "decoder.linear_pred (13 cls) is shape-mismatched vs the new 8-cls head "
                "-> skipped by strict=False and trained from scratch",
    }
    if args.audit_only:
        (out / "TRAINABLE_AUDIT.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps(rep, indent=2)); return

    model = build_model(device=device)
    lora, head, frozen = split_trainable(model)
    lora_params = [p for _, p in lora]; head_params = [p for _, p in head]
    rep["trainable"] = {
        "lora_tensors": len(lora), "lora_params": int(sum(p.numel() for p in lora_params)),
        "head_tensors": len(head), "head_params": int(sum(p.numel() for p in head_params)),
        "frozen_params": int(frozen),
        "trainable_total": int(sum(p.numel() for p in lora_params + head_params)),
    }
    (out / "TRAINABLE_AUDIT.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["trainable"], indent=2))

    ds = build_stanford_loader(args.train_list)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True,
                                     num_workers=4, drop_last=True)
    print("train images:", len(ds), "windows/img:", len(ds[0][0]))

    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.wd)
    total = args.steps
    warm = args.head_warmup

    def lr_scale(step):
        if step < warm:
            return 1.0
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))

    model.train()
    step = 0; micro = 0; t0 = time.time()
    losses = []
    model.memory_bank._init_output_dict_source()
    while step < total:
        for img, lbl, size, name in dl:
            if micro == 0:
                # warmup: freeze LoRA, only head trains
                for p in lora_params:
                    p.requires_grad_(step >= warm)
            img = img[0].to(device)      # (W,3,512,512)
            lbl = lbl[0].to(device)      # (W,512,512)
            model.memory_bank._init_output_dict_source()
            total_loss = 0.0
            for w in range(img.shape[0]):
                out_w, _, _ = model(w, img[w:w+1].contiguous())
                total_loss = total_loss + window_loss(out_w, lbl[w:w+1])
            loss = total_loss / img.shape[0]
            (loss / args.accum).backward()
            losses.append(float(loss.detach()))
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, args.clip)
                for g, base in zip(opt.param_groups, [args.lr_lora, args.lr_head]):
                    g["lr"] = base * lr_scale(step)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    msg = {"step": step, "loss": round(float(np.mean(losses[-80:])), 4),
                           "lr_head": opt.param_groups[1]["lr"],
                           "lr_lora": opt.param_groups[0]["lr"],
                           "elapsed_s": round(time.time() - t0, 1)}
                    print(json.dumps(msg), flush=True)
                    (out / "train_log.jsonl").open("a", encoding="utf-8").write(json.dumps(msg) + "\n")
                if step >= total:
                    break
        micro = 0
    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "indoor_lora_last.pth")
    print("SAVED", out / "indoor_lora_last.pth")


if __name__ == "__main__":
    main()
