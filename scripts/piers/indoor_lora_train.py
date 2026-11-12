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
# The vendored SAM2 package lives one level deeper than the upstream comment
# suggests: effective_omnisam/sam/sam/build_sam.py, with the hydra configs
# under effective_omnisam/sam/sam/configs/sam2.1/.
sys.path.insert(0, str(EFF))
sys.path.insert(0, str(EFF / "sam"))
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


def build_model(num_classes=NUM_CLASSES, device="cuda", ckpt=CKPT,
                text_protos=None, lambda_txt=0.0):
    """Build OmniSAM_adapt with an 8-class head, load the frozen 13-class ckpt.

    `text_protos` (optional) attaches the indoor SPAN8 text prototypes so the
    I1 text branch is *trained* together with the head.  This matters: the
    frozen street checkpoint contains **no** text-branch weights (0 of its 1806
    tensors), so an untrained `txt_head` projector would be pure noise and
    would poison every arm that claims to use I1.
    """
    import argparse as _ap
    from sam.build_sam import build_sam2
    from model.model import OmniSAM_adapt
    from utils.build import load_checkpoint

    sam_vendor = EFF / "sam" / "sam"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_ckpt = sam_vendor / "checkpoints" / "sam2.1_hiera_small.pt"
    if not sam2_ckpt.is_file():
        # fall back to the weights that ship outside the vendored package
        alt = Path("/root/autodl-tmp/weights/sam2.1_hiera_small.pt")
        if alt.is_file():
            sam2_ckpt = alt
        else:
            raise FileNotFoundError(f"SAM2 backbone weight not found: {sam2_ckpt}")
    sam = build_sam2(model_cfg, str(sam2_ckpt), device=torch.device(device))
    model = OmniSAM_adapt(
        sam, num_classes=num_classes, num_maskmem=9,
        text_prototypes=text_protos,
        d_txt=int(text_protos.shape[-1]) if text_protos is not None else None,
        lambda_txt=float(lambda_txt),
        txt_tau=0.07,
    ).to(device)
    # Upstream smart_load_checkpoint uses strict=False, which still raises on a
    # shape mismatch (13-cls classifier vs our 8-cls head).  indoor_ckpt
    # filters that tensor out and reports what happened.
    from indoor_ckpt import load_indoor_checkpoint
    load_indoor_checkpoint(model, ckpt)
    return model


def split_trainable(model):
    """Return (trainable_params, lora_params, head_params, frozen_count).

    The LoRA adapters are named `...qkv.linear_a_q` / `linear_b_q` /
    `linear_a_v` / `linear_b_v` (192 tensors over 16 trunk blocks).  Note that
    `LoRA_Sam.w_As` / `w_Bs` are plain Python lists holding the *same* Linear
    modules that `_LoRA_qkv` already owns, so `named_parameters()` yields 64
    unique entries while `state_dict()` holds 192 -- both refer to the same
    parameters, and the optimizer must use the `named_parameters()` view.
    """
    lora, head, frozen = [], [], 0
    seen = set()
    for n, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if ".linear_a_" in n or ".linear_b_" in n:
            p.requires_grad_(True); lora.append((n, p))
        elif n.startswith("decoder."):
            p.requires_grad_(True); head.append((n, p))
        elif n.startswith("txt_head."):
            # The I1 text branch.  The frozen street checkpoint has no text
            # weights at all, so this projector starts from scratch and MUST be
            # trained together with the head, otherwise the text logits are
            # random noise added to the visual logits.
            p.requires_grad_(True); head.append((n, p))
        else:
            p.requires_grad_(False); frozen += p.numel()
    return lora, head, frozen


def window_loss(logits, labels):
    """Per-window CE with ignore=255. logits: (B,C,H,W), labels: (B,H,W).

    A window whose labels are *entirely* ignore (255) produces NaN with the
    plain cross_entropy (0/0).  Such windows do occur in the SPAN8 panoramas
    (a crop can land entirely on unlabelled pixels), so we mask them out
    explicitly and return a zero loss instead of poisoning the run with NaN.
    """
    ce = F.cross_entropy(logits, labels, ignore_index=IGNORE, reduction="none")
    valid = (labels != IGNORE)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return ce[valid].mean()


def forward_pred(model, frame_idx, x):
    """Call the model and return only the prediction logits.

    `OmniSAM_adapt.forward` returns a bare tensor when the text branch is
    inactive, but a `(pred, feat, feat_de, TextAlignAux[, ...])` tuple as soon
    as text prototypes are present (the frozen checkpoint's model does have
    them).  We only need the logits, so unpack defensively.
    """
    out = model(frame_idx, x)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


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
    ap.add_argument("--text-proto", default=None,
                    help="Indoor SPAN8 text prototypes (.pt).  When given, the I1 "
                         "text branch is attached and trained jointly with the head.")
    ap.add_argument("--lambda-txt", type=float, default=0.5,
                    help="I1 text fusion weight used during training.")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {"args": vars(args), "device": device}

    # ---------- indoor text prototypes (I1) ----------
    text_protos = None
    if args.text_proto:
        payload = torch.load(args.text_proto, map_location="cpu")
        text_protos = payload["text_prototypes"] if isinstance(payload, dict) else payload
        if text_protos.shape[0] != NUM_CLASSES:
            raise ValueError(f"text prototypes class count {text_protos.shape[0]} "
                             f"!= {NUM_CLASSES}")
        rep["text_branch"] = {"protos": str(args.text_proto),
                              "shape": list(text_protos.shape),
                              "lambda_txt": args.lambda_txt}

    # ---------- CPU-only audit of what will be trainable ----------
    raw = torch.load(CKPT, map_location="cpu")
    ks = list(raw.keys())
    rep["ckpt_audit"] = {
        "total_tensors": len(ks),
        "lora_tensors": len([k for k in ks if ".linear_a_" in k or ".linear_b_" in k]),
        "decoder_tensors": len([k for k in ks if k.startswith("decoder.")]),
        "memory_tensors": len([k for k in ks if "memory" in k.lower()]),
        "head_shape_13cls": list(raw["decoder.linear_pred.weight"].shape),
        "note": "The frozen checkpoint already carries trained LoRA weights "
                "(192 tensors) plus a trained 13-class classifier.  For the "
                "indoor run the 13-cls classifier is dropped (shape mismatch) "
                "and the new 8-cls head is trained from scratch; the LoRA "
                "adapters continue from their checkpoint values.",
    }
    if args.audit_only:
        (out / "TRAINABLE_AUDIT.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps(rep, indent=2)); return

    model = build_model(device=device, text_protos=text_protos,
                        lambda_txt=args.lambda_txt if text_protos is not None else 0.0)
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
        for batch in dl:
            # Stanford360Span8 returns (image, label, size, name, <extra>)
            img, lbl = batch[0], batch[1]
            if micro == 0:
                # warmup: freeze LoRA, only head trains
                for p in lora_params:
                    p.requires_grad_(step >= warm)
            img = img[0].to(device)      # (W,3,512,512)
            lbl = lbl[0].to(device)      # (W,512,512)
            model.memory_bank._init_output_dict_source()
            total_loss = 0.0
            n_valid = 0
            for w in range(img.shape[0]):
                # frame_idx must stay 0: memory_bank.forward only concatenates
                # past frames when frame_idx != 0, which would leak the previous
                # window's memory into this one.  Each indoor window is an
                # independent crop, so 0 is the correct semantic.
                pred = forward_pred(model, 0, img[w:w+1].contiguous())
                if not torch.isfinite(pred).all():
                    # a non-finite forward poisons every parameter update
                    raise FloatingPointError(
                        f"non-finite logits at window {w} of image {batch[3] if len(batch) > 3 else '?'}")
                lw = window_loss(pred, lbl[w:w+1])
                if bool(torch.isfinite(lw)):
                    total_loss = total_loss + lw
                    n_valid += 1
            if n_valid == 0:
                continue
            loss = total_loss / n_valid
            (loss / args.accum).backward()
            losses.append(float(loss.detach()))
            micro += 1
            if micro % args.accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(lora_params + head_params, args.clip)
                if not torch.isfinite(gnorm):
                    # a non-finite gradient would poison every parameter; drop
                    # this micro-batch group instead of corrupting the run.
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                    continue
                for g, base in zip(opt.param_groups, [args.lr_lora, args.lr_head]):
                    g["lr"] = base * lr_scale(step)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == total:
                    msg = {"step": step, "loss": round(float(np.mean(losses[-80:])), 4),
                           "grad_norm": round(float(gnorm), 4),
                           "lr_head": opt.param_groups[1]["lr"],
                           "lr_lora": opt.param_groups[0]["lr"],
                           "elapsed_s": round(time.time() - t0, 1)}
                    print(json.dumps(msg), flush=True)
                    (out / "train_log.jsonl").open("a", encoding="utf-8").write(json.dumps(msg) + "\n")
                # periodic checkpoint so a long run is never lost
                if step % 300 == 0:
                    torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
                               out / "indoor_lora_last.pth")
                    print(f"CHECKPOINT step={step}", flush=True)
                if step >= total:
                    break
        micro = 0
    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "indoor_lora_last.pth")
    # Also export the LoRA adapters on their own, in the upstream key format
    # (w_a_XXX / w_b_XXX), so the indoor adapter is portable and comparable
    # with the outdoor one.
    enc = model.encoder
    lora_state = {}
    for i, w_a in enumerate(enc.w_As):
        lora_state[f"w_a_{i:03d}"] = w_a.weight.detach().cpu()
    for i, w_b in enumerate(enc.w_Bs):
        lora_state[f"w_b_{i:03d}"] = w_b.weight.detach().cpu()
    torch.save(lora_state, out / "indoor_lora_adapters.pth")
    print("SAVED", out / "indoor_lora_last.pth")
    print("SAVED", out / "indoor_lora_adapters.pth")


if __name__ == "__main__":
    main()
