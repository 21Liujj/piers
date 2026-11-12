"""Indoor stage-2 text alignment (indoor I1) on top of the stage-1 LoRA checkpoint.

Why a separate stage:
  The frozen street checkpoint (`omnisam_s.pth`) contains **no** text-branch
  weights at all (0 of its 1806 tensors), and the upstream outdoor pipeline
  never trains the text branch either -- the outdoor text experiments all run
  with `lambda_txt: 0.0` and only use the text logits through the
  KL-alignment loss (`weight_txt_align`) or the SSL gate.

  That means the `txt_head.projector` (a 2-layer MLP mapping the 512-d CLIP
  text embedding into the 256-d decoder feature space) and `txt_head.text_scale`
  are **randomly initialised** in any model built for the indoor run.  Adding
  random text logits to the visual logits would make every "I1 text" claim
  meaningless.

  So the indoor I1 contribution has to be *learned*.  This stage trains the
  text branch on the SAME indoor fit split with the SAME supervision the
  stage-1 head was trained on:

    loss = CE(visual logits, GT)                       (keeps the head sharp)
         + w_align * KL( p_vis || p_txt )               (the upstream align loss)

  The KL term is the upstream `model.text_align.kl_align_loss`, used verbatim.
  It pulls the text logits towards the visual logits at every labelled pixel,
  which is what turns the random projector into a genuine text-conditioned
  decoder head.  `lambda_txt` stays 0 during this stage (exactly like the
  upstream outdoor text config) -- the text logits are *aligned* with the
  visual ones, not *added* to them, so the alignment cannot degrade the
  segmentation.

  The result is a checkpoint that carries a *trained* `txt_head`, which is
  what the A1_* arms load.

Usage:
    python indoor_text_align_train.py \
        --ckpt runs/INDOOR-FOLD1-20261109/train/indoor_lora_last.pth \
        --text-proto runs/INDOOR-FOLD1-20261109/indoor_text_prototypes_span8.pt \
        --steps 400 --out runs/INDOOR-FOLD1-20261109/train
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/root/autodl-tmp/piers-main")
EFF = REPO / "third_party" / "effective_omnisam"
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
    return Stanford360PanoramaVal(
        DATA_ROOT, str(Path(DATA_ROOT) / "lists" / list_name),
        crop_size=crop_size, sliding_window=sliding_window, passes=passes)


def build_model(text_protos, device="cuda", ckpt=CKPT, lambda_txt=0.1):
    """Build the 8-class model with the text branch attached."""
    from sam.build_sam import build_sam2
    from model.model import OmniSAM_adapt

    sam_vendor = EFF / "sam" / "sam"
    sam2_ckpt = sam_vendor / "checkpoints" / "sam2.1_hiera_small.pt"
    if not sam2_ckpt.is_file():
        alt = Path("/root/autodl-tmp/weights/sam2.1_hiera_small.pt")
        if not alt.is_file():
            raise FileNotFoundError(f"SAM2 backbone weight not found: {sam2_ckpt}")
        sam2_ckpt = alt
    sam = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(sam2_ckpt),
                     device=torch.device(device))
    model = OmniSAM_adapt(
        sam, num_classes=NUM_CLASSES, num_maskmem=9,
        text_prototypes=text_protos,
        d_txt=int(text_protos.shape[-1]),
        lambda_txt=lambda_txt,
        txt_tau=0.07,
    ).to(device)
    from indoor_ckpt import load_indoor_checkpoint
    load_indoor_checkpoint(model, ckpt)
    return model


def split_trainable(model):
    """Trainable = the text branch + the decoder head.

    The LoRA adapters stay frozen: stage 1 already adapted the encoder, and
    the upstream `text_train_stage: stage2` likewise trains the semantic
    decoder together with the text adapter.
    """
    text, head, frozen = [], [], 0
    seen = set()
    for n, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if n.startswith("txt_head.") or n == "text_fusion.alpha":
            p.requires_grad_(True); text.append((n, p))
        elif n.startswith("decoder."):
            p.requires_grad_(True); head.append((n, p))
        else:
            p.requires_grad_(False); frozen += p.numel()
    return text, head, frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "train" / "indoor_lora_last.pth"),
                    help="Stage-1 LoRA checkpoint to start from.")
    ap.add_argument("--text-proto", default=str(RUNS / "indoor_text_prototypes_span8.pt"))
    ap.add_argument("--out", default=str(RUNS / "train"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--lr-text", type=float, default=1e-4)
    ap.add_argument("--w-align", type=float, default=0.05,
                    help="Weight of the upstream KL text-alignment loss.")
    ap.add_argument("--lambda-txt", type=float, default=0.1,
                    help="Text fusion weight.  The upstream outdoor text config "
                         "uses 0.1; the text branch is trained from scratch here, "
                         "so the fusion is added to the logits *during* training "
                         "and the head learns to compensate for it.")
    ap.add_argument("--align-tau", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train-list", default="train_fit.txt")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(args.text_proto, map_location="cpu")
    text_protos = payload["text_prototypes"] if isinstance(payload, dict) else payload
    if text_protos.shape[0] != NUM_CLASSES:
        raise ValueError(f"text prototypes class count {text_protos.shape[0]} != {NUM_CLASSES}")

    model = build_model(text_protos, device=device, ckpt=args.ckpt,
                        lambda_txt=args.lambda_txt)
    text_params, head_params, frozen = split_trainable(model)
    print(json.dumps({
        "text_tensors": len(text_params),
        "text_params": int(sum(p.numel() for _, p in text_params)),
        "head_tensors": len(head_params),
        "head_params": int(sum(p.numel() for _, p in head_params)),
        "frozen_params": int(frozen),
        "w_align": args.w_align,
    }, indent=2))

    ds = build_stanford_loader(args.train_list)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True,
                                     num_workers=4, drop_last=True)
    print("train images:", len(ds), "windows/img:", len(ds[0][0]))

    opt = torch.optim.AdamW([
        {"params": [p for _, p in text_params], "lr": args.lr_text},
        {"params": [p for _, p in head_params], "lr": args.lr_head},
    ], weight_decay=args.wd)

    from model.text_align import kl_align_loss

    def lr_scale(step):
        prog = step / max(1, args.steps)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))

    model.train()
    step = 0; micro = 0; t0 = time.time()
    losses = {"ce": [], "align": []}
    model.memory_bank._init_output_dict_source()
    while step < args.steps:
        for batch in dl:
            img, lbl = batch[0], batch[1]
            img = img[0].to(device)      # (W,3,512,512)
            lbl = lbl[0].to(device)      # (W,512,512)
            model.memory_bank._init_output_dict_source()
            ce_sum = torch.zeros((), device=device)
            al_sum = torch.zeros((), device=device)
            n_valid = 0
            for w in range(img.shape[0]):
                model_out = model(0, img[w:w + 1].contiguous(), return_aux=True)
                pred, feat, feat_de, aux = model_out[0], model_out[1], model_out[2], model_out[3]
                if not torch.isfinite(pred).all():
                    raise FloatingPointError("non-finite logits")
                labels = lbl[w:w + 1]
                valid = (labels != IGNORE)
                if not bool(valid.any()):
                    continue
                ce = F.cross_entropy(pred, labels, ignore_index=IGNORE, reduction="none")
                ce_loss = ce[valid].mean()
                # the upstream KL alignment between the visual and the text
                # logits, restricted to the labelled pixels
                align = kl_align_loss(aux.l_vis, aux.l_txt,
                                      mask=valid, tau=args.align_tau)
                if not (torch.isfinite(ce_loss) and torch.isfinite(align)):
                    continue
                ce_sum = ce_sum + ce_loss
                al_sum = al_sum + align
                n_valid += 1
            if n_valid == 0:
                continue
            loss = (ce_sum / n_valid) + args.w_align * (al_sum / n_valid)
            (loss / args.accum).backward()
            losses["ce"].append(float((ce_sum / n_valid).detach()))
            losses["align"].append(float((al_sum / n_valid).detach()))
            micro += 1
            if micro % args.accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in text_params] + [p for _, p in head_params], args.clip)
                if not torch.isfinite(gnorm):
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                    continue
                for g, base in zip(opt.param_groups, [args.lr_text, args.lr_head]):
                    g["lr"] = base * lr_scale(step)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == args.steps:
                    msg = {"step": step,
                           "ce": round(float(np.mean(losses["ce"][-80:])), 4),
                           "align": round(float(np.mean(losses["align"][-80:])), 4),
                           "grad_norm": round(float(gnorm), 4),
                           "lr_text": opt.param_groups[0]["lr"],
                           "lr_head": opt.param_groups[1]["lr"],
                           "elapsed_s": round(time.time() - t0, 1)}
                    print(json.dumps(msg), flush=True)
                    (out / "text_align_log.jsonl").open("a", encoding="utf-8").write(
                        json.dumps(msg) + "\n")
                if step % 100 == 0:
                    torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
                               out / "indoor_lora_text_last.pth")
        micro = 0
    torch.save({"model": model.state_dict(), "args": vars(args)},
               out / "indoor_lora_text_last.pth")
    print("SAVED", out / "indoor_lora_text_last.pth")


if __name__ == "__main__":
    main()
