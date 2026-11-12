"""Unified indoor evaluation runner for the Fold-1 experiment matrix.

Runs one arm of `indoor_experiment_matrix.py` over a split and writes:
  * per-image predictions (uint8 class ids, 1024x512) into <out>/preds/<arm>/
  * the scored metrics (mIoU / mAcc / per-class IoU) into <out>/<arm>_metrics.json

Design rules that keep the comparison honest:
  * Ground truth is read ONLY by `indoor_scorer`, never by any model path.
  * The Z arms use the frozen street 13-class head and map to SPAN8 through the
    official LUT, so classes with no street counterpart are structurally
    unreachable (reported as such, not hidden).
  * The A arms use the fine-tuned 8-class head.
  * Every arm uses the same indoor geometry (indoor_geometry): 512x512 windows,
    stride 64, 9 starts x 2 passes, plus I3 seam windows when i3_budget > 0.

Memory semantics (I2) -- the important part:
  The memory bank is a *temporal* module: `MemoryBank.forward` only reads the
  previous frames when `frame_idx != 0` (`if frame_idx != 0:` guards the whole
  history concatenation).  A forward with `frame_idx=0` therefore never sees
  any memory, no matter how the memory bank is configured -- so a pipeline that
  calls every window with `frame_idx=0` silently degenerates to "no memory at
  all" and the I2 arm would be identical to the dense arm.

  The indoor pipeline therefore drives the windows with an *increasing*
  `frame_idx` (0,1,...,N-1) inside each pass, exactly like the outdoor
  `full_model_semantic.py` runtime does.  That is what makes the memory bank
  (and hence the I2 sparse retrieval) actually do something.

  The window order of a pass is the semantic schedule (left to right, then
  right to left), so the memory history of window k contains the k previously
  seen FoVs -- the same causal structure as the outdoor sliding window.

Usage:
    python indoor_eval_runner.py --arm A1_B2 --split heldout \
        --ckpt runs/INDOOR-FOLD1-20261109/train/indoor_lora_last.pth \
        --out runs/INDOOR-FOLD1-20261109/eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
EFF = REPO / "third_party" / "effective_omnisam"
for p in (str(EFF), str(EFF / "sam"), str(REPO / "scripts" / "piers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from indoor_geometry import (  # noqa: E402
    INDOOR_W, INDOOR_H, INDOOR_WIN, INDOOR_STRIDE, INDOOR_PASSES,
    semantic_starts, MAP13_TO_8, i3_candidate_starts,
)
from indoor_seam_i3 import seam_mask, choose, fuse, base_coverage  # noqa: E402
from indoor_experiment_matrix import arm_specs, compare_table  # noqa: E402

SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]
NUM_CLASSES = 8
IGNORE = 255
DATA_ROOT = "/root/autodl-tmp/data_extra/Stanford2d3d_Seg"
PREP_ROOT = "/root/autodl-tmp/data_extra/SPAN8_PREP"
FROZEN_CKPT = "/root/autodl-tmp/weights/omnisam_s.pth"
RUN_DIR = Path("/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109")

SPLIT_LISTS = {
    "fit": "train_fit.txt",
    "calib": "train_calib.txt",
    "val": "train_val.txt",
    "heldout": "val.txt",
}


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_manifest_for_split(split):
    """Return [(image_name, label_path_absolute), ...] for a split."""
    p = RUN_DIR / f"split_{split}.jsonl"
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    out = []
    for r in recs:
        name = Path(r["image"]).name
        out.append((name, str(Path(PREP_ROOT) / r["area"] / r["label"])))
    return out


TEXT_PROTO_INDOOR = str(RUN_DIR / "indoor_text_prototypes_span8.pt")
TEXT_PROTO_STREET = "/root/autodl-tmp/eff_omni/assets/text_prototypes_cityscapes13_vitb32.pt"

# Checkpoint produced by the indoor stage-2 text alignment
# (`indoor_text_align_train.py`).  It carries the stage-1 LoRA + the trained
# 8-class head + the trained `txt_head`, so the A1_* arms get a *learned*
# text branch instead of a random projector.
CKPT_TEXT = str(RUN_DIR / "train" / "indoor_lora_text_last.pth")
CKPT_LORA = str(RUN_DIR / "train" / "indoor_lora_last.pth")


def build_model(arm, ckpt, device, text_proto_path=None, lambda_txt=0.0):
    """Build OmniSAM_adapt according to the arm spec; load the right checkpoint.

    Z arms keep the frozen street 13-class head (so their predictions have to go
    through the official 13->8 LUT and the classes without a street counterpart
    are structurally unreachable).  A arms use the fine-tuned 8-class head.

    The text branch (indoor I1) is enabled by passing text prototypes whose
    class count matches the head:
      * Z arms -> the frozen street 13-class prototypes (the original model's
        own text knowledge; the indoor I1 contribution does not exist there,
        which is exactly the "original vs ours" contrast).
      * A arms -> the indoor SPAN8 prototypes built by
        `indoor_text_prototypes.py`.

    IMPORTANT: the frozen street checkpoint carries **no** text-branch weights
    (0 of its 1806 tensors), so `txt_head.projector` and `txt_head.text_scale`
    are randomly initialised for the Z arms.  The Z arms therefore must NOT
    claim a text contribution: they run with `lambda_txt=0`, i.e. the text
    branch is present but its logits are not added to the visual logits
    (`model.py`: `fusion_type = "none"` when `lambda_txt == 0`).  This is
    stated in the arm spec and in the results table.
    """
    from sam.build_sam import build_sam2
    from model.model import OmniSAM_adapt

    spec = arm_specs()[arm]
    n_cls = 13 if spec["head"] == "street13" else 8

    text_prototypes = None
    if spec.get("text_branch"):
        path = text_proto_path
        if path is None:
            path = TEXT_PROTO_STREET if n_cls == 13 else TEXT_PROTO_INDOOR
        if Path(path).is_file():
            payload = torch.load(path, map_location="cpu")
            text_prototypes = payload["text_prototypes"] if isinstance(payload, dict) else payload
            if text_prototypes.shape[0] != n_cls:
                raise ValueError(
                    f"text prototypes class count mismatch: {text_prototypes.shape[0]} vs {n_cls}")
        else:
            print(f"[warn] text prototypes not found at {path}; text branch disabled for {arm}")

    sam_vendor = EFF / "sam" / "sam"
    sam2_ckpt = sam_vendor / "checkpoints" / "sam2.1_hiera_small.pt"
    if not sam2_ckpt.is_file():
        alt = Path("/root/autodl-tmp/weights/sam2.1_hiera_small.pt")
        if alt.is_file():
            sam2_ckpt = alt
        else:
            raise FileNotFoundError(f"SAM2 backbone weight not found: {sam2_ckpt}")
    sam = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(sam2_ckpt),
                     device=torch.device(device))
    model = OmniSAM_adapt(
        sam, num_classes=n_cls, num_maskmem=9,
        text_prototypes=text_prototypes,
        d_txt=int(text_prototypes.shape[-1]) if text_prototypes is not None else None,
        lambda_txt=lambda_txt,
        txt_tau=0.07,
    ).to(device)
    # Z arms: the frozen street checkpoint (13-cls head loads cleanly).
    # A arms: the fine-tuned checkpoint (the 13-cls classifier is dropped and
    # the 8-cls head comes from the fine-tune).
    #
    # The arms whose spec declares a *trained* text branch must load the
    # stage-2 text-aligned checkpoint, otherwise `txt_head` stays random.
    from indoor_ckpt import load_indoor_checkpoint
    if ckpt:
        chosen = ckpt
    elif spec.get("text_branch_trained") and Path(CKPT_TEXT).is_file():
        chosen = CKPT_TEXT
    elif spec.get("adapted") and Path(CKPT_LORA).is_file():
        chosen = CKPT_LORA
    else:
        chosen = FROZEN_CKPT
    print(f"[{arm}] checkpoint={chosen}")
    load_indoor_checkpoint(model, chosen, verbose=False)
    model.eval()
    return model, spec


def to_span8(logits):
    """Map a logits tensor (C,H,W) to SPAN8 class ids.

    For an 8-class head the mapping is identity.
    For a 13-class street head the official LUT is applied after argmax; the
    street classes without an indoor counterpart (beam, board, bookcase,
    clutter, column) map to 255 = "outside the SPAN8 label set".  Those pixels
    count as wrong for their GT class but must not create false positives on
    any other class, so the scorer treats them as a separate bucket.
    """
    pred = logits.argmax(0)
    if logits.shape[0] == 13:
        lut = torch.tensor(MAP13_TO_8, dtype=torch.long, device=logits.device)
        return lut.to(logits.device)[pred]
    return pred


def one_hot8(pred):
    """One-hot over the 8 SPAN8 classes; pixels outside the set (255) map to
    a zero vector so they contribute to no class."""
    p = pred.clone()
    p[p >= NUM_CLASSES] = 0
    oh = torch.nn.functional.one_hot(p, NUM_CLASSES).permute(2, 0, 1).float()
    outside = (pred >= NUM_CLASSES).unsqueeze(0).float()
    return oh * (1.0 - outside)


def wrap_window(tensor, start, width):
    """Crop `width` columns starting at `start`, wrapping around the canvas.

    The I3 seam windows straddle x=0 (the ERP seam), so a plain slice would
    run past the right edge and return a short window.  The outdoor seam
    windows have the same property and are handled by the modulo indexing in
    `support()`; here we materialise the wrapped crop explicitly.
    """
    idx = (torch.arange(width, device=tensor.device) + start) % tensor.shape[-1]
    return tensor.index_select(-1, idx)


def run_arm(arm, split, ckpt, out_dir, device="cuda", limit=0, dtype=torch.float32,
            lambda_txt=0.0, text_proto_path=None):
    """Run one arm over a split.

    The frozen model runs in fp32 (the outdoor pipeline benchmarks it that way),
    so the default dtype is fp32; bf16 would need the weights cast too.
    """
    spec = arm_specs()[arm]
    items = load_manifest_for_split(split)
    if limit:
        items = items[:limit]
    # The fusion weight comes from the arm spec: the Z arms keep it at 0 (the
    # frozen checkpoint has no trained text branch, so its text logits would be
    # random noise), the A1_* arms use the upstream value 0.1.
    lambda_txt = spec.get("lambda_txt", 0.0)
    model, spec = build_model(arm, ckpt, device, text_proto_path=text_proto_path,
                               lambda_txt=lambda_txt)

    # ---- I2 memory configuration ----------------------------------------
    from indoor_memory_i2 import install_indoor_i2, install_dense_memory
    if spec["memory"] == "i2":
        mem_audit = install_indoor_i2(model, stage="eval")
    else:
        mem_audit = install_dense_memory(model)
    print(f"[{arm}] memory={spec['memory']} audit={json.dumps(mem_audit)[:400]}")

    starts = semantic_starts(INDOOR_W, INDOOR_WIN, INDOOR_STRIDE)
    passes = INDOOR_PASSES
    pred_dir = Path(out_dir) / "preds" / arm
    pred_dir.mkdir(parents=True, exist_ok=True)

    stats = {"arm": arm, "spec": spec, "split": split, "n": len(items),
             "forwards_per_image": 0, "wall_s": 0.0,
             "memory_audit": mem_audit}
    t0 = time.time()

    from PIL import Image
    fwd_last = 0
    for idx, (name, lbl_path) in enumerate(items):
        img = Image.open(str(Path(DATA_ROOT) / "images" / name)).convert("RGB")
        if img.size != (INDOOR_W, INDOOR_H):
            img = img.resize((INDOOR_W, INDOOR_H), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

        canvas = torch.zeros((NUM_CLASSES, INDOOR_H, INDOOR_W), dtype=torch.float32)
        # pixels that fell outside the SPAN8 label set (only possible for the
        # frozen street head); kept apart so they are reported as "wrong but
        # not a false positive on some other class".
        outside = torch.zeros((1, INDOOR_H, INDOOR_W), dtype=torch.float32)
        model.memory_bank._init_output_dict_source()
        n_fwd = 0
        # Per-pass memory context + pseudo labels, exactly like the outdoor
        # `full_model_semantic.py`: the I3 seam window is re-observed *with*
        # the memory history of the pass it belongs to, which is what makes it
        # a "look again with context" rather than a cold re-run.
        contexts = []

        for p in range(passes):
            order = starts if p == 0 else list(reversed(starts))
            # The memory bank is temporal: it only reads the history when
            # `frame_idx != 0`.  We therefore drive the windows of a pass with
            # an increasing frame index so the I2 memory is actually used.
            #
            # The forwards are deliberately SEQUENTIAL (batch size 1), exactly
            # like the outdoor `full_model_semantic.py` runtime.  Batching the
            # windows of a pass would collapse the whole history into a single
            # `frame_idx` (the memory bank stores one entry per frame index) and
            # silently disable the memory attention -- measured on the heldout
            # split, batched and sequential outputs agree on only 68% of the
            # pixels.  The 5.4x speed-up is not worth changing the numbers.
            model.memory_bank._init_output_dict_source()
            labels = {}
            for f, s in enumerate(order):
                x = tensor[:, :, :, s:s + INDOOR_WIN].to(device=device, dtype=dtype)
                out = forward_pred(model, f, x)
                logits = out.float().cpu()
                n_fwd += 1
                labels[f] = out.detach().argmax(1).cpu()
                pred = to_span8(logits[0])
                canvas[:, :, s:s + INDOOR_WIN] += one_hot8(pred)
                outside[:, :, s:s + INDOOR_WIN] += (pred >= NUM_CLASSES).unsqueeze(0).float()
            contexts.append((
                {k: v for k, v in model.memory_bank.output_dict_source["frame_outputs"].items() if k > 0},
                {k: v for k, v in labels.items() if k > 0},
            ))

        # I3 seam re-observation: pick the extra windows from the risk map and
        # fuse them with the base prediction exactly like the outdoor I3 does
        # (indoor_seam_i3.fuse), instead of overwriting the window.
        if spec["i3_budget"] > 0:
            with torch.no_grad():
                base_probs = canvas.clone()
                base_label = canvas.argmax(0).numpy().astype(np.uint8)
                risk = semantic_risk_from_logits(canvas).numpy()
                sel, _ = choose(risk, i3_candidate_starts(), INDOOR_WIN,
                                B=spec["i3_budget"],
                                min_separation=max(1, INDOOR_WIN // 4))
                extras = []
                if sel:
                    # The I3 window is an *extra look* at a seam region that the
                    # base pass may have covered poorly.  It is re-observed
                    # with the memory history of the pass it belongs to (the
                    # outdoor runtime restores `output_dict_source` from the
                    # stored context and calls the model with frame_idx = 9 so
                    # the memory bank can attend over the whole pass), and its
                    # result is fused into the base prediction by the
                    # coverage-aware fusion -- never overwritten.
                    for k, s in enumerate(sel):
                        # The outdoor runtime pairs the B extra windows with the
                        # B passes (plan[pas] -> contexts[pas]); with B<=2 and
                        # 2 passes the pairing is one-to-one.
                        pas = min(k, len(contexts) - 1)
                        context, plabels = contexts[pas]
                        model.memory_bank._init_output_dict_source()
                        model.memory_bank.output_dict_source = {"frame_outputs": dict(context)}
                        x = wrap_window(tensor, s, INDOOR_WIN).to(device=device, dtype=dtype)
                        out = forward_pred(model, len(contexts[pas][0]), x,
                                           historical_pseudo_labels=plabels)
                        logits = out.float().cpu()
                        n_fwd += 1
                        # indoor_seam_i3.fuse works on numpy arrays
                        extras.append((s, one_hot8(to_span8(logits[0])).numpy()))
                    model.memory_bank._init_output_dict_source()
                fused, label, info = fuse(base_probs.numpy(), base_label, extras,
                                          base_coverage())
                canvas = torch.from_numpy(fused)
                final = torch.from_numpy(label)
                out_mask = (final >= NUM_CLASSES)
                final[out_mask] = 255
                out_np = final.numpy().astype(np.uint8)
                np.save(pred_dir / (Path(name).stem + ".npy"), out_np)
                fwd_last = n_fwd
                if (idx + 1) % 25 == 0:
                    print(f"[{arm}/{split}] {idx+1}/{len(items)} fwd/img={n_fwd}", flush=True)
                continue

        pred = canvas.argmax(0).to(torch.uint8)
        # restore the "outside the label set" pixels
        out_mask = outside[0] > 0
        pred[out_mask] = 255
        out_np = pred.cpu().numpy()
        np.save(pred_dir / (Path(name).stem + ".npy"), out_np)
        fwd_last = n_fwd
        if (idx + 1) % 25 == 0:
            print(f"[{arm}/{split}] {idx+1}/{len(items)} fwd/img={n_fwd}", flush=True)

    stats["forwards_per_image"] = fwd_last
    stats["wall_s"] = round(time.time() - t0, 1)
    stats["memory_audit_final"] = {
        k: v for k, v in (mem_audit or {}).items()
        if k in ("calls", "fallbacks", "tokens_before", "tokens_after", "max_tokens")
    }
    return stats


def forward_pred(model, frame_idx, x, historical_pseudo_labels=None):
    """Return only the prediction logits from OmniSAM_adapt.forward.

    The frozen model has text prototypes attached, so forward returns
    `(pred, feat, feat_de, TextAlignAux[, ...])`; a bare tensor when the text
    branch is inactive.  Only the logits are needed here.

    `historical_pseudo_labels` is forwarded so the I3 seam window can be
    re-observed with the pseudo labels of the pass it belongs to (the outdoor
    runtime passes them the same way); the memory bank uses them for the
    class-aware / boundary-guarded sparse selection.
    """
    if historical_pseudo_labels is None:
        out = model(frame_idx, x)
    else:
        out = model(frame_idx, x, historical_pseudo_labels=historical_pseudo_labels)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def semantic_risk_from_logits(canvas):
    """Normalized entropy + margin risk over the accumulated class votes."""
    q = canvas / torch.clamp(canvas.sum(0, keepdim=True), min=1e-8)
    ent = -(q * torch.log(torch.clamp(q, min=1e-8))).sum(0) / np.log(q.shape[0])
    top = torch.topk(q, 2, dim=0).values
    margin = top[0] - top[1]
    return 0.5 * ent + 0.5 * (1 - margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(arm_specs().keys()))
    ap.add_argument("--split", default="heldout", choices=list(SPLIT_LISTS.keys()))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=str(RUN_DIR / "eval"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--lambda-txt", type=float, default=0.5,
                    help="I1 text fusion weight for the arms that enable the text branch.")
    ap.add_argument("--text-proto", default=None,
                    help="Override the text prototype path for the arm.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    stats = run_arm(args.arm, args.split, args.ckpt, out, device=device, limit=args.limit,
                    lambda_txt=args.lambda_txt, text_proto_path=args.text_proto)
    print(json.dumps(stats, indent=2))
    (out / f"{args.arm}_run_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Score the predictions.  The scorer is the ONLY place that reads GT.
    if not args.no_score:
        from indoor_scorer import score_predictions
        manifest = RUN_DIR / f"split_{args.split}.jsonl"
        res = score_predictions(str(out / "preds" / args.arm), str(manifest),
                                PREP_ROOT, out_json=str(out / f"{args.arm}_{args.split}_metrics.json"))
        res.pop("per_image", None)
        print(json.dumps(res, indent=2))
        (out / f"{args.arm}_{args.split}_metrics.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
