"""Build text prototypes for the 8 indoor SPAN8 classes (indoor I1).

The upstream `tools/build_text_prototypes.py` only resolves Cityscapes 13/19
class names.  This module reuses the exact same open_clip recipe (ViT-B-32 /
openai, mean of the prompt templates, L2-normalised) but with indoor class
names and indoor prompt templates, and writes a payload with the same schema
(`text_prototypes`, `class_names`, `templates`, `text_model`,
`text_pretrained`) so the base model can consume it unchanged.

The templates deliberately describe *indoor* scenes; the street templates would
embed a strong "urban driving scene" prior that does not apply to a room.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]

# Class-specific descriptions, in the same spirit as the outdoor
# openclip_class_prompts_13_optimized.json asset.
CLASS_DESCRIPTIONS = {
    "ceiling": "the upper flat surface of an indoor room seen from below",
    "chair": "a single chair, seat with backrest, used for sitting indoors",
    "door": "a door or doorway in an indoor wall, an entrance to a room",
    "floor": "the horizontal ground surface inside a building, covered by carpet or tiles",
    "sofa": "a sofa or couch, a long upholstered seat for several people indoors",
    "table": "a table, a flat horizontal surface with legs standing inside a room",
    "wall": "a vertical interior wall surface of a room, painted or wallpapered",
    "window": "a window in an interior wall, glass opening admitting daylight",
}

TEMPLATES = [
    "a photo of a {cls} in an indoor room",
    "an indoor scene containing {cls}",
    "a 360 degree indoor panorama of {cls}",
    "the {cls} of an interior space",
    "an indoor room with {cls}",
]


def build(out_path, device="cuda", text_model="ViT-B-32", text_pretrained="openai"):
    import open_clip

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(text_model, pretrained=text_pretrained)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(text_model)

    protos = []
    per_class_prompts = {}
    for cls in SPAN8:
        prompts = [t.format(cls=CLASS_DESCRIPTIONS[cls]) for t in TEMPLATES]
        per_class_prompts[cls] = prompts
        tokens = tokenizer(prompts).to(dev)
        txt = model.encode_text(tokens)
        txt = torch.nn.functional.normalize(txt, dim=-1)
        proto = torch.nn.functional.normalize(txt.mean(dim=0), dim=-1)
        protos.append(proto)

    text_prototypes = torch.stack(protos, dim=0).cpu()  # (8, 512)
    payload = dict(
        text_prototypes=text_prototypes,
        class_names=SPAN8,
        templates=TEMPLATES,
        text_model=text_model,
        text_pretrained=text_pretrained,
        class_descriptions=CLASS_DESCRIPTIONS,
        per_class_prompts=per_class_prompts,
        note="Indoor I1 text prototypes for SPAN8; same open_clip recipe as the "
             "outdoor build_text_prototypes.py but with indoor prompts.",
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109/"
                                     "indoor_text_prototypes_span8.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    p = build(args.out, args.device)
    print(json.dumps({
        "out": args.out,
        "shape": list(p["text_prototypes"].shape),
        "class_names": p["class_names"],
        "norms": [round(float(v), 4) for v in p["text_prototypes"].norm(dim=1)],
    }, indent=2))


if __name__ == "__main__":
    main()
