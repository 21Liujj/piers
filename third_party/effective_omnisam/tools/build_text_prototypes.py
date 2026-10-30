#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

import torch


def _default_templates() -> List[str]:
    # Keep templates short & consistent; ensemble helps prompt sensitivity.
    return [
        "a photo of a {cls} in an urban driving scene",
        "a panoramic urban scene containing {cls}",
        "a street-view image of {cls}",
        "an equirectangular panorama of {cls}",
        "a driving scene with {cls}",
    ]


def _load_templates(path: str) -> List[str]:
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict) and "templates" in data:
                return [str(x) for x in data["templates"]]
            raise ValueError("JSON must be a list or {templates: [...]}.")
        # txt/yaml-lite: one template per line, '#' comments allowed
        out = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
        return out


def _resolve_class_names(dataset_name: str, class_set: int) -> List[str]:
    """
    Lightweight class resolver for prototype building.
    Avoids importing training-side utils/build.py (which pulls heavy SAM deps).
    """
    cityscapes_13 = [
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
        "car",
    ]
    cityscapes_19 = [
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
    if class_set == 13:
        return cityscapes_13
    if class_set == 19:
        return cityscapes_19
    raise ValueError(f"Unsupported class_set={class_set} for dataset={dataset_name}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("Build text prototypes for class names")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. Cityscapes/DensePASS/WildPASS")
    parser.add_argument("--class_set", type=int, default=13, choices=[13, 19], help="Label space (13 or 19)")
    parser.add_argument("--text_model", type=str, default="ViT-B-32", help="Text encoder model name (open_clip)")
    parser.add_argument("--text_pretrained", type=str, default="openai", help="Text encoder pretrained tag (open_clip)")
    # Backward-compatible aliases (deprecated)
    parser.add_argument("--clip_model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--clip_pretrained", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--templates", type=str, default="", help="Optional templates file (txt/json)")
    parser.add_argument("--out", type=str, required=True, help="Output .pt path (tensor CxD + metadata dict)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Backward compatibility for previous CLI flags.
    if args.clip_model:
        args.text_model = args.clip_model
    if args.clip_pretrained:
        args.text_pretrained = args.clip_pretrained

    class_names = _resolve_class_names(args.dataset, args.class_set)
    templates = _load_templates(args.templates) if args.templates else _default_templates()
    if not templates:
        raise ValueError("No templates provided.")

    try:
        import open_clip
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "open_clip_torch is required. Please install it (see requirements.txt)."
        ) from e

    device = torch.device(args.device)
    model, _, _ = open_clip.create_model_and_transforms(args.text_model, pretrained=args.text_pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.text_model)

    # Freeze everything; we only need text embeddings.
    for p in model.parameters():
        p.requires_grad_(False)

    # Build prototypes: mean over templates per class.
    protos = []
    per_class_prompts: Dict[str, List[str]] = {}
    for cls in class_names:
        prompts = [t.format(cls=cls) for t in templates]
        per_class_prompts[cls] = prompts
        tokens = tokenizer(prompts).to(device)
        txt = model.encode_text(tokens)
        txt = torch.nn.functional.normalize(txt, dim=-1)
        proto = txt.mean(dim=0)
        proto = torch.nn.functional.normalize(proto, dim=-1)
        protos.append(proto)

    text_prototypes = torch.stack(protos, dim=0).cpu()  # (C, D)

    payload = dict(
        text_prototypes=text_prototypes,
        class_names=class_names,
        templates=templates,
        text_model=args.text_model,
        text_pretrained=args.text_pretrained,
    )
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"[OK] Saved text prototypes to: {out_path}")
    print(f"     shape: {tuple(text_prototypes.shape)} (C,D)")


if __name__ == "__main__":
    main()

