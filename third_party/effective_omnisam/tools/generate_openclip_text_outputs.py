#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def default_class_prompts() -> Dict[str, str]:
    return {
        "road": "a drivable road surface in a panoramic street scene",
        "sidewalk": "a pedestrian sidewalk beside the road",
        "building": "building facade in an urban street scene",
        "pole": "a thin vertical pole or street lamp",
        "traffic light": "a small traffic light at an intersection",
        "traffic sign": "a traffic sign beside the road",
        "vegetation": "trees, grass, bushes and plants",
        "sky": "open sky region",
        "person": "pedestrian walking or standing",
        "car": "car or vehicle on the road",
    }


def load_class_prompts(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("class prompts json must be a dict: {class_name: prompt}")

    cleaned: Dict[str, str] = {}
    for k, v in data.items():
        cls = str(k).strip()
        prompt = str(v).strip()
        if not cls or not prompt:
            raise ValueError("class name and prompt must be non-empty strings")
        cleaned[cls] = prompt
    if not cleaned:
        raise ValueError("class prompts cannot be empty")
    return cleaned


def resolve_legacy_pt_in(user_value: str) -> str:
    if user_value:
        return user_value
    default_candidates = [
        "assets/text_prototypes_cityscapes13_vitb32.pt",
    ]
    for p in default_candidates:
        if os.path.exists(p):
            return p
    return ""


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("Generate OpenCLIP text embeddings from class prompts")
    parser.add_argument("--class_prompt_json", type=str, default="", help="Optional class prompt json file")
    parser.add_argument(
        "--legacy_pt_in",
        type=str,
        default="",
        help="Optional legacy text prototypes .pt. If set, skip OpenCLIP encoding and convert this file to new outputs.",
    )
    parser.add_argument("--text_model", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--text_pretrained", type=str, default="openai", help="OpenCLIP pretrained tag")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="aux_outputs/openclip_text",
        help="Output directory for npy/json artifacts",
    )
    parser.add_argument(
        "--legacy_pt_out",
        type=str,
        default="assets/text_prototypes_openclip_text_latest.pt",
        help="Optional output path for legacy .pt payload, compatible with existing multimodal training scripts.",
    )
    parser.add_argument(
        "--disable_legacy_auto",
        action="store_true",
        help="Disable automatic fallback to default legacy .pt when --legacy_pt_in is not provided.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    class_prompts = load_class_prompts(args.class_prompt_json) if args.class_prompt_json else default_class_prompts()
    class_names: List[str] = list(class_prompts.keys())
    prompts: List[str] = [class_prompts[k] for k in class_names]

    source_mode = "openclip_encode"
    device = torch.device(args.device)

    legacy_pt_in = "" if args.disable_legacy_auto else resolve_legacy_pt_in(args.legacy_pt_in)

    if legacy_pt_in:
        payload = torch.load(legacy_pt_in, map_location="cpu")
        source_mode = "legacy_pt_convert"

        if isinstance(payload, dict) and "text_prototypes" in payload:
            text_features = payload["text_prototypes"].float()
            if "class_names" in payload and isinstance(payload["class_names"], list) and payload["class_names"]:
                class_names = [str(x) for x in payload["class_names"]]
                if "templates" in payload and isinstance(payload["templates"], list) and payload["templates"]:
                    templates = [str(x) for x in payload["templates"]]
                    if len(templates) == len(class_names):
                        prompts = templates
                    else:
                        prompts = [class_prompts.get(k, f"{k} in a street scene") for k in class_names]
                else:
                    prompts = [class_prompts.get(k, f"{k} in a street scene") for k in class_names]
        elif isinstance(payload, torch.Tensor):
            text_features = payload.float()
        else:
            raise ValueError("legacy_pt_in must be a tensor or dict containing key 'text_prototypes'")

        if text_features.dim() != 2:
            raise ValueError(f"text prototypes must be 2D [C,D], got {tuple(text_features.shape)}")
        text_features = F.normalize(text_features, dim=-1)
    else:
        try:
            import open_clip
        except Exception as e:
            raise RuntimeError("open_clip_torch is required. Install via pip install open_clip_torch.") from e

        model, _, _ = open_clip.create_model_and_transforms(args.text_model, pretrained=args.text_pretrained)
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(args.text_model)

        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1).detach().cpu()

    similarity = text_features @ text_features.t()

    text_embedding_np = text_features.detach().cpu().numpy().astype(np.float32)
    similarity_np = similarity.detach().cpu().numpy().astype(np.float32)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    emb_path = os.path.join(out_dir, "class_text_embedding.npy")
    meta_path = os.path.join(out_dir, "class_prompt_metadata.json")
    sim_path = os.path.join(out_dir, "class_similarity_matrix.npy")

    np.save(emb_path, text_embedding_np)
    np.save(sim_path, similarity_np)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "legacy_pt_in": legacy_pt_in or None,
        "text_model": args.text_model,
        "text_pretrained": args.text_pretrained,
        "device": str(device),
        "num_classes": len(class_names),
        "embedding_dim": int(text_embedding_np.shape[1]),
        "class_names": class_names,
        "class_prompts": {k: p for k, p in zip(class_names, prompts)},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if args.legacy_pt_out:
        legacy_payload = {
            "text_prototypes": torch.from_numpy(text_embedding_np),
            "class_names": class_names,
            "templates": prompts,
            "text_model": args.text_model,
            "text_pretrained": args.text_pretrained,
            "source_mode": source_mode,
        }
        legacy_out_abs = os.path.abspath(args.legacy_pt_out)
        os.makedirs(os.path.dirname(legacy_out_abs), exist_ok=True)
        torch.save(legacy_payload, legacy_out_abs)
        print(f"[OK] Saved: {legacy_out_abs}")

    print(f"[OK] Saved: {emb_path}")
    print(f"[OK] Saved: {meta_path}")
    print(f"[OK] Saved: {sim_path}")
    print(f"[INFO] embedding shape: {tuple(text_embedding_np.shape)}")
    print(f"[INFO] similarity shape: {tuple(similarity_np.shape)}")


if __name__ == "__main__":
    main()

