"""Inspect a trained checkpoint: what does it actually contain?

Usage: python scripts/piers/inspect_ckpt.py <path/to/final.pth>
"""
from __future__ import annotations

import sys

import torch


def main(path: str) -> int:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        obj = torch.load(path, map_location="cpu")

    print(f"=== {path} ===")
    print(f"top-level type: {type(obj).__name__}")

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"top-level keys ({len(keys)}): {keys[:20]}")
        # Common layouts: {"model": state_dict}, {"state_dict": ...}, raw state_dict
        for wrapper in ("model", "state_dict", "model_state_dict", "net"):
            if wrapper in obj and isinstance(obj[wrapper], dict):
                sd = obj[wrapper]
                print(f"\n-> found '{wrapper}' with {len(sd)} tensors")
                total = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
                print(f"-> total parameters: {total:,} ({total/1e6:.2f} M)")
                # look for the modules we touched
                interesting = [k for k in sd
                               if any(t in k for t in ("reliab", "cooperat", "bridge",
                                                       "retriev", "seam", "guide",
                                                       "semantic_bridge", "revision",
                                                       "pixel"))]
                print(f"-> innovation-related tensors ({len(interesting)}):")
                for k in interesting[:30]:
                    shape = tuple(sd[k].shape)
                    print(f"     {k}  {shape}")
                return 0
        # raw state dict
        total = sum(v.numel() for v in obj.values() if hasattr(v, "numel"))
        print(f"\nlooks like a raw state_dict: {len(obj)} tensors, "
              f"{total:,} params ({total/1e6:.2f} M)")
        print("sample keys:", keys[:15])
    else:
        print("not a dict")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
