from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_DEPTH_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".npy", ".pt")


@dataclass
class DepthPriorState:
    warned_missing_once: bool = False
    missing_count: int = 0


def _to_hw(size: Sequence[int] | torch.Size | Tuple[int, int]) -> Tuple[int, int]:
    if len(size) < 2:
        raise ValueError(f"target_size must have at least 2 dims, got {size}")
    return int(size[-2]), int(size[-1])


def find_depth_path(rgb_path: str, depth_dir: str, file_exts: Optional[Iterable[str]] = None) -> Optional[str]:
    if not depth_dir:
        return None
    depth_dir = os.path.abspath(depth_dir)
    stem = os.path.splitext(os.path.basename(rgb_path))[0]
    rgb_norm = rgb_path.replace("\\", "/")
    rgb_parent = os.path.basename(os.path.dirname(rgb_norm))
    rel_after_left = None
    if "leftImg8bit/" in rgb_norm:
        rel_after_left = rgb_norm.split("leftImg8bit/", 1)[1]
        rel_after_left = os.path.splitext(rel_after_left)[0]
    sub_roots = ("", "depth_png", "depth_npy", "leftImg8bit", "depth_png/leftImg8bit", "depth_npy/leftImg8bit")
    exts = tuple(file_exts) if file_exts is not None else DEFAULT_DEPTH_EXTS
    for sub in sub_roots:
        base = os.path.join(depth_dir, sub) if sub else depth_dir
        for ext in exts:
            candidates = [
                os.path.join(base, f"{stem}{ext}"),
                os.path.join(base, rgb_parent, f"{stem}{ext}"),
            ]
            if rel_after_left:
                candidates.append(os.path.join(base, f"{rel_after_left}{ext}"))
            for p in candidates:
                if os.path.isfile(p):
                    return p
    return None


def _normalize_minmax(depth: torch.Tensor) -> torch.Tensor:
    dmin = torch.amin(depth)
    dmax = torch.amax(depth)
    denom = dmax - dmin
    if float(denom.item()) < 1e-6:
        return torch.zeros_like(depth)
    return (depth - dmin) / denom


def load_depth(
    depth_path: str,
    target_size: Optional[Sequence[int]] = None,
    normalize: str = "minmax",
) -> torch.Tensor:
    ext = os.path.splitext(depth_path)[1].lower()
    if ext in {".npy"}:
        arr = np.load(depth_path)
        depth = torch.from_numpy(arr)
    elif ext in {".pt"}:
        obj = torch.load(depth_path, map_location="cpu")
        if isinstance(obj, dict):
            if "depth" in obj:
                obj = obj["depth"]
            else:
                raise ValueError(f"Unsupported .pt depth payload keys: {list(obj.keys())}")
        depth = obj if torch.is_tensor(obj) else torch.tensor(obj)
    else:
        img = Image.open(depth_path)
        depth = torch.from_numpy(np.array(img))

    depth = depth.float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    elif depth.ndim == 3:
        # Handle HWC or CHW with 3 channels.
        if depth.shape[0] in (1, 3, 4) and depth.shape[-1] not in (1, 3, 4):
            depth = depth[:1]
        else:
            depth = depth[..., :1].permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported depth tensor shape {tuple(depth.shape)} from {depth_path}")

    if target_size is not None:
        th, tw = _to_hw(target_size)
        depth = F.interpolate(depth.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False).squeeze(0)

    if str(normalize).lower() == "minmax":
        depth = _normalize_minmax(depth)

    return depth.to(dtype=torch.float32)

