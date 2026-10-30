#!/usr/bin/env python3
"""
Generate offline pseudo-depth priors from RGB images using Depth Anything V2.

Notes:
- This tool produces pseudo-depth / relative depth, NOT metric depth.
- Generated depth priors are for downstream pseudo-label reliability estimation,
  not semantic supervision labels.
- This is an offline preprocessing utility; it does not participate in OmniSAM
  training backpropagation and does not increase OmniSAM training memory usage.
"""

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
EXCLUDE_KEYWORDS = {
    "label",
    "mask",
    "annotation",
    "gt",
    "depth",
    "boundary",
    "json",
}

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


@dataclass
class ImageItem:
    image_path: Path
    relative_path: Path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _norm_to_uint8_per_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    denom = max(vmax - vmin, 1e-12)
    y = (x - vmin) / denom
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def _is_excluded(path: Path) -> bool:
    joined = "/".join(path.parts).lower()
    return any(k in joined for k in EXCLUDE_KEYWORDS)


def discover_images(scan_root: Path) -> List[ImageItem]:
    if not scan_root.exists():
        raise FileNotFoundError(f"Image scan root not found: {scan_root}")

    items: List[ImageItem] = []
    for p in sorted(scan_root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if _is_excluded(p.relative_to(scan_root)):
            continue
        rel = p.relative_to(scan_root)
        items.append(ImageItem(image_path=p, relative_path=rel))
    return items


def build_depth_anything_model(encoder: str, ckpt: Path, device: torch.device):
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        raise RuntimeError(
            "Failed to import Depth Anything V2. Please ensure the package is importable. "
            "Tip: set PYTHONPATH to Depth-Anything-V2 root, or run in that environment."
        ) from e

    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported encoder={encoder}. Choices: {list(MODEL_CONFIGS.keys())}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def infer_relative_depth(
    model,
    raw_bgr: np.ndarray,
    input_size: int,
    precision: str,
    use_cuda: bool,
) -> np.ndarray:
    with torch.no_grad():
        if use_cuda and precision in {"fp16", "bf16"}:
            dtype = torch.float16 if precision == "fp16" else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=dtype):
                depth = model.infer_image(raw_bgr, input_size=input_size)
        else:
            depth = model.infer_image(raw_bgr, input_size=input_size)
    return np.asarray(depth, dtype=np.float32)


def compute_depth_gradient(depth: np.ndarray, use_scharr: bool) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    norm = depth - float(depth.min())
    norm = norm / max(float(norm.max()), 1e-12)

    if use_scharr:
        gx = cv2.Scharr(norm, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(norm, cv2.CV_32F, 0, 1)
    else:
        gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    return np.asarray(grad, dtype=np.float32)


def maybe_smooth_depth(depth: np.ndarray, smooth: str, kernel: int) -> np.ndarray:
    """
    Optional smoothing before gradient/boundary extraction to reduce texture noise
    (e.g., foliage / brick patterns). This does NOT change the saved depth_npy;
    it only affects grad/boundary computation.
    """
    if smooth == "none":
        return depth
    k = int(kernel)
    if k <= 1:
        return depth
    if k % 2 == 0:
        k += 1
    if smooth == "gaussian":
        return cv2.GaussianBlur(depth.astype(np.float32), (k, k), sigmaX=0.0)
    raise ValueError(f"Unsupported smooth mode: {smooth}")


def apply_morphology(binary_01: np.ndarray, morph: str, kernel_size: int) -> np.ndarray:
    if morph == "none":
        return binary_01
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    src = (binary_01 * 255).astype(np.uint8)
    if morph == "open":
        out = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    elif morph == "close":
        out = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Unsupported morph mode: {morph}")
    return (out > 0).astype(np.uint8)


def compute_boundary(
    grad: np.ndarray,
    boundary_percentile: float,
    morph: str,
    morph_kernel: int,
) -> Tuple[np.ndarray, float]:
    thr = float(np.percentile(grad, boundary_percentile))
    boundary = (grad >= thr).astype(np.uint8)
    boundary = apply_morphology(boundary, morph=morph, kernel_size=morph_kernel)
    return boundary.astype(np.uint8), thr


def make_compare_vis(
    rgb: np.ndarray,
    depth_png: np.ndarray,
    grad_png: np.ndarray,
    boundary_png: np.ndarray,
) -> np.ndarray:
    if depth_png.ndim == 2:
        depth_png = cv2.cvtColor(depth_png, cv2.COLOR_GRAY2BGR)
    if grad_png.ndim == 2:
        grad_png = cv2.cvtColor(grad_png, cv2.COLOR_GRAY2BGR)
    if boundary_png.ndim == 2:
        boundary_png = cv2.cvtColor(boundary_png, cv2.COLOR_GRAY2BGR)
    tiles = [rgb, depth_png, grad_png, boundary_png]
    return cv2.hconcat(tiles)


def resolve_scan_root(image_root: Optional[str], image_dir: Optional[str]) -> Path:
    if image_dir:
        return Path(image_dir).expanduser().resolve()
    if image_root:
        return Path(image_root).expanduser().resolve()
    raise ValueError("Either --image-root or --image-dir must be provided.")


def default_out_dir(dataset_name: str) -> Path:
    return (REPO_ROOT / "aux_outputs" / "depth_anything_v2" / dataset_name).resolve()


def build_output_paths(out_dir: Path, rel: Path) -> Dict[str, Path]:
    stem_png = rel.with_suffix(".png")
    stem_npy = rel.with_suffix(".npy")
    return {
        "depth_npy": out_dir / "depth_npy" / stem_npy,
        "depth_png": out_dir / "depth_png" / stem_png,
        "depth_grad_npy": out_dir / "depth_grad_npy" / stem_npy,
        "depth_grad_png": out_dir / "depth_grad_png" / stem_png,
        "depth_boundary_npy": out_dir / "depth_boundary_npy" / stem_npy,
        "depth_boundary_png": out_dir / "depth_boundary_png" / stem_png,
        "vis_compare": out_dir / "vis_compare" / stem_png,
    }


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        _ensure_parent(self.log_path)
        self.fp = open(self.log_path, "a", encoding="utf-8")

    def write(self, msg: str) -> None:
        line = msg.rstrip("\n")
        print(line, flush=True)
        self.fp.write(line + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()


def parse_args() -> argparse.Namespace:
    default_root = str((REPO_ROOT / "data" / "WildPASS").resolve())
    ap = argparse.ArgumentParser(
        description="Generate pseudo-depth priors (depth/grad/boundary) from RGB images.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "python tools/generate_wildpass_depth_prior.py \\\n"
            "  --image-root /root/autodl-tmp/OmniSAM-main/data/WildPASS \\\n"
            "  --out-dir /root/autodl-tmp/OmniSAM-main/aux_outputs/depth_anything_v2/WildPASS \\\n"
            "  --encoder vitb \\\n"
            "  --ckpt /root/autodl-tmp/checkpoints/depth_anything_v2_vitb.pth \\\n"
            "  --precision fp16 \\\n"
            "  --boundary-percentile 90 \\\n"
            "  --skip-existing \\\n"
            "  --save-vis-num 50\n\n"
            "python tools/generate_wildpass_depth_prior.py \\\n"
            "  --image-root /root/autodl-tmp/OmniSAM-main/data/WildPASS \\\n"
            "  --out-dir /root/autodl-tmp/OmniSAM-main/aux_outputs/depth_anything_v2/WildPASS_debug \\\n"
            "  --encoder vits \\\n"
            "  --ckpt /root/autodl-tmp/checkpoints/depth_anything_v2_vits.pth \\\n"
            "  --precision fp16 \\\n"
            "  --max-images 20 \\\n"
            "  --save-vis-num 20"
        ),
    )
    ap.add_argument("--image-root", type=str, default=default_root, help="Dataset root for recursive image scan.")
    ap.add_argument("--image-dir", type=str, default="", help="Explicit image directory to scan.")
    ap.add_argument("--dataset-name", type=str, default="WildPASS", help="Dataset tag used in metadata/output defaults.")

    ap.add_argument("--out-dir", type=str, default="", help="Output directory. Default: aux_outputs/depth_anything_v2/<dataset-name>")
    ap.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--ckpt", type=str, required=True, help="Depth Anything V2 checkpoint path.")
    ap.add_argument("--input-size", type=int, default=518, help="Depth Anything V2 inference input size.")
    ap.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])

    ap.add_argument("--boundary-percentile", type=float, default=90.0, help="Percentile threshold for depth boundary.")
    ap.add_argument("--morph", type=str, default="none", choices=["none", "open", "close"])
    ap.add_argument("--morph-kernel", type=int, default=3)
    ap.add_argument("--grad-method", type=str, default="sobel", choices=["sobel", "scharr"])
    ap.add_argument(
        "--grad-depth-smooth",
        type=str,
        default="none",
        choices=["none", "gaussian"],
        help="Optional depth smoothing (for grad/boundary only) to reduce texture noise.",
    )
    ap.add_argument(
        "--grad-smooth-kernel",
        type=int,
        default=5,
        help="Kernel size for --grad-depth-smooth (odd number recommended).",
    )

    ap.add_argument("--skip-existing", action="store_true", help="Skip when depth/grad/boundary npy already exist.")
    ap.add_argument("--max-images", type=int, default=0, help="Process only first N images (0 means all).")
    ap.add_argument("--save-vis-num", type=int, default=0, help="Save first N comparison visualizations.")
    ap.add_argument("--empty-cache-every", type=int, default=50, help="Call torch.cuda.empty_cache every N images.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scan_root = resolve_scan_root(args.image_root, args.image_dir if args.image_dir else None)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir(args.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(out_dir / "process_log.txt")
    manifest_path = out_dir / "manifest.json"
    metadata_path = out_dir / "metadata.json"

    resolved_ckpt = Path(args.ckpt).expanduser().resolve()

    log.write("=== WildPASS/Generic depth prior generation started ===")
    log.write(f"scan_root={scan_root}")
    log.write(f"dataset_name={args.dataset_name}")
    log.write(f"out_dir={out_dir}")
    log.write(f"encoder={args.encoder}, ckpt={resolved_ckpt}, input_size={args.input_size}, precision={args.precision}")
    log.write(f"boundary_percentile={args.boundary_percentile}, morph={args.morph}, morph_kernel={args.morph_kernel}")
    log.write(f"grad_method={args.grad_method}, grad_depth_smooth={args.grad_depth_smooth}, grad_smooth_kernel={args.grad_smooth_kernel}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        log.write(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        log.write("WARNING: CUDA not available, using CPU inference (slower).")
        if args.precision in {"fp16", "bf16"}:
            log.write("WARNING: fp16/bf16 requested but CUDA unavailable, fallback to fp32-style inference.")

    model = build_depth_anything_model(args.encoder, resolved_ckpt, device=device)
    use_scharr = args.grad_method == "scharr"

    items = discover_images(scan_root)
    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]
    if not items:
        raise FileNotFoundError(f"No valid RGB images found under: {scan_root}")

    for sub in [
        "depth_npy",
        "depth_png",
        "depth_grad_npy",
        "depth_grad_png",
        "depth_boundary_npy",
        "depth_boundary_png",
        "vis_compare",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    failed_files: List[Dict[str, str]] = []
    success_count = 0
    fail_count = 0
    skipped_count = 0
    vis_saved = 0

    pbar = tqdm(items, desc="depth-prior", total=len(items))
    for idx, item in enumerate(pbar):
        paths = build_output_paths(out_dir, item.relative_path)

        entry: Dict[str, object] = {
            "image_path": str(item.image_path),
            "relative_path": str(item.relative_path.as_posix()),
            "depth_npy_path": str(paths["depth_npy"]),
            "depth_grad_npy_path": str(paths["depth_grad_npy"]),
            "depth_boundary_npy_path": str(paths["depth_boundary_npy"]),
            "image_height": None,
            "image_width": None,
            "status": "pending",
            "boundary_threshold": None,
        }

        if args.skip_existing and paths["depth_npy"].is_file() and paths["depth_grad_npy"].is_file() and paths["depth_boundary_npy"].is_file():
            entry["status"] = "skipped"
            skipped_count += 1
            results.append(entry)
            if (idx + 1) % 100 == 0:
                log.write(f"[skip] {idx + 1}/{len(items)} {item.relative_path}")
            continue

        try:
            raw_bgr = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                raise RuntimeError("cv2.imread returned None")
            h, w = raw_bgr.shape[:2]
            entry["image_height"] = int(h)
            entry["image_width"] = int(w)

            depth = infer_relative_depth(
                model=model,
                raw_bgr=raw_bgr,
                input_size=args.input_size,
                precision=args.precision,
                use_cuda=use_cuda,
            ).astype(np.float32)

            depth_for_grad = maybe_smooth_depth(
                depth,
                smooth=args.grad_depth_smooth,
                kernel=args.grad_smooth_kernel,
            )
            grad = compute_depth_gradient(depth_for_grad, use_scharr=use_scharr).astype(np.float32)
            boundary, threshold = compute_boundary(
                grad=grad,
                boundary_percentile=args.boundary_percentile,
                morph=args.morph,
                morph_kernel=args.morph_kernel,
            )
            entry["boundary_threshold"] = float(threshold)

            depth_png = _norm_to_uint8_per_image(depth)
            grad_png = _norm_to_uint8_per_image(grad)
            boundary_png = (boundary * 255).astype(np.uint8)

            _ensure_parent(paths["depth_npy"])
            np.save(paths["depth_npy"], depth.astype(np.float32))
            _ensure_parent(paths["depth_grad_npy"])
            np.save(paths["depth_grad_npy"], grad.astype(np.float32))
            _ensure_parent(paths["depth_boundary_npy"])
            np.save(paths["depth_boundary_npy"], boundary.astype(np.uint8))

            _ensure_parent(paths["depth_png"])
            cv2.imwrite(str(paths["depth_png"]), depth_png)
            _ensure_parent(paths["depth_grad_png"])
            cv2.imwrite(str(paths["depth_grad_png"]), grad_png)
            _ensure_parent(paths["depth_boundary_png"])
            cv2.imwrite(str(paths["depth_boundary_png"]), boundary_png)

            if vis_saved < args.save_vis_num:
                rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                compare = make_compare_vis(
                    rgb=rgb,
                    depth_png=depth_png,
                    grad_png=grad_png,
                    boundary_png=boundary_png,
                )
                compare_bgr = cv2.cvtColor(compare, cv2.COLOR_RGB2BGR)
                _ensure_parent(paths["vis_compare"])
                cv2.imwrite(str(paths["vis_compare"]), compare_bgr)
                vis_saved += 1

            entry["status"] = "success"
            success_count += 1

        except Exception as e:
            fail_count += 1
            entry["status"] = "failed"
            err = f"{type(e).__name__}: {e}"
            failed_files.append({"image_path": str(item.image_path), "relative_path": str(item.relative_path), "error": err})
            log.write(f"[error] {item.relative_path}: {err}")
            log.write(traceback.format_exc().rstrip("\n"))

        results.append(entry)

        if use_cuda and args.empty_cache_every > 0 and ((idx + 1) % args.empty_cache_every == 0):
            torch.cuda.empty_cache()

        pbar.set_postfix(success=success_count, skipped=skipped_count, failed=fail_count)

    metadata = {
        "dataset_name": args.dataset_name,
        "image_root": str(Path(args.image_root).expanduser().resolve()) if args.image_root else "",
        "image_dir": str(Path(args.image_dir).expanduser().resolve()) if args.image_dir else "",
        "scan_root": str(scan_root),
        "out_dir": str(out_dir),
        "encoder": args.encoder,
        "ckpt": str(resolved_ckpt),
        "precision": args.precision,
        "input_size": int(args.input_size),
        "boundary_percentile": float(args.boundary_percentile),
        "morph": args.morph,
        "morph_kernel": int(args.morph_kernel),
        "grad_method": args.grad_method,
        "grad_depth_smooth": args.grad_depth_smooth,
        "grad_smooth_kernel": int(args.grad_smooth_kernel),
        "skip_existing": bool(args.skip_existing),
        "max_images": int(args.max_images),
        "save_vis_num": int(args.save_vis_num),
        "total_images": int(len(items)),
        "success_count": int(success_count),
        "failed_count": int(fail_count),
        "skipped_count": int(skipped_count),
        "failed_files": failed_files,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"dataset_name": args.dataset_name, "scan_root": str(scan_root), "items": results}, f, ensure_ascii=False, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.write("=== Finished ===")
    log.write(f"total={len(items)}, success={success_count}, skipped={skipped_count}, failed={fail_count}")
    log.write(f"manifest={manifest_path}")
    log.write(f"metadata={metadata_path}")
    log.close()


if __name__ == "__main__":
    main()
