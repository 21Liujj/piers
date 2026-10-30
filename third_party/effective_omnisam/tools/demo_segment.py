import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# Ensure repo root is on sys.path when running as a script from tools/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.sam2_bootstrap  # noqa: F401

from utils.build import build_model, load_checkpoint, DATASET_NAME2CLASSES_13
from utils.tools import merge_width_scatter_np_uncertainty
from utils.vis import COLOR_MAP


DEFAULT_CLASS_NAMES = DATASET_NAME2CLASSES_13["Cityscapes"]

# Legend / panel typography: 200% of earlier defaults (22/26/30 → 44/52/60)
_FONT_PANEL_TITLE = 44
_FONT_LEGEND_ROW = 52
_FONT_LEGEND_TITLE = 60
_FONT_ID_ON_IMAGE = 56
_LEGEND_COLUMNS = 3


@dataclass
class DemoConfig:
    backbone: str = "sam2_s"
    num_maskmem: int = 9
    dataset: str = "Cityscapes"  # only affects class names here
    image_size: Tuple[int, int] = (4096, 1024)  # (W,H)
    window_size: Tuple[int, int] = (1024, 1024)  # (W,H)
    stride_w: int = 384
    passes: int = 2
    uc_threshold: float = 0.0  # 0.0 => keep all pixels; set 0.85 to mimic UDA PL filtering
    ignore_index: int = 255
    alpha: int = 110  # overlay alpha


def _resolve_ckpt_path(p: str) -> Path:
    path = Path(p)
    if path.is_file():
        return path.resolve()
    cand = (REPO_ROOT / p).resolve()
    if cand.is_file():
        return cand
    return path.resolve()


def _safe_slug(s: str, max_len: int = 48) -> str:
    t = re.sub(r"[^\w\-.]+", "_", s.strip(), flags=re.UNICODE)
    t = t.strip("_") or "ckpt"
    return t[:max_len]


def _sliding_patches(img_t: torch.Tensor, window_w: int, stride_w: int) -> torch.Tensor:
    _, h, w = img_t.shape
    patches = []
    for x in range(0, w, stride_w):
        if x + window_w <= w:
            patches.append(img_t[:, :, x : x + window_w])
    if not patches:
        raise ValueError(f"Image width={w} too small for window_w={window_w}.")
    return torch.stack(patches, dim=0)


def _normalize(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std
    return torch.from_numpy(arr)


def _make_overlay(mask: np.ndarray, alpha: int) -> Image.Image:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cid, rgb in COLOR_MAP.items():
        if cid == 255:
            continue
        m = mask == cid
        if not np.any(m):
            continue
        rgba[m, 0] = rgb[0]
        rgba[m, 1] = rgb[1]
        rgba[m, 2] = rgb[2]
        rgba[m, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def _centroid(mask: np.ndarray) -> Tuple[int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def _draw_ids(base_rgba: Image.Image, pred: np.ndarray, class_names: List[str]) -> Image.Image:
    out = base_rgba.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", _FONT_ID_ON_IMAGE)
    except Exception:
        font = ImageFont.load_default()

    for cid, cname in enumerate(class_names):
        m = pred == cid
        c = _centroid(m)
        if c is None:
            continue
        x, y = c
        text = str(cid)
        draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 255), font=font)
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    return out


def _compose_side_by_side_with_legend(
    original_rgb: Image.Image,
    overlay_ids_rgba: Image.Image,
    class_names: List[str],
    bg: Tuple[int, int, int] = (16, 16, 16),
    legend_columns: int = _LEGEND_COLUMNS,
) -> Image.Image:
    """
    Layout:
      [ original | overlay+ids ]
      [ legend: class names in legend_columns columns (row-major) ]
    """
    left = original_rgb.convert("RGB")
    right = overlay_ids_rgba.convert("RGBA")

    if left.size != right.size:
        right = right.resize(left.size, Image.BICUBIC)

    w, h = left.size
    pad = 18
    mid_gap = 18
    legend_pad = 28
    swatch = 48
    gap = 24
    line_h = 62

    legend_title_h = 72
    n = len(class_names)
    cols = max(1, int(legend_columns))
    num_rows = (n + cols - 1) // cols
    legend_h = legend_pad * 2 + legend_title_h + num_rows * line_h + 12

    canvas_w = pad * 2 + (w * 2) + mid_gap
    canvas_h = pad * 2 + h + legend_h

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (*bg, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_panel_title = ImageFont.truetype("DejaVuSans.ttf", _FONT_PANEL_TITLE)
        font_legend = ImageFont.truetype("DejaVuSans.ttf", _FONT_LEGEND_ROW)
        font_legend_title = ImageFont.truetype("DejaVuSans.ttf", _FONT_LEGEND_TITLE)
    except Exception:
        font_panel_title = ImageFont.load_default()
        font_legend = font_panel_title
        font_legend_title = font_panel_title

    x0 = pad
    y0 = pad
    canvas.paste(left, (x0, y0))
    canvas.paste(right, (x0 + w + mid_gap, y0), mask=right.split()[-1])

    draw.text((x0, y0 - 4), "Input", fill=(235, 235, 235, 255), font=font_panel_title)
    draw.text((x0 + w + mid_gap, y0 - 4), "Segmentation", fill=(235, 235, 235, 255), font=font_panel_title)

    legend_y0 = y0 + h + pad
    legend_x0 = pad
    legend_x1 = canvas_w - pad
    legend_y1 = legend_y0 + legend_h
    draw.rounded_rectangle((legend_x0, legend_y0, legend_x1, legend_y1), radius=14, fill=(24, 24, 24, 255))

    tx0 = legend_x0 + legend_pad
    ty0 = legend_y0 + legend_pad
    draw.text((tx0, ty0), "Legend (id: class)", fill=(255, 255, 255, 255), font=font_legend_title)
    ty_grid = ty0 + legend_title_h

    inner_w = (legend_x1 - legend_pad) - tx0
    col_gap = 32
    col_w = (inner_w - col_gap * (cols - 1)) // cols

    for row in range(num_rows):
        for col in range(cols):
            idx = row * cols + col
            if idx >= n:
                break
            cid = idx
            cname = class_names[cid]
            x = tx0 + col * (col_w + col_gap)
            y = ty_grid + row * line_h
            rgb = COLOR_MAP.get(cid, (200, 200, 200))
            draw.rectangle((x, y + 10, x + swatch, y + 10 + swatch), fill=(*rgb, 255))
            draw.text((x + swatch + gap, y), f"{cid:>2}: {cname}", fill=(240, 240, 240, 255), font=font_legend)

    return canvas


@torch.no_grad()
def predict_mask(
    model,
    img: Image.Image,
    class_names: List[str],
    device: torch.device,
    cfg: DemoConfig,
) -> np.ndarray:
    img = img.convert("RGB").resize(cfg.image_size, Image.BICUBIC)
    img_t = _normalize(img)

    W_patch, H_patch = cfg.window_size
    stride = cfg.stride_w

    patches_fwd = _sliding_patches(img_t, window_w=W_patch, stride_w=stride)
    if cfg.passes == 2:
        patches = torch.cat([patches_fwd, torch.flip(patches_fwd, dims=[0])], dim=0)
    else:
        patches = patches_fwd

    frame_cnt = patches_fwd.shape[0]
    C = len(class_names)
    H_full = cfg.image_size[1]
    logits_maps = np.zeros((1, cfg.passes * frame_cnt, C, H_full, W_patch), dtype=np.float32)

    model.eval()
    model = model.to(device)
    model.memory_bank._init_output_dict_source()
    model.memory_bank._init_output_dict_target()

    for i in range(cfg.passes * frame_cnt):
        frame_idx = 0
        frame = patches[i].unsqueeze(0).to(device)
        logits, *_ = model(frame_idx, frame, mem_type="target")
        logits_maps[0, i] = logits.detach().cpu().numpy()

    pred = merge_width_scatter_np_uncertainty(
        logits_np=logits_maps,
        orig_size=(H_full, cfg.image_size[0]),
        W_patch=W_patch,
        stride_w=stride,
        passes=cfg.passes,
        device="cpu",
        return_probs=False,
        threshold=cfg.uc_threshold,
        ignore_index=cfg.ignore_index,
    )[0]
    return pred.astype(np.uint8)


def _preview_scale_for_composed(composed: Image.Image, preview_max_width: int) -> float:
    if preview_max_width <= 0:
        return 1.0
    w, _ = composed.size
    if w <= preview_max_width:
        return 1.0
    return preview_max_width / float(w)


def _resize_pred_nearest(pred: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return pred
    h, w = pred.shape
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = Image.fromarray(pred, mode="L")
    img = img.resize((nw, nh), Image.NEAREST)
    return np.array(img, dtype=np.uint8)


def composed_from_pred(
    img: Image.Image,
    pred: np.ndarray,
    cfg: DemoConfig,
    class_names: List[str],
) -> Image.Image:
    base_rgb = img.convert("RGB").resize(cfg.image_size, Image.BICUBIC)
    base = base_rgb.convert("RGBA")
    overlay = _make_overlay(pred, alpha=cfg.alpha)
    blended = Image.alpha_composite(base, overlay)
    with_ids = _draw_ids(blended, pred, class_names=class_names)
    return _compose_side_by_side_with_legend(base_rgb, with_ids, class_names)


def apply_preview(
    pred: np.ndarray,
    composed: Image.Image,
    preview_max_width: int,
) -> Tuple[np.ndarray, Image.Image]:
    scale = _preview_scale_for_composed(composed, int(preview_max_width))
    if scale < 1.0:
        cw, ch = composed.size
        composed = composed.resize(
            (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
            Image.LANCZOS,
        )
    pred_out = _resize_pred_nearest(pred, scale)
    return pred_out, composed


def save_demo_outputs(
    pred: np.ndarray,
    composed: Image.Image,
    out_dir: Path,
    stem: str,
    preview_max_width: int,
) -> None:
    pred_out, composed_out = apply_preview(pred, composed, preview_max_width)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred_out, mode="L").save(out_dir / f"{stem}_pred_labelTrainIds.png", optimize=True)
    composed_out.save(out_dir / f"{stem}_demo_left_input_right_seg_bottom_legend.png", optimize=True)


def _run_one_ckpt(
    img: Image.Image,
    ckpt_path: str,
    cfg: DemoConfig,
    class_names: List[str],
    device: torch.device,
):
    model = build_model(cfg, num_classes=len(class_names), num_maskmem=cfg.num_maskmem, device=device)
    load_checkpoint(model, ckpt_path)
    pred = predict_mask(model=model, img=img, class_names=class_names, device=device, cfg=cfg)
    composed = composed_from_pred(img, pred, cfg, class_names)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred, composed


def stack_vertical_panels(
    panels: List[Image.Image],
    labels: List[str],
    bg: Tuple[int, int, int] = (16, 16, 16),
) -> Image.Image:
    """Stack preview-sized RGBA panels vertically with a title bar per row."""
    panels = [p.convert("RGBA") for p in panels]
    inner_w = max(p.width for p in panels)
    pad = 14
    bar_h = 84  # ~200% of prior 42
    gap = 12

    def center_paste(canvas: Image.Image, im: Image.Image, x0: int, y0: int) -> None:
        x = x0 + (inner_w - im.width) // 2
        canvas.paste(im, (x, y0), mask=im.split()[-1])

    total_w = pad * 2 + inner_w
    total_h = pad
    for im, _lab in zip(panels, labels):
        total_h += bar_h + im.height + gap
    total_h += pad - gap

    out = Image.new("RGBA", (total_w, total_h), (*bg, 255))
    draw = ImageDraw.Draw(out)
    try:
        font_bar = ImageFont.truetype("DejaVuSans.ttf", 48)
    except Exception:
        font_bar = ImageFont.load_default()

    y = pad
    for im, lab in zip(panels, labels):
        draw.text((pad + 4, y + 18), lab, fill=(240, 240, 240, 255), font=font_bar)
        y += bar_h
        center_paste(out, im, pad, y)
        y += im.height + gap
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Panorama segmentation demo: 1 ckpt → pred + demo; N ckpts → N demo pairs; optional vertical stack."
    )
    ap.add_argument("--image", type=str, required=True, help="Input image path (jpg/png).")
    ap.add_argument(
        "--ckpt",
        action="append",
        dest="ckpts",
        required=True,
        metavar="PATH",
        help="Checkpoint path. Repeat for multiple ckpts (one demo pair per ckpt).",
    )
    ap.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=None,
        help="Row title for each --ckpt in order. Default: checkpoint file stem.",
    )
    ap.add_argument(
        "--no_compare_stack",
        action="store_true",
        help="With 2+ ckpts, skip the single vertical stack PNG (only per-ckpt pred+demo files).",
    )
    ap.add_argument("--out_dir", type=str, default="demo_out", help="Output directory.")
    ap.add_argument("--backbone", type=str, default="sam2_s", choices=["sam2_t", "sam2_s", "sam2_b+", "sam2_l"])
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--uc_threshold", type=float, default=0.0)
    ap.add_argument("--alpha", type=int, default=110)
    ap.add_argument(
        "--preview_max_width",
        type=int,
        default=2048,
        help="Max width (px) of saved PNGs; 0 = full resolution.",
    )
    args = ap.parse_args()

    ckpts: List[str] = args.ckpts
    labels: List[str] = list(args.labels) if args.labels else []
    for i in range(len(labels), len(ckpts)):
        labels.append(Path(ckpts[i]).stem)
    if len(labels) != len(ckpts):
        ap.error("Number of --label must match number of --ckpt")

    resolved = []
    for i, c in enumerate(ckpts):
        p = str(_resolve_ckpt_path(c))
        if not Path(p).is_file():
            raise FileNotFoundError(f"--ckpt[{i}] not found: {c} -> {p}")
        resolved.append(p)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    class_names = DEFAULT_CLASS_NAMES
    cfg = DemoConfig(
        backbone=args.backbone,
        uc_threshold=float(args.uc_threshold),
        alpha=int(args.alpha),
    )

    img = Image.open(args.image)
    out_dir = Path(args.out_dir)
    stem = Path(args.image).stem
    pw = int(args.preview_max_width)

    preview_panels: List[Image.Image] = []
    last_composed = None

    for ckpt_path, label in zip(resolved, labels):
        pred, composed = _run_one_ckpt(img, ckpt_path, cfg, class_names, device)
        last_composed = composed
        slug = _safe_slug(label)
        out_stem = f"{stem}_{slug}" if len(ckpts) > 1 else stem
        save_demo_outputs(pred, composed, out_dir, out_stem, pw)
        _, panel = apply_preview(pred, composed, pw)
        preview_panels.append(panel)

    if len(ckpts) >= 2 and not args.no_compare_stack:
        slugs = "_".join(_safe_slug(lb) for lb in labels)
        stack_path = out_dir / f"{stem}_compare_stack_{slugs}.png"
        stack_path.parent.mkdir(parents=True, exist_ok=True)
        stacked = stack_vertical_panels(preview_panels, labels)
        stacked.save(stack_path, optimize=True)
        print(f"Compare stack: {stack_path.resolve()}")

    scale = _preview_scale_for_composed(last_composed, pw) if last_composed is not None else 1.0
    print(f"Saved under: {out_dir.resolve()} (preview scale≈{scale:.4f}, preview_max_width={pw})")


if __name__ == "__main__":
    main()
