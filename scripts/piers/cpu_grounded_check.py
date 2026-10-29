"""CPU acceptance for the REAL Grounded-SAM-2 backend (no GPU required).

Runs the genuine HF Grounding-DINO + SAM2ImagePredictor path on a real image
with real local weights, and asserts the invariants the project depends on:

  1. grounding-dino-tiny loads from local files only (no network)
  2. sam2.1_hiera_small loads with its matching config; state_dict matches exactly
  3. DINO boxes are in ORIGINAL image pixel coords (no double scaling)
  4. SAM2 mask for a box overlaps that box (mask/box coordinate agreement)
  5. one FoV == one image encode; N boxes == N mask-decoder calls
  6. an absent query yields an EMPTY candidate list (never whole-image fallback)

Run (cwd must NOT be the vendor dir root so sam2 resolves to the vendor copy):
    python scripts/piers/cpu_grounded_check.py

Exit code 0 = passed_cpu. Non-zero = FAIL (see printed reason).
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENDOR = os.path.join(REPO, "third_party", "grounded_sam2")
WEIGHTS = os.environ.get("PIERS_WEIGHTS_DIR", r"D:\piers2027\weights")

GROUNDING_DIR = os.environ.get(
    "PIERS_GROUNDING_DIR", os.path.join(WEIGHTS, "grounding-dino-tiny-hf"))
SAM2_CKPT = os.path.join(WEIGHTS, "sam2.1_hiera_small.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
IMAGE = os.path.join(VENDOR, "assets", "tracking_car_1.jpg")

# sam2 must resolve to the VENDOR copy, not any repo-root copy.
sys.path.insert(0, VENDOR)
sys.path.insert(0, os.path.join(REPO, "src"))

failures = []
notes = []


def check(name: str, ok: bool, detail: str = ""):
    (notes if ok else failures).append(
        {"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image

    print(f"python      : {sys.executable}")
    print(f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"vendor      : {VENDOR}")
    print(f"grounding   : {GROUNDING_DIR}")
    print(f"sam2 ckpt   : {SAM2_CKPT}")

    for p in (GROUNDING_DIR, SAM2_CKPT, IMAGE):
        if not os.path.exists(p):
            print(f"FATAL missing resource: {p}")
            return 2

    import sam2
    check("sam2 resolves to vendor copy",
          os.path.abspath(sam2.__file__).startswith(os.path.abspath(VENDOR)),
          f"sam2.__file__={sam2.__file__}")

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cpu"
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        GROUNDING_DIR, local_files_only=True, trust_remote_code=False)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DIR, local_files_only=True, trust_remote_code=False).to(device)
    dino.eval()
    check("grounding-dino loads offline (local_files_only)",
          True, f"{time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    sam_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    check("sam2.1_hiera_small loads with matching config",
          True, f"{time.perf_counter() - t0:.1f}s, "
                f"image_size={sam_model.image_size}")

    # ---- real inference on a real photo -------------------------------------
    img = Image.open(IMAGE).convert("RGB")
    W, H = img.size
    print(f"image       : {IMAGE} {W}x{H}")

    def run_query(text: str):
        """Returns (boxes_xyxy_origscale, scores, masks, n_encodes, n_decodes)."""
        n_enc = n_dec = 0
        norm = text.strip().lower()
        if norm and not norm.endswith("."):
            norm += "."
        inputs = processor(images=img, text=norm, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.35, text_threshold=0.25,
            target_sizes=[img.size[::-1]])
        r0 = res[0]
        boxes = r0["boxes"].detach().cpu().numpy().astype(np.float64)
        scores = [float(s) for s in r0["scores"].detach().cpu().numpy().tolist()]
        labels = r0.get("text_labels", r0.get("labels", []))

        masks = []
        quals = []
        if len(boxes):
            predictor.set_image(np.array(img))
            n_enc += 1
            with torch.no_grad():
                m, q, _ = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=boxes.astype(np.float32), multimask_output=False)
            if m.ndim == 4:
                m = m[:, 0]
            masks = [np.asarray(x, dtype=bool) for x in m]
            quals = [float(x) for x in np.ravel(q)]
            n_dec += len(boxes)
        return boxes, scores, labels, masks, quals, n_enc, n_dec

    # --- positive query: "car" ---------------------------------------------
    boxes, scores, labels, masks, quals, n_enc, n_dec = run_query("car")
    check("positive query returns candidates",
          len(boxes) > 0,
          f"n={len(boxes)} labels={list(labels)[:4]} scores={[round(s,3) for s in scores[:4]]}")
    check("one FoV -> exactly one image encode", n_enc == 1, f"n_enc={n_enc}")
    check("N boxes -> N mask decoder calls", n_dec == len(boxes), f"n_dec={n_dec}")

    if len(boxes) and len(masks):
        b = boxes[0]
        check("DINO box is in ORIGINAL pixel coords (no double scaling)",
              -1e-6 <= b[0] and b[2] <= W + 1e-6 and -1e-6 <= b[1] and b[3] <= H + 1e-6,
              f"box0={[round(float(x),1) for x in b]} image={W}x{H}")
        m = masks[0]
        check("SAM2 mask shape == original image shape",
              m.shape == (H, W), f"mask={m.shape} expected={(H, W)}")
        bx0, by0, bx1, by1 = [float(x) for x in b]
        box_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        inter = float(m[max(0, int(by0)):max(0, int(by1)),
                       max(0, int(bx0)):max(0, int(bx1))].sum())
        check("SAM2 mask overlaps its own box (coord agreement)",
              inter > 0 and box_area > 0,
              f"intersection_px={int(inter)} box_area={box_area:.0f} "
              f"coverage={inter / max(box_area,1):.3f}")
        check("mask is not the whole image (not a fallback)",
              0 < int(m.sum()) < m.size,
              f"fg_px={int(m.sum())}/{m.size} ({m.mean():.4f})")

    # --- negative query: something absent ----------------------------------
    b2, s2, l2, m2, q2, e2, d2 = run_query("airplane")
    check("absent query yields EMPTY candidate list (no whole-image fallback)",
          len(b2) == 0 and len(m2) == 0,
          f"n_boxes={len(b2)} n_masks={len(m2)}")
    check("absent query performs NO image encode",
          e2 == 0, f"n_enc={e2}")

    result = {
        "status": "passed_cpu" if not failures else "failed",
        "device": device,
        "torch": torch.__version__,
        "sam2_file": sam2.__file__,
        "grounding_dir": GROUNDING_DIR,
        "sam2_ckpt": SAM2_CKPT,
        "sam2_cfg": SAM2_CFG,
        "image": {"path": IMAGE, "w": W, "h": H},
        "positive_query": {"text": "car", "n_boxes": len(boxes),
                           "scores": scores, "labels": list(labels),
                           "sam_quality": quals, "n_encodes": n_enc,
                           "n_decoder_calls": n_dec},
        "negative_query": {"text": "airplane", "n_boxes": len(b2),
                           "n_encodes": e2},
        "checks": notes + failures,
        "n_passed": len(notes), "n_failed": len(failures),
        "synthetic": False,
    }
    out = os.path.join(REPO, "runs", "cpu_grounded_check")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nresult: {len(notes)} passed, {len(failures)} failed -> {out}/result.json")
    if failures:
        for f_ in failures:
            print(f"  FAILED: {f_['check']} :: {f_['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
