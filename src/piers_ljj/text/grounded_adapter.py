"""Innovation-1 text branch adapter (Grounded-SAM-2 HF path).

Mirrors grounded_sam2_hf_model_demo.py as importable functions (no globals,
no argparse-at-import, no auto CUDA). Real weights/backend live in the
piers-grounded env + third_party/grounded_sam2 vendor snapshot (pending).

Pipeline per FoV crop (PIL/RGB):
  1. Grounding-DINO (HF transformers) -> boxes (orig-size xyxy), detector scores
  2. SAM2ImagePredictor.set_image once per FoV -> per-box mask decode (sam quality)
  3. Empty candidate list is legal -> return [] (never whole-image fallback)
  4. Caller remaps via geometry.box_local_to_global + paste_erp_wrap, then
     text.assoc.associate + collaboration rerank.

Coordinate rule: DINO orig-size xyxy vs normalized cxcywh vs SAM pixel prompts
must not be double-scaled; record raw/norm/tokenizer text separately.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GroundedConfig:
    grounding_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam2_cfg: str = "sam2_hiera_l.yaml"
    sam2_ckpt: str = ""          # local path; never fake if missing
    device: str = "cuda"        # or cpu (fp32 path)
    dtype: str = "fp32"
    box_thr: float = 0.35
    text_thr: float = 0.25
    disable_custom_kernels: bool = True
    local_files_only: bool = True


@dataclass
class TextCandidate:
    box_xyxy: tuple      # FoV-local pixel coords
    detector_score: float
    sam_quality: float
    mask: object         # HxW bool (numpy) or path per contracts
    phrase: str


def normalize_text(raw: str) -> str:
    """Lowercase + strip + trailing-dot convention; NEVER map to fixed ontology here."""
    t = raw.strip().lower()
    if t and not t.endswith("."):
        t += "."
    return t


def run_fov_text_branch(pil_crop, raw_text: str, cfg: GroundedConfig,
                        backend=None):
    """Run one FoV. backend: injected (dino, sam_predictor) pair for testability.

    Returns {"norm_text":..., "candidates": [TextCandidate], "meta": {...}}.
    Raises FileNotFoundError on missing ckpt; never substitutes synthetic backend.
    """
    norm = normalize_text(raw_text)
    if backend is None:
        raise RuntimeError(
            "real backend missing: vendor snapshot + weights required "
            "(see docs/piers/gpu_runbook.md); refusing synthetic fallback")
    dino, sam = backend
    image_enc = sam.set_image(pil_crop)  # one encode per FoV; caller counts calls
    # The dino backend returns (boxes, scores) or (boxes, scores, phrases).
    dino_out = dino(pil_crop, norm, cfg.box_thr, cfg.text_thr)
    if len(dino_out) == 3:
        boxes, det_scores, phrases = dino_out
    else:
        boxes, det_scores = dino_out
        phrases = [norm] * len(boxes)
    out = []
    for b, s, ph in zip(boxes, det_scores, phrases):
        masks, sam_q = sam.predict_boxes(image_enc, [b])
        for m, q in zip(masks, sam_q):
            out.append(TextCandidate(box_xyxy=tuple(b), detector_score=float(s),
                                     sam_quality=float(q), mask=m, phrase=str(ph)))
    return {"norm_text": norm, "candidates": out,
            "meta": {"image_encodes": 1, "mask_decoder_calls": len(out)}}
