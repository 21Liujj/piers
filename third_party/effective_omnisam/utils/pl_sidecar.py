"""
utils/pl_sidecar.py

Save and load lightweight sidecar files alongside pseudo-label PNGs.

Layout:
  {pl_root}/gtFine/{rel_name}_labelTrainIds.png   ← existing PL file (unchanged)
  {pl_root}/pseudo_meta/{rel_name}.npz            ← new sidecar

Sidecar NPZ keys (first version):
  pl_confidence : float16 [H, W]   – normalised confidence from PL merge
  coverage      : uint8  [H, W]    – (debug only) patch coverage count

The module is intentionally minimal:
  - save_sidecar()   writes the NPZ
  - load_sidecar()   reads it with fallback to None if missing
  - rel_name_from_img_name()  derives the relative key used for both PL and sidecar
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from utils.io import ensure_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rel_name_from_img_name(img_name: str) -> str:
    """
    img_name examples (as used in WildPASSforAdaptation):
      "Amsterdam/leftImg8bit/Amsterdam_001.jpg"
      "someCity/img.png"
    Strips leading slash, keeps structure.
    """
    return img_name.lstrip("/")


def _sidecar_path(pl_root: str, img_name: str) -> str:
    rel = rel_name_from_img_name(img_name)
    stem, _ = os.path.splitext(rel)
    return os.path.join(pl_root, "pseudo_meta", stem + ".npz")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_sidecar(
    pl_root: str,
    img_name: str,
    confidence: np.ndarray,       # float32 [H, W] – will be stored as float16
    coverage: Optional[np.ndarray] = None,  # uint8 [H, W] – stored only in debug mode
    save_coverage: bool = False,
) -> str:
    """
    Write sidecar NPZ next to the PL PNG.
    Returns the path written.
    """
    path = _sidecar_path(pl_root, img_name)
    ensure_dir(os.path.dirname(path))
    payload: Dict[str, np.ndarray] = {
        "pl_confidence": confidence.astype(np.float16),
    }
    if save_coverage and coverage is not None:
        payload["coverage"] = coverage.astype(np.uint8)
    np.savez_compressed(path, **payload)
    return path


# ---------------------------------------------------------------------------
# Load (with fallback)
# ---------------------------------------------------------------------------

def load_sidecar(
    pl_root: str,
    img_name: str,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load sidecar NPZ.  Returns None if the file does not exist.
    Caller must handle None (= fallback to old behaviour).
    """
    path = _sidecar_path(pl_root, img_name)
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path)
        return {k: data[k] for k in data.files}
    except Exception:
        return None
