"""C-generic taxonomy helpers for indoor SPAN8 (isolated; frozen street files untouched)."""
from __future__ import annotations

SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]
MAP13_TO_8 = [255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 5, 6, 7]
NATIVE13_14 = ["<UNK>", "beam", "board", "bookcase", "ceiling", "chair", "clutter",
               "column", "door", "floor", "sofa", "table", "wall", "window"]

assert len(SPAN8) == 8 and len(MAP13_TO_8) == 13


def map13_to_span8(ids):
    import numpy as np
    lut = __import__("numpy").array(MAP13_TO_8, dtype="int32")
    a = __import__("numpy").asarray(ids, dtype="int32")
    out = __import__("numpy").full(a.shape, 255, dtype="int32")
    m = (a >= 0) & (a < 13)
    out[m] = lut[a[m]]
    return out
