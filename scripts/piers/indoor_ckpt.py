"""Checkpoint loading for the indoor adaptation branch.

Why this module exists
----------------------
`utils.checkpoint_adapt.smart_load_checkpoint` (upstream) calls
`model.load_state_dict(raw, strict=False)`.  PyTorch's `strict=False` only
tolerates *missing* and *unexpected* keys -- it does **not** tolerate a shape
mismatch.  The frozen street checkpoint carries a 13-class decoder classifier
(`decoder.linear_pred.weight` = [13, 256, 1, 1]) while the indoor model has an
8-class one ([8, 256, 1, 1]), so the upstream loader raises:

    RuntimeError: size mismatch for decoder.linear_pred.weight

We therefore filter the classifier tensors out *before* calling
`load_state_dict`, and then report exactly what was loaded, what was skipped
and what was re-initialised.  Nothing else in the upstream code is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load_indoor_checkpoint(model, ckpt_path, verbose=True) -> dict:
    """Load a checkpoint into the indoor model, adapting the class head.

    * 8-class head + 13-class checkpoint -> the 13-class classifier is dropped
      and the 8-class classifier stays randomly initialised (fully trained).
    * 8-class head + 8-class checkpoint -> everything loads as-is.
    * 13-class head + 13-class checkpoint -> everything loads as-is (Z arms).

    Returns an audit dict describing the outcome.
    """
    ckpt_path = str(ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    raw = {k.replace("module.", ""): v for k, v in raw.items()}

    model_state = model.state_dict()
    dropped = []
    resized = {}
    for key in list(raw.keys()):
        if key in model_state and raw[key].shape != model_state[key].shape:
            ck, mk = raw[key].shape, model_state[key].shape
            if len(ck) == len(mk) and all(a == b for a, b in zip(ck[1:], mk[1:])):
                # same feature shape, different class count -> re-init the head
                dropped.append(key)
                del raw[key]
            else:
                raise RuntimeError(
                    f"Incompatible checkpoint tensor {key}: ckpt {tuple(ck)} vs model {tuple(mk)}"
                )

    missing, unexpected = model.load_state_dict(raw, strict=False)

    audit = {
        "ckpt": ckpt_path,
        "ckpt_tensors": len(raw) + len(dropped),
        "loaded": len(raw),
        "dropped_shape_mismatch": dropped,
        "missing_keys": len(list(missing)),
        "unexpected_keys": len(list(unexpected)),
        "head_state": "reinitialised" if dropped else "loaded_from_ckpt",
        "missing_sample": list(missing)[:10],
        "unexpected_sample": list(unexpected)[:10],
    }
    if verbose:
        print(json.dumps(audit, indent=2))
    return audit
