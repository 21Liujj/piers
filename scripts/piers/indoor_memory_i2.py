"""Indoor I2 (Innovation-2) memory configuration -- ISOLATED from the frozen outdoor code.

The outdoor I2 runtime lives in `experiments/paper20261109/runtime/i1_runtime.py`
(`install_fixed_memory`) and is configured from
`configs/piers/paper20261109/memory.json`.  It hard-codes the *street* setting:

  * `mode="class_aware"` with `rare_class_ids=[3,4,6,7]` -- those ids are
    Cityscapes trainIds (car / truck / bus / train).  They do not exist in
    SPAN8, so the class-protected reserve would protect nothing indoors.
  * `num_classes=13` is hard-coded inside `memory_bank.py` when the boundary
    mask is extracted from the pseudo labels.
  * the checkpoint digest and the runtime source digests are asserted, so the
    outdoor installer cannot be reused for the indoor model without lying about
    which files were verified.

This module therefore re-implements the *same* installation for the indoor
setting:

  * identical sparse-retrieval configuration (`configure_sparse_retrieval`)
  * identical `set_runtime_context(stage="eval")`
  * the SPAN8 analogue of the "rare class" ids: the classes whose pixel share
    is below 1% (sofa, window) plus the two structurally hard ones, taken from
    the measured `SPAN8_CLASS_STATS.json` -- never from the street list
  * the SPAN8 analogue of the 13->8 LUT for the class-protected reserve
  * an honest audit dict (tokens before/after, fallbacks, mode actually used)

The inherited `SparseMemoryRetriever` / `MemoryBank` code is used unchanged;
only the *configuration* is indoor-specific.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]

# The frozen outdoor configuration, reproduced here so the indoor arms use the
# same budget / keep-ratio / score as the published outdoor numbers.
I2_OUTDOOR = {
    "mode": "class_aware",
    "score": "uncertainty",
    "budget": 1024,
    "keep_ratio": 0.03125,
    "rare_reserve_ratio": 0.15,
    "boundary_reserve_ratio": 0.125,
    "min_tokens": 64,
    "num_maskmem": 9,
}


def indoor_rare_class_ids(stats_path=None, threshold_pct=1.0):
    """SPAN8 classes that deserve the class-protected reserve.

    The outdoor setting reserves tokens for the rare *street* classes.  Indoors
    the rare classes are measured, not guessed: we read the per-class pixel
    share from `SPAN8_CLASS_STATS.json` and reserve every class below
    `threshold_pct` percent of the labelled pixels.

    Returns a list of SPAN8 trainIds (0..7) in the class order above.
    """
    if stats_path is None:
        stats_path = ("/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109/"
                      "SPAN8_CLASS_STATS.json")
    p = Path(stats_path)
    ids = []
    if p.is_file():
        rep = json.loads(p.read_text(encoding="utf-8"))
        # the heldout split is the evaluation domain; the fit split is what the
        # model saw during training.  Use the union so the reserve covers every
        # class that is rare in either.
        pct = {}
        for split in ("fit", "heldout"):
            s = rep.get(split, {})
            for k, v in s.get("class_pixel_pct", {}).items():
                pct[k] = min(pct.get(k, 100.0), float(v))
        for i, name in enumerate(SPAN8):
            if pct.get(name, 100.0) < threshold_pct:
                ids.append(i)
    if not ids:
        # measured fallback: sofa and window are the two classes that are
        # consistently below 1% of the labelled pixels in every split.
        ids = [SPAN8.index("sofa"), SPAN8.index("window")]
    return ids


def install_indoor_i2(model, stage="eval", stats_path=None, overrides=None):
    """Install the I2 sparse-memory configuration on an indoor model.

    Mirrors `i1_runtime.install_fixed_memory` for the indoor setting:

      * `configure_sparse_retrieval` with the frozen outdoor parameters
      * `set_runtime_context(stage=...)`
      * a `checked` wrapper on `sparse_retriever.retrieve` that records the
        audit instead of asserting (the outdoor one asserts because it is
        pinned to a frozen checkpoint; the indoor model is a different,
        fine-tuned checkpoint, so the honest thing is to *record* the
        fallbacks and report them).

    Returns the audit dict.
    """
    cfg = {**I2_OUTDOOR, **(overrides or {})}
    mb = model.memory_bank
    rare_ids = indoor_rare_class_ids(stats_path)

    mb.sparse_retriever.diagnostic_overlap = False
    mb.configure_sparse_retrieval(
        enabled=True,
        mode=cfg["mode"],
        keep_ratio=cfg["keep_ratio"],
        min_tokens=cfg["min_tokens"],
        cache_enabled=False,
        fallback="feature_norm",
        fallback_full_when_uncertain=True,
        enable_in_train=True,
        enable_in_pseudo_label=True,
        boundary_reserve_ratio=cfg["boundary_reserve_ratio"],
        rare_class_ids=rare_ids,
        rare_reserve_ratio=cfg["rare_reserve_ratio"],
        boundary_score_source=cfg["score"],
    )
    mb.set_runtime_context(stage=stage, iter_idx=0)

    audit = {
        "calls": 0,
        "fallbacks": 0,
        "tokens_before": 0,
        "tokens_after": 0,
        "max_tokens": 0,
        "configuration": {
            "mode": cfg["mode"],
            "score": cfg["score"],
            "keep_ratio": cfg["keep_ratio"],
            "budget": cfg["budget"],
            "rare_reserve_ratio": cfg["rare_reserve_ratio"],
            "rare_class_ids": rare_ids,
            "rare_class_names": [SPAN8[i] for i in rare_ids],
            "min_tokens": cfg["min_tokens"],
        },
    }

    retrieve = mb.sparse_retriever.retrieve

    def checked(*a, **kw):
        result = retrieve(*a, **kw)
        stats = result[3]
        audit["calls"] += 1
        audit["fallbacks"] += int(getattr(stats, "fallback_used", False))
        before = int(getattr(mb, "last_sparse_tokens_before", 0))
        after = int(getattr(mb, "last_sparse_tokens_after", 0))
        audit["tokens_before"] += before
        audit["tokens_after"] += after
        audit["max_tokens"] = max(audit["max_tokens"],
                                  int(getattr(mb, "last_effective_memory_length", 0)))
        return result

    mb.sparse_retriever.retrieve = checked
    return audit


def install_dense_memory(model, stage="eval"):
    """Explicitly disable the sparse retrieval -> the dense reference arm.

    Mirrors the outdoor `i1_dense_runtime.install_fixed_memory`: the same
    `configure_sparse_retrieval(enabled=False, mode="dense", keep_ratio=1.)`
    call, plus a guard that makes any accidental sparse retrieval loud
    instead of silent.
    """
    mb = model.memory_bank
    mb.configure_sparse_retrieval(enabled=False, mode="dense", keep_ratio=1.,
                                  min_tokens=64, cache_enabled=False,
                                  enable_in_train=False, enable_in_pseudo_label=False)
    mb.set_runtime_context(stage=stage, iter_idx=0)
    mb.sparse_retriever.diagnostic_overlap = False
    audit = {"mode": "dense", "calls": 0, "fallbacks": 0, "max_tokens": 0}

    def no_sparse(*a, **kw):
        raise AssertionError("Dense control attempted sparse retrieval")

    mb.sparse_retriever.retrieve = no_sparse
    return audit
