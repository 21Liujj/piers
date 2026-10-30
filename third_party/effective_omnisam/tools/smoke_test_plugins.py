#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import torch

# Ensure repo root is on PYTHONPATH when running from workspace.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adaptation.plugins.pseudo_conf_mask import masked_unsup_ce
from adaptation.plugins.depth_confidence import compute_depth_confidence_weight
from model.depth_adapter import LightDepthAdapter
from model.modules.sparse_memory_retriever import SparseMemoryRetriever


def _assert_finite(x: torch.Tensor, name: str) -> None:
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise RuntimeError(f"{name} has NaN/Inf")


def test_pseudo_conf_mask() -> None:
    torch.manual_seed(0)
    b, c, h, w = 2, 5, 16, 16
    logits = torch.randn(b, c, h, w)
    pseudo = torch.randint(low=0, high=c, size=(b, h, w))
    pseudo[:, :2, :2] = 255  # ignore corner
    probs = torch.softmax(logits, 1)
    conf, _ = probs.max(1)
    base_mask = torch.ones_like(pseudo, dtype=torch.bool)
    out = masked_unsup_ce(
        pred_logits=logits,
        pseudo_label=pseudo,
        conf_map=conf,
        base_mask=base_mask,
        threshold=0.9,
        ignore_index=255,
        min_valid_ratio=0.0,
        min_valid_pixels=0,
        fallback="skip",
        weight_map=None,
    )
    _assert_finite(out.loss_raw, "loss_raw")
    _assert_finite(out.loss_masked, "loss_masked")


def test_sparse_memory_retriever() -> None:
    torch.manual_seed(0)
    retr = SparseMemoryRetriever()
    l, b, c = 256, 1, 32
    tokens = torch.randn(l, b, c)
    pos = torch.randn(l, b, c)
    sel_tokens, sel_pos, idx, stats = retr.retrieve(
        memory_tokens=tokens,
        pos_tokens=pos,
        spatial_shape=(16, 16),
        memory_id="m0",
        mode="feature_norm",
        keep_ratio=0.25,
        min_tokens=64,
        cache_enabled=True,
        fallback="feature_norm",
    )
    assert sel_tokens.shape[0] >= 64
    assert sel_tokens.shape == sel_pos.shape
    assert stats.selected_num_tokens == int(sel_tokens.shape[0])


def test_depth_confidence_weight() -> None:
    torch.manual_seed(0)
    depth = torch.rand(2, 1, 128, 256)
    pseudo = torch.randint(low=0, high=13, size=(2, 128, 256))
    cfg = {
        "edge_consistency": True,
        "smooth_region_consistency": True,
        "weight_min": 0.3,
        "weight_max": 1.0,
        "edge_beta": 0.5,
        "smooth_beta": 0.5,
        "ignore_index": 255,
    }
    out = compute_depth_confidence_weight(depth=depth, pseudo_label=pseudo, cfg=cfg)
    w = out.weight_map
    assert w.shape == pseudo.shape
    assert float(w.min().item()) >= 0.3 - 1e-6
    assert float(w.max().item()) <= 1.0 + 1e-6
    out_no_depth = compute_depth_confidence_weight(depth=None, pseudo_label=pseudo, cfg=cfg)
    assert torch.allclose(out_no_depth.weight_map, torch.ones_like(pseudo, dtype=torch.float32))


def test_depth_adapter() -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = torch.randn(2, 256, 64, 128, device=dev)
    depth = torch.rand(2, 1, 512, 1024, device=dev)
    adapter = LightDepthAdapter(out_channels=256, gate_init=0.0).to(dev)

    out = adapter(feature, depth)
    assert out.shape == feature.shape

    out_none = adapter(feature, None)
    assert torch.allclose(out_none, feature)

    # alpha init = 0, should be no-op up to floating error.
    assert torch.allclose(out, feature, atol=1e-6)
    assert adapter.alpha.requires_grad

    if dev.type == "cuda":
        with torch.cuda.amp.autocast(enabled=True):
            out_amp = adapter(feature, depth)
        assert out_amp.shape == feature.shape


def main() -> None:
    test_pseudo_conf_mask()
    test_depth_confidence_weight()
    test_depth_adapter()
    test_sparse_memory_retriever()
    print("[smoke_test_plugins] OK")


if __name__ == "__main__":
    main()

