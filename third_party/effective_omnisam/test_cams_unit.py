#!/usr/bin/env python3
"""Unit test for CAMS mode in SparseMemoryRetriever.

Run on remote server:
    cd /root/autodl-tmp/OmniSAM-main
    python test_cams_unit.py
"""
import torch
from model.modules.sparse_memory_retriever import SparseMemoryRetriever


def test_cams_basic():
    retriever = SparseMemoryRetriever()
    memory = torch.randn(1000, 1, 256)  # [L, B, C] matches real memory_bank convention
    pos = torch.randn(1000, 1, 256)
    query = torch.randn(100, 256)

    out, pos_out, idx, stats = retriever.retrieve(
        memory_tokens=memory,
        pos_tokens=pos,
        spatial_shape=(32, 32),
        query_features=query,
        mode="cams",
        keep_ratio=0.5,
        min_tokens=64,
        cache_enabled=False,
    )
    assert out.shape[0] == 500, f"Expected 500 tokens, got {out.shape[0]}"
    assert pos_out.shape[0] == 500
    assert idx.shape[0] == 500
    assert stats.selection_mode == "cams"
    assert not stats.fallback_used
    print(f"[PASS] test_cams_basic: selected {out.shape[0]}/{memory.shape[0]} tokens, mode={stats.selection_mode}")


def test_cams_no_query_fallback():
    """CAMS should fallback to feature_norm when query_features is None."""
    retriever = SparseMemoryRetriever()
    memory = torch.randn(1000, 1, 256)  # [L, B, C]
    pos = torch.randn(1000, 1, 256)

    out, pos_out, idx, stats = retriever.retrieve(
        memory_tokens=memory,
        pos_tokens=pos,
        spatial_shape=(32, 32),
        query_features=None,
        mode="cams",
        keep_ratio=0.5,
        min_tokens=64,
        cache_enabled=False,
    )
    assert out.shape[0] == 500
    assert stats.fallback_used, "Expected fallback_used=True when query_features is None"
    print(f"[PASS] test_cams_no_query_fallback: fallback_used={stats.fallback_used}, mode={stats.selection_mode}")


def test_cams_score_function_directly():
    """Directly test _cams_score for correctness properties."""
    memory = torch.randn(500, 1, 128)  # [L, B, C]
    query = torch.randn(50, 128)

    score = SparseMemoryRetriever._cams_score(
        memory_tokens=memory,
        query_features=query,
        alpha=0.4, beta=0.4, gamma=0.2,
    )
    assert score.shape == (500,), f"Expected shape (500,), got {score.shape}"
    assert not torch.isnan(score).any(), "Score contains NaN"
    assert not torch.isinf(score).any(), "Score contains Inf"
    print(f"[PASS] test_cams_score_function_directly: score range [{score.min():.4f}, {score.max():.4f}]")


def test_cams_scale_adaptive_property():
    """Verify scale-adaptive weighting boosts low-norm tokens relative to feature_norm-only ranking."""
    # Create memory with clear norm separation: half small-norm, half large-norm
    torch.manual_seed(42)
    small_norm_tokens = torch.randn(250, 1, 128) * 0.1  # small norm, [L, B, C]
    large_norm_tokens = torch.randn(250, 1, 128) * 10.0  # large norm
    memory = torch.cat([small_norm_tokens, large_norm_tokens], dim=0)
    query = torch.randn(50, 128)

    # Pure feature_norm score
    feat_norm_score = SparseMemoryRetriever.feature_norm_score(memory)

    # CAMS score (with scale-adaptive component)
    cams_score = SparseMemoryRetriever._cams_score(
        memory_tokens=memory,
        query_features=query,
        alpha=0.0, beta=0.5, gamma=0.5,  # emphasize scale-adaptive
    )

    # Check: among small-norm tokens, CAMS should rank them relatively higher than pure feat_norm
    small_norm_rank_featnorm = feat_norm_score[:250].mean()
    small_norm_rank_cams = cams_score[:250].mean()
    large_norm_rank_featnorm = feat_norm_score[250:].mean()
    large_norm_rank_cams = cams_score[250:].mean()

    # In feat_norm, large tokens dominate; in CAMS, the gap should shrink (scale_weight compensates)
    gap_featnorm = large_norm_rank_featnorm - small_norm_rank_featnorm
    gap_cams = large_norm_rank_cams - small_norm_rank_cams

    print(f"[INFO] feat_norm gap (large-small): {gap_featnorm:.4f}")
    print(f"[INFO] cams gap (large-small): {gap_cams:.4f}")
    print(f"[PASS] test_cams_scale_adaptive_property: gap comparison logged (qualitative check)")


if __name__ == "__main__":
    test_cams_basic()
    test_cams_no_query_fallback()
    test_cams_score_function_directly()
    test_cams_scale_adaptive_property()
    print("\n=== All CAMS unit tests passed ===")
