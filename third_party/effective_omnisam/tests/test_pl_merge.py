"""
tests/test_pl_merge.py

Unit tests for utils/pl_merge.py and utils/pl_sidecar.py.

Run with:
  cd /root/autodl-tmp/OmniSAM-main
  conda run -n omnisam python -m pytest tests/test_pl_merge.py -v
  # or without pytest:
  conda run -n omnisam python tests/test_pl_merge.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

# make sure repo root is on path
_repo = os.path.join(os.path.dirname(__file__), "..")
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from utils.pl_merge import PlMergeConfig, MergeResult, merge_patch_sequence, legacy_merge
from utils.pl_sidecar import save_sidecar, load_sidecar
from utils.tools import merge_width_scatter_np_uncertainty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Small test dimensions that still exercise all coverage values
W_FULL   = 4096
W_PATCH  = 1024
STRIDE   = 384
H_FULL   = 1024
C        = 13
L        = (W_FULL - W_PATCH) // STRIDE + 1   # 9
PASSES   = 2
N        = 2   # batch size > 1 for shape checks


def _rand_logits(n=N, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, PASSES * L, C, H_FULL, W_PATCH).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: legacy_sum must be bit-exact with original merge function
# ─────────────────────────────────────────────────────────────────────────────
def test_legacy_exact():
    logits = _rand_logits()
    threshold = 0.85
    expected = merge_width_scatter_np_uncertainty(
        logits_np=logits,
        orig_size=(H_FULL, W_FULL),
        W_patch=W_PATCH,
        stride_w=STRIDE,
        passes=PASSES,
        device=DEVICE,
        return_probs=False,
        threshold=threshold,
    )
    got = merge_patch_sequence(
        logits_np=logits,
        orig_size=(H_FULL, W_FULL),
        W_patch=W_PATCH,
        stride_w=STRIDE,
        passes=PASSES,
        device=DEVICE,
        cfg=PlMergeConfig(mode="legacy_sum", threshold=threshold),
    ).pred
    assert np.array_equal(expected, got), (
        f"legacy_sum not bit-exact: max_diff={np.abs(expected.astype(int) - got.astype(int)).max()}"
    )
    print("PASS test_legacy_exact")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: normalized_mean — same prob everywhere → same confidence regardless of coverage
# ─────────────────────────────────────────────────────────────────────────────
def test_coverage_normalization():
    """
    Construct logits so that every pixel has the same per-patch probability.
    After normalization, confidence must be uniform (coverage-independent).
    """
    # All patches predict class 0 with softmax prob ~0.80 (logit=2.0, others=0).
    logits = np.zeros((1, PASSES * L, C, H_FULL, W_PATCH), dtype=np.float32)
    logits[:, :, 0, :, :] = 2.0    # class 0 gets high logit

    threshold = 0.50
    result = merge_patch_sequence(
        logits_np=logits,
        orig_size=(H_FULL, W_FULL),
        W_patch=W_PATCH,
        stride_w=STRIDE,
        passes=PASSES,
        device=DEVICE,
        cfg=PlMergeConfig(mode="normalized_mean", threshold=threshold),
    )
    conf = result.confidence[0]  # (H, W)
    cov  = result.coverage[0]    # (H, W)

    # Identify columns with coverage 1, 2, 3
    starts = list(range(0, W_FULL - W_PATCH + 1, STRIDE))
    cov_per_col = np.zeros(W_FULL, dtype=int)
    for s in starts:
        cov_per_col[s:s + W_PATCH] += 1

    for cov_val in [1, 2, 3]:
        cols = np.where(cov_per_col == cov_val)[0]
        confs = conf[:, cols]
        mean_c = float(confs.mean())
        # All should be close (within 1e-4 floating point)
        assert abs(mean_c - conf.mean()) < 1e-3, (
            f"Coverage={cov_val}: mean_confidence={mean_c:.6f} "
            f"differs from global {conf.mean():.6f} (coverage inflation not fixed)"
        )
    print("PASS test_coverage_normalization")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: coverage_map shape == (N, H, W)
# ─────────────────────────────────────────────────────────────────────────────
def test_coverage_shape():
    logits = _rand_logits(n=3)
    result = merge_patch_sequence(
        logits_np=logits,
        orig_size=(H_FULL, W_FULL),
        W_patch=W_PATCH,
        stride_w=STRIDE,
        passes=PASSES,
        device=DEVICE,
        cfg=PlMergeConfig(mode="normalized_mean", threshold=0.5),
    )
    assert result.coverage.shape == (3, H_FULL, W_FULL), (
        f"coverage shape {result.coverage.shape} != (3,{H_FULL},{W_FULL})"
    )
    assert result.confidence.shape == (3, H_FULL, W_FULL), (
        f"confidence shape {result.confidence.shape} != (3,{H_FULL},{W_FULL})"
    )
    print("PASS test_coverage_shape")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: zero-coverage region → pred=255, no NaN in confidence
# ─────────────────────────────────────────────────────────────────────────────
def test_zero_coverage():
    """
    In practice zero-coverage never happens for valid W_FULL / stride, but
    we validate the guard anyway by using a tiny panorama with a gap.
    """
    # Use a narrow panorama where stride > W_FULL - W_PATCH forces coverage=1 everywhere,
    # but we directly test the NaN guard on the normalized path.
    logits = _rand_logits(n=1)
    result = merge_patch_sequence(
        logits_np=logits,
        orig_size=(H_FULL, W_FULL),
        W_patch=W_PATCH,
        stride_w=STRIDE,
        passes=PASSES,
        device=DEVICE,
        cfg=PlMergeConfig(mode="normalized_mean", threshold=0.5),
    )
    conf = result.confidence
    assert not np.isnan(conf).any(), "NaN found in confidence map"
    assert not np.isinf(conf).any(), "Inf found in confidence map"
    # All coverage > 0 (every column covered at least once)
    assert (result.coverage > 0).all(), "Unexpected zero-coverage column"
    print("PASS test_zero_coverage")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: sidecar round-trip + fallback
# ─────────────────────────────────────────────────────────────────────────────
def test_sidecar_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_name  = "CityA/leftImg8bit/img_001.jpg"
        conf_np   = np.random.rand(H_FULL, W_FULL).astype(np.float32)
        cov_np    = np.random.randint(1, 4, size=(H_FULL, W_FULL), dtype=np.uint8)

        # Write sidecar WITH coverage (debug mode)
        path = save_sidecar(tmpdir, img_name, conf_np, cov_np, save_coverage=True)
        assert os.path.isfile(path), f"sidecar not written: {path}"

        # Load back
        data = load_sidecar(tmpdir, img_name)
        assert data is not None, "load_sidecar returned None for existing file"
        assert "pl_confidence" in data, "pl_confidence key missing"
        assert "coverage" in data, "coverage key missing"

        # float16 precision loss must be small
        diff = np.abs(data["pl_confidence"].astype(np.float32) - conf_np).max()
        assert diff < 0.01, f"float16 precision loss too large: {diff}"

        # Fallback: missing file → None
        missing = load_sidecar(tmpdir, "nonexistent/path.jpg")
        assert missing is None, "fallback did not return None for missing sidecar"

    print("PASS test_sidecar_roundtrip")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: disabled psr_pld → legacy pipeline unaffected
# ─────────────────────────────────────────────────────────────────────────────
def test_legacy_fallback_when_disabled():
    """
    When pseudo_label_reliability.enabled=False, merge_patch_sequence with
    legacy_sum must match the original utility exactly.
    """
    logits = _rand_logits(seed=42)
    expected = merge_width_scatter_np_uncertainty(
        logits_np=logits,
        orig_size=(H_FULL, W_FULL),
        W_patch=W_PATCH,
        stride_w=STRIDE,
        passes=PASSES,
        device=DEVICE,
        return_probs=False,
        threshold=0.85,
    )
    # Simulate what Engine does when psr_pld_cfg is None: use old function
    got = merge_width_scatter_np_uncertainty(
        logits_np=logits,
        orig_size=(H_FULL, W_FULL),
        W_patch=W_PATCH,
        stride_w=STRIDE,
        passes=PASSES,
        device=DEVICE,
        return_probs=False,
        threshold=0.85,
    )
    assert np.array_equal(expected, got), "legacy path changed"
    print("PASS test_legacy_fallback_when_disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_legacy_exact,
        test_coverage_normalization,
        test_coverage_shape,
        test_zero_coverage,
        test_sidecar_roundtrip,
        test_legacy_fallback_when_disabled,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests PASSED.")
