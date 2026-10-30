from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model.memory_bank import MemoryBank
from model.modules.pano_smb_allocator import PanoSMBAllocator
from model.modules.sparse_memory_retriever import SparseMemoryRetriever


def _allocate(values, legacy=512, cap=1024):
    tensor = torch.tensor(values, dtype=torch.float32)
    allocator = PanoSMBAllocator(min_tokens=256)
    return allocator.allocate(tensor, [legacy] * len(values), [cap] * len(values))


def test_uniform_importance():
    result = _allocate([1] * 8)
    assert result.quotas == [512] * 8
    assert result.allocated_total == 4096 and result.budget_error == 0


def test_nonuniform_importance():
    result = _allocate([1, 1, 1, 3])
    assert result.quotas[-1] > result.quotas[0]
    assert sum(result.quotas) == 2048
    assert min(result.quotas) >= 256 and max(result.quotas) <= 1024


def test_zero_and_nonfinite_fallback():
    assert _allocate([0, 0, 0, 0]).quotas == [512] * 4
    assert _allocate([1, float("nan"), 2, 3]).quotas == [512] * 4
    assert _allocate([1, float("inf"), 2, 3]).quotas == [512] * 4


def test_extreme_imbalance():
    result = _allocate([1000, 0.01, 0.01, 0.01])
    assert sum(result.quotas) == 2048
    assert min(result.quotas) >= 256 and max(result.quotas) <= 1024
    assert result.quotas[0] == 1024


def test_single_fov():
    assert _allocate([3]).quotas == [512]


def test_all_legal_history_lengths():
    generator = torch.Generator().manual_seed(17)
    allocator = PanoSMBAllocator(min_tokens=256)
    for count in range(1, 9):
        importance = torch.rand(count, generator=generator) + 0.01
        first = allocator.allocate(importance, [512] * count, [1024] * count)
        second = allocator.allocate(importance, [512] * count, [1024] * count)
        assert first.quotas == second.quotas
        assert sum(first.quotas) == 512 * count
        assert min(first.quotas) >= 256 and max(first.quotas) <= 1024


def test_selector_override_and_position_sync():
    retriever = SparseMemoryRetriever()
    tokens = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4)
    positions = tokens + 1000
    selected, selected_pos, indices, stats = retriever.retrieve(
        tokens, positions, (2, 3), mode="feature_norm", keep_ratio=0.5,
        min_tokens=1, cache_enabled=False, target_tokens=2,
    )
    assert indices is not None and indices.numel() == 2
    assert torch.equal(selected, torch.index_select(tokens, 0, indices))
    assert torch.equal(selected_pos, torch.index_select(positions, 0, indices))
    assert torch.equal(selected_pos - selected, torch.full_like(selected, 1000))
    assert stats.selected_num_tokens == 2


def test_cache_recomputes_for_changed_quota():
    retriever = SparseMemoryRetriever()
    tokens = torch.arange(80, dtype=torch.float32).reshape(20, 1, 4)
    positions = tokens + 100
    first = retriever.retrieve(
        tokens, positions, (4, 5), memory_id="same", mode="feature_norm",
        min_tokens=1, cache_enabled=True, target_tokens=8,
    )
    second = retriever.retrieve(
        tokens, positions, (4, 5), memory_id="same", mode="feature_norm",
        min_tokens=1, cache_enabled=True, target_tokens=13,
    )
    third = retriever.retrieve(
        tokens, positions, (4, 5), memory_id="same", mode="feature_norm",
        min_tokens=1, cache_enabled=True, target_tokens=13,
    )
    assert first[0].shape[0] == 8 and not first[3].cache_hit
    assert second[0].shape[0] == 13 and not second[3].cache_hit
    assert third[0].shape[0] == 13 and third[3].cache_hit


def test_disabled_selector_legacy_equivalence():
    generator = torch.Generator().manual_seed(23)
    tokens = torch.randn(1024, 2, 8, generator=generator)
    positions = torch.randn(1024, 2, 8, generator=generator)
    legacy = SparseMemoryRetriever().retrieve(
        tokens, positions, (32, 32), mode="attn_topk", keep_ratio=0.5,
        min_tokens=64, cache_enabled=False, fallback="feature_norm",
    )
    disabled = SparseMemoryRetriever().retrieve(
        tokens, positions, (32, 32), mode="attn_topk", keep_ratio=0.5,
        min_tokens=64, cache_enabled=False, fallback="feature_norm", target_tokens=None,
    )
    assert torch.equal(legacy[2], disabled[2])
    assert torch.equal(legacy[0], disabled[0])
    assert torch.equal(legacy[1], disabled[1])
    assert legacy[3].selected_token_shape == disabled[3].selected_token_shape


def test_incompatible_stage_is_rejected():
    class FakeSam(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = SimpleNamespace(neck=SimpleNamespace(d_model=64))
            self.memory_encoder = SimpleNamespace(out_proj=nn.Linear(64, 64, bias=False))

    bank = MemoryBank(FakeSam(), num_maskmem=1)
    bank.configure_sparse_retrieval(enabled=True, mode="feature_norm", enable_in_pseudo_label=False)
    bank.configure_pano_smb(enabled=True)
    bank.set_runtime_context(stage="pseudo_label")
    try:
        bank._pano_smb_enabled_now()
    except ValueError as error:
        assert "current runtime stage" in str(error)
    else:
        raise AssertionError("Pano-SMB accepted a runtime stage where SMR is disabled")


def test_memory_bank_smoke():
    class FakeCompressor(nn.Module):
        def configure(self, **kwargs):
            pass

        def compress(self, feats, pos, hw):
            length, batch, channels = feats.shape
            compressed = feats.reshape(32, 2, 32, 2, batch, channels).mean((1, 3)).reshape(1024, batch, channels)
            compressed_pos = pos.reshape(32, 2, 32, 2, batch, channels).mean((1, 3)).reshape(1024, batch, channels)
            stats = SimpleNamespace(
                original_num_tokens=length,
                compressed_num_tokens=1024,
                compression_ratio=0.25,
                original_hw=hw,
                compressed_hw=(32, 32),
                compression_mode="pool",
                compression_location="before_attention",
                skipped=False,
                skip_reason="",
            )
            return compressed, compressed_pos, (32, 32), stats

    class FakeSam(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = SimpleNamespace(neck=SimpleNamespace(d_model=64))
            self.memory_encoder = SimpleNamespace(out_proj=nn.Linear(64, 64, bias=False))
            self.last_memory_length = 0

        def memory_attention(self, curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens):
            assert memory.shape == memory_pos.shape
            self.last_memory_length = int(memory.shape[0])
            return curr

    sam = FakeSam()
    bank = MemoryBank(sam, num_maskmem=8)
    bank.memory_compressor = FakeCompressor()
    bank.configure_compression(
        enabled=True, compress_mode="pool", compress_location="before_attention",
        kernel_size=2, stride=2, min_tokens=64,
    )
    bank.configure_sparse_retrieval(
        enabled=True, mode="feature_norm", keep_ratio=0.5, min_tokens=64,
        cache_enabled=True, fallback="feature_norm",
    )
    bank.configure_pano_smb(enabled=True, min_tokens=256)
    generator = torch.Generator().manual_seed(31)
    for frame_idx in range(8):
        scale = float(frame_idx + 1)
        bank.output_dict_target["frame_outputs"][frame_idx] = {
            "maskmem_features": torch.randn(4096, 1, 64, generator=generator) * scale,
            "maskmem_pos_enc": torch.randn(4096, 1, 64, generator=generator),
            "importance": 0.0,
            "mem_hw": (64, 64),
        }
    current = torch.randn(1, 64, 2, 2, generator=generator)
    current_pos = torch.randn(1, 64, 2, 2, generator=generator)
    output = bank(8, current, current_pos, "target")
    assert output.shape == current.shape
    assert sam.last_memory_length == 4096
    assert bank.last_pano_smb_stats["allocated_total"] == 4096
    assert bank.last_pano_smb_stats["budget_error"] == 0
    assert sum(bank.last_pano_smb_stats["quotas"]) == 4096


def test_allocation_micro_profile():
    allocator = PanoSMBAllocator(min_tokens=256)
    importance = torch.tensor([1.0, 1.1, 0.8, 1.5, 2.0, 0.7, 1.3, 0.9])
    for _ in range(50):
        allocator.allocate(importance, [512] * 8, [1024] * 8)
    start = time.perf_counter()
    repeats = 1000
    for _ in range(repeats):
        allocator.allocate(importance, [512] * 8, [1024] * 8)
    elapsed_us = (time.perf_counter() - start) * 1e6 / repeats
    assert elapsed_us < 10000, f"allocation overhead unexpectedly high: {elapsed_us:.1f} us"
    print(f"allocation_only_cpu_us_per_call={elapsed_us:.1f}")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
