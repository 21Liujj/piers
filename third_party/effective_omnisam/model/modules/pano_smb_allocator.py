from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math
import time

import torch


@dataclass
class PanoSMBAllocation:
    quotas: List[int]
    importance: List[float]
    legacy_total_budget: int
    allocated_total: int
    budget_error: int
    min_quota: int
    max_quota: int
    mean_quota: float
    quota_std: float
    importance_std: float
    fallback_used: bool
    fallback_reason: str
    allocation_time: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PanoSMBAllocator:
    """Parameter-free, batch-shared allocation of a fixed FoV token budget."""

    def __init__(self, min_tokens: int = 256) -> None:
        self.min_tokens = max(1, int(min_tokens))

    @staticmethod
    def importance_from_scores(score_vectors: Sequence[torch.Tensor]) -> torch.Tensor:
        if not score_vectors:
            return torch.empty(0, dtype=torch.float32)
        return torch.stack([score.detach().float().mean() for score in score_vectors])

    @staticmethod
    def _std(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    def _result(self, quotas, importance, legacy_total, t0, fallback_reason=""):
        quota_list = [int(value) for value in quotas]
        importance_list = [float(value) for value in importance.detach().cpu().tolist()]
        allocated_total = int(sum(quota_list))
        return PanoSMBAllocation(
            quotas=quota_list,
            importance=importance_list,
            legacy_total_budget=int(legacy_total),
            allocated_total=allocated_total,
            budget_error=allocated_total - int(legacy_total),
            min_quota=min(quota_list) if quota_list else 0,
            max_quota=max(quota_list) if quota_list else 0,
            mean_quota=(allocated_total / len(quota_list)) if quota_list else 0.0,
            quota_std=self._std(quota_list),
            importance_std=self._std(importance_list),
            fallback_used=bool(fallback_reason),
            fallback_reason=str(fallback_reason),
            allocation_time=float(time.perf_counter() - t0),
        )

    def allocate(self, importance: torch.Tensor, legacy_quotas: Sequence[int], capacities: Sequence[int]):
        t0 = time.perf_counter()
        legacy = [int(value) for value in legacy_quotas]
        caps = [int(value) for value in capacities]
        count = len(legacy)
        if count == 0 or len(caps) != count or int(importance.numel()) != count:
            raise ValueError("importance, legacy_quotas, and capacities must have equal non-zero length")
        if any(quota < 1 for quota in legacy) or any(cap < quota for cap, quota in zip(caps, legacy)):
            raise ValueError("legacy quotas must be positive and cannot exceed FoV capacities")
        if any(cap < self.min_tokens for cap in caps):
            raise ValueError("every FoV capacity must be at least the Pano-SMB minimum quota")

        importance = importance.detach().float().reshape(-1)
        legacy_total = int(sum(legacy))
        if count == 1:
            return self._result(legacy, importance, legacy_total, t0, "single_fov")

        finite = bool(torch.isfinite(importance).all().item())
        safe_importance = torch.clamp(importance, min=0.0) if finite else importance
        if not finite:
            return self._result(legacy, importance, legacy_total, t0, "non_finite_importance")
        max_importance = float(safe_importance.max().item())
        min_importance = float(safe_importance.min().item())
        if max_importance <= torch.finfo(torch.float32).eps:
            return self._result(legacy, safe_importance, legacy_total, t0, "zero_importance")
        equal_tol = max(1e-8, 1e-6 * max_importance)
        if max_importance - min_importance <= equal_tol:
            return self._result(legacy, safe_importance, legacy_total, t0, "equal_importance")

        base = [min(self.min_tokens, cap) for cap in caps]
        base_total = int(sum(base))
        if base_total > legacy_total or legacy_total > sum(caps):
            return self._result(legacy, safe_importance, legacy_total, t0, "infeasible_bounds")

        device = safe_importance.device
        alloc = torch.tensor(base, device=device, dtype=torch.float64)
        cap_tensor = torch.tensor(caps, device=device, dtype=torch.float64)
        weights = safe_importance.to(dtype=torch.float64)
        remaining = float(legacy_total - base_total)
        active = cap_tensor > alloc

        for _ in range(count + 1):
            if remaining <= 1e-9 or not bool(active.any().item()):
                break
            active_weights = torch.where(active, weights, torch.zeros_like(weights))
            weight_sum = float(active_weights.sum().item())
            if weight_sum <= torch.finfo(torch.float64).eps:
                return self._result(legacy, safe_importance, legacy_total, t0, "zero_active_importance")
            proposed = remaining * active_weights / weight_sum
            room = torch.clamp(cap_tensor - alloc, min=0.0)
            added = torch.minimum(proposed, room)
            alloc = alloc + added
            used = float(added.sum().item())
            remaining = max(0.0, remaining - used)
            active = (cap_tensor - alloc) > 1e-9

        integer = torch.floor(alloc).to(torch.int64)
        tokens_left = legacy_total - int(integer.sum().item())
        fractional = alloc - integer.to(alloc.dtype)
        order = torch.argsort(fractional, descending=True, stable=True).detach().cpu().tolist()
        while tokens_left > 0:
            progressed = False
            for index in order:
                if int(integer[index].item()) < caps[index]:
                    integer[index] += 1
                    tokens_left -= 1
                    progressed = True
                    if tokens_left == 0:
                        break
            if not progressed:
                raise RuntimeError("Pano-SMB integer correction exhausted FoV capacity")

        quotas = [int(value) for value in integer.detach().cpu().tolist()]
        if sum(quotas) != legacy_total:
            raise RuntimeError("Pano-SMB failed to conserve the legacy token budget")
        if any(value < base[index] or value > caps[index] for index, value in enumerate(quotas)):
            raise RuntimeError("Pano-SMB produced a quota outside configured bounds")
        return self._result(quotas, safe_importance, legacy_total, t0)
