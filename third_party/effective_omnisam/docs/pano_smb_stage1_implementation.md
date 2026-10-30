# Pano-SMB Stage 1 Implementation

## 1. Stage

Stage 1 implements fixed-budget adaptive FoV allocation in the historical-memory token-selection layer only. It does not start training, validation, or accuracy evaluation.

## 2. Scope

The change sits between MTC compression and legacy SMR Top-K selection. Backbone, memory encoder, SAM2 memory attention, decoder, losses, pseudo labels, data, evaluation, optimizer, and schedule remain unchanged.

## 3. Retrieval Pipeline

```text
Historical FoV memory
  -> existing MTC (normally 4096 -> 1024 tokens per FoV)
  -> Pano-SMB quota allocation (K_j per FoV)
  -> existing SMR feature-norm Top-K(K_j)
  -> historical-memory concatenation
  -> unchanged SAM2 memory attention
```

## 4. Feature Importance

Pano-SMB exposes and reuses the existing SMR feature-norm score:

```python
torch.linalg.norm(tokens_lbc.float(), dim=-1).mean(dim=1)
```

Each FoV importance is the mean of that FoV's token-score vector. Both the score vector and the resulting FoV quota are averaged across the batch dimension. The quota vector is therefore batch-shared, not sample-wise dynamic.

## 5. Legacy Budget

For each available historical FoV, Stage 1 computes the legacy SMR Top-K count with the existing `_safe_topk_count` rule. The allocation target is the sum of those legacy counts. Under the canonical MTC/SMR configuration, this is exactly:

```text
sum(K_j) = 512 * J
```

where `J` is the number of available historical FoVs. `4096` is not hard-coded; it occurs only when `J=8`.

## 6. Bounds

The configured lower bound is `K_min=256`. Each allocated quota satisfies:

```text
256 <= K_j <= compressed FoV capacity
```

The canonical compressed capacity is 1024 tokens, hence `K_j <= 1024`.

## 7. Allocation Rule

The allocator starts every FoV at the minimum quota, distributes the residual budget proportionally to nonnegative FoV importance, saturates any FoV that reaches capacity, and redistributes the residual over unsaturated FoVs.

## 8. Integer Correction

Continuous allocations are floored. Remaining integer tokens are assigned by descending fractional remainder with stable index-order tie breaking. The allocator asserts exact total-budget conservation and all configured bounds before returning.

## 9. Deterministic Fallbacks

The allocator returns the legacy quota vector for a single FoV, non-finite importance, all-zero importance, effectively equal importance, infeasible bounds, or zero residual importance after saturation. These paths are deterministic and preserve the legacy total.

## 10. Legacy SMR Reuse

`SparseMemoryRetriever.retrieve` accepts an optional `target_tokens`. When absent, the original keep-ratio/min-token K calculation runs unchanged. When present, only K changes; scoring, Top-K ranking, sorted token indices, and synchronized feature/position indexing remain the existing SMR behavior.

Adaptive cache reuse requires the cached index count to equal the requested K. A different K triggers score recomputation and replaces the cached indices, preventing stale variable-length selections.

## 11. Configuration Gate

The plugin configuration adds:

```yaml
pano_smb:
  enabled: false
  min_tokens: 256
  verbose: false
  log_interval: 200
```

The feature is disabled by default and requires `efficient_sam2_sparse.enabled=true`. It also requires whole-chunk `memory select_mode=none`; otherwise initialization/runtime rejects the incompatible configuration because post-allocation chunk removal would violate the fixed budget.

## 12. Instrumentation

`MemoryBank.last_pano_smb_stats` records FoV importance, quotas, legacy budget, allocated total, budget error, min/max/mean/std quota, importance std, fallback reason, and allocation-only time. Optional console output is verbose-gated, interval-gated, and suppressed for eval.

## 13. Tests

`tests/test_pano_smb.py` covers uniform, nonuniform, zero, NaN/Inf, extreme imbalance, J=1, every J from 1 through 8, deterministic integer correction, capacity saturation, exact budget conservation, synchronized positional indexing, optional-K disabled equivalence, and a warmed allocation-only CPU micro-profile.

A minimal model-level smoke constructs a small fake SAM2 shell around the real `MemoryBank`, loads historical chunks with the canonical token dimensions, enables MTC plus SMR plus Pano-SMB, and verifies the concatenated memory length and output shape without loading data or checkpoints.

## 14. Compatibility

Pano-SMB adds no trainable parameters and no checkpoint keys. Disabled-by-default configuration preserves checkpoint loading. With `enabled=false`, `target_tokens=None`, so the original SMR K path is used. The MTC geometry output remains ignored in the before-attention integration path, preserving legacy diagnostics and downstream arguments.

## 15. Stage 1 Boundary

No formal training, full epoch, validation, mIoU/BFScore measurement, seam or overlap method, PQR, language guidance, or hyperparameter sweep was run. Stage 1 makes no accuracy or superiority claim. Its verdict is based only on implementation checks, exact-budget tests, disabled-path equivalence, smoke execution, and allocation-only profiling.
