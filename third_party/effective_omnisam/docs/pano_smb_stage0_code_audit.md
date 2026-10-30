# Pano-SMB Stage 0 Code Audit

## Audit Scope and Reproducibility Boundary

This is a static audit of the current working tree at `/root/autodl-tmp/OmniSAM-main` on branch `omnisam-base-20260410`, HEAD `57b7654`. The working tree contains pre-existing modified and untracked files, so all conclusions describe that working tree rather than clean HEAD. No Python, configuration, loss, model, Top-K, token budget, checkpoint, branch, dependency, or experiment was modified or run during this audit.

Evidence is classified as: **code fact**, **existing-result fact**, **reasonable hypothesis**, or **unsupported claim**. Stage 0 does not treat hypotheses as measured conclusions.

## 1. Current OmniSAM Memory Pipeline

**Code fact.** DensePASS panoramas are processed as horizontal view sequences. `utils/build.py` configures ERP size `(4096, 1024)`, window `1024`, and stride `384`; `dataset/DensePASS.py::_sliding_windows` therefore yields nine overlapping `1024x1024` views:

`(4096 - 1024) / 384 + 1 = 9`.

`WithMemStrategy.forward` iterates tensor-order `frame_idx` over this sequence. For every view, `OmniSAM_adapt.forward` performs:

1. Encode the current image into current vision features and positional embeddings.
2. Call `MemoryBank.forward` to read historical memory.
3. Decode the current semantic prediction.
4. Convert the argmax prediction into `mask_for_mem`.
5. Call `MemoryBank._encode_new_memory` to write the current view.

The lifecycle is therefore **read history -> predict current view -> write current view**. A view never attends to its own newly encoded memory. Memory dictionaries are reset after the panorama sequence. Object-pointer tokens are disabled in this path (`num_obj_ptr_tokens=0`).

The actual memory unit is one historical tensor-order view, batch-packed as one `(L,B,C)` chunk stored under `frame_outputs[frame_idx]`. It is not one whole panorama and not one object.

## 2. Tensor Shape Trace

For SAM2.1 Hiera Small at image size 1024:

| Stage | Shape / count | Meaning |
|---|---:|---|
| Current query grid | `64x64` | high-level current-view grid |
| Flattened query | `(4096,B,256)` | query sequence for memory attention |
| Raw encoded memory map | `(B,64,64,64)` | 64 memory channels |
| Raw historical chunk | `(4096,B,64)` | one stored FoV after flatten/permute |
| MTC output per chunk | `(1024,B,64)` | `64x64 -> 32x32` average pooling |
| SMR output per chunk | `(512,B,64)` | 50% Top-K of compressed chunk |
| Max history at view 9 | 8 chunks | current view is not yet written |
| Concatenated max memory | `(4096,B,64)` | `8 x 512` memory tokens |

Memory positional encodings undergo the same flattening, pooling, indexing, and concatenation as memory features. Temporal memory position embeddings are added to each historical chunk before MTC and SMR.

## 3. Actual MTC Implementation

**Code fact.** MTC is `SpatialMemoryCompressor` in `model/modules/memory_compressor.py`. In the audited configuration it is a parameter-free two-dimensional average pool:

```python
F.avg_pool2d(feat_bchw, kernel_size=2, stride=2, ceil_mode=False)
```

It is independently applied to each historical memory chunk. Both the memory feature map and memory positional map are pooled, reducing each chunk from 4096 to 1024 tokens. It introduces no learned parameters.

Two implementation qualifications matter:

- `apply_to_pos_enc` is configured but not consulted inside `compress()`; positional maps are always pooled. This matches the audited run configuration but makes the flag misleading.
- `MemoryBank.forward` discards the compressed spatial shape returned by MTC and later supplies the original `mem_hw` to SMR. Feature-norm retrieval is unaffected because spatial shape is diagnostic in that path, but future spatial/geometry-aware selection could use incorrect geometry.

## 4. Actual SMR Implementation

**Code fact.** SMR is `SparseMemoryRetriever` in `model/modules/sparse_memory_retriever.py`. For a chunk with `L` compressed tokens, configured K is:

`K = clamp(round(L * keep_ratio), min_keep_tokens, max_tokens if set, L)`.

The audited configuration uses `mode: attn_topk`, `keep_ratio: 0.5`, and `min_keep_tokens: 64`. However, the caller passes `attn_scores=None` and `confidence_map=None`, and the fallback saliency source is `feature_norm`. The active score is therefore:

```python
score = torch.linalg.norm(tokens_lbc.float(), dim=-1).mean(dim=1)
```

This produces one score vector of shape `(L,)`, averaged across batch. `torch.topk` selects one shared index set, the indices are sorted into sequence order, and identical indices are applied to feature and positional tokens.

Consequences:

- Selection is based on feature norm, not measured attention.
- All batch samples share K and selected token locations.
- Confidence, overlap, boundary, and hybrid pathways are interfaces or dormant alternatives, not active evidence for the standard run.

## 5. Global vs Per-FoV Retrieval

**Code fact. Retrieval scope is independently per historical FoV/chunk.** `MemoryBank.forward` loops over historical entries; for each entry it adds temporal position, runs MTC, and runs SMR. Only afterward does it concatenate all selected chunks with `torch.cat` and call SAM2 memory attention.

The operative order is:

`historical chunk -> temporal position -> per-chunk MTC -> per-chunk Top-K -> concatenate chunks -> memory attention`.

There is no single global Top-K over all historical tokens. Existing whole-chunk `select_mode` logic can retain or drop complete chunks, but it is separate from token-level SMR and is not a shared-token-budget allocator.

## 6. Current Token Budget Allocation

The canonical audited MTC+SMR allocation is uniform:

| History chunks available | Tokens per chunk | Total memory tokens |
|---:|---:|---:|
| 1 | 512 | 512 |
| 2 | 512 | 1024 |
| ... | ... | ... |
| 8 | 512 | 4096 |

Before MTC/SMR, maximum historical memory is `8 x 4096 = 32768` tokens. MTC reduces it to `8 x 1024 = 8192`; SMR reduces it to `8 x 512 = 4096`. Base SAM YAML declares `num_maskmem: 7`, but `run_adaptation.py` overrides it from `memory_bank_size`; canonical timing evidence records `memory_bank_size=9`.

## 7. Evidence of Uniform FoV Budget

Uniformity is established directly by control flow and K computation:

1. Each historical entry is processed separately in the same loop.
2. All standard chunks have the same raw `64x64` shape.
3. MTC applies the same stride-2 pool, yielding 1024 tokens per chunk.
4. SMR computes K solely from that chunk length and common configuration, yielding 512.
5. No cross-chunk normalization, quota assignment, residual budget, per-view cap, or exact-total correction exists.

Therefore current capacity is equal per historical tensor-order FoV. This is a code fact. It does **not** prove that equal allocation causes the measured accuracy loss.

## 8. Candidate Shared-Budget Injection Point

The minimal later injection point is inside `MemoryBank.forward`, **after every historical chunk has undergone MTC and before SMR selects tokens**.

A parameter-free, fixed-total design can be:

```text
collect all compressed chunks
-> reuse their feature-norm score vectors
-> derive one batch-shared importance value per chunk
-> assign K_min to each available chunk
-> distribute the residual shared budget by normalized importance
-> apply deterministic integer correction so sum(K_j) equals legacy total
-> run existing per-chunk Top-K using K_j
-> index memory and position tensors identically
-> concatenate
-> call unchanged SAM2 memory attention
```

At maximum history, the invariant should be `sum(K_j) = 4096`, matching the current endpoint. For shorter histories, the fixed comparator should match the legacy total `512 x number_of_available_chunks`, unless a later experiment explicitly defines another causal comparison.

SAM2 memory attention accepts variable memory sequence lengths shared across a batch. The current RoPE working-tree code repeats/crops frequencies for arbitrary key lengths. No padding is required when one batch-shared allocation is used. Per-sample ragged allocation is incompatible with the present attention interface unless padding plus a key mask, or separate calls, is introduced.

## 9. Existing Low-cost Importance Signals

At most three signals are recommended:

1. **Normalized high-score mass:** sum or mean of the top-q feature-norm scores in each compressed chunk, normalized across chunks. It reuses the score already needed by SMR.
2. **Score dispersion:** coefficient of variation or normalized entropy of feature-norm scores. It distinguishes concentrated salient structure from uniformly weak/strong chunks with only reductions.
3. **Low-resolution mask complexity:** boundary or class-transition density computed from the already available predicted `mask_for_mem`, preferably after cheap downsampling.

The stored scalar `importance = mask_for_mem.float().mean()` should not be used as semantic importance: it averages categorical class IDs whose numeric order has no semantic meaning.

## 10. Panorama Metadata Availability

Available or derivable geometry includes ERP width/height, window size, stride, sequence index, token row, and diagnostic latitude. Crop x-ranges are deterministically derivable only if sequence order is known to be the forward order.

`DensePASS._sliding_windows` returns tensors without explicit physical patch ID, x-range, direction, or reversal flag. When `passes == 2`, patch lists are extended in reverse order. Consequently tensor-order `frame_idx` is reliable, but physical longitude, seam identity, and overlap ownership are not always reliable metadata.

Existing geo/seam settings and `_build_geo_context` are diagnosis-only in the audited path. They do not affect active MTC, SMR, or token quotas. Geometry-aware budgeting should be blocked until physical crop/order metadata is carried explicitly.

## 11. Expected Runtime Overhead

For a batch-shared allocator that reuses existing feature-norm scores, extra work is limited to one scalar reduction per chunk, normalization across at most eight chunks, integer quota correction, and passing variable K values to existing Top-K calls. It adds no parameters and does not increase the concatenated memory-attention token count.

Expected extra overhead is small relative to memory attention, but cannot be called zero. Existing timing shows selection itself has measurable cost:

- Full memory attention: about 117.48 s for the profiled run.
- MTC+SMR attention: about 51.92 s, approximately 55.8% lower.
- Full-memory step: 2.4089 s; MTC+SMR step: 2.3153 s, only 3.89% lower overall.
- Derived pool/saliency/Top-K overhead is about 0.08 s/iteration and was not independently instrumented.
- A 75% E-only point is 2.4614 s/step, 2.18% slower than full memory, proving sparse retrieval is not free.

A valid later claim requires direct profiling, not only unchanged attention token count.

## 12. Compatibility Risks

1. **Batch semantics:** current saliency is averaged over B and one index set is shared. Pano-SMB can remain compatible only with batch-shared importance and quotas; sample-specific quotas require a larger attention-interface change.
2. **RoPE dependency:** arbitrary concatenated key lengths rely on uncommitted working-tree frequency repeat/crop handling absent from clean HEAD `57b7654`.
3. **Spatial-shape bug:** compressed `hw` is discarded, risking incorrect geometry-aware scoring.
4. **Metadata ambiguity:** tensor-order view identity is not always physical FoV identity when reverse passes are present.
5. **Minimum coverage:** aggressive allocation can starve a historical FoV; `K_min` and exact-total correction are required.
6. **Checkpoint compatibility:** parameter-free quota logic need not alter state dictionaries, but implementation must avoid registering new parameters/buffers or changing existing tensor contracts.
7. **Comparison validity:** changing total tokens, pseudo-labels, model weights, or memory-attention implementation would confound the shared-allocation question.

## 13. Existing Accuracy-Efficiency Failure Evidence

**Existing-result fact.** Fixed-pseudo-label tables report:

| Method | Max memory tokens | mIoU | BFScore | Memory-attention time |
|---|---:|---:|---:|---:|
| Full memory | 32768 | 60.26 | 39.59 | 117.48 s |
| MTC+SMR | 4096 | 60.02 | 39.23 | 51.92 s |

This is an 8x token reduction with `-0.24` mIoU and `-0.36` BFScore. Largest listed class drops versus `A_ccpm` are traffic light `-1.65`, fence `-0.77`, terrain `-0.39`, traffic sign `-0.29`, and pole `-0.28`. Road is unchanged; sky is `+0.01`, sidewalk `+0.27`, and wall `+0.62`.

Confirmed visualization notes include small-object misses, thin-structure breaks, boundary erosion, one seam inconsistency, class confusion, and suspected pseudo-label noise. These panels used standard inference with D/E inactive, so they reflect training outcomes rather than direct inference-time token-selection maps. One seam case does not establish systematic seam degradation.

Pseudo-label refiner records at the same 4096-token budget report only slight recovery: R0 60.02/39.35, class-balanced R2 60.07/39.54, and R2+prototype 60.05/39.57 mIoU/BFScore, with step time increasing from 3.393 to 3.717/3.749 s. Their rows are marked `done=False`, so provenance is incomplete and claims must remain qualified.

**Reasonable hypothesis:** equal quotas waste capacity on simple FoVs and underfund complex FoVs; fixed-total redistribution may recover some quality.

**Unsupported:** current evidence does not isolate uniform allocation as the cause, prove any FoV is intrinsically more important, or establish systematic seam damage.

## 14. Pano-SMB Feasibility Verdict

**Verdict: CONDITIONAL.**

Pano-SMB is technically feasible as a parameter-free, fixed-total, batch-shared cross-FoV quota allocator inserted between MTC and SMR. It can preserve the existing total attention memory length, use existing score computations, keep synchronized memory/position selection, and leave SAM2 memory attention unchanged.

Conditions are:

- Allocation is batch-shared, not per-sample ragged.
- Total K exactly equals the equal-budget comparator at every history length.
- Every FoV receives a minimum quota.
- RoPE variable-key handling is retained and verified.
- Initial design uses tensor-order chunks, not unverified physical longitude/seam claims.
- Later evaluation isolates allocation from pseudo-label, model, loss, MTC, and total-budget changes.

The largest implementation risk is batch-averaged saliency causing shared quotas/indices to fit no individual sample well. The largest research risk is that gains are absent or the method is judged generic heuristic budgeting rather than a panorama-specific contribution.

## 15. Minimal Experiment Recommendation

Stage 0 starts no experiment. If Stage 1 is separately approved, the minimal causal test should be:

1. Freeze the same trained checkpoint and pseudo-label set.
2. Keep MTC, total memory tokens, memory-bank size, decoder, losses, and inference protocol unchanged.
3. Compare equal `512/chunk` allocation against one parameter-free shared-budget allocator with `K_min` and exact-total correction.
4. Report mIoU, BFScore, per-class IoU, total step time, memory-attention time, allocator/Top-K time, actual per-chunk K, and exact total K.
5. Inspect small objects, thin structures, boundaries, and overlap/seam cases without claiming physical seam causality unless crop/order metadata is made explicit.

A signal ablation should occur only after the fixed-total feature-norm allocator establishes value. Combining three signals immediately would prevent attribution.

## 16. Alternative Directions if Pano-SMB Is Rejected

1. **Boundary-preserving MTC at the same 4096-token endpoint:** replace or augment uniform average pooling with parameter-free boundary-aware retention while holding final token count fixed. This targets the observed thin-structure and BFScore losses more directly.
2. **Overlap-consistency deduplication with budget recycling:** identify redundant overlap tokens across adjacent FoVs and recycle the saved quota to unique regions. This is panorama-specific but should not proceed until explicit crop/order metadata is available.

## Final Audit Answer

The current implementation assigns equal capacity to every standard historical tensor-order panoramic FoV: each is independently pooled from 4096 to 1024 tokens and independently reduced to 512 by feature-norm Top-K before concatenation. A fixed total budget can be redistributed adaptively without new model parameters and with low expected incremental cost, but feasibility is conditional on batch-shared quotas, exact budget preservation, minimum coverage, retained variable-length RoPE support, and disciplined causal evaluation.
