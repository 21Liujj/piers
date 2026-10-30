# OmniSAM 中的 Efficient-SAM2-Style Sparse Memory Retrieval（SMR）

本文档记录 OmniSAM 中参考 Efficient-SAM2 思想实现的**稀疏记忆检索（Sparse Memory Retrieval, SMR）**加速：在 memory attention 中，仅让历史 memory bank 的**显著（salient）空间记忆 tokens**参与 K/V 计算，从而减少 cross-attention 的计算开销。

> 注意：本仓库任务是全景语义分割（patch sliding-window 序列），不是单目标 prompt tracking。因此本实现不写死 foreground object 逻辑，采用 **feature-norm 主线**并预留 attention/confidence/overlap/boundary 的扩展接口。

## 1. 范围与目标（Scope & Objective）

- **不改** OmniSAM 主干 / decoder 接口与 checkpoint 兼容性。
- 通过“从历史空间 memory tokens 中检索 top-k”降低 memory attention 的 K/V token 数。
- `enabled=False` 时保持 baseline 行为完全不变（no-op）。
- 低风险、易消融：
  - 不引入 Perceiver
  - 不引入 learnable latent queries
  - 不引入额外可训练压缩器
- 语义安全：
  - 不压缩 current-frame query tokens
  - 不破坏 object pointer / special tokens 的语义（本 repo 的 SMR 只作用于历史 spatial memory tokens）

## 2. 核心流程（Core Pipeline）

默认路径（`before_attention`）：

1. 从 memory bank 读取历史 spatial memory tokens（`maskmem_features`）与对应位置编码（`maskmem_pos_enc`）。
2. 计算显著性分数（当前稳定默认：`feature_norm`）。
3. 按 keep ratio 选 top-k（受 `keep_ratio / min_tokens / max_tokens` 约束）。
4. 用**同一组 indices**同时 gather `memory_tokens` 与 `pos_tokens`（避免 pos-enc 错位）。
5. 将检索后的稀疏 memory 送入 SAM2 memory attention，current query 不变。
6. 记录 token keep ratio、cache hit ratio、selection time、memory attention time 等指标。

与 EfficientTAM-style memory pooling（`mem_compression`）的组合：

- 支持顺序配置：
  - `pool_then_smr`：先 pooling 再做 SMR（默认）
  - `smr_then_pool`：先 SMR 再 pooling

## 3. 代码实现文件（Implementation Files）

- `model/modules/sparse_memory_retriever.py`
  - `SparseMemoryRetriever`（非学习式检索/选择）
  - mode 支持：`none`, `feature_norm`, `attn_topk`（占位 + fallback）, `confidence_topk`, `hybrid`, `random`
  - 安全回退（min token、NaN、空选择、异常保护）
  - First Recollection Cache（缓存 selected indices，CPU + detach）

- `model/memory_bank.py`
  - SMR 插入点：`before_attention`
  - 训练/伪标签阶段开关：`set_runtime_context(stage=...)`
  - selection 失败自动回退 full memory
  - 复用已有 memory attention 计时埋点（CUDA sync）

- `run_adaptation.py` + `utils/build.py`
  - CLI 配置面与参数接线
  - baseline-safe 默认值（`enabled=False`）

- `adaptation/strategies.py` / `adaptation/pl_generator.py` / `adaptation/engine.py`
  - 训练 iter 日志 + 伪标签统计 + JSONL 落盘聚合

## 4. 配置项（Config Parameters）

本实现使用 `run_adaptation.py` 中 `sparse_memory_retrieval_*` CLI 参数。

关键参数：

- `sparse_memory_retrieval_enabled`（默认 `False`）
- `sparse_memory_retrieval_location`（当前仅 `before_attention`）
- `sparse_memory_retrieval_mode`（`none|feature_norm|attn_topk|confidence_topk|hybrid|random`）
- `sparse_memory_retrieval_fallback_mode`（`feature_norm|random`）
- `sparse_memory_retrieval_min_tokens`
- `sparse_memory_retrieval_max_tokens`（0 表示不限制）
- `sparse_memory_retrieval_cache_enabled`
- `sparse_memory_retrieval_cache_update_interval`
- `sparse_memory_retrieval_apply_in_train`
- `sparse_memory_retrieval_apply_in_pseudo_label`
- `sparse_memory_retrieval_train_keep_ratio`
- `sparse_memory_retrieval_pl_keep_ratio`
- `sparse_memory_retrieval_verbose`
- `sparse_memory_retrieval_log_interval`
- `sparse_memory_retrieval_order_with_pooling`（`pool_then_smr|smr_then_pool`）

预留参数（本版本不启用 / 后续扩展用）：

- `sparse_memory_retrieval_return_attn`（attn_topk 需要上游返回 attention score）
- `sparse_memory_retrieval_preserve_overlap_tokens`
- `sparse_memory_retrieval_preserve_boundary_tokens`
- `sparse_memory_retrieval_preserve_confident_tokens`

## 5. 检索模式说明（Retrieval Modes）

- `none`
  - 完全 no-op，等价 baseline。

- `feature_norm`（当前稳定默认）
  - `score = mean_B ||token||_2`，不依赖 attention 权重，工程风险最低。

- `attn_topk`（占位 + 自动回退）
  - 设计目标：用 memory cross-attention 响应估计 token 重要性。
  - 当前状态：SAM2 attention 使用 `scaled_dot_product_attention`，默认不返回权重；因此当 `attn_scores` 不可用时会**自动 fallback**到 `fallback_mode`（默认 `feature_norm`）。

- `confidence_topk`（可选）
  - 当外部提供 confidence map 时可用；否则回退。

- `hybrid`（轻量融合）
  - 在 feature_norm 基础上可叠加 attn/conf/overlap/boundary priors（若提供且 shape 匹配），否则退化为 feature_norm。

- `random`
  - 负对照/消融 sanity check。

## 6. First Recollection Cache（首次回忆缓存）

目标：

- 同一条历史 memory 在多次被读取时，避免重复计算 selection。

当前实现：

- cache key：`memory_id`（由 `MemoryBank` 生成，如 `source:<frame_idx>` / `target:<frame_idx>`）。
- cache 内容：selected indices（`detach().cpu()`），避免显存泄漏与梯度追踪。
- cache 清理：
  - 当 memory reset（`_init_output_dict_source/_target`）时，移除对应 memory_id 的 cache 条目。

说明：

- 通过 `sparse_memory_retrieval_cache_update_interval` 控制更新频率（降低额外开销）。

## 7. 精度与稳定性保护（Safety Guards）

- `min_tokens` 保护：token 数过少时跳过 SMR。
- top-k 至少保留 `min_tokens` 且至少 1 token。
- `memory_tokens` 与 `pos_tokens` 必须用同一 indices gather。
- NaN / 空选择 / 异常：自动回退 full memory。
- 默认 `enabled=False`，baseline 行为不变。

## 8. 日志与指标（Logging & Metrics）

### 训练 online 指标（iter 级）

在 `adaptation/strategies.py` 的 `metrics` 中新增：

- `smr_enabled`
- `smr_mode`
- `smr_tokens_before`
- `smr_tokens_after`
- `smr_keep_ratio`
- `smr_selection_time`
- `smr_cache_hit`
- `smr_cache_hit_ratio`

并继续复用 memory attention 计时：

- `memory_attention_time`
- `memory_attention_time_ms`
- `memory_attention_time_avg`

### JSONL 汇总（epoch/stage 级）

- `metrics/pl_timing.jsonl`
  - 新增：`smr_tokens_before/after`, `smr_keep_ratio`, `smr_cache_hit_ratio`, `smr_selection_time`

- `metrics/train_timing.jsonl`
  - 新增：`smr_enabled`, `smr_mode`, `smr_cache_hit_ratio`

- `checkpoints/.../metrics.jsonl`
  - 新增 SMR 字段，便于自动化脚本汇总

计时说明：

- memory attention 时间在 `MemoryBank` 内部做 CUDA sync 后测量。
- SMR selection time 在 `SparseMemoryRetriever.retrieve(...)` 内部测量。

## 9. 推荐消融组合（Recommended Ablations）

至少跑：

1. Baseline
   - `--no-mem_compression_enabled`
   - `--no-sparse_memory_retrieval_enabled`

2. SMR Conservative（稳定默认）
   - `--sparse_memory_retrieval_enabled`
   - `--sparse_memory_retrieval_mode feature_norm`
   - `--sparse_memory_retrieval_train_keep_ratio 0.7`
   - `--sparse_memory_retrieval_pl_keep_ratio 0.5`

可选：

3. SMR Aggressive（偏伪标签阶段）
   - `--sparse_memory_retrieval_train_keep_ratio 0.7`
   - `--sparse_memory_retrieval_pl_keep_ratio 0.4`（或 0.3）

4. SMR + pooling（组合实验）
   - `--mem_compression_enabled`
   - `--sparse_memory_retrieval_enabled`
   - `--sparse_memory_retrieval_order_with_pooling pool_then_smr`

5. Random sanity check
   - `--sparse_memory_retrieval_mode random`

## 10. 组会表述建议（Meeting Positioning）

在 EfficientTAM-style memory pooling 的基础上，OmniSAM 进一步参考 Efficient-SAM2 的 Sparse Memory Retrieval 思想，利用 memory attention 中 token-level 冗余：在读取历史 patch 的 spatial memory 后，根据 token 重要性（如 feature norm，后续可扩展至 attention/confidence/overlap/boundary priors），仅保留 top-k salient memory tokens 作为 K/V 参与 memory attention，从而降低跨 patch 记忆建模的计算开销。与 EdgeTAM 的 learnable Perceiver 压缩不同，该方法不引入额外学习式压缩器，主要通过后处理式稀疏检索实现低风险加速，并在关闭开关时保持 baseline 完全一致。

