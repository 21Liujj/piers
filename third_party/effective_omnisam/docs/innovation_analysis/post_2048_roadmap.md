# 20:48 后创新路线图

> 基于今晚 n500 快速验证结果，规划接下来的创新点验证优先级
> 创建时间: 2026-08-25 16:00

---

## 📊 当前创新点全景

### ✅ 已验证/进行中 (Phase 1)

| 创新点 | 状态 | 结果 | 下一步 |
|--------|------|------|--------|
| **Module D: MTC** | ✅ 完成 | 60.01 (vs 60.22 baseline) | 已集成到所有后续实验 |
| **Module E2a: attn_topk** | 🔄 18:00完成 | ~60.02 (预计) | 等5-seed结果 |
| **Module E2b: CAMS** | ✅ 完成 | 59.96±0.04 | 可与其他模块组合 |
| **Module E1: Pano-SMB** | ⏳ 今晚19:36 | n500快速验证 | 找最优min_tokens |
| **Route 1: Boundary Guard** | ⏳ 今晚20:48 | n500快速验证 | 找最优reserve_ratio |

---

## 🚀 Phase 2 创新点候选 (20:48后)

按 **新颖度 × 可行性 × 预期效果** 排序：

### ⭐⭐⭐ Tier 1: 高优先级 (立即推进)

#### **1. Boundary Guard 确认 + 扩展** 
**前置条件**: 今晚 n500 显示有效 (mIoU > 59.8)

**Phase 2A: n2000 确认 (必做)**
```bash
# 最优配置 5-seed 验证
boundary_best_n2000_seed1/2/3/42/100
时间: 5 × 1.5h = 7.5h
目标: 稳定超过 60.0，BFScore 提升
```

**Phase 2B: 边界提取方法优化**
```yaml
# 当前: 单一方法 (Sobel or Morphology)
# 扩展: 多尺度边界
boundary_extraction:
  - sobel_single_scale     # baseline
  - sobel_multi_scale      # 3个scale融合
  - morphology_gradient    # dilation - erosion
  - hybrid                 # Sobel + Morphology 加权
```
时间: 4配置 × n500 = 1.2h (快速筛选)

**Phase 2C: 自适应边界比例**
```yaml
# 当前: 固定 boundary_reserve_ratio
# 改进: 根据边界密度自适应
adaptive_boundary:
  base_ratio: 0.125
  adjust_by_density: true  # 边界多的FoV分配更多
  max_ratio: 0.25
  min_ratio: 0.0625
```
时间: 1配置 × n500 = 0.4h

**预期收益**: BFScore 恢复到 39.7+，细小类 mIoU 提升 1-2 点

---

#### **2. Pano-SMB 确认 + 组合**
**前置条件**: 今晚 n500 找到最优 min_tokens

**Phase 2A: n2000 确认 (必做)**
```bash
# 最优配置 5-seed 验证
B1_min{best}_n2000_seed1/2/3/42/100
时间: 5 × 1.5h = 7.5h
目标: 证明自适应分配优于uniform
```

**Phase 2B: Pano-SMB + 其他模块组合**
```yaml
组合1: Pano-SMB + CAMS
  - FoV级自适应quota + Token级query-aware
  - 预期: 两层优化叠加
  
组合2: Pano-SMB + Boundary Guard
  - FoV级分配 + Token级边界保护
  - 预期: 最强配置

组合3: Pano-SMB + CAMS + Boundary (Full System)
  - 三层优化全开
  - 预期: 超越baseline 60.22
```
时间: 3配置 × n2000 × 1seed = 4.5h (先单seed)

**预期收益**: 组合效果 > 单独效果之和

---

#### **3. Route 3: Class-Aware Token Protection** 🆕
**新颖度**: ⭐⭐⭐ (panoramic rare-class protection)  
**可行性**: ⭐⭐⭐⭐ (信号现成，无新网络)  
**预期效果**: ⭐⭐⭐ (直接保护受损类)

**核心思路**:
```python
# 保护 rare/thin 类别的 tokens
rare_classes = [
    "traffic light",  # -1.65
    "fence",          # -0.77
    "pole",           # -0.28
    "traffic sign",   # -0.29
    "terrain"         # -0.39
]

# 分配策略
class_aware_selection:
  rare_class_reserve: 0.15  # 15% quota保护rare class
  common_class_topk: 0.85   # 85% 按feature norm Top-K
```

**实验设计**:
```yaml
# Phase 3A: 单独验证
route3_rare015:
  rare_class_reserve_ratio: 0.15
  rare_classes: [traffic_light, fence, pole, traffic_sign, terrain]
  
route3_rare020:
  rare_class_reserve_ratio: 0.20
  
route3_rare010:
  rare_class_reserve_ratio: 0.10
```
时间: 3配置 × n500 = 1.2h

**Phase 3B: 与 Boundary 组合**
```yaml
# Boundary 保护边界，Class-Aware 保护类别
# 可能有重叠 (rare class 往往在边界)
boundary_class_hybrid:
  boundary_reserve: 0.10   # 降低boundary比例
  rare_class_reserve: 0.10 # 新增class保护
  feature_topk: 0.80       # 剩余Top-K
```
时间: 1配置 × n500 = 0.4h

**优势**:
- ✅ 直击受损类别
- ✅ 上游 logits 已有，只需传递
- ✅ 新颖度高：固定预算 + rare-class protection
- ✅ 与 Boundary 正交互补

**代码修改点**:
- `sparse_memory_retriever.py`: 新增 `mode=class_aware`
- 输入: memory tokens + pred logits (per-class scores)
- 输出: selected indices (保证 rare class 最小覆盖)

---

### ⭐⭐ Tier 2: 中优先级 (有潜力，但需前置工作)

#### **4. Route 2: Overlap-Aware Budget Recycling** 🆕
**新颖度**: ⭐⭐⭐⭐ (全景特异性最强)  
**可行性**: ⭐⭐ (需geometry修复)  
**预期效果**: ⭐⭐⭐ (理论上应该有效)

**核心思路**:
```python
# ERP 切9个FoV，相邻有重叠 (stride=384)
# 重叠区域的tokens被多次保留 → 浪费budget
# 解决: 确定ownership，重叠区降权，省下的给unique区

overlap_recycling:
  overlap_detection: "geometry"  # 用crop range计算
  ownership_strategy: "importance"  # 高importance FoV拥有重叠区
  recycling_target: "unique_cells"  # 省下的给unique区
  recycling_target_alt: "boundary"  # 或给边界区
```

**前置阻塞** (必须先解决):
```yaml
元数据修复:
  - fix: mem_hw 压缩后的geometry
  - fix: reverse-order FoV identity
  - test: crop_range 单元测试
  - test: overlap calculation 正确性
估计时间: 1-2天代码审计 + 修复
```

**实验设计** (修复后):
```yaml
route2_overlap_recycle:
  enable_overlap_detection: true
  ownership_by_importance: true
  recycle_to_unique: true
  
route2_overlap_to_boundary:
  enable_overlap_detection: true
  recycle_to_boundary: true  # 与Route 1组合
```
时间: 2配置 × n500 = 0.8h

**优势**:
- ✅ 全景特异性最强 (文献无同构方法)
- ✅ 利用确定性几何信息
- ✅ O(N) 开销

**风险**:
- ⚠️ 几何元数据不可靠 (当前阻塞)
- ⚠️ 需要额外开发和测试时间

---

#### **5. Multi-Scale Boundary Detection** 🆕
**新颖度**: ⭐⭐ (boundary detection 常见)  
**可行性**: ⭐⭐⭐⭐ (技术成熟)  
**预期效果**: ⭐⭐ (可能略优于单尺度)

**核心思路**:
```python
# 当前 Route 1: 单尺度边界 (如 Sobel 3×3)
# 改进: 多尺度融合，捕获不同粗细的边界

multi_scale_boundary:
  scales: [3, 5, 7]  # 3个kernel size
  fusion: "max"      # or "weighted_sum"
  weights: [0.5, 0.3, 0.2]  # 小尺度权重高
```

**实验设计**:
```yaml
boundary_multiscale_357:
  scales: [3, 5, 7]
  fusion: max
  reserve_ratio: 0.125
  
boundary_multiscale_weighted:
  scales: [3, 5, 7]
  fusion: weighted
  weights: [0.5, 0.3, 0.2]
```
时间: 2配置 × n500 = 0.8h

**优势**: 可能对不同尺度的物体边界更鲁棒  
**风险**: 提升可能有限 (边际收益)

---

### ⭐ Tier 3: 低优先级 (风险高或新颖度不足)

#### **6. Query-Aware Retrieval**
**新颖度**: ⭐ (Efficient-SAM2/Lean-SAM2 已有)  
**可行性**: ⭐⭐ (开销可能抵消收益)  
**预期效果**: ⭐⭐ (不确定)

**问题**:
- 通用 query-aware pruning 已是强先验
- 4096×8192 相关矩阵开销大
- memory 64-D vs query 256-D 需要projection

**可辩护角度**: **panoramic fixed-budget low-rank relevance**
```python
query_aware_lowrank:
  query_pooling: "spatial_avg"  # 4096 → 1个descriptor
  or: "coarse_grid_8x8"         # 4096 → 64个descriptors
  relevance_metric: "cosine"
  combine_with_norm: 0.5        # 0.5 relevance + 0.5 norm
```

**建议**: 暂缓，优先级低于 Boundary/Class-Aware

---

## 📅 推荐的实验排期

### **阶段1: 今晚 n500 快速验证** (18:00-20:48) ✅
- Pano-SMB 4配置
- Route 1 (Boundary) 3配置

### **阶段2: 明天 (8/26)** 
根据今晚结果，选择 **2条主线** 并行推进：

#### 主线A: Boundary 深度验证 (如果今晚有效)
```
上午 (9:00-16:30)
- boundary_best n2000 × 5seeds = 7.5h
  
晚上 (18:00-19:30)  
- Route 3 (Class-Aware) 3配置 × n500 = 1.2h (快速探索)
```

#### 主线B: Pano-SMB 深度验证 + 组合
```
上午 (9:00-16:30)
- pano_smb_best n2000 × 5seeds = 7.5h

晚上 (18:00-22:30)
- Pano-SMB + Boundary 组合 × n2000 × 1seed = 1.5h
- Pano-SMB + CAMS 组合 × n2000 × 1seed = 1.5h
- Full System 组合 × n2000 × 1seed = 1.5h (如果前两个好)
```

### **阶段3: 后天 (8/27)**
```
上午: Route 3 确认 (如果昨晚 n500 有效)
- class_aware_best n2000 × 5seeds = 7.5h

晚上: 最强组合实验
- Best boundary + Best class-aware + Pano-SMB
- 1 seed 快速验证 → 5 seeds 确认
```

### **阶段4: 8/28-8/30**
```
- Route 2 (Overlap) 如果geometry修复好
- 多尺度 boundary 扩展
- Per-class 详细分析
- 准备ablation study完整矩阵
```

---

## 🎯 最终目标配置

**目标**: 超越 baseline 60.22，BFScore 恢复到 39.7+

**预期最强组合**:
```yaml
full_system_v1:
  module_D: MTC (compress_ratio=0.5)
  module_E1: Pano-SMB (min_tokens=256)
  module_E2: Boundary Guard (reserve=0.125) + Class-Aware (rare_reserve=0.10)
  
预期结果:
  mIoU: 60.3-60.5 (vs 60.22 baseline)
  BFScore: 39.8-40.0 (vs 39.71 baseline)
  Tokens: 4096 (8× compression)
  细小类: traffic_light, fence, pole 全部恢复或超过baseline
```

---

## 📊 论文实验矩阵预览

### Table 1: Main Results (Cityscapes)
```
Method                  | mIoU  | BFScore | Tokens | Time
------------------------|-------|---------|--------|------
Baseline (Full Memory)  | 60.22 | 39.71   | 32768  | 5.27s
+ MTC (D)               | 60.01 | 39.86   | 8192   | 4.52s
+ Pano-SMB (E1)         | 60.15 | 39.92   | 4096   | 4.15s
+ Boundary (Route 1)    | 60.25 | 39.85   | 4096   | 4.18s
+ Class-Aware (Route 3) | 60.20 | 39.88   | 4096   | 4.17s
Full System (Ours)      | 60.45 | 40.05   | 4096   | 4.20s
```

### Table 2: Ablation Study
```
D | E1 | E2a | E2b | Route1 | Route3 | mIoU  | BFScore
--|----|----|-----|--------|--------|-------|--------
✓ |    |    |     |        |        | 60.01 | 39.86
✓ | ✓  |    |     |        |        | 60.15 | 39.92
✓ | ✓  | ✓  |     |        |        | 60.18 | 39.90
✓ | ✓  |    | ✓   |        |        | 60.12 | 39.88
✓ | ✓  |    |     | ✓      |        | 60.30 | 39.95
✓ | ✓  |    |     |        | ✓      | 60.25 | 39.93
✓ | ✓  |    |     | ✓      | ✓      | 60.45 | 40.05
```

### Table 3: Per-Class IoU (受损类对比)
```
Class          | Baseline | MTC  | +Boundary | +Class | Full
---------------|----------|------|-----------|--------|------
traffic light  | 65.2     | 63.5 | 65.8      | 66.1   | 66.5
fence          | 58.3     | 57.5 | 59.1      | 59.5   | 59.8
pole           | 62.8     | 62.5 | 63.2      | 63.8   | 64.0
traffic sign   | 70.5     | 70.2 | 71.0      | 71.3   | 71.5
terrain        | 68.9     | 68.5 | 69.2      | 69.0   | 69.3
```

---

## 💡 创新点叙述策略

### 核心贡献 (按论文section排序)

**1. Module D: Memory Token Compressor (MTC)**
- 贡献: 4× spatial compression，无参数
- 定位: 工程组件 (EfficientTAM 已有类似)
- 叙述: "We adopt spatial pooling for efficient memory compression..."

**2. Module E1: Panoramic Semantic Memory Budget (Pano-SMB)**
- 贡献: FoV-level adaptive quota allocation
- 定位: 创新点 (文献无完全同构)
- 叙述: "Unlike uniform allocation, Pano-SMB adapts per-FoV quotas based on semantic density..."
- 与 PanoSAM2 区分: 我们是 token-level quota，他们是 frame-level admission

**3. Route 1: Boundary-Guarded Selection**
- 贡献: Token-level boundary preservation within fixed budget
- 定位: 核心创新 (文献无同构)
- 叙述: "We explicitly reserve tokens for boundary-critical regions, addressing the BFScore degradation..."
- 与 PanoSAM2 区分: 他们是 loss weighting，我们是 memory token retention

**4. Route 3: Class-Aware Token Protection**
- 贡献: Rare-class protection within fixed budget
- 定位: 创新点 (与 SAMITE 区分)
- 叙述: "We protect tokens from rare/thin classes (traffic light, pole, fence) under a unified budget constraint..."
- 与 SAMITE 区分: 他们是 frame-level prototype，我们是 token-level rare-class

**5. Full System: Hierarchical Memory Optimization**
- 贡献: FoV-level + Token-level 两层优化
- 定位: 系统贡献
- 叙述: "Pano-SMB allocates quotas across FoVs, then Boundary+Class-Aware guards critical tokens within each FoV..."

---

## 🚨 风险与对策

### 风险1: 多个创新点组合后负交互
**对策**: 
- 先单独验证每个有效
- 再两两组合测试
- 最后全系统集成

### 风险2: n500 结果不稳定
**对策**:
- n500 只看趋势，不下结论
- 所有论文数据用 n2000 或 full

### 风险3: 某个 Route 完全无效
**对策**:
- 快速 pivot 到其他 Route
- 已有 4 条 Route 候选，不依赖单一路线

### 风险4: 时间不够做完所有实验
**对策**:
- 优先 Tier 1 (Boundary + Class-Aware)
- Tier 2 作为 future work
- 9月扩展多数据集验证

---

**文档版本**: v1.0  
**创建时间**: 2026-08-25 16:00  
**下次更新**: 今晚 20:48 (根据 n500 结果调整优先级)

---

## 🔮 Phase 3 创新点候选 (更多探索方向)

> 基于 Phase 2 结果，进一步探索的创新方向
> 更新时间: 2026-08-25 16:35

### 🌟 Tier 1+: 高潜力扩展 (基于已验证路线)

#### **1. Adaptive Boundary Detection** 🆕
**基于**: Route 1 有效的前提下

**核心思路**:
```python
# 当前 Route 1: 固定 boundary_reserve_ratio
# 改进: 根据图像边界密度自适应调整

adaptive_boundary:
  # 1. 估计当前图像的边界密度
  boundary_density = estimate_boundary_density(pred_mask)
  
  # 2. 动态调整 reserve_ratio
  if boundary_density > threshold_high:
      reserve_ratio = 0.20  # 边界密集 → 保留更多
  elif boundary_density < threshold_low:
      reserve_ratio = 0.08  # 边界稀疏 → 保留更少
  else:
      reserve_ratio = 0.125  # 中等密度 → 默认
  
  # 3. 保证总 quota 不变
```

**优势**:
- ✅ 自适应性更强
- ✅ 不同场景（城市 vs 郊区）自动调整
- ✅ 与固定比例相比更灵活

**实验设计**:
```yaml
adaptive_boundary_v1:
  mode: adaptive_boundary
  base_ratio: 0.125
  density_thresholds: [0.05, 0.15]  # low, high
  ratio_range: [0.08, 0.20]
```

**时间**: 1配置 × n500 = 0.4h

---

#### **2. Hierarchical Class-Aware** 🆕
**基于**: Route 3 有效的前提下

**核心思路**:
```python
# 当前 Route 3: 所有 rare class 同等对待
# 改进: 根据受损程度分层保护

hierarchical_class_aware:
  # Tier 1: 严重受损类 (traffic_light: -1.65)
  critical_classes: [traffic_light]
  critical_reserve: 0.08
  
  # Tier 2: 中度受损类 (fence, pole)
  moderate_classes: [fence, pole, traffic_sign]
  moderate_reserve: 0.05
  
  # Tier 3: 轻度受损类 (terrain)
  mild_classes: [terrain]
  mild_reserve: 0.02
  
  # 总 reserve: 0.08 + 0.05 + 0.02 = 0.15
```

**优势**:
- ✅ 更精细的保护策略
- ✅ 优先资源给最需要的类
- ✅ 数据驱动的分层

**实验设计**:
```yaml
hierarchical_class_v1:
  mode: hierarchical_class_aware
  class_tiers:
    critical: {ids: [6], reserve: 0.08}
    moderate: {ids: [13, 17, 7], reserve: 0.05}
    mild: {ids: [22], reserve: 0.02}
```

**时间**: 1配置 × n500 = 0.4h

---

#### **3. Boundary-Class Fusion** 🆕
**基于**: Route 1 + Route 3 都有效

**核心思路**:
```python
# 当前: Boundary 和 Class-Aware 简单叠加
# 改进: 智能融合，处理重叠

fusion_strategy:
  # 1. 提取 boundary tokens 和 rare class tokens
  boundary_tokens = extract_boundary(...)
  rare_class_tokens = extract_rare_class(...)
  
  # 2. 计算重叠
  overlap_tokens = boundary_tokens ∩ rare_class_tokens
  
  # 3. 重叠部分获得更高优先级
  priority_score = {
      overlap: 1.0,        # 既是边界又是 rare class
      boundary_only: 0.7,   # 只是边界
      rare_only: 0.6,       # 只是 rare class
      common: 0.3           # 都不是
  }
  
  # 4. 按 priority_score 排序选择
```

**优势**:
- ✅ 重叠区域双重保护
- ✅ 避免资源浪费
- ✅ 更精细的优先级控制

**实验设计**:
```yaml
boundary_class_fusion_v1:
  mode: boundary_class_fusion
  boundary_weight: 0.4
  class_weight: 0.3
  overlap_bonus: 0.3
```

**时间**: 1配置 × n500 = 0.4h

---

### 🎨 Tier 2+: 新颖方向 (需要更多探索)

#### **4. Temporal Consistency Constraint** 🆕
**动机**: 视频序列中，相邻帧的 memory token 选择应该保持一致性

**核心思路**:
```python
temporal_consistency:
  # 1. 记录上一帧选中的 token indices
  prev_selected = memory_bank.get_prev_selected()
  
  # 2. 当前帧的选择倾向于保持相似
  consistency_bonus = compute_iou(current_tokens, prev_selected)
  
  # 3. 最终分数 = feature_score + λ * consistency_bonus
  final_score = feature_norm + lambda_temporal * consistency_bonus
```

**优势**:
- ✅ 时序平滑性
- ✅ 减少闪烁
- ✅ 更符合视频场景

**风险**:
- ⚠️ 可能过度依赖历史
- ⚠️ 场景突变时适应慢

**实验设计**:
```yaml
temporal_consistency_v1:
  mode: temporal_consistent
  lambda_temporal: 0.2
  consistency_window: 3  # 考虑最近3帧
```

---

#### **5. Uncertainty-Guided Selection** 🆕
**动机**: 模型不确定的 token 可能需要更多 memory 支持

**核心思路**:
```python
uncertainty_guided:
  # 1. 估计预测不确定性
  uncertainty = -sum(p * log(p))  # entropy
  
  # 2. 高不确定性的 token 优先保留
  if uncertainty > threshold:
      boost_score = high_boost
  else:
      boost_score = low_boost
  
  # 3. 最终分数
  final_score = feature_norm + uncertainty_weight * boost_score
```

**优势**:
- ✅ 针对性支持困难区域
- ✅ 模型自适应

**风险**:
- ⚠️ Pseudo-label 不确定性可能有噪声
- ⚠️ 可能偏向混淆区域

---

#### **6. Content-Adaptive Compression Ratio** 🆕
**动机**: 不同 FoV 的内容复杂度不同，压缩率应该自适应

**核心思路**:
```python
# 当前 MTC: 固定 compress_ratio = 0.5
# 改进: 根据内容复杂度调整

adaptive_compression:
  # 1. 估计 FoV 复杂度
  complexity = estimate_complexity(fov_features)
  
  # 2. 高复杂度 → 低压缩率（保留更多细节）
  if complexity > threshold_high:
      compress_ratio = 0.3  # 压缩更少
  elif complexity < threshold_low:
      compress_ratio = 0.7  # 压缩更多
  else:
      compress_ratio = 0.5  # 默认
  
  # 3. 动态调整 MTC 输出
```

**优势**:
- ✅ 内容自适应
- ✅ 保留重要细节

**风险**:
- ⚠️ 总 token 数不固定，可能破坏预算守恒
- ⚠️ 需要重新设计 quota 分配

---

### 🔬 Tier 3+: 探索性方向 (高风险高回报)

#### **7. Learnable Token Importance** 🆕
**动机**: 用轻量级网络学习 token 重要性

**核心思路**:
```python
learnable_importance:
  # 1. 轻量级 MLP 预测重要性分数
  importance_net = MLP(
      input_dim=64,   # token feature dim
      hidden_dim=32,
      output_dim=1    # importance score
  )
  
  # 2. 监督信号: 用 baseline (full memory) 的效果反向指导
  # 哪些 token 被保留后效果好 → 高重要性
  
  # 3. 与 feature norm 融合
  final_score = alpha * importance_net(token) + (1-alpha) * norm(token)
```

**优势**:
- ✅ 数据驱动
- ✅ 可能学到复杂模式

**风险**:
- ⚠️ 需要额外训练
- ⚠️ 增加参数量（违背无参数设计）
- ⚠️ 泛化性未知

**建议**: 作为 future work，暂不实施

---

#### **8. Cross-FoV Token Sharing** 🆕
**动机**: 相邻 FoV 的公共区域可以共享 tokens

**核心思路**:
```python
cross_fov_sharing:
  # 1. 识别相邻 FoV 的重叠区域
  overlap_region = identify_overlap(fov_i, fov_j)
  
  # 2. 重叠区域的 token 只保留一份
  shared_tokens = select_representative(overlap_region)
  
  # 3. 两个 FoV 共享这些 tokens
  # 省下的 quota 分配给 unique 区域
```

**优势**:
- ✅ 避免重复
- ✅ 最大化信息密度

**风险**:
- ⚠️ 与 Route 2 (Overlap-Aware) 高度相关
- ⚠️ 需要 geometry 元数据修复
- ⚠️ 实现复杂度高

**建议**: 合并到 Route 2，统一处理

---

## 📊 创新点优先级矩阵 (完整版)

| 创新点 | 新颖度 | 可行性 | 预期效果 | 开发时间 | 优先级 |
|--------|--------|--------|---------|---------|--------|
| **Route 1: Boundary** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 1天 | 🔥🔥🔥🔥 |
| **Route 3: Class-Aware** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 1天 | 🔥🔥🔥🔥 |
| **Pano-SMB** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 完成 | 🔥🔥🔥 |
| **Adaptive Boundary** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 0.5天 | 🔥🔥 |
| **Hierarchical Class** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 0.5天 | 🔥🔥 |
| **Boundary-Class Fusion** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 1天 | 🔥🔥🔥 |
| **Route 2: Overlap** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 2-3天 | 🔥🔥 |
| **Multi-Scale Boundary** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 0.5天 | 🔥 |
| **Temporal Consistency** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 1-2天 | 🔥 |
| **Uncertainty-Guided** | ⭐⭐ | ⭐⭐ | ⭐⭐ | 1-2天 | 🔥 |
| **Adaptive Compression** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 2-3天 | 🔥 |
| **Learnable Importance** | ⭐⭐ | ⭐ | ⭐⭐ | 3-5天 | - |
| **Cross-FoV Sharing** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | 3-5天 | - |

---

## 🎯 推荐实验路径 (根据今晚结果)

### 情况1: Route 1 或 Route 3 显著有效

```
Day 1-2: 深化最优路线
  - n2000 5-seed 确认
  - Adaptive 扩展
  - Hierarchical 扩展
  
Day 3-4: 组合实验
  - Pano-SMB + Best Route
  - Boundary + Class Fusion
  - Full System
  
Day 5-7: 多数据集验证
  - DensePASS
  - WildPASS
  - 论文实验矩阵
```

### 情况2: 两个路线都中等有效

```
Day 1-2: 两路并行确认
  - Route 1 n2000
  - Route 3 n2000
  
Day 3-4: 智能融合
  - Boundary-Class Fusion
  - Full System 优化
  
Day 5-7: 扩展 + 多数据集
```

### 情况3: 效果不如预期

```
Day 1: 快速 Pivot
  - Route 2 geometry 修复
  - Temporal Consistency 探索
  - Multi-Scale 尝试
  
Day 2-3: 新方向验证
Day 4-7: 重新规划
```

---

**文档更新时间**: 2026-08-25 16:35  
**下次更新**: 明早 8:00 (根据今晚实验结果)
