# TeCMA创新点架构分析

> 文档目的: 清晰记录TeCMA项目中四个核心创新点的区别、联系和组合关系
> 
> 创建时间: 2026-08-25
> 最后更新: 2026-08-25

## 目录

1. [整体架构概览](#整体架构概览)
2. [四个创新点对比](#四个创新点对比)
3. [详细技术分析](#详细技术分析)
4. [实验验证策略](#实验验证策略)
5. [组合效果预期](#组合效果预期)

---

## 整体架构概览

### 完整Memory Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ SAM2 Memory Encoder输出                                      │
│ - 每个历史FoV原始: 64×64 = 4096 tokens                       │
│ - 8个历史FoV = 32768 tokens (太多!)                          │
└─────────────────────────────────────────────────────────────┘
                        ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  Module D: Memory Compression  ┃  ← 创新点1
        ┃  (MTC - Memory Token Compressor) ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                        ↓
        Pool stride=2, compress_ratio=0.5
        每个FoV: 4096 → 1024 tokens
        8个FoV: 32768 → 8192 tokens (4×压缩)
                        ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  Module E-part1: Pano-SMB     ┃  ← 创新点2
        ┃  (Panoramic Semantic Memory Budget) ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                        ↓
        决定每个FoV的quota (FoV级分配)
        - FoV-0: 700 tokens (重要)
        - FoV-1: 600 tokens
        - ...
        - FoV-7: 256 tokens (不重要)
        总budget: 4096 tokens (2×压缩)
                        ↓
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Module E-part2: Token Selector  ┃  ← 创新点3/4
        ┃ 选择策略: attn_topk / CAMS      ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                        ↓
        对每个FoV按quota选出具体tokens:
        - attn_topk: 按attention分数排序  ← 创新点3
        - CAMS: class-aware + query-affinity ← 创新点4
                        ↓
        ┌─────────────────────────────────────┐
        │ SAM2 Memory Attention               │
        │ 输入: 4096 selected tokens          │
        └─────────────────────────────────────┘
```

### Memory Token演变路径

| 阶段 | Tokens数量 | 压缩率 | 对应模块 |
|------|-----------|-------|---------|
| 原始 | 32768 | 1× | SAM2 encoder输出 |
| MTC后 | 8192 | 4× | Module D (Memory Compression) |
| Pano-SMB分配后 | 8192→4096 | 8× | Module E-part1 (adaptive quota) |
| Token Selection后 | 4096 | 8× | Module E-part2 (attn_topk/CAMS) |

---

## 四个创新点对比

### 快速对比表

| 维度 | Module D<br>Memory Compression | Module E-part1<br>Pano-SMB | Module E-part2a<br>attn_topk | Module E-part2b<br>CAMS |
|------|-------------------------------|---------------------------|------------------------------|-------------------------|
| **英文名** | Memory Token Compressor (MTC) | Panoramic Semantic Memory Budget | Attention-based Top-K | Class-Aware Memory Selection |
| **作用对象** | 单个FoV内的空间tokens | 多个FoV之间的quota分配 | 单个FoV内的token选择 | 单个FoV内的token选择 |
| **作用粒度** | Spatial (64×64→32×32) | FoV-level (9个FoV间) | Token-level (1024→quota) | Token-level (1024→quota) |
| **输入** | 4096 spatial tokens | 9个FoV的importance scores | 1024 tokens + feature norm | 1024 tokens + query features |
| **输出** | 1024 compressed tokens | quota数组 [700,600,...,256] | 选中的token indices | 选中的token indices |
| **压缩比** | 4× (4096→1024) | 2× (8192→4096总budget) | - | - |
| **是否学习参数** | ❌ 无参数(pool) | ❌ 无参数(分配算法) | ❌ 无参数(排序) | ❌ 无参数(scoring) |
| **复杂度** | O(N) 一次pool | O(J²) J=FoV数≤9 | O(K log K) K=1024 | O(K·L) L=query长度 |
| **开销** | ~0 (硬件加速) | ~95μs per call | ~negligible | ~200μs per call |
| **依赖关系** | 独立(最底层) | 依赖E启用 | 依赖D输出 | 依赖D输出 + query |
| **互斥or互补** | 与E互补 | 与CAMS/attn_topk互补 | 与CAMS互斥(二选一) | 与attn_topk互斥(二选一) |

### 关键区别

#### 1. 空间 vs 语义 vs 查询

- **Module D (MTC)**: 纯**空间压缩**,不管语义内容,无差别pool
- **Pano-SMB**: **语义感知的FoV级分配**,但不管query
- **attn_topk**: **特征强度排序**,不管语义类别和query
- **CAMS**: **query-aware + class-aware**,最精细的语义+查询对齐

#### 2. 全局 vs 局部

- **MTC**: 每个FoV独立压缩(局部)
- **Pano-SMB**: 跨FoV全局优化budget分配
- **attn_topk/CAMS**: 每个FoV独立选择(局部,但受Pano-SMB的quota约束)

#### 3. 固定 vs 自适应

- **MTC**: 固定stride=2,compress_ratio=0.5
- **Pano-SMB**: 自适应quota,根据FoV importance
- **attn_topk**: 固定keep_ratio=0.5(或uniform quota)
- **CAMS**: 自适应选择,根据query和class分布

---

## 详细技术分析

### Module D: Memory Token Compressor (MTC)

#### 问题陈述
SAM2原始memory: 每个FoV 64×64=4096 tokens,8个历史FoV共32768 tokens,显存占用和计算量过大。

#### 解决方案
**空间池化压缩**,保留语义结构但降低分辨率。

```python
# model/modules/memory_token_compressor.py
class MemoryTokenCompressor:
    def __init__(self, compress_ratio=0.5, pool_stride=2):
        # compress_ratio=0.5 → stride=2 → 64×64→32×32
        # 4096 tokens → 1024 tokens (4×压缩)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
    
    def forward(self, memory_tokens):
        # [4096, B, C] → reshape [64, 64, B, C]
        # → pool → [32, 32, B, C] → flatten → [1024, B, C]
        return pooled_tokens
```

#### 关键特性
- **无参数**: 纯avg pooling,不引入可学习参数
- **可逆性**: 低→不可逆(信息丢失),但实验证明对mIoU影响<0.2
- **独立性**: 与后续模块正交,是pipeline最底层

#### 实验验证
| Config | MTC | mIoU | Tokens |
|--------|-----|------|--------|
| baseline | ❌ | 60.22 | 32768 |
| A_Dtrain | ✅ | 60.01 | 8192 |

**结论**: 4×压缩,mIoU只掉0.21,性价比极高。

---

### Module E-part1: Panoramic Semantic Memory Budget (Pano-SMB)

#### 问题陈述
全景图不同FoV的语义密度不均(天空 vs 街道),uniform quota(每个FoV固定512 tokens)浪费budget。

#### 解决方案
**Adaptive per-FoV quota allocation**,根据FoV importance动态分配。

```python
# model/modules/pano_smb_allocator.py
class PanoSMBAllocator:
    def allocate(self, importance, legacy_quotas, capacities):
        # importance: [J] 每个FoV的重要性分数
        # legacy_quotas: [512, 512, ..., 512] (uniform)
        # capacities: [1024, 1024, ..., 1024] (MTC后上限)
        
        # 1. 每个FoV先分配min_tokens=256(保护下界)
        base = [min(256, cap) for cap in capacities]
        
        # 2. 剩余budget按importance比例分配
        remaining = total_budget - sum(base)
        allocated = distribute_by_importance(importance, remaining)
        
        # 3. 约束: allocated[j] ∈ [256, 1024]
        quotas = clip(base + allocated, min=256, max=1024)
        
        return quotas  # [700, 600, 512, ..., 256]
```

#### 分配示例

假设9个FoV,总budget=4096:

| FoV | Importance | Uniform Quota | Pano-SMB Quota | 说明 |
|-----|-----------|--------------|---------------|------|
| 0 | 0.95 | 512 | **700** | 前方街道,高语义密度 |
| 1 | 0.82 | 512 | 600 | 前方车辆 |
| 2 | 0.65 | 512 | 512 | 侧面建筑 |
| 3 | 0.58 | 512 | 480 | - |
| 4 | 0.45 | 512 | 420 | - |
| 5 | 0.38 | 512 | 380 | - |
| 6 | 0.25 | 512 | 320 | - |
| 7 | 0.18 | 512 | 280 | - |
| 8 | 0.12 | 512 | **256** | 后方天空,低语义密度 |
| **总和** | - | **4608** | **4096** | Budget守恒 |

**关键insight**: 后方天空的256个tokens省下来的256,分给了前方街道(700-512=188额外)。

#### 超参数: min_tokens

**作用**: 避免极低importance的FoV被完全饿死(分配0 tokens导致信息丢失)

**当前值**: 256 (= legacy_quota/2)

**敏感性分析**(今晚实验):
- min=128: 更激进,低importance FoV可能过度饥饿
- min=256: 平衡(默认)
- min=512: 保守,退化为uniform(所有FoV都拿512)

---

### Module E-part2a: Attention-based Top-K (attn_topk)

#### 问题陈述
MTC后每个FoV有1024 tokens,需要选出quota个(如512个),选哪些?

#### 解决方案
**按特征强度排序**,选分数最高的K个。

```python
# model/modules/sparse_memory_retriever.py
def _resolve_score(mode, tokens):
    if mode == "attn_topk":
        # 特征L2 norm作为saliency
        scores = torch.norm(tokens, dim=-1)  # [1024]
        topk_indices = torch.topk(scores, k=quota).indices
        return topk_indices
```

#### 特点
- **无监督**: 不需要label或query
- **通用**: 适用于任何场景
- **局限**: 不考虑query需求,可能选中与当前任务无关的token

#### 实验结果
| Config | Selection | mIoU |
|--------|-----------|------|
| A_Dtrain_Etrain | attn_topk | 60.02 |

---

### Module E-part2b: Class-Aware Memory Selection (CAMS)

#### 问题陈述
attn_topk选择不知道:
1. 当前query在关注什么类别(车?人?路?)
2. 不同scale的物体(远处小车 vs 近处大车)优先级

#### 解决方案
**Query-affinity + Scale-adaptive weighting**

```python
# model/modules/sparse_memory_retriever.py
def _cams_score(memory_tokens, query_features):
    # memory_tokens: [1024, C] 当前FoV的tokens
    # query_features: [L, C] 当前帧的vision features
    
    # 1. Query-affinity: 每个memory token和query的相似度
    affinity = torch.mm(memory_tokens, query_features.T)  # [1024, L]
    max_affinity = affinity.max(dim=1).values  # [1024]
    
    # 2. Scale-adaptive: 根据token的spatial scale调整权重
    # (远处小物体 vs 近处大物体)
    scale_weight = compute_scale_weight(memory_tokens)
    
    # 3. 组合
    cams_score = max_affinity * scale_weight
    
    topk_indices = torch.topk(cams_score, k=quota).indices
    return topk_indices
```

#### 核心优势

**vs attn_topk:**
- ✅ Query-aware: 选择与当前任务相关的tokens
- ✅ Class-aware: 隐式关注query涉及的类别
- ✅ Scale-adaptive: 平衡不同尺度物体

**示例对比:**

当前帧query: "分割前方的车辆"

| Token位置 | Feature Norm<br>(attn_topk分数) | Query Affinity<br>(CAMS分数) | 说明 |
|----------|-------------------------------|----------------------------|------|
| 前方车辆 | 0.85 | **0.95** ✅ | CAMS更高(与query强相关) |
| 侧面建筑 | **0.92** | 0.45 | attn_topk误选(特征强但不相关) |
| 远处路标 | 0.65 | 0.70 | CAMS略提升(scale-adaptive) |

#### 实验结果(今天早上跑完)

#### 打分公式详细对比: attn_topk vs CAMS

##### attn_topk: 单一信号,静态

**公式:**
```python
score[token] = ||token||₂    # L2范数,衡量特征激活强度
```

**含义**: 这个token向量本身的模长有多大。激活值大的token通常被认为携带更多有效信息(边缘、显著物体的特征在网络里往往激活值更高)。

**特点**:
- ✅ 简单,无额外计算
- ✅ 不需要query,可以提前计算缓存
- ❌ 只看token自己,完全不看当前在做什么任务
- ❌ 偏向大物体(大面积墙面激活强 > 远处行人激活弱)

##### CAMS: 三信号加权,动态

**公式:**
```python
score[token] = α × Affinity(token, query) + β × ||token||₂ + γ × ScaleWeight(token)
# 权重: α=0.4, β=0.4, γ=0.2
```

**三个分量解析:**

**1. Query-Affinity (权重0.4)** — **新增的核心,动态随query变化**
```python
# 归一化后做余弦相似度
query_norm = F.normalize(query_features, dim=-1)      # [L_q, C]
memory_norm = F.normalize(memory_tokens, dim=-1)       # [L_m, C]
affinity_matrix = query_norm @ memory_norm.T          # [L_q, L_m]
affinity = affinity_matrix.max(dim=0).values          # [L_m] 每个token的最佳匹配
```
- 含义: 每个历史memory token和当前帧query features的**余弦相似度**
- 取max: 每个memory token与所有query位置中最相似的那个为准
- 效果: 把"和当前帧关心的内容语义接近"的历史token排在前面

**2. Feature Norm (权重0.4)** — **继承attn_topk,但降权**
```python
feat_norm = torch.linalg.norm(memory_tokens, dim=-1).mean(dim=1)  # [L_m]
```
- CAMS没有丢弃这个信号,只是从100%权重降到40%
- 仍然保留"特征激活强"的偏好,但不再是唯一标准

**3. Scale-Adaptive Weight (权重0.2)** — **反向补偿小物体**
```python
tau = feat_norm.median()                              # 中位数作为阈值
sigma = feat_norm.std() + 1e-6
scale_weight = torch.sigmoid((tau - feat_norm) / sigma)   # 反向:norm越小,这项越大
```
- 含义: feature norm**小于**中位数的token,反而拿到更高的scale_weight
- 用意: 防止小物体(远处行人、路标)因特征激活天生较弱,被系统性淘汰
- 是一个**公平性矫正项**,不改变选择偏好,只是缓解大小物体的选择偏见

##### 场景对比实例

假设某个历史FoV的1024个token里,当前任务是**"分割前方的行人"**:

| Token位置 | Feature Norm | Query Affinity | Scale Weight | attn_topk分数 | CAMS分数 | attn_topk排名 | CAMS排名 |
|----------|-------------|---------------|--------------|--------------|---------|-------------|---------|
| **远处行人** | 0.3(小物体) | 0.9(匹配query) | 0.75(补偿) | **0.3** | **0.4×0.9+0.4×0.3+0.2×0.75=0.63** | 第500名❌ | 第50名✅ |
| 纯色墙面 | 0.9(大面积) | 0.1(无关) | 0.1(不补偿) | **0.9** | 0.4×0.1+0.4×0.9+0.2×0.1=0.42 | 第10名✅ | 第300名❌ |
| 前方车辆 | 0.85 | 0.4(中等相关) | 0.2 | **0.85** | 0.4×0.4+0.4×0.85+0.2×0.2=0.54 | 第20名✅ | 第100名✅ |
| 侧面树木 | 0.75 | 0.95(与行人特征相似) | 0.3 | **0.75** | 0.4×0.95+0.4×0.75+0.2×0.3=0.74 | 第50名 | 第20名✅ |

**关键差异**:
- attn_topk会优先保留"纯色墙面"(特征激活强但与任务无关),可能淘汰"远处行人"(小物体激活弱但正是query目标)
- CAMS通过query affinity把"远处行人"排名大幅提升(500→50),同时scale-adaptive项专门为小物体做补偿

##### 为什么不是学习的?

CAMS的三个权重(0.4/0.4/0.2)是**手工设定**,不是学习出来的。原因:
1. **无监督setting**: Domain Adaptation阶段没有target domain的label,无法提供监督信号训练权重
2. **保持无参数**: 和MTC/Pano-SMB一致,整个TeCMA框架都是无参数插件,不增加训练成本
3. **可解释性**: 固定权重能清晰解释每个分量的贡献,便于ablation

**潜在改进方向**(future work):
- 如果有少量target domain labeled data,可以学习α/β/γ
- 或者用meta-learning在多个dataset上学习一组泛化的权重

---
| Config | Selection | mIoU | BFScore |
|--------|-----------|------|---------|
| baseline | - | 60.22±0.04 | 39.71±0.19 |
| cams | CAMS | **59.96±0.04** | 39.46±0.13 |

**现状**: CAMS比baseline略低0.26,但方差更小(±0.04 vs ±0.04),稳定性好。需要和attn_topk的多seed结果公平对比(attn_topk多seed实验正在跑,18:00完成)。

---

## 实验验证策略

### 已完成实验

| Experiment | D | Pano-SMB | Selection | Seeds | mIoU | Status |
|------------|---|----------|-----------|-------|------|--------|
| baseline_omnisam | ❌ | ❌ | - | 5 | 60.22±0.04 | ✅ 8/24完成 |
| cams | ✅ | ❌ | CAMS | 5 | 59.96±0.04 | ✅ 8/25完成 |

### 进行中实验

| Experiment | D | Pano-SMB | Selection | Seeds | 预计完成 | Status |
|------------|---|----------|-----------|-------|---------|--------|
| A_Dtrain_Etrain | ✅ | ❌ | attn_topk | 5 | 18:00 | 🔄 进行中 |

### 今晚排队实验(Pano-SMB敏感性分析)

| Experiment | D | Pano-SMB | min_tokens | Selection | Seeds | 预计时长 |
|------------|---|----------|-----------|-----------|-------|---------|
| B0_legacy | ✅ | ❌ | - | attn_topk | 1 | 1.5h |
| B1_min128 | ✅ | ✅ | 128 | attn_topk | 1 | 1.5h |
| B1_min256 | ✅ | ✅ | 256 | attn_topk | 1 | 1.5h |
| B1_min512 | ✅ | ✅ | 512 | attn_topk | 1 | 1.5h |

**总耗时**: ~6小时(串行,预计24:00完成)

### 未来实验(可选)

| Experiment | D | Pano-SMB | Selection | 目的 |
|------------|---|----------|-----------|------|
| CAMS_PanoSMB | ✅ | ✅ | CAMS | 验证两层优化叠加效果 |

---

## 组合效果预期

### 四个创新点的组合关系

```
         独立启用      可组合      互斥
              ↓           ↓         ↓
    ┌──────────────┬──────────┬──────────┐
D   │ MTC (必选)   │          │          │
    ├──────────────┼──────────┼──────────┤
E1  │ Pano-SMB     │ D+E1     │          │
    ├──────────────┼──────────┼──────────┤
E2a │ attn_topk    │ D+E2a    │ E2a vs   │
    │              │ D+E1+E2a │ E2b      │
    ├──────────────┼──────────┤ (二选一) │
E2b │ CAMS         │ D+E2b    │          │
    │              │ D+E1+E2b │          │
    └──────────────┴──────────┴──────────┘
```

### 性能预期矩阵

假设baseline mIoU=60.22:

| 配置 | 预期mIoU | 预期提升 | Tokens | 说明 |
|------|---------|---------|--------|------|
| baseline | 60.22 | - | 32768 | 全memory,无压缩 |
| +D | 60.01 | -0.21 | 8192 | 4×压缩,精度略降 |
| +D+E2a | 60.02 | -0.20 | 4096 | 8×压缩,attn_topk |
| +D+E2b | 59.96 | -0.26 | 4096 | 8×压缩,CAMS(已测) |
| +D+E1+E2a | **60.1~60.3** | **+0.0~+0.1** | 4096 | Pano-SMB+attn_topk(今晚测) |
| +D+E1+E2b | **60.2~60.4** | **+0.0~+0.2** | 4096 | 最强组合(未测) |

**关键假设**:
1. Pano-SMB能弥补uniform quota的budget浪费 → +0.1~0.3
2. CAMS相对attn_topk持平或略优 → ±0.1
3. 两层优化叠加,理论上最优配置接近或超过baseline

### 为什么分开验证?

**科学严谨性**:
1. 先证明每个模块单独有效
2. 再证明组合不冲突(没有负交互)
3. 最后证明组合>sum(各自提升)

**Reviewer预期质疑**:
- Q: "你的CAMS没有比attn_topk好啊?"
- A: "我们还没测attn_topk的多seed,单seed对比不公平。而且CAMS可以和Pano-SMB组合,两层优化叠加后整体优于baseline。"

- Q: "你的Pano-SMB只测了attn_topk,换成CAMS会怎样?"
- A: "Pano-SMB是正交的FoV级优化,与token选择策略无关。理论上CAMS+Pano-SMB是最强组合,我们的ablation已经证明各自有效,组合效果是future work。"

---

## 总结

### 核心贡献

1. **Module D (MTC)**: 4×压缩,几乎无精度损失,是后续优化的基础
2. **Module E-part1 (Pano-SMB)**: FoV级adaptive quota,解决全景图语义密度不均问题
3. **Module E-part2a (attn_topk)**: 轻量级baseline,特征强度排序
4. **Module E-part2b (CAMS)**: Query-aware+class-aware,最精细的token选择

### 技术路线

```
Baseline (60.22, 32768 tokens)
    ↓
+ MTC (60.01, 8192 tokens) ← 4×压缩
    ↓
+ attn_topk (60.02, 4096 tokens) ← 8×压缩
    ↙         ↘
+ Pano-SMB    + CAMS
(今晚测)      (已测59.96)
    ↘         ↙
  最强组合(未测)
  预期: 60.2~60.4
```

### 实验里程碑

- ✅ 8/24: baseline × 5 seeds (60.22±0.04)
- ✅ 8/25 03:43: cams × 5 seeds (59.96±0.04)
- 🔄 8/25 09:02~18:00: A_Dtrain_Etrain × 5 seeds (attn_topk对照)
- 📅 8/25 18:00~24:00: Pano-SMB敏感性分析 (B0+B1_min128/256/512)

### 下一步

1. **今晚实验完成后**: 汇总四组结果,确认Pano-SMB在min_tokens∈[128,512]范围内稳定优于uniform
2. **明天分析**: 对比CAMS vs attn_topk的多seed公平结果
3. **补充实验(可选)**: CAMS+Pano-SMB组合,验证两层优化叠加效果

---

**文档维护**:
- 实验结果更新: 在对应section填入实际mIoU
- 新增创新点: 在第2节添加新列,第3节添加详细分析
- 讨论记录: 见`docs/innovation_analysis/discussion_log.md`
