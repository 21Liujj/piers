# OmniSAM 2026 最新技术与下一阶段创新方向调研

> **Stage R0：Research-only / Literature-only / Innovation Audit**  
> 文献与代码核查截止：2026-08-17。本文只进行论文检索、代码审计与候选方向判断；未修改模型、代码、配置，未启动训练或新实验。
>
> 证据标签：
> - `CONFIRMED_BY_PAPER`：已核对论文正文、方法、图表、实验或官方代码。
> - `INFERENCE`：基于多篇论文、OmniSAM 真实代码和已有实验作出的推断。
> - `UNVERIFIED`：尚未完成足够深度的一手资料核查。
> - `NOVELTY COLLISION`：与已有工作存在实质性方法重合。

## 1. Current Research Baseline

### 1.1 真实 MemoryBank 路径

经 `model/memory_bank.py`、`memory_compressor.py`、`sparse_memory_retriever.py`、`pano_smb_allocator.py` 和相关测试核查，当前历史记忆路径为：

```text
historical FoV chunk (4096, B, 64)
  -> temporal position
  -> MTC：非学习式 2x2 average pooling，4096 -> 1024
  -> 可选 Pano-SMB：在历史 FoV 间分配 K_j
  -> SMR：每个 FoV 内 feature-norm Top-K(K_j)
  -> 拼接历史记忆
  -> SAM2 memory attention，current query = (4096, B, 256)
```

代码事实：

- 一张 4096x1024 ERP 全景图切成 9 个重叠的 1024x1024 FoV，水平 stride 为 384。
- 顺序为“读取历史 -> 预测当前 -> 写入当前”。
- 最多约 8 个历史 FoV。
- Legacy MTC+SMR 每个历史 FoV 保留 512 tokens，最大端点为 `8 x 512 = 4096`。
- Retrieval 是 **per-FoV Top-K**，不是 panorama-global Top-K。
- 当前 token ranking 和 Pano-SMB quota 均为 batch-shared。
- Pano-SMB 为每个 FoV 保底 256 tokens，再按重要性分配剩余预算，并严格保证 `sum(K_j)=512J`。

### 1.2 已测结果

完整协议：

| 方法 | Memory tokens | mIoU | BFScore | Step time | Peak memory |
|---|---:|---:|---:|---:|---:|
| Full memory | 32768 | 60.26 | 39.97 | 5.27 s/iter | 26.82 GB |
| MTC+SMR | 4096 | 60.16 | 39.86 | 4.08 s/iter | 25.34 GB |

固定 pseudo-label 控制实验：

| 方法 | Memory tokens | mIoU | BFScore | Memory-attention time |
|---|---:|---:|---:|---:|
| Full memory | 32768 | 60.26 | 39.59 | 117.48 s |
| MTC+SMR | 4096 | 60.02 | 39.23 | 51.92 s |

`INFERENCE`：当前系统已形成较强 Pareto 点：历史 tokens 减少 87.5%，完整协议 step time 下降约 22.6%，mIoU/BFScore 只下降 0.10/0.11。下一步应研究“同样 4096 tokens 下保留哪些信息”，而不是叠加重型网络。

## 2. Current Method Weaknesses

1. **MTC content-blind**：固定 2x2 平均池化不区分边界、类别、重叠区，细杆和小目标可能在 SMR 前已被抹平。
2. **SMR query-blind**：活动分数仅为 batch-mean feature norm；attention、confidence、overlap、boundary、hybrid 接口虽存在，但实际调用没有提供这些信号。
3. **Batch-shared selection**：一个 ranking/quota 服务整个 batch，可能忽略样本特有的小目标。
4. **Pano-SMB 重要性语义弱**：mean feature norm 不等于当前 query 相关性、独特全景覆盖或类别价值。
5. **损伤集中于细小/边界类**：traffic light -1.65、fence -0.77、terrain -0.39、traffic sign -0.29、pole -0.28。
6. **选择开销不可忽略**：pooling、norm、Top-K 已约增加 0.08 s/iter；memory attention 仅占控制实验总 step 的 9.76%，昂贵 query-token scoring 可能吃掉收益。
7. **几何元数据暂不可靠**：MTC 输出压缩几何后续被丢弃，reverse pass 下物理 crop identity/longitude 不稳定，阻塞可信 overlap 实验。
8. **最终效率证据仍不完整**：缺多 seed、完整 inference throughput、FLOPs/算子成本及第二数据设置。

`INFERENCE`：最有可能得到正向结果的目标不是继续压 tokens，而是在固定 4096 下恢复 BFScore 和细小类别。

## 3. Latest SAM2 Memory Efficiency Research

### Efficient-SAM2（ICLR 2026，arXiv:2602.08224）

`CONFIRMED_BY_PAPER`：官方实现对 memory-attention map 归一化并取平均，再对每帧 4096 memory tokens 采用固定比例 Top-K mask，属于 object-aware / attention-derived memory retrieval。

`NOVELTY COLLISION`：不能再声称 OmniSAM 首次提出 SAM2 sparse memory retrieval。可区分点只剩：全景 FoV 序列、严格跨 FoV 总预算以及 boundary/unique-coverage 保护。

### Lean-SAM2（arXiv:2607.19811）

`CONFIRMED_BY_PAPER`：Target-Anchored Memory Pruning 联合 memory-attention usage 与 prompt-anchor cosine similarity；Temporal Condensation 使用 insurance memory；另有 risk-aware encoder routing。

`NOVELTY COLLISION`：泛化的 query-aware、semantic-anchor-aware、long/short memory pruning 已有强先验。OmniSAM 当前 dense semantic UDA 路径没有等价 object prompt anchor，不能直接照搬。

### EfficientTAM / Efficient Track Anything（2024 preprint / CVPR 2025）

`CONFIRMED_BY_PAPER`：官方代码对 memory key/value 使用 stride-2、2x2 non-learned average pooling。

`NOVELTY COLLISION`：与 MTC 的核心操作直接重合。MTC 可作为有效工程组件，但不能单独作为算法创新。

### PanoSAM2（arXiv:2604.07901）

`CONFIRMED_BY_PAPER`：Long-Short Memory Module 只在长期库中保存采样的 object pointers，丢弃长期 dense features；按预测 occlusion score 的指数权重采样历史帧；用 long/short pointer similarity 及 FiLM scale/bias 重加权短期 dense memory。

关键实验：

- LSMM 单独使 J&F 从 59.4 提升到 61.6（+2.2）。
- 长期大小 `L=2` 最佳：64.3；无 LSMM 为 63.4；`L=3` 降至 63.7，说明过多长期信息会稀释短期信息。
- Hiera-T 完整模型 memory 约从 3.6 GB 增至 3.7 GB，FLOPs 为 728.0 G；FPS 从 SAM2 的 42.6 变为完整 PanoSAM2 的 34.8。
- 完整模型相对 SAM2 在 360VOTS 提升 +5.6 J&F，在 PanoVOS validation 提升 +6.7。

结论：PanoSAM2 是**相邻先验，不是 Pano-SMB 的直接碰撞**。它做的是帧/object-pointer admission 与短期记忆调制，不是重叠 FoV 间的 token-level quota、不保持 OmniSAM 的 `sum(K_j)=512J`，也不做 overlap budget recycling。但任何“首个 panorama-aware SAM2 memory module”表述都会 `NOVELTY COLLISION`。

## 4. Latest Dynamic Token Research

- `CONFIRMED_BY_PAPER`：ToMe（ICLR 2023）以 bipartite soft matching 和 size-weighted aggregation 合并 token，是相似度 token merging 的基础先验。
- `CONFIRMED_BY_PAPER`：Efficient-SAM2 为每个 memory frame 使用固定局部保留率，没有跨全景 FoV 重新分配一个守恒总预算。
- `CONFIRMED_BY_PAPER`：Lean-SAM2 根据 target anchor 和 risk 动态剪枝/压缩，但目标提示设定不同于 dense panoramic semantic segmentation。
- `INFERENCE`：对最多 `8 x 1024` 候选 tokens 做 all-pairs matching 不适合当前系统；匹配与 memory traffic 可能高于节省的 attention 成本。
- `INFERENCE`：OmniSAM 更适合 O(N) 或 O(JN) 的确定性几何 ownership + scalar quota，而不是通用 learned token router。
- `UNVERIFIED`：本次已核查集合中，没有发现“严格守恒全局 token budget + 重叠全景 FoV 间自适应分配 + semantic UDA”的完全同构方法；这不是全领域不存在的证明。

## 5. Latest Video / Multi-view Memory Research

### HiM2SAM（PRCV 2025，arXiv:2507.07603）

`CONFIRMED_BY_PAPER`：使用运动估计、短期/长期 memory policy，主要是 frame-level admission。CoTracker 依赖和 object-tracking 假设使其不适合作为 OmniSAM 首个实验。

### TSMS-SAM2（arXiv:2508.05829）

`CONFIRMED_BY_PAPER`：根据 feature similarity 剪除整帧 memory，建立了 redundancy-based frame pruning 先验，但不做 token-level 固定总预算分配。

### SAMITE（arXiv:2507.21732）

`CONFIRMED_BY_PAPER`：比较历史 foreground prototypes 并进行 frame-level Top-K，形成 prototype-guided frame selection 先验；不等同于固定 4096 tokens 下的 dense class-aware retention。

### 综合判断

`INFERENCE`：coarse FoV admission 技术上容易接入，但整 FoV 丢弃可能损失小而独特的类别。它只能作为带 256-token insurance floor 的低优先候选，不能替代 token-level guarded selection。

## 6. Panorama-specific Efficient Modeling

- `CONFIRMED_BY_PAPER`：PanoSAM2 通过 left-right wrap padding 建模 ERP seam，并使用 previous mask 逐步修正 distortion；PA Decoder 单独带来 +3.3 J&F，是其最大增益模块。
- `CONFIRMED_BY_PAPER`：PanoSAM2 distortion-guided loss 按 ERP distortion、foreground balance 和 boundary distance 加权。
- `CONFIRMED_BY_PAPER`：PanoVOS（ECCV 2024）与 360VOTS（2025）确立全景视频分割任务和 projection-specific 难点，但不是 fixed-budget memory 方法。
- `UNVERIFIED`：GoodSAM（2024）是 panoramic semantic segmentation/domain gap 的相关先验，但本次未重新完成其全文和官方代码深读，不能据此作细粒度 novelty 结论。
- `INFERENCE`：仅写“panorama-aware”已不足够。OmniSAM 必须把贡献绑定到九视图重叠拓扑、总容量守恒、semantic UDA，以及 thin-class boundary recovery。

## 7. Overlap / Redundancy-aware Methods

通用 redundancy removal 已拥挤：ToMe 合并相似 tokens，TSMS-SAM2 剪相似帧，HiM2SAM/SAMITE 管理历史观察。因此泛称“删除冗余记忆”会 `NOVELTY COLLISION`。

可保留的 OmniSAM-specific 角度是：

```text
已知 ERP crop geometry
  -> 确定相邻 FoV 重复覆盖的物理 longitude
  -> 对重复 cell 设单一 owner 或降权
  -> 把释放 quota 分给 unique / boundary-critical cells
  -> 保持 sum(K_j) 和最大 4096 不变
```

可行性判断：

- 所需 crop range/stride 理论上已有且确定。
- 当前 retrieval 点的 reverse-order identity 与 compressed `mem_hw` 不可靠，必须先修复并做几何单测。
- 确定性 ownership mask 近似 O(N)，不会明显吃掉 attention savings。
- 不推荐 learned correspondence 或 all-pairs cosine matching。

`INFERENCE`：这是全景特异性最强的路线，但不是第一个实验；它有明确的元数据前置阻塞。

## 8. Boundary-preserving Compression

`CONFIRMED_BY_PAPER`：PanoSAM2 的 distortion-guided loss 对 foreground 均匀加权，并让 background 权重随其到 object boundary 的距离衰减，证明边界/畸变加权对全景分割有效；但该工作是 loss weighting，不是 memory-token retention。

`INFERENCE`：OmniSAM 的真实损伤恰好集中于 traffic light、pole、fence、traffic sign、terrain 和 BFScore。上游已有预测 mask/logits，可产生 Sobel 或 morphology boundary band，无需新网络。小比例 boundary reservation 后再用 legacy Top-K 填满预算，仍保持 4096。

代码可行性：

- 可保留 `MemoryBank.forward` 输出接口和最大 token 数。
- boundary signal 上游存在，但目前没有传入活动 SMR，需要轻量 signal plumbing。
- O(N) edge lookup + 两组 Top-K 应明显低于 dense query-memory matching。
- 目标域 pseudo-boundary 有噪声，不能采用纯 boundary hard selection；应保留小比例 insurance reservation。

`UNVERIFIED`：已深读集合中未发现“固定全景总预算下显式保留 semantic-boundary SAM2 memory tokens”的完全同构方案。这是机会，不是绝对首创证明。

## 9. Query-aware Retrieval

`CONFIRMED_BY_PAPER`：Efficient-SAM2 已使用 attention-derived saliency；Lean-SAM2 已联合 attention usage 与 prompt-anchor similarity。通用 query/target-aware pruning 已有强先验。

当前代码适配：

- current FoV query 已在 `MemoryBank.forward` 中，shape 为 `(4096, B, 256)`。
- memory token 为 64-D，不能在没有兼容 projection 的情况下直接 cosine。
- 预剪枝阶段构造 `4096 x up to 8192` 完整相关矩阵相当于重复昂贵 attention，会抵消效率收益。
- 可行形式应为单个 pooled descriptor、最多 8x8 coarse query grid，或复用 SAM2 frozen projections。

`INFERENCE`：正向效果有可能，因为 norm ranking 完全 query-blind；但 novelty 和 overhead 风险均高于 boundary guarding。可辩护的表述应是 **panoramic fixed-budget low-rank relevance**，而非 generic query-aware pruning。

## 10. Semantic / Language-guided Selection

### Semantic / class-aware

`CONFIRMED_BY_PAPER`：SAMITE 使用 foreground prototype 选历史帧；Lean-SAM2 使用 target anchor。泛化 prototype-guided retrieval 并不新。

`INFERENCE`：OmniSAM 更具体的空间是，在统一全局预算内保护 rare/thin semantic-class cells。现有 logits 或 class prototypes 比新模型更合适，且结果可直接由五个受损类别判定。信号上游存在，但需要缓存/传递到 SMR。

### Language-guided

- 当前 `MemoryBank.forward` 没有 text anchor，也没有与文本空间兼容的 memory embedding。
- 冻结 text embedding 不能直接给 64-D SAM2 memory 打分，仍需视觉编码器或 learned projector。
- CLIP/VLM dense features 会引入额外计算、显存和目标域偏移风险。
- DensePASS 类名提供的信息很可能不超过现有 classifier logits/prototypes。

**结论：停止作为主路线。** `INFERENCE`：在单张 RTX 4090 和 1-2 周窗口内，language guidance 不太可能以合理成本显著胜过 class-logit guidance。只有当系统未来“免费”产生兼容 vision-language feature 时才值得作为附加消融。

## 11. UDA Training Efficiency

- `CONFIRMED_BY_MEASUREMENT`：MTC+SMR 完整协议 step time 下降 22.6%，mIoU/BFScore 变化约 -0.10/-0.11。
- `CONFIRMED_BY_MEASUREMENT`：控制实验中 memory-attention time 下降 55.84%，但 total step 只改善 3.89%，因为 memory attention 占比有限且 selection 有开销。
- `INFERENCE`：token reduction 只有在同时报告 end-to-end wall time、GPU hours、peak memory、throughput、attention-only time 和统一训练协议时，才构成可信效率证据。
- `INFERENCE`：普通 pseudo-label threshold/prototype filtering 不能继续作为核心。现有结果 Identity 60.02、Class-balanced 60.07、Prototype 60.05，杠杆过小。
- `UNVERIFIED`：本次未完成 2024-2026 全部 UDA training-efficiency 论文的系统 meta-analysis，论文不应声称“first efficient UDA”。

最终最低证据要求：主结果 3 seeds、固定 pseudo-label 对照、完整 wall time/GPU hours、allocated/reserved peak memory、train/inference throughput、token counts、FLOPs 或明确算子成本、per-class mIoU、BFScore，以及可行时的第二 transfer setting 或第二 panorama dataset。

## 12. Novelty Audit of MTC

**结论：GENERIC / KNOWN；保留为效率组件，不作为核心创新。**

- `NOVELTY COLLISION`：EfficientTAM 已执行同类 non-learned stride-2 memory pooling。
- 空间平均池化本身也远早于 SAM2。
- OmniSAM 中 MTC 的工程价值真实：它参与 32768 -> 4096 的高效路径。
- 论文应把它描述为简单有效的 compressor，而不是新压缩算法。
- 真正创新必须来自池化后如何在全景固定预算中分配和保护信息。

## 13. Novelty Audit of SMR

**结论：拒绝 broad novelty；作为 legacy lightweight baseline 保留。**

- `NOVELTY COLLISION`：Efficient-SAM2 已做基于 attention saliency 的 sparse SAM2 memory retrieval。
- `NOVELTY COLLISION`：Lean-SAM2 已做 target-anchored memory pruning。
- Feature-norm Top-K 更简单、便宜，但语义依据也更弱；“简单”是效率优势，不是更强 novelty。
- per-FoV scope 和固定端点是有价值的系统属性。
- 下一版应把 SMR 写成轻量 norm baseline，只对 panorama/boundary-specific policy 主张创新。

## 14. Novelty Audit of Pano-SMB

**结论：CONTINUE BUT REFRAME，并只允许一次生死实验。**

仍可能独特的部分：

- 在可变数量历史 FoV 之间守恒一个精确总预算；
- 先 inter-FoV quota，再 intra-FoV token selection；
- 256-token insurance floor、capacity saturation、stable largest-remainder correction；
- 不超过 legacy `512J` 和最大 4096；
- 与现有 MTC+SMR 直接兼容。

碰撞边界：

- Efficient-SAM2 使用每帧固定 retention，不共享跨帧 quota。
- Lean-SAM2、PanoSAM2 管理 long/short memory，但未实现该 inter-FoV exact-token allocation。
- PanoSAM2 已否定“首个 panorama-aware SAM2 memory”说法。
- 泛化 dynamic allocation / frame selection 不能作为宽泛 claim。

当前科学弱点：

- mean feature norm 可能产生近似均匀 quota，或放大高能量但无关区域；
- quota batch-shared；
- 尚无 accuracy 证据；
- 未显式使用 panorama overlap、distortion、boundary 或 class 信息。

**停止规则**：若多数 multi-history steps 的 `quota CV < 0.10`，或固定预算实验没有达到 `mIoU +0.15` 或 `BFScore +0.20` 任一阈值，就停止 **norm-only Pano-SMB**，不做 temperature/min-quota 扫参。只允许一次基于 boundary/unique-coverage 的 signal replacement。

## 15. Candidate Innovation Matrix

评分 1-5；Compute Cost 越高代表额外成本越低。Practical Priority 为综合判断，不是机械平均。

| ID | Candidate | Type | Novelty | Code Fit | Compute Cost | Experiment Speed | Paper Value | Practical Priority | 正向效果可能性 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A | Pano-SMB / adaptive FoV budget | I | 4 | 5 | 5 | 5 | 4 | 4.5 | 中；取决于 quota 是否真正非均匀 |
| B | Overlap deduplication + recycling | I/II | 5 | 2 | 5 | 2 | 5 | 4.0 | 中高；先修复几何元数据 |
| C | Low-cost query-aware SMR | I | 2 | 3 | 3 | 3 | 3 | 3.0 | 中；碰撞和开销风险较高 |
| D | Boundary-preserving memory | I | 4 | 3 | 4 | 4 | 5 | 5.0 | 高；直接对应 BFScore/细类损伤 |
| E | Semantic / rare-class-aware | I | 3 | 3 | 4 | 4 | 4 | 4.0 | 中高；目标类别明确 |
| F | Language-guided selection | III | 2 | 1 | 2 | 1 | 3 | 1.0 | 低；当前路径无可用信号 |
| G | Coarse FoV admission + insurance | I/II | 2 | 4 | 5 | 4 | 3 | 2.5 | 中；可能丢小型 unique objects |

只新增 Candidate G。它受 TSMS-SAM2、HiM2SAM、Lean-SAM2、PanoSAM2 启发，但先验碰撞较强，不进入主路线。

## 16. Top 3 Recommended Directions

### Route 1：Boundary- and rare-class-guarded Pano-SMB（最推荐）

- **Research question**：一个守恒全景记忆预算，能否通过 boundary/rare-class insurance 恢复 content-blind pooling 和 norm retrieval 损失的细小结构？
- **复用模块**：MTC、SMR Top-K/位置索引、Pano-SMB exact allocator、4096 endpoint。
- **新模块**：从已有 mask/logits 产生 O(N) guard；先保留 12.5% boundary tokens，剩余预算沿 legacy 路径选择。Boundary 单独 PASS 后再加 rare-class。
- **预期正向原因**：直接对应 BFScore 与 traffic-light/pole/fence/sign 的实测下降。
- **成本目标**：token 不增加，end-to-end overhead <2%。
- **Kill condition**：一次控制实验中 BFScore 未提高 0.20，且五个受损类平均 IoU 未提高 0.30。

### Route 2：Geometry-exact overlap deduplication + budget recycling（备选）

- **Research question**：能否确定性删除重复 ERP longitude 的 tokens，并把 quota 转给 unique/boundary-critical 区域？
- **复用模块**：MTC、Pano-SMB exact conservation、SMR indexing。
- **新模块**：physical crop ID/longitude、compressed-grid ownership、neighbor-only duplicate suppression、quota recycling。
- **预期正向原因**：九视图重叠产生可证明冗余，比 feature all-pairs matching 更低成本、更具全景特异性。
- **前置条件**：先修复并验证 reverse/wrap/compressed geometry。
- **Kill condition**：两天内无法通过几何单测，或重复覆盖下降后 mIoU/BFScore 均无恢复。

### Route 3：Low-rank query-aware semantic retrieval（高风险高收益）

- **Research question**：能否用 coarse current-FoV descriptor，以低于完整 attention 的成本选出比 norm Top-K 更相关的历史 tokens？
- **复用模块**：current query、可复用 frozen SAM2 projection、SMR Top-K、4096 budget。
- **新模块**：最多 64 个 pooled query descriptors，或单一 class/prototype descriptor，与 norm score 混合。
- **预期正向原因**：消除当前 retrieval 最大概念缺陷——query blindness。
- **风险**：Efficient-SAM2/Lean-SAM2 碰撞、256-D/64-D projection mismatch、overhead 抵消收益。
- **Kill condition**：一次实验未达到 mIoU +0.15/BFScore +0.20，或 step overhead >3%。

## 17. Minimal Experiments

统一约束：相同数据、pseudo-label、schedule、seed、评估代码；最大 endpoint 仍为 4096。只有单 seed 控制实验 PASS 后，才进入 3 seeds/full protocol。

### A. Pano-SMB

- **Baseline**：MTC+SMR，所有 `K_j=512`。
- **唯一变量**：启用现有 norm-based Pano-SMB，`sum(K_j)=512J`，minimum 256。
- **Metrics**：mIoU、BFScore、五个受损类、quota CV/entropy/saturation、step time、peak memory。
- **成本**：1 run，预计 <=1.01x baseline GPU-hours。
- **PASS**：mIoU +0.15 或 BFScore +0.20；受损类无一下降 >0.30；overhead <1%；多数 multi-history steps 的 quota CV >=0.10。
- **FAIL**：精度阈值均未达到，或 quota 近似均匀。
- **Stop rule**：停止 norm-only Pano-SMB；只允许一次 Route 1 signal replacement。

### B. Overlap deduplication + recycling

- **Baseline**：4096-token MTC+SMR。
- **唯一变量**：确定性 neighbor-overlap ownership；释放 slots 由非重复区域 legacy norm Top-K 填充。
- **Metrics**：mIoU、BFScore、seam/non-seam、duplicate physical-coverage ratio、step time。
- **成本**：最多 2 天几何验证 + 1 run，<=1.01x GPU-hours。
- **PASS**：重复物理覆盖减少 >=20%，且 mIoU +0.15 或 BFScore +0.20，overhead <1%。
- **FAIL**：元数据仍有歧义，或 dedup 后无精度恢复。
- **Stop rule**：synthetic crop/reverse/wrap ownership 不精确就不训练；禁止退化为 all-pairs matching。

### C. Query-aware SMR

- **Baseline**：4096-token norm Top-K。
- **唯一变量**：使用兼容 frozen projection，将 norm 与 <=64 pooled-query cosine relevance 混合。
- **Metrics**：mIoU、BFScore、ranking overlap、scoring time、total step、peak memory。
- **成本**：1 次 profile + 1 run，<=1.03x GPU-hours。
- **PASS**：mIoU +0.15 或 BFScore +0.20，总 overhead <=3%。
- **FAIL**：必须引入较大学习模块，或精度/开销阈值失败。
- **Stop rule**：不构建 dense pre-pruning query-memory matrix；失败后不尝试第二个架构版本。

### D. Boundary-preserving memory

- **Baseline**：4096-token norm Top-K。
- **唯一变量**：保留 12.5%（最大历史下 512 tokens）给现有 mask 的最强 boundary-band cells，其余由 norm 填满。
- **Metrics**：mIoU、BFScore、traffic light/pole/fence/sign/terrain、boundary-token retention、step time。
- **成本**：1 run，<=1.02x GPU-hours。
- **PASS**：BFScore +0.20，五个受损类平均 +0.30，overall mIoU 不低于 -0.05，overhead <2%。
- **FAIL**：boundary retention 上升但 BFScore/细类不升，或 noisy pseudo-boundary 使 mIoU 下降 >0.10。
- **Stop rule**：只试一个 reservation ratio；失败后停止 boundary-only，不扫比例。

### E. Semantic / rare-class-aware selection

- **Baseline**：4096-token norm Top-K。
- **唯一变量**：保留 12.5% 给现有 logits 预测为五个受损/稀有类的 cells，其余由 norm 填满。
- **Metrics**：mIoU、BFScore、per-class IoU、保留 token 数、pseudo-label precision proxy、step time。
- **成本**：1 run，<=1.02x GPU-hours。
- **PASS**：保护类平均 +0.40，且 overall mIoU +0.10 或 BFScore +0.15，overhead <2%。
- **FAIL**：保护 token 增多但 class IoU 不升。
- **Stop rule**：不调 class threshold；仅当 D 或 E 单独 PASS 时才合并。

### F. Language-guided selection

- **Baseline**：Candidate E，而不是较弱的 norm baseline。
- **唯一变量**：以 frozen vision-language similarity 替代 class-logit importance。
- **Metrics**：mIoU、BFScore、protected-class IoU、feature extraction time、peak memory。
- **成本**：至少 1 次 integration profile + 1 run，预计 1.10-1.25x GPU-hours。
- **PASS**：相对 E 提升 >=0.30 mIoU 或 >=0.30 BFScore，且 step overhead <=8%。
- **FAIL**：需要 learned alignment/dense external encoder，或收益/成本阈值失败。
- **Stop rule**：本次审计已建议在实现前停止，除非未来已有免费兼容特征。

### G. FoV admission + insurance

- **Baseline**：uniform 512/FoV 或 Pano-SMB，相同总预算。
- **唯一变量**：pooled descriptor 标记冗余 FoV，但保留 256-token insurance；其余 quota 回收。
- **Metrics**：mIoU、BFScore、按 history age 的 admission、unique-class loss、step time。
- **成本**：1 run，<=1.01x GPU-hours。
- **PASS**：mIoU +0.15，或精度不降且实际 tokens 再降 >=15%；保护类无一下降 >0.30。
- **FAIL**：只复现时间顺序，或丢失小型 unique classes。
- **Stop rule**：不训练 frame selector，不扫多种 policy。

**第一个应运行的实验：Candidate D，12.5% boundary reservation。** 它与真实损伤的因果联系最强，单次实验即可判生死。Pano-SMB quota CV 可同时仅做日志诊断，但在 D 判定前不启动第二个训练实验。

## 18. Directions Not Recommended

1. Language-guided selection：当前无兼容信号，成本高于 class logits。
2. norm-only Pano-SMB 失败后的温度/min-quota 扫参：是在优化弱 proxy。
3. full pre-pruning query-memory attention：可能抵消全部 attention savings。
4. all-pairs token merging/deduplication：匹配成本高，且忽略已知 crop geometry。
5. 无 insurance 的整 FoV hard eviction：容易丢失小型 unique objects。
6. 新 backbone、更大 SAM2、大 VLM、full distillation、重型 auxiliary network、architecture search、新生成模型：超出时间算力且冲淡 memory 主线。
7. 纯 INT8/TensorRT：可做工程补充，不能解决 novelty。
8. 继续 pseudo-label threshold/prototype filtering：现有提升过小。
9. 把大规模多数据集重训包装成方法创新：只能作为通过生死实验后的证据扩展。

## 19. Reviewer-risk Assessment

| 风险 | 等级 | 原因 | 必须补救 |
|---|---|---|---|
| Novelty | HIGH | MTC、broad SMR 有直接碰撞 | 以 guarded panoramic allocation/recycling 为核心 |
| Related-work collision | HIGH | EfficientTAM、Efficient-SAM2、Lean-SAM2、PanoSAM2 | 逐机制对照，不作宽泛 first claim |
| Panorama specificity | HIGH（当前） | norm pooling/Top-K 仍通用 | 引入 overlap geometry 或 boundary/rare-class insurance |
| Experimental scale | HIGH | 单 transfer setting、seed 不完整 | 主结果 3 seeds，尽量增加第二 setting |
| Single dataset | HIGH | DensePASS-only 限制泛化 | 增加第二 panorama benchmark/controlled transfer |
| Single seed | HIGH | 0.1 级变化可能在方差内 | 报 mean/std |
| Efficiency evidence | MEDIUM-HIGH | attention gain 与 end-to-end gain 差异大 | GPU hours、step、throughput、peak、tokens 同时报 |
| Accuracy drop | MEDIUM | 总体小，但集中于安全相关细类 | Route 1 恢复 BFScore/细类 |
| Missing inference | MEDIUM | 只报训练效率不完整 | 固定硬件 inference latency/throughput |
| Missing FLOPs | MEDIUM | token 数不能表达 Top-K/memory traffic | 报算子 FLOPs + wall time |
| Missing peak memory | LOW | 主表已有初步 peak memory | 明确 allocated/reserved 定义并重复测量 |

`INFERENCE`：如果论文只有 MTC + norm Top-K SMR + 22% training acceleration + 约 0.1 mIoU drop，CCF-C 或相近 venue 最大拒稿原因会是增量 novelty，而不是 0.1 的精度下降。速度结果是有价值的支撑证据，但不足以独立成为核心贡献。

## 20. Final Recommendation

1. **冻结 MTC 和当前 SMR 为高效 baseline**，不再主张独立算法创新。
2. **第一个实验运行 12.5% boundary reservation**，这是正向效果依据最强、判定最快的方向。
3. **Pano-SMB 继续，但定位为 exact-budget infrastructure 和 panorama-level allocator**；先确认 quota 是否真正非均匀，norm-only policy 一次失败即停。
4. **若 boundary guarding PASS，形成 Route 1**：exact shared FoV budget + boundary insurance；rare-class insurance 只作为第二消融。
5. **Route 2 先修几何再实验**：deterministic overlap ownership + recycling 是最具全景特异性的备选。
6. **Route 3 保留为高风险路线**：低秩 query-aware retrieval 有潜力，但碰撞与 overhead 风险高。
7. **停止 language-guided selection**：不适配当前信号路径和单卡时间窗口。

若 Route 1 PASS，建议论文核心命题为：

> 在不改变最大 4096-token 端点的条件下，将全景 memory 视为守恒资源：在历史 FoV 间自适应分配容量，同时为 content-blind pooling 和 norm retrieval 易丢失的 boundary/rare structures 提供记忆保险。

该表述在投稿前仍需做一次 claim-level 最终文献更新；不能声称首个 panorama-aware SAM2 memory，因为 PanoSAM2 已构成明确先验。

# Key References

| Title | Year / Venue | Paper | Code | 与本项目关系 |
|---|---|---|---|---|
| Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval | 2026 / ICLR | https://arxiv.org/abs/2602.08224 | https://github.com/jingjing0419/Efficient-SAM2 | attention-saliency SAM2 retrieval 的直接碰撞 |
| Lean-SAM2: Target-Anchored Memory and Encoder Acceleration for SAM2 | 2026 / arXiv | https://arxiv.org/abs/2607.19811 | https://github.com/DeawhaleQwQ/Lean-SAM2 | target-aware pruning、temporal condensation、insurance memory |
| PanoSAM2: Lightweight Distortion- and Memory-aware Adaptions of SAM2 for 360 Video Object Segmentation | 2026 / arXiv | https://arxiv.org/abs/2604.07901 | `UNVERIFIED`：本次未确认官方仓库 | 最近的 panorama/SAM2 memory 先验，但不是 Pano-SMB 同构方法 |
| Efficient Track Anything | 2024 preprint / CVPR 2025 | https://arxiv.org/abs/2411.18933 | https://github.com/yformer/EfficientTAM | non-learned 2x2 memory pooling 的直接先验 |
| HiM2SAM | 2025 / PRCV | https://arxiv.org/abs/2507.07603 | https://github.com/LouisFinner/HiM2SAM | short/long memory 与 frame admission 先验 |
| TSMS-SAM2 | 2025 / arXiv | https://arxiv.org/abs/2508.05829 | https://github.com/apple1986/TSMS-SAM2 | similarity-based whole-frame memory pruning |
| SAMITE: Position Prompted SAM2 with Calibrated Memory for Visual Object Tracking | 2025 / arXiv | https://arxiv.org/abs/2507.21732 | https://github.com/Sam1224/SAMITE | prototype-guided frame Top-K 先验 |
| Token Merging: Your ViT But Faster | 2023 / ICLR | https://arxiv.org/abs/2210.09461 | https://github.com/facebookresearch/ToMe | token-similarity merging 基础先验 |
| SAM 2: Segment Anything in Images and Videos | 2024 / arXiv | https://arxiv.org/abs/2408.00714 | https://github.com/facebookresearch/sam2 | 基础 memory architecture |
| PanoVOS | 2024 / ECCV | https://arxiv.org/abs/2309.12303 | `UNVERIFIED` in this audit | panoramic VOS benchmark/task 先验 |
| GoodSAM | 2024 | https://arxiv.org/abs/2403.16370 | `UNVERIFIED` in this audit | panoramic semantic adaptation 相关先验，投稿前需最终深查 |
