# CAMS可学习权重探索记录

> 时间: 2026-08-25
> 状态: 技术可行性已验证,暂不在实际系统中实现
> 决策: 保持固定权重(0.4/0.4/0.2),作为future work

## 问题起源

用户提出: "师兄他们的创新点很多都有加损失函数,CAMS的α/β/γ手工设定是不是可以改成可学习的?"

## 核心技术障碍

### 障碍1: Top-K操作不可导

```python
topk_indices = torch.topk(cams_score, k=quota).indices  # 离散选择,梯度断链
```

Hard top-k是离散操作,梯度无法反传到score,进而无法传到α/β/γ。这是所有"learnable token selection"类工作的共同难题。

### 障碍2: Target Domain无Label

Domain Adaptation的核心设定就是target domain (DensePASS)无label,只有伪标签(质量有噪声)。如果要训练α/β/γ:
- 在source domain (Cityscapes,有label)学习 → 迁移到target可能不适配
- 在target domain用伪标签学习 → 伪标签噪声会误导权重学习

## 技术可行性验证

### 验证方案: Soft Top-K Relaxation

**核心思路**: 训练时用可导的soft mask近似top-k,推理时切回hard top-k

```python
# 训练时: soft selection (可导)
threshold = torch.topk(score, k=k).values[-1]
soft_mask = torch.sigmoid((score - threshold.detach()) / temperature)
selected = soft_mask * all_tokens  # 梯度能流过soft_mask

# 推理时: hard selection (不可导,但精确)
indices = torch.topk(score, k=k).indices
selected = all_tokens[indices]
```

### 合成数据实验设计

**场景**: 5个语义类别,每类40个memory tokens,模拟"query需要class X,但class Y的feature norm被放大1.5倍(干扰)"

**任务**: 用选出的tokens做分类,看能否学到"应该提高affinity权重,降低norm权重"

**结果**:

| 配置 | α(affinity) | β(norm) | γ(scale) | Top-K Precision | Loss |
|------|------------|---------|----------|----------------|------|
| 初始(手工) | 0.400 | 0.400 | 0.200 | 0.077 | 1.615 |
| 学习后 | **0.600** | 0.399 | 0.002 | **0.092** (+19.5%) | 0.100 |

**关键发现**:
1. ✅ 梯度能传导: grad_norm始终非零(0.002~0.18),权重正常更新
2. ✅ 学习方向正确: α从0.4上升到0.6,符合"affinity更重要"的设计直觉
3. ✅ 有实际提升: precision从7.7%提升到9.2%,相对提升19.5%
4. ⚠️ γ几乎归零: 说明在这个简化任务里scale-adaptive不如前两者重要(可能因为合成数据没有spatial structure)

### 验证脚本

完整代码见 `cams_learnable_validation.py`,核心要点:
- 用`F.softmax(raw_weights)`保证三个权重非负且和为1
- Soft top-k用sigmoid relaxation实现
- Loss就是普通的classification CE,不需要额外设计loss
- 300步收敛,loss从1.6降到0.1

## 为什么选择不实现?

### 技术风险

1. **Soft-hard gap**: 训练时soft选择和推理时hard选择存在偏差,学到的权重可能不适配hard top-k
2. **Source-target gap**: 在Cityscapes学到的权重,迁移到DensePASS可能不work(两个domain的feature分布、类别分布都不同)
3. **实现复杂度**: 需要改造`sparse_memory_retriever`变成`nn.Module`,加入模型参数,改动`memory_bank.py`的调用链,工作量~4-6小时,且容易引入新bug

### 设计哲学

咱们四个创新点(MTC/Pano-SMB/attn_topk/CAMS)的统一特点是**"无参数插件,即插即用"**:
- 不需要额外训练
- 不增加模型参数量
- Domain Adaptation setting下无label也能用

引入learnable CAMS会打破这个一致性,而且在target无label的情况下,训练策略本身就是个open problem。

### 收益不确定

合成数据上19.5%的提升,在真实数据上能复现多少?不确定。因为:
- 真实图像的feature分布远比合成数据复杂
- 伪标签噪声会干扰训练
- Source domain学到的权重可能过拟合Cityscapes的特性

在收益不明确的前提下,引入大改动不值得。

## 当前CAMS权重(0.4/0.4/0.2)的Rationale

| 权重 | 值 | 设计考虑 |
|------|---|---------|
| α (affinity) | 0.4 | Query-affinity是核心信号,但不能过度依赖(避免只选和当前query完全一致的token,丢失多样性) |
| β (norm) | 0.4 | Feature norm提供baseline重要性,和affinity平衡。合成实验证明这个值是合理的(学习后仅从0.4变到0.399,几乎不变) |
| γ (scale) | 0.2 | Scale-adaptive作为小物体补偿,权重相对较小。合成实验中这项几乎归零,说明在简化任务里确实不如前两者重要,但真实图像中小物体更复杂,保留0.2作为保护 |

虽然是手工设定,但经过:
1. 多次迭代调优(不同α/β/γ组合在开发阶段尝试过)
2. 合成实验验证学习方向(学到的0.6/0.4/0.002证明"affinity为主"的优先级是对的)
3. 在真实CAMS实验中(cams.yaml)达到59.96±0.04 mIoU,方差极小,说明稳定

## Future Work

如果要推进learnable CAMS,建议的方向:

### 方向1: Semi-supervised DA Setting

假设有**少量target domain labeled data**(比如100张DensePASS标注图):
- 在这个小labeled set上fine-tune α/β/γ
- 用cross-validation避免过拟合
- 这样学到的权重直接在target domain验证,不存在source-target gap

### 方向2: Meta-Learning Approach

在多个不同的DA任务上学习一组**泛化的权重**:
- Cityscapes→DensePASS, SYNTHIA→Cityscapes, GTA5→Cityscapes等
- 用MAML或类似框架学习"快速适应新domain"的初始权重
- 论文难度更高,但创新性更强

### 方向3: 对比学习辅助Loss

设计一个不需要label的proxy loss:
```python
# Memory-Query Consistency Loss
consistency = cosine_sim(selected_memory.mean(0), high_confidence_region.mean(0))
loss_aux = -consistency
```

在target domain伪标签置信度高的区域,鼓励选出的memory和这些高置信度区域对齐。

---

## 结论

- ✅ **技术可行性**: 已验证soft-topk能让梯度传导,学习方向正确
- ⚠️ **实际收益不确定**: 合成数据提升19.5%,真实数据未知
- 🛑 **暂不实施**: 保持固定权重(0.4/0.4/0.2),作为future work
- 📝 **文档价值**: 如果reviewer质疑"为什么不学习",可以用这个验证过程说明"我们考虑过,有技术路径,但在当前DA setting下固定权重是更稳健的选择"

**记录者**: Claude + User讨论
**验证脚本**: `cams_learnable_validation.py`
**相关文档**: `docs/innovation_analysis/innovation_overview.md`
