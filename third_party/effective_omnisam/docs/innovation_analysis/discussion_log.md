# 创新点讨论记录

> 记录关于四个创新点(Memory Compression/Pano-SMB/attn_topk/CAMS)的讨论过程、决策依据和待解决问题
>
> 创建时间: 2026-08-25

## 讨论时间线

### 2026-08-24: Baseline多seed实验

**背景**: 验证baseline_omnisam(full memory, 无压缩)的稳定性

**执行**: seed 1/2/3/42/100,串行跑完
- seed1: 60.22, seed2: 60.16, seed3: 60.19, seed42: 60.23, seed100: 60.24
- 聚合: mIoU=60.22±0.04, BFScore=39.71±0.19

**结论**: baseline在不同随机种子下高度稳定,方差极小,是可靠的对照组。

---

### 2026-08-25 上午: CAMS创新点讨论

**问题1**: "Pano-SMB和CAMS有什么区别?"

**分析过程**:
1. 先确认CAMS的代码位置: `model/modules/sparse_memory_retriever.py:64` `_cams_score`
2. 确认Pano-SMB的代码位置: `model/modules/pano_smb_allocator.py`
3. 检查两者在`memory_bank.py`中的调用顺序(行525-585)
4. 发现: Pano-SMB先决定每个FoV的quota,再调用sparse_retriever(CAMS/attn_topk)对每个FoV选具体token

**结论**: 两者是**正交的两层机制**
- Pano-SMB: FoV粒度(决定"多少个")
- CAMS/attn_topk: Token粒度(决定"哪些个")

详见 [innovation_overview.md](innovation_overview.md) 第2-3节完整对比表。

---

### 2026-08-25: CAMS结果 vs attn_topk公平性问题

**问题**: CAMS(5-seed, 59.96±0.04) 是否优于attn_topk?

**争议点**: attn_topk现有数据只有**单seed**(6月16日, mIoU=60.02),不能直接和CAMS的5-seed均值比较。

**决策**: 补齐attn_topk的5-seed实验(`A_Dtrain_Etrain`配置),预计9小时,09:02启动。

**用户澄清**: 用户最初以为8/24下午跑的5-seed就是attn_topk对照组,但核实后发现8/24下午跑的全部是`baseline_omnisam`(不是`A_Dtrain_Etrain`)。这是本次讨论中纠正的一个关键误解。

---

### 2026-08-25: min_tokens=256硬编码问题

**问题**: Pano-SMB的`min_tokens`超参数当前硬编码为256,代码中有两处强制检查:
```python
# model/modules/pano_smb_allocator.py:77-78 (已删除)
if self.min_tokens != 256:
    raise ValueError("Pano-SMB Stage 1 requires min_tokens=256")

# model/memory_bank.py:285-286 (已删除)
if self.pano_smb_min_tokens != 256:
    raise ValueError("Pano-SMB Stage 1 requires min_tokens=256")
```

**会议论文标准审视**:
用户提出"按照可以发会议论文的标准来"评估这个设计。

**分析**:
1. Reviewer必然会问: "256是怎么定的?是grid search还是随便选的?"
2. 从Stage 0审计文档推导: 256 = legacy单FoV quota(512) / 2,是一个保护下界,不是精心调优的值
3. 硬编码意味着**从未验证过其他值是否也work**,这是论文中的薄弱点

**决策**: 解除硬编码,补充min_tokens敏感性分析(128/256/512三个值),形成一条曲线证明方法的鲁棒性,而非依赖单一超参数。

**执行**:
1. 删除两处硬编码检查(allocator.py + memory_bank.py)
2. 重新跑12个单元测试确认无回归 → 全部PASS
3. 创建`B1_min128`, `B1_min512`两套config(复制`B1_pano_smb`并替换min_tokens)
4. 编写`run_pano_smb_sensitivity.sh`批处理脚本,串行跑B0→B1_min128→B1_min256→B1_min512

**待验证**: 今晚18:00(attn_topk跑完后)手动启动此批处理,预计24:00完成。

---

## 关键决策记录

### 决策1: 实验优先级排序

**背景**: 用户提醒"后续实验方向"清单中CAMS的优先级本应高于Pano-SMB Stage2,但对话一度偏向讨论Token Budget实验(Route非当前重点)。

**修正**: 重新确认当前实验重点顺序:
1. CAMS完整集成测试(创新点验证) ← 当前阶段
2. Pano-SMB Stage 2实验(B0/B1对比) ← 下一阶段
3. Token Budget/speed-accuracy曲线 ← 延后,非当前重点

### 决策2: GPU资源调度原则

**用户要求**: "白天做创新,晚上排实验" — 白天用于代码/设计层面的推进(不占或少占GPU),晚上串行跑长时间训练任务。

**已遵循**:
- 白天(8/25上午): 解除硬编码、写config、单元测试 — 不占GPU
- 白天GPU仍在跑: `A_Dtrain_Etrain`×5seed(9:02启动,不受影响)
- 晚上18:00后: 手动启动Pano-SMB敏感性分析(用户要求手动触发,不自动化)

### 决策3: 不做后台自动监控

**用户明确要求**: "你不要后台监管，或者我直接把你这个窗口关掉？" + "在跑脚本的时候你不要实时监控，不要浪费token"

**响应**: 停止主动轮询进度,只在用户主动询问时才去SSH检查状态。所有长跑训练用`nohup ... &`在远程独立运行,不依赖本地会话存活。

---

## 待解决问题 / Open Questions

1. **attn_topk多seed结果**(预计18:00出): 和CAMS(59.96±0.04)对比后,谁更优?
2. **Pano-SMB敏感性分析结果**(预计24:00出): min_tokens在128/256/512下是否都优于uniform baseline?
3. **CAMS + Pano-SMB组合**: 尚未测试,是否需要补充实验验证两层优化叠加效果?
4. **Token Budget曲线实验**(`A_Etrain_keep075`/`keep025`): 已确认延后,配置文件已存在但未执行。

---

## 相关文档索引

- [innovation_overview.md](innovation_overview.md) — 四个创新点完整技术对比
- `docs/pano_smb_stage0_code_audit.md` — Pano-SMB代码审计(静态分析)
- [cams_learnable_discussion.md](cams_learnable_discussion.md) — CAMS可学习权重探索(技术验证+暂不实施决策)
- [cams_learnable_validation.py](cams_learnable_validation.py) — 合成数据验证脚本
- `docs/pano_smb_stage1_implementation.md` — Pano-SMB实现细节
- `docs/pano_smb_stage2_experiment.md` — B0/B1对照实验协议
- `experiments/pano_smb_stage2/run_pano_smb_sensitivity.sh` — min_tokens敏感性分析脚本
