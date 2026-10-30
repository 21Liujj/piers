# SG-DRF 当前状态记录

**记录日期**：2026-08-31  
**创新点**：SG-DRF（Semantic-Guided Dynamic Receptive Field）  
**本地项目**：`/home/lightwheel/Desktop/TECMA-PASS`  
**远端项目**：`/root/autodl-tmp/OmniSAM-main`  
**远端分支**：`feature/scm-circular-memory`

## 1. 当前结论

SG-DRF 的核心模块和训练接线已经完成一轮高质量修复，模块级集成测试和静态编译检查均已通过；但目前还不能宣称第三个创新点已经完成最终实验验证。

当前准确结论是：

- **算法模块实现**：基本完成。
- **关键结构性错误修复**：已完成。
- **mock SAM 集成测试**：已通过，13 项全部通过。
- **真实 SAM2 tiny 构造**：已通过。
- **真实 SAM2 + OmniSAM_adapt 完整 forward**：尚未完成。
- **真实 backward、optimizer.step、MemoryBank attention 闭环**：尚未完成。
- **E0-E5 小规模消融实验**：尚未开始。
- **最终 mIoU、small-object mIoU、延迟、显存和路由分布结论**：尚无真实实验数据支持。

因此，SG-DRF 目前处于“核心实现完成，真实链路验证进行中，效果实验待执行”阶段。

## 2. 已完成的代码工作

### 2.1 Dynamic FSMC

文件：`model/modules/dynamic_fsmc.py`

已修复原实现把 compression ratio map 错当成 density map 的问题。原错误会使路由几乎固定在 4×，无法体现动态压缩。

当前实现：

- 先根据 density 计算连续压缩比例。
- 再离散映射到 4×、8×、16×三档。
- 支持 batch 内逐样本路由。
- 支持 horizontal stripe 级路由。
- 每个样本按实际压缩宽度处理，再对齐到 batch 最大宽度。
- 解压阶段恢复原始 feature map 空间尺寸。
- 保存 `last_ratio_map` 和 `last_stripe_ratios`。
- 返回原始宽度、压缩宽度、压缩比例、平均 density 和离散路由统计。

已验证的确定性路由结果：

```text
输入：(3, 8, 4, 64)
样本 1 stripe 路由：4×
样本 2 stripe 路由：8×
样本 3 stripe 路由：16×
输出尺寸：torch.Size([3, 8, 4, 64])
backward：可执行
```

### 2.2 Dynamic SCM

文件：`model/modules/dynamic_scm.py`

已修复 source-image pixels 与 feature-map columns 混用造成的 padding 宽度错误。

当前实现：

- 将 source-image padding 通过 `feature_stride` 转换为 feature-map columns。
- 默认 `feature_stride=16`，待真实 SAM2 encoder 输出进一步确认。
- 使用 circular indexing，支持 padding 大于原始 feature width。
- batch 内不同样本允许使用不同 padding。
- 使用 batch 最大 padding 对齐输出张量。
- 保存 per-sample padding metadata。
- 生成并保存 attention mask。
- 提供 `pad_tensor()`，用于 position embedding 采用相同 padding 几何。
- `remove_circular_padding()` 会校验输入宽度和 padding 合法性。

已验证：

```text
输入：(3, 8, 4, 64)
per-sample pad widths：[6, 4, 2]
输出宽度：76
attention mask：torch.Size([3, 1, 4, 76])
移除 padding 后：torch.Size([3, 8, 4, 64])
```

### 2.3 OmniSAM_adapt 接线

文件：`model/model.py`

已将 SG-DRF 接入 `OmniSAM_adapt`，支持以下配置：

- `enable_sgdrf`
- `sgdrf_enable_dynamic_fsmc`
- `sgdrf_enable_dynamic_scm`
- `sgdrf_backbone_channels`
- FSMC 最小和最大压缩比例
- SCM source-image padding 范围
- `sgdrf_training_stage`

已修复 feature 与 position embedding 的 padding 不一致问题。当前流程为：

```text
current_vision_feats
    -> DynamicSCM.pad
current_vision_pos_embeds
    -> DynamicSCM.pad_tensor
两者一起进入 MemoryBank
    -> MemoryBank attention
    -> DynamicSCM.remove_circular_padding
    -> decoder
```

此外已加入 SG-DRF consistency loss：

- 对 segmentation logits 做 softmax。
- 计算归一化 entropy 作为 segmentation uncertainty。
- 将 uncertainty resize 到 density map 分辨率。
- 与 density map 使用 MSE 对齐。

已修复真实环境下两种 SAM LoRA 包布局的 import 兼容性：

```python
try:
    from sam.sam_lora_image_encoder import LoRA_Sam
except ModuleNotFoundError:
    from sam_lora_image_encoder import LoRA_Sam
```

### 2.4 DDP builder

文件：`utils/build_ddp.py`

已将 SG-DRF 模式下的模型构造统一为 `OmniSAM_adapt`，不再使用与训练策略接口不兼容的 `SGDRFNet(OmniSAM)` 包装路径。

已保留：

- `SyncBatchNorm.convert_sync_batchnorm`
- `DistributedDataParallel`
- `find_unused_parameters=True`

训练策略所需的以下接口现在位于模型本体：

- `enable_sgdrf`
- `sgdrf_training_stage`
- `density_estimator`
- `compute_sgdrf_consistency_loss`
- SG-DRF auxiliary outputs

### 2.5 两阶段训练与优化器

文件：`train/trainer.py`、`train/strategies.py`、`train.py`

当前设计：

- `cold_start`：默认前 2 个 epoch 只训练 density estimator。
- `joint`：从第 2 个 epoch 开始联合训练主网络和 density estimator。
- 联合损失：

```text
loss = seg_loss + 0.1 * density_loss + 0.05 * consistency_loss
```

- 主网络默认学习率：`1e-4`。
- density estimator 默认学习率：`5e-5`。
- 优化器：AdamW。
- 梯度裁剪：`max_norm=1.0`。
- optimizer 使用两个参数组，density estimator 使用独立学习率。

epoch 开始时会依据 `sgdrf_cold_start_epochs` 自动切换训练阶段。

当前仍需通过真实 `Trainer.setup()` 和一次实际训练步确认：

- 两个参数组内容正确。
- cold-start 阶段主网络确实被冻结或不产生更新。
- joint 阶段主网络恢复更新。
- 梯度裁剪实际生效。

### 2.6 Density pseudo-label

文件：`utils/generate_density_pseudo_labels.py`

当前伪标签流程包含：

- Sobel 边界检测。
- 二值边界图。
- Distance Transform。
- 距离反向归一化。
- Gaussian Blur。
- 最终使用 `np.clip(..., 0.0, 1.0).astype(np.float32)` 保证输出范围。

已验证输出范围位于 `[0, 1]`。

## 3. 已完成的验证

### 3.1 模块级集成测试

文件：`test_sgdrf_integration.py`

执行方式：

```bash
/root/miniconda3/envs/omnisam/bin/python test_sgdrf_integration.py
```

已通过 13 项：

1. SG-DRF 模块导入。
2. `OmniSAM_adapt` 导入。
3. density pseudo-label generator 导入。
4. training strategies 导入。
5. 三个 SG-DRF 模块创建。
6. density estimator forward。
7. Dynamic FSMC。
8. Dynamic SCM。
9. density pseudo-label 生成。
10. SG-DRF disabled 的 `OmniSAM_adapt`。
11. SG-DRF enabled 的 `OmniSAM_adapt`。
12. cold-start/joint 阶段切换。
13. consistency loss。

测试最终输出：

```text
✓ ALL TESTS PASSED!
```

该测试使用 mock SAM，因此只能证明模块接口和局部逻辑正确，不能代替真实模型闭环验证。

### 3.2 静态编译检查

已对以下文件执行 `py_compile`，无编译错误：

```text
model/modules/dynamic_fsmc.py
model/modules/dynamic_scm.py
model/model.py
utils/build_ddp.py
train/strategies.py
train/trainer.py
train.py
utils/generate_density_pseudo_labels.py
```

### 3.3 真实 SAM2 tiny 构造

已在远端 RTX 5090 环境成功构造：

```text
配置：sam2.1_hiera_t.yaml
权重：sam2.1_hiera_tiny.pt
GPU：NVIDIA GeForce RTX 5090
显存：约 31.36 GiB
encoder neck d_model：256
```

真实 SAM2 backbone 构造成功，说明配置、checkpoint、GPU 和基础包布局可用。

## 4. 尚未完成与当前风险

### 4.1 真实 forward 闭环尚未完成

还没有完整跑通：

```text
真实 SAM2 tiny
→ LoRA_Sam
→ OmniSAM_adapt
→ density estimator
→ Dynamic FSMC
→ Dynamic SCM
→ position embedding 同步 padding
→ MemoryBank
→ remove padding
→ SegFormer decoder
→ segmentation logits
```

需要首先记录真实 encoder 返回的：

- `vision_features` shape。
- `vision_pos_enc` shape。
- `backbone_fpn` 三个尺度的 shape。
- 真实 feature stride。

### 4.2 MemoryBank attention mask 尚未完全接入

DynamicSCM 已经生成并保存 attention mask，但当前 mask 尚未完全传入 MemoryBank 的 attention 计算。需要确认：

- padded columns 是否应该在 memory attention 中被屏蔽。
- batch 最大 padding 对齐后，各样本 mask 是否正确生效。
- mask 是否需要转换为 MemoryBank 使用的 token 维度格式。

### 4.3 FSMC 当前实现以 Python loop 为主

当前优先保证路由和形状正确，代码中存在逐 sample、逐 stripe 的 Python loop。真实训练通过后再评估是否需要向量化，否则不要在真实验证前进行大范围性能重构。

### 4.4 Density pseudo-label 仍可能成为训练瓶颈

当前生成器主要依赖 CPU/OpenCV 路径。真实小规模训练时需要单独记录 pseudo-label 生成耗时，不能将 PL、Train、Val、E2E 时间混为一个数字。

### 4.5 validation 闭环尚未完成

还没有获得 SG-DRF 在 DensePASS val 上的真实：

- overall mIoU。
- small-object mIoU。
- per-class IoU。
- boundary 指标。
- latency。
- peak GPU memory。
- 平均 compression ratio。
- 4×、8×、16×路由分布。

## 5. 下一步执行顺序

### 阶段 A：真实最小 forward/backward smoke test

使用真实 SAM2 tiny、单 GPU、batch size 1 和尽可能小的输入，完成一到两个 frame 的真实训练步：

```text
build real SAM2
→ build OmniSAM_adapt(enable_sgdrf=True)
→ forward
→ segmentation loss
→ density loss
→ consistency loss
→ total loss
→ backward
→ gradient clipping
→ optimizer.step
```

必须记录：

- 各级 feature shape。
- density map shape 和数值范围。
- FSMC 路由统计。
- SCM per-sample pad width。
- padded feature/position token 数。
- decoder output shape。
- 三项 loss 和 total loss。
- 梯度范数。
- optimizer 参数组及学习率。
- 是否出现 NaN、Inf 或 shape mismatch。

### 阶段 B：补齐自动化断言

在测试中加入：

- FSMC 必须实际产生 4×、8×、16×。
- optimizer 必须包含两个参数组。
- density estimator 参数组学习率为 `5e-5 * world_size`。
- cold-start 阶段只有 density estimator 可训练。
- joint 阶段主网络恢复可训练。
- DDP SG-DRF builder 返回的内部模型为 `OmniSAM_adapt`。
- feature 和 position embedding flatten 后 token 数完全一致。
- SCM remove padding 后空间尺寸恢复。

### 阶段 C：小规模消融

按照设计文档建立 E0-E5：

| 实验 | 配置 |
|---|---|
| E0 | baseline，不启用 SG-DRF |
| E1 | 仅 Dynamic FSMC |
| E2 | 仅 Dynamic SCM |
| E3 | Full SG-DRF，不使用 cold-start |
| E4 | Full SG-DRF，不使用 consistency loss |
| E5 | Full SG-DRF，包含 cold-start 和 consistency loss |

先执行单 seed、小数据子集、短 epoch；确认链路稳定后再对 E0、E4、E5 做多 seed 重复。

每个实验必须保存：

- 配置 YAML 快照。
- seed。
- GPU 型号和显存信息。
- 当前 git branch / commit 或工作树标识。
- 完整启动命令。
- stdout/stderr 日志。
- `metrics.jsonl`。
- 路由比例统计。
- density map 可视化。
- Train / PL / Val / E2E 时间。
- latency 和 peak memory。

建议目录：

```text
logs/sgdrf_e0_baseline/
logs/sgdrf_e1_fsmc/
logs/sgdrf_e2_scm/
logs/sgdrf_e3_full_no_cold_start/
logs/sgdrf_e4_full_no_consistency/
logs/sgdrf_e5_full/
```

## 6. 当前文件变更清单

远端当前核心修改和新增文件包括：

```text
M model/model.py
M train.py
M train/strategies.py
M train/trainer.py
M utils/build_ddp.py
M validation.py
?? model/modules/dynamic_fsmc.py
?? model/modules/dynamic_scm.py
?? model/modules/semantic_density_estimator.py
?? model/modules/sgdrf_net.py
?? utils/generate_density_pseudo_labels.py
?? test_sgdrf_integration.py
?? docs/SG-DRF_INTEGRATION_GUIDE.md
```

远端工作树还包含其他已有实验资产和用户修改，后续操作不得回退、覆盖或删除与 SG-DRF 无关的内容。

## 7. 版本与操作纪律

- 当前没有执行 git commit 或 push。
- 没有执行 `git reset`、`git checkout` 或文件回退。
- 没有覆盖已有 checkpoint 和伪标签。
- 真实实验不得覆盖已有实验目录，必须使用新的 `logs/sgdrf_*` 目录。
- 本次状态文档允许在本地 Desktop 项目中编辑，并同步到远端。
- 后续代码施工和实验仍应以远端 `/root/autodl-tmp/OmniSAM-main` 为主，避免本地和远端出现未记录的分叉。
- 后续任何结论必须以实际日志、测试输出或 `metrics.jsonl` 为依据；尚未验证的内容必须明确标注为“尚未验证”。

## 8. 最终判定标准

只有同时满足以下条件，才能将 SG-DRF 标记为“完成”：

1. 真实 SAM2 tiny 或 small 的 forward 闭环通过。
2. 真实 MemoryBank feature/position token 对齐通过。
3. 真实 segmentation、density、consistency loss 均 finite。
4. 真实 backward 和 optimizer.step 通过。
5. cold-start 与 joint 阶段行为通过自动化断言。
6. E0-E5 小规模实验全部完成且日志完整。
7. 至少完成 E0、E4、E5 的重复实验或给出明确的单 seed 限制说明。
8. validation 指标、速度、显存和动态路由统计均有可复查记录。

在上述条件完成前，正确表述应为：

> SG-DRF 核心实现和关键错误修复已完成，模块级测试通过，真实训练链路与效果实验仍在验证中。
