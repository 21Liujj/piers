# PSR-PLD 施工前代码审计报告

**日期**: 2026-06-04  
**范围**: A0 Baseline Audit — 第 4 章全部内容  
**审计者**: Claude Code  
**仓库**: `/root/autodl-tmp/OmniSAM-main`

---

## 4.1 PL 生成链路（真实调用链）

```
run_adaptation.py (argparse + YAML)
  └─ Engine.__init__()                           engine.py:~50
  └─ Engine.run()
       └─ Engine.train(epoch)                    engine.py:489
            ├─ Engine.need_pseudo_labels(epoch)  每 pseudo_every 轮触发一次
            ├─ Engine.generate_pseudo_labels(epoch)  engine.py:441
            │    └─ PseudoLabelGenerator.run(
            │         dataloader=target_pl_loader,   # WildPASSforPL
            │         save_dir=pd_root/DensePASS/.../epoch{N},
            │         uc_threshold=args.uc_threshold, # 默认 0.85
            │         passes=2 (forward + backward)
            │       )
            │         ├─ 对每个 batch 的 images: (B, 2*frame_cnt, C, H, W)
            │         │    ├─ Forward pass: frame_idx in [0, frame_cnt)
            │         │    │    └─ model(frame_idx, frame_img) → logits_maps[:, frame_idx]
            │         │    ├─ Backward pass: frame_idx in [0, frame_cnt) 反向
            │         │    │    └─ model(frame_idx, frame_img) → logits_maps[:, frame_cnt+frame_idx]
            │         │    └─ merge_width_scatter_np_uncertainty(logits_maps, ...)
            │         │         → merged_pred: (B, H_full, W_full) uint8，低置信位置=255
            │         └─ mask_vis(merged_pred, ...) → PNG 写盘
            │              路径: {save_dir}/gtFine/{name}_labelTrainIds.png
            └─ Engine._ensure_pseudo_labels_ready(epoch)
                 └─ DataBuilder.build_tgt_from_pd(save_dir)
                      └─ WildPASSforAdaptation(pd_dir, used_path.txt, ...)
                           └─ 读取 PNG pseudo-label → 滑窗切片 → DataLoader
```

**Patch sequence 细节**：

| 参数 | 值（DensePASS/WildPASS） |
|---|---|
| 全景尺寸 | 4096×1024 |
| patch W×H | 1024×1024 |
| stride_w | 384 |
| patches per pass (L) | (4096-1024)//384+1 = **9** |
| passes | 2（forward + reverse） |
| frame_cnt | 9（每 pass 9 个 patch） |
| logits_maps shape | (B, 18, C, 1024, 1024) |

**Forward pass**：patch 从左到右（index 0→8）  
**Backward pass**：patch 从右到左（index 8→0，即 reversed）  
`images[:, :frame_cnt]` = forward，`images[:, frame_cnt:]` = backward

**Ignore index**：255（确认）

**PL 何时重新生成**：每 `pseudo_every` 轮（默认每轮），保存到 `epoch{N}` 子目录

---

## 4.2 Merge 逻辑（核心发现）

函数位置：`utils/tools.py:86` `merge_width_scatter_np_uncertainty()`

```python
# Step 1: 每个 patch 独立 softmax
probs = torch.softmax(logits, dim=2)        # (N, B, C, H, Wp)

# Step 2: 重排为 (N, passes, L, C, H, Wp)
probs = probs.view(N, passes, L, C, H, Wp)

# Step 3: forward + backward 对应 patch 取平均
forward  = probs[:, 0]              # (N, L, C, H, Wp)
backward = probs[:, 1].flip(1)      # 反转 L 维度，对齐到 forward 顺序
probs = 0.5 * (forward + backward)  # (N, L, C, H, Wp) ← 平均 passes

# Step 4: scatter_add_ 累积
out.scatter_add_(3, idx_r, vals_r)  # out: (N, C, H, W), 无除法

# Step 5: 阈值过滤
probs_max, pred = out.max(dim=1)
pred[probs_max < threshold] = ignore_index
```

### ⚠️ 核心工程问题：Coverage 未归一化

DensePASS 4096 宽，stride=384，patch=1024，L=9 的覆盖分布：

| 覆盖次数 | 像素列数 | 占比 |
|---|---|---|
| 1× | 768 | 18.75% |
| 2× | 1536 | 37.50% |
| 3× | 1792 | 43.75% |

累积后 `out[c, y, x]` 是该位置 covering patches 对类别 c 的概率之和（非均值）。  
`probs_max` 实际取值范围：

- 覆盖 1×：`[0, 1.0]`
- 覆盖 2×：`[0, 2.0]`  
- 覆盖 3×：`[0, 3.0]`

以 `threshold=0.85` 为例：
- **覆盖 1× 边缘像素**：需要单 patch 置信度 ≥ 0.85 才能通过，要求严格
- **覆盖 3× 中央像素**：3 个 patch 各 0.30 概率即可通过，等效置信度仅 0.30，过滤形同虚设

**结论**：
- 没有 coverage 归一化
- 没有统计覆盖次数
- 没有区分 seam（边缘）和 non-seam 区域
- 阈值行为非均匀，覆盖多的区域置信度虚高

### forward/backward merge 是否正确？

是。`backward = probs[:, 1].flip(1)` 将反向 pass 的 L 维度翻转，使其与 forward 的 patch 索引对齐，然后取 0.5 平均。这部分逻辑正确。

---

## 4.3 loss_unsup 逻辑

### 位置

`adaptation/strategies.py:WithMemStrategy.forward()`

### 两层独立机制

**第一层（PL 生成时）**：`uc_threshold=0.85` 在 `merge_width_scatter_np_uncertainty` 中过滤，低置信像素写成 255 存入 PNG。

**第二层（Train 时，online）**：

```python
# strategies.py:183-199
probs = torch.softmax(pred_t, 1)
conf_t, pseudo_t = probs.max(1)               # 当前模型对每个 patch 的在线置信度
valid_mask = (frame_pl_t != ignore_index)     # PL 文件中非 255 的位置
visual_mask = (conf_t > conf_thr) & valid_mask # 在线置信度再次过滤

# 最终 mask 应用
frame_pl_t_f = frame_pl_t.clone()
frame_pl_t_f[~mask_t] = ignore_index
loss_unsup_raw = criteria['t'](pred_t, frame_pl_t_f)
```

**第二层 pseudo_conf_mask 插件（`plugins.enabled("pseudo_conf_mask")` 时）**：

```python
# pseudo_conf_mask.py: masked_unsup_ce()
keep = valid & conf_keep                  # valid = (pl != 255) & base_mask
                                          # conf_keep = conf_t >= threshold
ce = F.cross_entropy(pred_logits, pseudo_label, ignore_index=255, reduction="none")
ce = ce * keep.float()
loss_masked = ce.sum() / float(valid_pixels)   # 正确归一化
```

### 检查结果

| 检查项 | 状态 |
|---|---|
| `reduction="none"` + 手动归一化 | ✅ 已实现（在 pseudo_conf_mask.py 中） |
| pixel-wise mask 进入梯度 | ✅ 是：`ce * keep.float()` 直接参与反向传播 |
| mask 全零时不产生 NaN | ✅ `valid_pixels==0` → `loss_masked = tensor(0.0)` |
| AMP 安全 | ✅ dtype 保持在 `ce` 上 |
| ignore_index 处理 | ✅ `F.cross_entropy(..., ignore_index=255)` |
| loss 分母归一化 | ✅ `/ float(valid_pixels)` |
| 旧 path（无插件）是否支持 weight | ⚠️ `criteria['t'](pred_t, frame_pl_t_f)` 是普通 CE，`reduction="mean"`，无 pixel weight |

**结论**：
- `pseudo_conf_mask` 插件已正确实现 pixel-wise weighted CE，mask 真正影响梯度
- 但它仅在 train 阶段对 ONLINE prediction 做阈值，与 PL 文件中的置信度无关
- 没有 sidecar confidence/weight 文件；train 时无法直接使用 PL 生成时的置信度图
- `weighted_unsup_loss`（text/depth 路径）也有正确实现但用 `.mean()` 不是 `/valid_pixels`

---

## 4.4 FPA Prototype 链路

### 位置

`adaptation/methods.py` → `PrototypicalAdaptationEMA`

### Prototype 状态

| 属性 | 说明 |
|---|---|
| shape | `(num_classes=13, feature_dim)` |
| 存储位置 | GPU tensor |
| 更新方式 | EMA momentum=0.99，仅从**源域特征**更新 |
| 特征来源 | `feat_s_de_ori`（源域模型 decoder head 输出，`SegFormerHead_adaption`） |
| 更新时机 | 每个 train iteration 的每个 frame |

### 在 PL 阶段是否可复用？

PL 生成时：
```python
# pl_generator.py
output, _, _ = self.model(frame_idx, frame_img, depth=frame_depth)
logits_maps[:, frame_idx] = output.detach().float().cpu().numpy()
# feat_de（decoder features）被丢弃（第二个 _ 和第三个 _）
```

模型 forward 已经计算了 decoder features，但**当前代码丢弃**了它们（返回值 `_, _` 中的中间两个）。

- **Prototype 本身**：存在 checkpoint 中，可在 PL 阶段通过加载 checkpoint 读取
- **Target decoder features**：PL 阶段已经 forward 计算，只需修改返回值收集
- **不需要额外完整 forward**：只需在 `pl_generator.py` 中接收 `feat_de` 并进行点积计算

---

## 总结：5 个关键发现

### 发现 1（高优先级）：Coverage 未归一化导致置信度虚高

`merge_width_scatter_np_uncertainty` 使用 `scatter_add_` 累积概率，但不除以覆盖次数。DensePASS 4096 宽场景下，43.75% 像素被覆盖 3 次，中央区域 probs_max 可达 3.0，以 0.85 阈值过滤完全失效。

**影响**：
- PL 质量低于预期：中央区域大量低置信像素被错误保留
- 边缘区域（coverage=1）被过度过滤，而这些区域往往包含重要小类别目标
- 阈值语义失去意义

### 发现 2（工程问题）：Train-time confidence 与 PL confidence 脱钩

PL 生成的置信度信息（coverage-inflated）未保存为 sidecar 文件。Train 时 `pseudo_conf_mask` 使用的是当前模型的 online 预测置信度，与 PL 文件的置信度完全无关。这意味着 PL 文件本身没有携带可靠性信号。

### 发现 3（正确实现，可扩展）：pseudo_conf_mask 已正确实现 pixel-wise loss

`masked_unsup_ce()` 使用 `reduction="none"` + `/ valid_pixels` 归一化，mask 真正进入梯度，无 NaN，AMP 安全。**PSR-PLD 的 R1-R3 可以在此基础上直接扩展**，只需传入新的 `weight_map`。

### 发现 4（可直接复用）：forward+backward 平均逻辑正确，但丢失了逐 patch 投票信息

`0.5 * (forward + backward)` 在 patch 内聚合了两次 pass，但完全丢失了逐 patch 投票一致性信息（每个覆盖像素的多个 patch 各自预测什么类别）。R3 需要在 `scatter_add_` 之前保留逐 patch argmax，用于构造 vote agreement map。

### 发现 5（R4 可行但需修改）：FPA prototype 可复用，decoder features 已计算但被丢弃

`feat_t_de` 在 PL forward 中已被计算，只需修改 `pl_generator.py` 的接收部分即可使用。不需要额外完整 forward。但需要在 PL 生成时有可用的 prototype（来自上一轮训练后的 checkpoint）。

---

## A0 工程正确性修复范围

基于以上发现，A0 阶段需要修复：

1. **Coverage normalization**（必须）：在 `merge_width_scatter_np_uncertainty` 中加入 `/ coverage.clamp_min(1)` 选项，默认保持旧行为，config 控制开关
2. **Loss pixel-wise 接入**（已存在，需验证）：`masked_unsup_ce` 已有正确实现，PSR-PLD 路径直接复用即可；旧路径 `criteria['t']` 使用标准 `reduction="mean"`，需确认是否需要统一

### 下一步动作

1. 实现 `merge_width_scatter_np_uncertainty` 的 coverage normalization 分支
2. 验证归一化后的 probs_max 分布符合预期（0-1 范围）
3. 确认 sidecar 保存逻辑设计（weight map 与 mask）
4. 在此基础上实现 R1 confidence mask（基于归一化后置信度）

---

*本报告基于直接代码读取生成，无推测成分。*
