# 创新点2成果文档：全景 Memory 按需分配（I2，不确定性排序 top-k + 类别/边界配额）

> 状态：DensePASS100 已冻结收尾；跨域受数据/架构限制未进主表。冻结口径以 `docs/piers/innov2_results.md`、
> `results/paper20261109/memory_component_accuracy.json`、`complete_table.json` 为准。
> 对应代码：`src/piers_ljj/memory/`（`PanoSMBAllocator` 窗间分钱 + `SparseMemoryRetriever` 窗内挑点）、
> `scripts/piers/full_model_semantic.py::SemanticEngine`（memory='i2'，keep_ratio 0.03125）、
> `experiments/paper20261109/runtime/i1_runtime.py::install_fixed_memory`。

## 0. 实际收益（一句话）

- I2 是**拿 attention 计算量换精度的压缩器，不是加速器/省存器**：单走 -0.15（60.2961→60.1447），但把单次 attention 从 O(32k) 降到 O(1k)，给 I3 腾出 2 个 seam forward 的预算，i2_B0→i2_B2 **+0.357**；同预算稀有类 light **+0.63** / sign **+0.67** / mAcc **+0.35**；峰值显存 4157 vs 4157 不变（逻辑剪枝，主动承认）。

## 1. 解决了什么问题

- OmniSAM 原版 dense memory：32,768 token（`num_maskmem=9` 帧历史全堆）全进 attention。
  全景 4096×1024 按滑窗（win 1024 / stride 384，正扫 9 + 反扫 9 = 18 forward）窗窗重叠，
  记忆里全是重复像素，又费显存又慢。
- 第一代压缩（EfficientTAM 式平均池化 `pool`）无脑压，好坏一起糊：灯牌类塌 4.2（见 §3）。

## 2. 做法（两级按需分配）

- 级一窗间分钱（`PanoSMBAllocator`，无参数不学习）：总预算锁死（如 2048），按每窗重要性
  `importance_from_scores` 按比例分，保底每窗 `min_tokens=256`，`budget_error=0`。
- 级二窗内挑点（`SparseMemoryRetriever`）：`uniform / pool / topk(feature_norm) / random` 四对照；
  `bg` 留 12.5% 给历史伪标签高熵/边界；`ca` 留 15% 给 traffic light/sign/pole/rider；
  `cams` 查询亲和度 + 范数自适应。
- 全景适配：历史伪标签空域前传（左窗 pred+entropy 传右窗）；预算守恒（总数恒 2048，
  与 dense 32768 比严格 16×）；位置去重（窗重叠区重复 token）。
- 冻结版（v3 整机）：不确定性排序 top-k，keep_ratio **0.03125（32,768→1,024，1/32）**，
  bank 全驻留、attention 时取 top-k（逻辑剪枝，非物理删除；`stored_dense_features_retained: true`）。

## 3. 成果数字（冻结，只报已验证）

### 3.1 主表（DensePASS100，N=2048 vs dense 32768，`innov2_results.md`）

| 指标 | dense | ca主推 | bg | pool | topk | uniform |
|---|---|---|---|---|---|---|
| tokens | 32768 | 2048（1/16） | 2048 | 2048 | 2048 | 2048 |
| mIoU | 60.46 | **60.30（-0.16）** | 60.27 | 60.07 | 59.83 | 59.80 |
| mAcc | 71.73 | **72.08（+0.35）** | 72.03 | 70.96 | 70.38 | 70.34 |

- 稀有类：traffic light 30.06→**30.69（+0.63）**、traffic sign 26.03→**26.69（+0.67）**；
  `pool` 把 light 压到 25.84（塌 4.2），ca 拉回 4.85。person 66.47→65.86（-0.61，大类让利）。
- 代价：wall time 100 张 dense 383s vs ca 465s（+20%，贵在伪标签打分）；pool/uniform 才又省存又省时但压塌稀有类。

### 3.2 整机中的 I2（`memory_component_accuracy.json`，100 张 13 类）

- dense_B0 **60.2961** → i2_B2 **60.5018**（**+0.2057**）；i2_B0 60.1447 → i2_B2 60.5018（+0.357，
  含 I3 的 B2 两窗贡献，见 I3 文档）。
- v3 整机 full **60.8555**（再含 I1 融合与开词表竞争，见 I1 文档）。

### 3.3 效率向诚实记录（11-11 晚实测，不算方法增益）

- attention 输入 32,768→1,024（1/32），但语义峰值分配 **4157 vs 4157 MiB 纹丝不动**
 （bank 全驻留，只算得少、没存得少）；整机语义 18 fwd 4.58s → 20 fwd 5.05s（每个 seam +0.2~0.26s，
  融合另 +1.49s）。I2 省的计算量被 I3+fusion 吃掉——I2 是 I3 的入场券，不是加速器。

## 4. 诚实局限

- 跨域未进主表：XW500-dense mIoU 0.09（manifest flat labels_trainid 把 void 当 road，GT 错，数作废）；
  SPAN8 报 `decoder 13 vs 8 mismatch`（13 类 ckpt 评 8 类数据，原仓库架构限制，表 3 不放 SPAN8）。
- 16×（2048）与 32×（1024）是两套预算：消融主表用 2048（1/16），v3 整机用 1024（1/32）；
  写论文时注明口径，不混用。

## 5. 论文写法建议

- 主张：**同预算下稀有类/边界不掉反涨**（light +0.63 / sign +0.67 / mAcc +0.35），不是全面超越 dense（-0.16）。
- 消融表：dense vs ca vs bg vs pool vs topk vs uniform（+cams）；`nocompress` 只划上限不参与排名。
- I2→I3 关系写 trade-off：拿 attention 计算量换 2 个 seam forward 的精度（§3.3 诚实披露）。
