# 创新点3成果文档：接缝按需补看（I3，自适应 seam forward + 接缝带门控融合）

> 状态：保底已成、涨点未成，论文中写"候选"不写"已验证正向"。冻结配置
> `configs/piers/innovation3_candidate.json`，运行记录 `results/I2_I3/`，
> 整机数见 `results/paper20261109/memory_component_accuracy.json`。
> 对应代码：`seam/adaptive.py + policy.py`（`select_extra` / `fuse_overlap`）、
> `scripts/piers/full_model_semantic.py::SemanticEngine`（budget 0/1/2 → 18/19/20 forward）、
> `experiments/paper20261109/runtime/run_i3_semantic.py`（`choose` + `fuse_cuda`）。

## 0. 实际收益（一句话）

- I3 是**花钱买接缝精度的候选模块**：i2_B0 60.1447→i2_B2 **60.5018（+0.357）**，每个 seam forward 约 +0.2~0.26s + 融合 1.49s；但独立消融 adaptive vs uniform 打平（60.4619 vs 60.4619），已验证的是 B=0 严格退化 + 融合无泄漏，精度增益写"候选、噪声级"，靠 I2 的 1/32 才跑得动。

## 1. 解决了什么问题

- 全景左右两边在 ERP 里断开、实际连着（0° 缝）。滑窗扫到缝附近时物体被切两半、两边各判各的，
  接缝带 IoU 低于全局。
- 继承基线 `fixed_scm`（DensePASS_circular + DynamicSCM）无脑全缝补，当对照留着；
  I3 只加按需策略：同样的额外开销预算 B 下，按需补看 ≥ 均匀补看，且 B=0 严格退化。

## 2. 做法（B 预算）

- B = 最多补几扇跨缝真窗（wrap crop，`x_start` 可负/超宽，绝不造全宽假框）。
  基础 18 窗不动：B=0→18 窗；B=1→19 窗；B=2→20 窗（`model_input_px=(18+B)×1024×1024`，多花如实记）。
- 选窗：`score_uda = 熵 + 截断 + 跨 FoV 不一致`；`score_text = 贴缝 + (1-质量) + assoc 不确定`；
  `select_extra` 取 ≤B 非冗余，同 B 下比 adaptive vs uniform。
- 融合：只融接缝带（`BAND_W=256` 左右各一条），可靠性门控 `ent>0.6 & econf>0.7`，
  `fuse_overlap` 按覆盖×置信加权。B=0 严格退化为 base（`fused_px=0`，逐比特一致，CPU 测试已锁）。

## 3. 成果数字（冻结，只报已验证；涨点未成如实写）

### 3.1 独立消融（`innovations_understanding.md` §2.3，门控版）

- 20 张：B=0 55.44 / seam 57.74；B=1 adapt 55.45 / 57.75（fused 807px）；
  B=1 uniform 55.4487；B=2 55.448（fused 2110px）。
- 100 张：S100-b0 **60.4607** / seam 57.0019；B=1 adapt **60.4619** / 57.0028
  （fused 1603px，多花 59s）；B=1 uniform 也是 60.4619（选窗没比随机好）。
- 旧版全窗覆盖 B=1 掉到 48.98 是 bug，已废弃不计入正表。
- 结论：**保底成（B=0 恒等、融合无泄漏）、涨点没成（+0.001 是噪声）**。

### 3.2 整机中的 I3（`memory_component_accuracy.json`，100 张 13 类）

- i2_B0 60.1447 → i2_B2 **60.5018**（+0.357，含 I2 的 1024 压缩底座）；
  dense_B0 60.2961 → i2_B2 60.5018（+0.206）。
- 代价（MB18，Dense 单图）：original 18 fwd 4.58s / i2b1 19 fwd 4.79s / i2b2 20 fwd 5.05s——
  每个 seam forward 约 +0.2~0.26s；融合另 +1.49s（FUSE12）。

## 4. 诚实局限

- adaptive vs uniform 在 S100 上打平（都是 60.4619）："按需"在当前 B/数据/门控下无增益，
  不得调参掩盖，不得换子集重跑。
- 接缝难例分析未做：只挑熵高/跨缝不一致图的 `seam_mIoU`、门控放宽让 fused_px 从 0.15% 涨到 1~2%
  看 B=1/2 能否涨 0.2~0.3——这是下一步，不是已得结果。

## 5. 论文写法建议

- 措辞保持"候选"：B=0 严格退化 + 融合无副作用是已验证性质；精度增益写"未观测到显著增益（+0.001，噪声级）"。
- 表格：B=0 / B=1-adapt / B=1-uniform / B=2 + fused_px + 多花时间 + seam_iou + full_miou 五列并列。
- 与 I2 绑定写：一句话说明 I3 的 2 个额外 forward 依赖 I2 的 1/32 压缩才跑得动（trade-off）。
