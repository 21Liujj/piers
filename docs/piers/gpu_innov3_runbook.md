# GPU 上卡实验手册（创新点 3：接缝按需补看 / R4）

适用对象：租到 CUDA 单卡后，跑通 **S suite 四变体等代价消融**。
前置：T1 文本通路已跑（S 的文本侧复用其候选与代价记账），E0/UDA 侧按需就绪。

状态基线（2026-10-30，CPU 已验证）：
`seam/policy.py`（`select_extra` / `fuse_overlap`）+ `seam/adaptive.py`
（接缝带打分、跨缝连续窗、按需融合、成本记账）代码完成，
`fixed_scm`（继承 `DensePASS_circular + DynamicSCM`）保持为基线、未重实现；
`test_innov23.py` 覆盖 B=0 严格退化、真 wrap 窗裁剪、UDA/文本双打分函数、成本计数。
GPU 上要做的是等代价实测 + 真实重叠融合。

---

## 0. 结论先行：要回答的问题与消融表

核心主张：**同样的额外开销预算 B 下，按需补看（adaptive_extra）接缝质量 ≥ 均匀补看**，
且 **B=0 时严格退化为基线**（无额外窗、无融合、输出逐比特一致）。

| 变体 | 含义 | 开销 | 是否参与等代价比较 |
|---|---|---|---|
| `linear` | 线性基线（无 seam 特殊处理） | B=0 | ✅ 下限对照 |
| `fixed_scm` | 继承 circular + DynamicSCM（基线） | B=0 | ✅ 主对照 |
| `uniform_extra` | 均匀加 B 个额外窗（位置固定） | B=2 | ✅ 关键基线：赢它才算“按需”有用 |
| `adaptive_extra` | 按不确定度选 ≤B 个跨缝连续窗补看 | B=2 | ✅ 创新点 |

固定量（所有变体相同）：`extra_budget_B: 2`（`S_seam.yaml`）、seed、数据子集、
FoV 切分（与 T1 同配置）、阈值。
报告量（`S_seam.yaml::report`）：`windows / extra_windows / dino_calls / sam_enc /
sam_dec / latency_ms / seam_iou / full_miou`。

**消融是必做的，而且是双任务的**：UDA 侧与文本侧各一张表。
打分函数不同（UDA：熵/截断/跨 FoV 分歧；文本：触边/SAM 质量/assoc 不确定），
但选择与融合机制同一套（`select_extra + fuse_overlap`），两张表都要能赢 `uniform_extra`。
另需单列 **B=0 退化校验**（`adaptive_extra` 配 B=0 时输出 == `fixed_scm` 输出）。

---

## 1. 上卡前置（沿用创新点 1 文档的 env + 路径）

```bash
conda activate piers-grounded   # S 的文本侧跑在此 env；UDA 侧用 piers-omni
nvidia-smi
piers_ljj doctor
# manifest 与 T1 共用一份（同一 dev-calib 子集，保证可比）
piers_ljj validate-data --manifest configs/piers/manifest.dev-calib.json --check-images
```

依赖确认：

```bash
piers_ljj sweep --suite S --check-only --runs-root runs
# S 无上游依赖；但文本侧比较要求 T1 已跑完（同一 manifest、同一阈值）
```

---

## 2. S 运行（四变体 + B=0 退化，共五次）

### 2.1 单变体命令（以 adaptive_extra 为例）

```bash
# 复制 config 改 variants 为单元素（B/seed/数据不动），逐个 run
# 例：configs/piers/S_seam_adaptive_extra.yaml  (variants: [adaptive_extra])
piers_ljj run --config configs/piers/S_seam_adaptive_extra.yaml --dry-run
piers_ljj run --config configs/piers/S_seam_adaptive_extra.yaml
```

`S_seam.yaml` 关键字段（改单变体 config 时只动 `variants`，其它不动）：

```yaml
suite: S-seam
run_id: S-seam
seed: 0
variants: [adaptive_extra]   # linear | fixed_scm | uniform_extra | adaptive_extra
extra_budget_B: 2
report: [windows, extra_windows, dino_calls, sam_enc, sam_dec, latency_ms, seam_iou, full_miou]
```

### 2.2 B=0 退化校验（必做，不计入排名）

```bash
# 同一份 adaptive_extra config，仅把 extra_budget_B 改为 0，跑一次
# 断言：输出 == fixed_scm 输出（逐像素一致），extra_windows == 0
```

CPU 侧该性质已由测试锁定（`test_seam_B0_strict_degradation`），GPU 上是端到端复核：
若不一致，说明融合分支有副作用泄漏，先修 bug 再谈精度。

### 2.3 跨缝窗的诚实性（实现约束，跑之前先理解）

- 额外窗必须是**真 wrap 连续窗**（`plan_cross_seam_windows` 生成，`crop_erp_wrap` 裁剪，
  `x_start` 可为负或 ≥W）；**禁止**用整宽伪 box 假装跨缝（CPU 测试已锁
  `test_cross_seam_windows_are_true_wrap`）。
- 融合只发生在**真重叠区**（`fuse_overlap` 按 coverage×confidence 加权），未覆盖区保持 base。
- 接缝带是左右列的**空间邻域**，不是同一像素（`seam_bands`，CPU 已锁）。

### 2.4 评测（双任务 + 接缝分段）

```bash
# 文本侧（与 T1 同口径）：
piers_ljj evaluate --pred-npz runs/S-seam/<variant>/pred.npz \
  --gt-npz runs/S-seam/<variant>/gt.npz --task text \
  --out runs/S-seam/<variant>/metrics_text.json
# UDA 语义侧：
piers_ljj evaluate --pred-npz runs/S-seam/<variant>/pred_uda.npz \
  --gt-npz runs/S-seam/<variant>/gt_uda.npz --task uda --n-cls 13 \
  --out runs/S-seam/<variant>/metrics_uda.json

piers_ljj summarize --runs-root runs --out runs/summary.json
```

报表要求（写论文 §3(3) 用）：

| 列 | 说明 |
|---|---|
| `seam_iou` | 接缝带内 IoU（左右 band 列，如 `seam_bands(W, band_w)` 定义） |
| `full_miou` | 整图 mIoU（确认补看没有拆东墙补西墙） |
| 代价五件套 | `windows / dino_calls / sam_enc / sam_dec / latency_ms` 并列，不折算加速比 |

---

## 3. 结果判读（写论文 §3(3) 与消融表用）

1. **主表（×2 任务）**：固定 B=2 下四变体的 `seam_iou / full_miou` + 代价；
   `adaptive_extra` 必须在 `seam_iou` 上赢 `uniform_extra`，且 `full_miou` 不跌。
2. **退化行**：B=0 的 `adaptive_extra` 与 `fixed_scm` 数值一致（单列声明，不参与排名）。
3. **失败即结论**：若打不过 `uniform_extra`，结论就是“按需策略在该 B/数据下无增益”，
   不得追加 B、不得换 band 宽度重跑（固定配置、无网格搜索）。
4. 与创新点 2 的 M 表一样：固定种子、固定子集；`nocompress` 式的不等代价参照不得进排名。

## 附录：故障速查

| 现象 | 最可能原因 | 查哪里 |
|---|---|---|
| B=0 输出与基线不一致 | 融合分支泄漏 | 查 `fuse_overlap` 是否在 `extra_w==0` 处短路 |
| 跨缝窗出现整宽 box | 有人绕过 `plan_cross_seam_windows` | 查 worker 日志的 window 生成路径 |
| `seam_iou` 高但 `full_miou` 跌 | 融合权重压过 base | 查 `extra_conf` 与 coverage 权重 |
| 代价对不上（sam_enc 偏大） | 同一 FoV 重复 `set_image` | 查 `calls.sam_image_encodes` 是否 == 窗数 |
