# GPU 上卡实验手册（创新点 2：ERP 位置约束的稀疏记忆 / R3）

适用对象：租到 CUDA 单卡后，跑通 **M suite 七变体等预算消融**。
前置：E0 基线已跑（`runs/E0-effective-start` 为 `completed_real`），数据/ckpt 就位。

状态基线（2026-10-30，CPU 已验证）：
`memory/policy.py`（M1 去重 / M2 配额预留 / M3 邻域排序）+
`memory/bridge.py`（ERP footprint、overlap 判定、variant→SMR 参数映射）代码完成，
`test_innov23.py` 覆盖去重保额/配额预留/邻域排序/nocompress 隔离/B=0 退化；
**重计算全部转调继承模块**（PanoSMB/SMR/MemoryBank），piers 只算元数据。
GPU 上要做的是等预算实测 + parity 对齐。

---

## 0. 结论先行：要回答的问题与消融表

核心主张：**同 token 预算下，位置约束的稀疏记忆（M1/M2/M3）精度 ≥ 无约束基线**，
代价（select/attn/step 时间、峰值显存）并列报告，不折算加速比。

| 变体 | 含义 | SMR 映射（`bridge.VARIANT_TO_SMR`） | 是否参与等预算比较 |
|---|---|---|---|
| `current` | 继承 `effective_start` 原样（对照） | `sparse_mode: None`（不覆盖） | ✅ 主对照 |
| `nocompress_ref` | 不压缩参照（精度/代价上限参照） | `sparse_mode: none` | ❌ 只做参照，token 不可比，单列报告 |
| `uniform_budget` | 等配额（PanoSMB 跳过）+ 特征范数 topk | `sparse_mode: feature_norm` | ✅ |
| `topk_fixed` | 同预算纯特征范数 topk（无位置） | `sparse_mode: feature_norm` | ✅ 关键基线：赢它才算位置有用 |
| `pos_dedup`（M1） | 同 bucket/overlap 去重，留高分代表 | `feature_norm` + overlap 预过滤 | ✅ 第一创新点 |
| `pos_dedup_priority`（M2） | M1 + 边界/稀有类配额预留 | `boundary_guarded`，fallback `class_aware` | ✅ 第二创新点 |
| `neighborhood`（M3） | 按与当前 FoV 邻接度排序，保全局尾部 | `cams`，fallback `hybrid` | ✅ 第三创新点 |

固定量（所有变体相同）：`budget_tokens: 2048`（`M_memory.yaml`）、seed、数据子集、ckpt。
报告量（`M_memory.yaml::report`）：`kept_tokens / participating_tokens / select_ms /
attn_ms / step_ms / peak_mem / miou / smallclass_iou`。

**消融是必做的**：M1→M2→M3 是逐级加码（`pos_dedup` → `+priority` → `neighborhood`），
必须能看到每一级相对上一级的增量；同时每个创新变体都要能赢 `topk_fixed`，
否则位置信息就没有贡献。`nocompress_ref` 只划定上限，不参与排名。

---

## 1. 上卡前置（沿用创新点 1 文档的 env + 路径）

```bash
conda activate piers-omni        # M 跑在 omni 侧（继承 MemoryBank/SMR 在此 env）
nvidia-smi                       # 记 driver/CUDA/显存
piers_ljj doctor                  # torch cuda=True，sam2 路径正确
cp configs/piers/paths.example.yaml configs/piers/paths.yaml   # 填 omni_ckpt + data_root
piers_ljj validate-data --paths configs/piers/paths.yaml --sha256
# 期望：errors 为空；ckpt SHA 与登记值一致
```

E0 基线（M 的共同起点，必须先是 `completed_real`）：

```bash
piers_ljj run --config configs/piers/effective_start.yaml --dry-run   # 验依赖
piers_ljj run --config configs/piers/effective_start.yaml             # 真实 E0（GPU）
# 确认 runs/E0-effective-start/status.json 为 completed_real
piers_ljj sweep --suite M --check-only --runs-root runs                # 应报 missing: none
```

---

## 2. M 运行（七变体，逐个跑，顺序随意但建议按表序）

### 2.1 单变体命令（以 M1 为例）

`M_memory.yaml` 的 `variants:` 是一次声明七个；运行时用环境变量/覆盖值一次跑一个
（避免一次全跑显存爆炸，也方便断点续跑）：

```bash
# 方法：复制 config 改 variants 为单元素（保留 seed/budget 不变），逐个 run
# 例：configs/piers/M_memory_pos_dedup.yaml  (variants: [pos_dedup])
piers_ljj run --config configs/piers/M_memory_pos_dedup.yaml --dry-run
piers_ljj run --config configs/piers/M_memory_pos_dedup.yaml
```

每个变体输出（`runs/M-memory/` 下按变体分子目录或 `metrics.json` 区分）：

| 文件 | 内容 |
|---|---|
| `status.json` | `completed_real` + `synthetic: false` |
| `metrics.json` | `miou / smallclass_iou` + `kept_tokens / participating_tokens` + `select_ms / attn_ms / step_ms / peak_mem` |
| `plan.json`（建议落盘） | `bridge.plan_selection` 的 `kept_ids/dropped_ids/smr_params/report`（CPU 侧计划，可复现） |

### 2.2 必须记录的诚实口径

- `kept_tokens` = 实际参与 attention 的 token 数；`participating_tokens` 同值并列报告
  （若实现只是把 dense tensor 置零而非真 gather，**必须**在报告中声明无加速，只谈精度）。
- `nocompress_ref` 的 token 数天然更大，**不得**与其它变体比“效率”，只比精度上限。
- frame-index 永不作为 ERP 位置：每个 history token 必须带 `(image_id, fov_id, erp_rect)`；
  跨图/跨 batch 不得共享 SMR cache（key = `image_id + memory_id + strategy + budget`）。

### 2.3 评测与汇总

```bash
# UDA 语义：dataset-level 混淆矩阵 -> mIoU（不是每图平均）
piers_ljj evaluate --pred-npz runs/M-memory/<variant>/pred.npz \
  --gt-npz runs/M-memory/<variant>/gt.npz --task uda --n-cls 13 \
  --out runs/M-memory/<variant>/metrics.json

piers_ljj summarize --runs-root runs --out runs/summary.json
# 只收 completed_real + synthetic=false；dry-run/pending_gpu/blocked 不聚合
```

---

## 3. 结果判读（写论文 §3(2) 与消融表用）

1. **主表**：固定 budget 下各变体 `miou / smallclass_iou` + 代价四件套；
   `smallclass_iou` 是 M2/M3 的主战场（边界/稀有类配额的意义在此）。
2. **逐级增量**：`topk_fixed → pos_dedup → pos_dedup_priority → neighborhood`
   每一跳都要有数；若某跳为负，如实报告并分析（例如邻域排序在稀疏场景失效）。
3. **参照位**：`nocompress_ref` 单列，说明压缩带来的精度损失上界。
4. **失败即结论**：若创新变体打不过 `topk_fixed`，结论就是“位置约束在该预算/数据下无增益”，
   不得调参掩盖，不得换子集重跑（固定种子、固定子集、无网格搜索，见 gpu_runbook §8-9）。

## 附录：故障速查

| 现象 | 最可能原因 | 查哪里 |
|---|---|---|
| `BLOCKED missing deps ['E0']` | E0 未跑完 | `runs/E0-effective-start/status.json` |
| 变体间 token 数对不上 | 有人改了 budget/seed | `resolved_config.yaml` 的 `config_hash` 是否一致 |
| `participating` 与 `kept` 不一致 | gather 语义不清 | 查 omni 侧 SMR 调用是真 gather 还是置零 |
| 跨图结果串扰 | SMR cache key 缺 image_id | 检查 key 组成 |
