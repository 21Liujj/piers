## OmniSAM 插件化消融说明（论文实验框架）

本仓库在 UDA/自训练链路中引入 3 个**默认关闭**的可插拔插件，用于伪标签质量提升与耗时加速，方便后续硕士论文做严格消融。

### 相关扩展文档（文本模态）

文本模态（V1→V3）方案与复现实验文档：
- `docs/text_modality_roadmap_zh.md`
- `docs/text_modality_v1_logit_add_zh.md`
- `docs/text_modality_v3_grounding_zh.md`

### 插件列表（3 个）

- **A. `pseudo_conf_mask`**：对 target 的 `loss_unsup` 做 pixel-wise 置信筛选，只在高置信像素上学习；支持 `min_valid_ratio/min_valid_pixels` 与 `fallback`（`skip/original/zero`）。
- **D. `efficient_tam_memory`**：对 memory tokens 做可插拔压缩（复用现有 `MemoryBank.configure_compression`），可分别控制训练/伪标签阶段是否启用。
- **E. `efficient_sam2_sparse`**：对 memory retrieval 做 token 选择（复用 `SparseMemoryRetriever`），在进入 `memory_attention` 前减少 tokens；带 cache 与安全回退。

### 关键实现位置（代码入口）

- **UDA 入口**：`run_adaptation.py`（新增 `--config/--opts` 读取 YAML 并覆盖 args，提取 `plugins:`）  
- **训练 unsup loss**：`adaptation/strategies.py`（`WithMemStrategy.forward()`）  
- **伪标签生成**：`adaptation/engine.py` → `PseudoLabelGenerator.run()`  
- **memory 压缩 / sparse retrieval**：`model/memory_bank.py`  
- **插件实现**：`adaptation/plugins/`  

### 如何开启/关闭（YAML + CLI）

本仓库支持 `--config` 加载 YAML（通过 OmegaConf），以及 `--opts` 做覆盖（逗号分隔 `k=v`，支持 dotted key）。

例如（开启所有保留模块）：

```bash
python run_adaptation.py --config configs/ablations/omnisam_all5_full.yaml \
  --data_root data --pd_root data_pd --dataset Cityscapes --backbone sam2_s \
  --tgt_ckpt exp2/Cityscapes/best_sam2_s_10.pth --src_ckpt exp2/Cityscapes/best_sam2_s_10.pth
```

或者用脚本：

```bash
DATA_ROOT=data PD_ROOT=data_pd TGT_CKPT=... SRC_CKPT=... bash scripts/run_ablation_all5.sh
```

### 消融配置（已生成）

目录：`configs/ablations/`

- `omnisam_base.yaml`
- `omnisam_base_plus1_confmask.yaml`
- `omnisam_base_plus4_efficienttam.yaml`
- `omnisam_base_plus5_efficientsam2_sparse.yaml`
- `omnisam_all5_full.yaml`

对应脚本：`scripts/run_ablation_*.sh`

### 重点日志指标（用于论文分析）

训练相关（每 iter）：\n
- `loss_sup` / `loss_unsup_raw` / `loss_unsup_masked` / `total_loss`\n
- `pseudo_valid_ratio` / `pseudo_mean_confidence` / `pseudo_filtered_ratio`\n
- `memory_tokens_before/after` / `memory_attention_time_ms`\n
- `sparse_tokens_before/after` / `sparse_keep_ratio`\n

伪标签生成相关（每次生成写入 `metrics/pl_timing.jsonl`）：\n
- `pseudo_label_generation_time`、`pl_*_time`、`valid_pseudo_ratio`\n

### Smoke test

快速功能自检（不依赖数据集）：\n

```bash
python tools/smoke_test_plugins.py
```

### 轻量实现说明与可扩展点

- `efficient_sam2_sparse` 当前做 **memory token selection**（不修改 attention kernel）；后续可扩展为真正 sparse attention / attention mask。\n
- `SparseMemoryRetriever` 的 cache（indices）当前不随 checkpoint 持久化；若论文需要严格可复现，可在 resume-state 中追加保存（注意体积）。\n

### 最终实验设计（全量 2000 张）

为兼顾论文结论可信度与工程时间成本，最终实验采用**短热身 + 正式双组对比**：

1. `base_first`（warmup，不参与最终对比）
   - `WARMUP_IMAGES=100`
   - `WARMUP_TRAIN_MAX_ITERS=20`
2. `A_Dtrain_Etrain`（正式）
3. `base_last`（正式基线）

最终结果只比较：`A_Dtrain_Etrain` vs `base_last`。  
该设计用于避免冷启动（CUDA/DataLoader/IO 预热）对正式结果的干扰。

#### 最终方案定义

- **A = `pseudo_conf_mask`**：主要作用于训练阶段的 `loss_unsup`（loss masking）
- **D = `efficient_tam_memory`**：仅训练阶段启用（`apply_in_train=true, apply_in_pl=false`）
- **E = `efficient_sam2_sparse`**：仅训练阶段启用（`apply_in_train=true, apply_in_pl=false`）

即：PL 阶段保持原始伪标签生成与置信/uncertain 机制，不引入 D/E 的 token 调度；训练阶段启用 A+D+E 以获取主要加速收益。

#### 对应脚本

- `scripts/run_plugin_final_2000_base_vs_ADEtrain.sh`

默认目录：

- 日志：`logs/omnisam_final_2000/`
- 输出：`data_pd_exp_final_2000/`

推荐启动命令（5090 32G）：

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=1 \
GPU=0 SEED=1 QUICK_IMAGES=2000 \
WARMUP_IMAGES=100 WARMUP_TRAIN_MAX_ITERS=20 \
NUM_EPOCHS=1 TRAIN_MAX_ITERS=0 \
AMP=true TF32=true \
BATCH_SIZE=6 PL_BATCH_SIZE=2 PL_NUM_WORKERS=6 \
SRC_CKPT=/root/autodl-tmp/omnisam_s.pth \
TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
bash scripts/run_plugin_final_2000_base_vs_ADEtrain.sh
```

如遇显存不足，可先降到：`BATCH_SIZE=4` 或 `PL_BATCH_SIZE=1`。

