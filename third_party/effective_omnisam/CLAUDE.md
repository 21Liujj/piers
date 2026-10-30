# OmniSAM — 全景语义分割无监督域适应

> 每次进入此仓库时自动加载。保持简洁：只放每次都值得读的信息。
> 详细内容见 `docs/claude/`，按需读取，不要无差别全加载。

---

## 1. Project Overview

**任务**：基于 OmniSAM 的全景语义分割无监督域适应（Panoramic Semantic Segmentation with Unsupervised Domain Adaptation）

注意：**panoramic segmentation（全景图语义分割）** ≠ **panoptic segmentation（实例级全景分割）**，不要混淆。

**设定**：
- 源域：Cityscapes（针孔图像，有像素级标签，13 类）
- 目标域：DensePASS（360° ERP 全景图，训练阶段无标签）
- 验证集：DensePASS val（100 张，有标签，仅用于评估）
- 额外数据集：WildPASS、Stanford360_SPAN8（补充实验）

**基础模型**：SAM2 (sam2.1_hiera_small，sam2_s backbone)，加 LoRA (r=4) 微调

**当前研究阶段**：
- 复现 OmniSAM 基线，已完成
- 插件模块消融，已完成主要实验
- 最终配置 `A_Dtrain_Etrain` 对比 baseline，已完成并确认结论

**仓库位置**：远程服务器 `/root/autodl-tmp/OmniSAM-main`（通过 paramiko SSH 访问）
- SSH: `root@connect.bjb2.seetacloud.com:18898`
- conda env: `omnisam`

---

## 2. Repository Structure

```
/root/autodl-tmp/OmniSAM-main/
├── run_adaptation.py        # UDA 训练主入口（argparse + YAML --config）
├── train.py                 # 源域预训练入口（OmniSAM，非 adapt 版本）
├── validation.py            # 验证入口（单模型）
├── validation_muti.py       # 多模型对比验证
├── adaptation/              # UDA 核心逻辑
│   ├── engine.py            # Engine：orchestrate PL + Train + Val
│   ├── trainer.py           # Trainer：train_one_epoch，loss 计算
│   ├── pl_generator.py      # PseudoLabelGenerator：伪标签生成
│   ├── strategies.py        # WithMemStrategy：forward() 含完整 loss 链路
│   ├── methods.py           # PrototypicalAdaptationEMA（FPA 原型库）
│   ├── patch_scheduler.py   # AdaptivePatchScheduler
│   ├── pl_update_controller.py
│   └── plugins/             # 插件系统（A/D/E/depth）
│       ├── manager.py       # PluginManager：统一开关管理
│       ├── pseudo_conf_mask.py  # A 模块
│       ├── depth_boundary_prior.py
│       └── depth_confidence.py
├── model/                   # 模型定义
│   ├── model.py             # OmniSAM / OmniSAM_adapt
│   ├── memory_bank.py       # MemoryBank（含 D/E 模块接口）
│   ├── modules/
│   │   ├── decoder_sam.py       # SegFormerHead / SegFormerHead_adaption
│   │   ├── memory_compressor.py # D 模块：SpatialMemoryCompressor
│   │   └── sparse_memory_retriever.py  # E 模块：SparseMemoryRetriever
│   ├── text_align.py        # 文本融合模块（多种 fusion 类型）
│   ├── text_semantic.py     # LightweightLanguageAdapter
│   └── depth_adapter.py     # LightDepthAdapter
├── dataset/                 # 数据集 loader
├── configs/
│   ├── ablations/           # 消融 YAML（omnisam_base.yaml 等）
│   ├── depth_prior/         # 深度先验配置
│   ├── experiments/         # 文本/深度实验配置
│   └── extra_exp/           # WildPASS/Stanford360 配置
├── scripts/                 # 启动脚本（按功能分类）
├── tools/                   # 独立工具脚本
├── data/                    # 数据集（cityscapes/DensePASS/WildPASS）
├── data_extra/              # 额外处理后数据（WildPASS13/Stanford360_SPAN8）
├── data_pd_exp_final_2000/  # 最终对比实验产物（PL + ckpt + metrics）★
├── logs/                    # 实验日志
│   └── omnisam_final_2000/  # 最终实验日志 ★
├── docs/
│   ├── claude/              # Claude 知识库（跨对话加载）
│   ├── reports/             # 正式实验报告
│   └── legacy/              # 已归档文档
├── sam/                     # SAM2 backbone（含 LoRA，旧版）
├── sam2/                    # SAM2（新版，部分实验）
├── assets/                  # 文本原型 .pt，prompt JSON
├── aux_outputs/             # 深度先验 .npy（Depth Anything V2 预生成）
├── markdown_archive/        # 已从工作树移除的 git tracked 文档
└── .claude/rules/           # Claude 项目规则
```

---

## 3. Core Algorithm Flow

```
Cityscapes（源域，有标签）
        ↓ source pretraining（train.py）
        ↓
Source checkpoint（omnisam_s.pth）
        ↓
Pseudo-label generation（adaptation/engine.py → PseudoLabelGenerator.run()）
    DensePASS 2000 张子集 → patch sequence（正向+反向滑窗）
    → LoRA encoder → MemoryBank → memory attention
    → SegFormerHead → softmax conf → uncertain mask → PL .png
        ↓
UDA adaptation（adaptation/engine.py → trainer.py → strategies.py）
    loss_sup（source CE）
    + loss_unsup（target CE，经 pseudo_conf_mask 过滤）
    + loss_fpa（FPA 原型对齐，PrototypicalAdaptationEMA）
        ↓
validation（validation.py）
    DensePASS val 100 张 → 双 pass 滑窗 → mIoU/mAcc/PixAcc/FWIoU/mDice/BFScore
```

---

## 4. Important Entry Points

| 功能 | 入口 | 说明 |
|---|---|---|
| UDA 训练 | `run_adaptation.py` | argparse + `--config YAML` |
| 源域预训练 | `train.py` | OmniSAM（非 adapt） |
| PL 生成 | `adaptation/engine.py` → `PseudoLabelGenerator.run()` | 在 Engine._run_pseudo_label() |
| 验证 | `validation.py` | 单模型 |
| 多模型对比 | `validation_muti.py` | 见 compare_omnisam_vs_60_36 |
| 模型构建 | `utils/build.py` → `build_model()` | 构建 OmniSAM_adapt + 数据集 |
| loss 计算 | `adaptation/strategies.py` → `WithMemStrategy.forward()` | loss_sup + loss_unsup + loss_fpa |
| checkpoint 加载 | `utils/checkpoint_adapt.py` → `smart_load_checkpoint()` | 兼容旧 ckpt |
| 指标计算 | `utils/metrics.py` | fast_hist, per_class_iu, boundary_f_score |

最终实验启动脚本：
```bash
# 最终对比实验（2000 张，seed=1）
bash scripts/run_plugin_final_2000_base_vs_ADEtrain.sh
# 消融实验
bash scripts/run_ablation_all5.sh  # 全部 5 个插件
```

---

## 5. Confirmed Experiment Summary

### 模块状态

| 模块 | 代码名 | 当前状态 | 配置文件 |
|---|---|---|---|
| A: pseudo_conf_mask | `pseudo_conf_mask` | **进入最终配置** | `omnisam_A_plus4train_plus5train.yaml` |
| D: Memory Compression | `efficient_tam_memory` | **进入最终配置（仅 train）** | 同上 |
| E: Sparse Memory Retrieval | `efficient_sam2_sparse` | **进入最终配置（仅 train）** | 同上 |
| 文本融合 | `text_fusion` | 默认 disabled，结论待确认 | `configs/experiments/text_modality/` |
| 深度先验 | `depth_adapter` | 默认 disabled，无稳定正向收益 | `configs/depth_prior/` |

### 已确认实验结果（2026-04-25/26，seed=1，DensePASS 2000 张）

| 指标 | A_Dtrain_Etrain | baseline | 差值 |
|---|--:|--:|--:|
| mIoU | 60.16 | 60.26 | -0.10 |
| mAcc | 71.42 | 71.32 | +0.10 |
| PixAcc | 88.63 | 88.69 | -0.06 |
| FWIoU | 80.71 | 80.76 | -0.05 |
| mDice | 72.72 | 72.80 | -0.08 |

**耗时**：

| 阶段 | A_Dtrain_Etrain | baseline | 节省 |
|---|--:|--:|--:|
| Train step | 4.08 s/it | 5.27 s/it | **22.5% 提速** |
| epoch_train_time | 24m57s | 35m0s | 10m3s |
| Memory attention | 93.6s | 263.0s | **64.4% 提速** |

---

## 6. Research Rules

1. **不要混淆** panoramic segmentation（全景图语义分割）与 panoptic segmentation（实例分割）
2. 不要混淆源域（Cityscapes，有标签）、目标域（DensePASS/WildPASS，无标签）和验证集
3. 不要用目标域测试标签参与训练
4. 不要覆盖 `data_pd_exp_final_2000/` 下的已有 checkpoint 和伪标签
5. 新增模块前必须确认插入点（MemoryBank / strategies / engine）
6. 每个实验必须保存配置 YAML 快照、seed、GPU、commit、启动命令
7. 速度实验必须包含 warm-up（已在脚本中用 `base_first` 做热身）
8. 必须区分 Train / PL / Val / E2E 时间，不能混在一起报告
9. 新增模块必须设计独立消融，默认 `enabled: false`
10. 结论必须由日志 / metrics.jsonl 支持，推测需标注"推测"

---

## 7. Repository Asset Rules

- 新正式实验脚本放在 `scripts/exp/` 或保持 `scripts/` 根目录风格（过渡期）
- 新数据准备脚本放在 `scripts/data/`
- 新评估脚本放在 `scripts/eval/`
- 新工具脚本放在 `scripts/utils/`
- 新实验日志放在 `logs/<EXPERIMENT_NAME>/`（不要散落在 logs/ 根目录）
- 不要创建命名不清晰的文件（`test.sh`, `new.sh`, `final2.sh`, 临时 Markdown 说明）
- 添加文件前先查阅 `docs/claude/script-index.md`、`docs/claude/document-index.md`、`docs/claude/log-index.md`
- 移动 legacy 资产前必须在 `docs/claude/legacy-assets.md` 中登记
- 不要直接删除未登记的资产

---

## 8. Documentation Index

按需读取，不要在每次对话中全部加载：

- Script index: `docs/claude/script-index.md`
- Document index: `docs/claude/document-index.md`
- Log index: `docs/claude/log-index.md`
- Legacy assets: `docs/claude/legacy-assets.md`
- Repository organization: `docs/claude/repository-organization.md`
- Experiment conventions: `docs/claude/experiment-conventions.md`
- Experiment operations: `docs/claude/experiment-operations.md`
- Cleanup audit: `docs/claude/repository-assets-cleanup-audit.md`
- Plugin system: `docs/plugin_ablation_readme.md`
- E module (SMR): `docs/efficient_sam2_style_sparse_memory_retrieval_zh.md`
- Text modality roadmap: `docs/text_modality_roadmap_zh.md`
