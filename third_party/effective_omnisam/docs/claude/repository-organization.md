# Repository Organization

## Canonical Structure

```
OmniSAM-main/
├── run_adaptation.py        # UDA 训练主入口
├── train.py                 # 源域预训练入口
├── validation.py            # 单模型验证入口
├── validation_muti.py       # 多模型对比验证
├── adaptation/              # UDA 核心逻辑
├── model/                   # 模型定义
├── dataset/                 # 数据集 loader
├── utils/                   # 工具函数
├── tools/                   # 独立工具脚本（eval/数据准备/ckpt管理）
├── configs/
│   ├── ablations/           # 消融 YAML（最终配置）
│   ├── depth_prior/         # 深度先验实验配置
│   ├── experiments/         # 文本多模态/深度先验实验配置
│   └── extra_exp/           # WildPASS/Stanford360 配置
├── scripts/
│   ├── exp/                 # 【新】正式实验启动脚本（未来使用）
│   ├── data/                # 【新】数据准备脚本（未来使用）
│   ├── eval/                # 【新】评估脚本（未来使用）
│   ├── utils/               # 工具脚本（cleanup 等）
│   ├── templates/           # 脚本模板
│   ├── legacy/              # 已归档的历史脚本
│   ├── extra_exp/           # WildPASS/Stanford360 实验脚本
│   ├── time_ablation/       # 速度消融脚本
│   └── *.sh                 # 现有正式实验脚本（保留原路径）
├── docs/
│   ├── claude/              # Claude 知识库（本目录）
│   ├── reports/             # 正式实验报告
│   ├── legacy/              # 已归档的历史文档
│   └── *.md                 # 现有设计/算法文档
├── logs/
│   ├── omnisam_final_2000/  # 最终对比实验日志 ★
│   ├── omnisam_time_ablation/        # 速度消融日志
│   ├── omnisam_time_ablation_seed1_triplet/  # seed=1 消融日志
│   ├── wildpass_results/    # WildPASS 补充实验日志
│   ├── Standford2d3d_results/ # Stanford360 补充实验日志
│   ├── cleanup/             # 清理操作记录
│   └── *.log / *.md         # 历史实验日志和文档
├── markdown_archive/        # 已从工作树删除的 git tracked 文档副本
├── data_pd_exp_final_2000/  # 最终实验产物（PL + ckpt + metrics）★ 不可删除
├── data/                    # 数据集（Cityscapes/DensePASS）
├── data_extra/              # 处理后数据（WildPASS13/Stanford360_SPAN8）
├── aux_outputs/             # 深度先验 .npy
├── assets/                  # 文本原型 .pt，prompt JSON
├── ckpt_zoo/                # ckpt 软链接管理目录
└── sam/                     # SAM2 backbone（含 LoRA，核心代码）
```

## Rules

- **新正式实验脚本** → `scripts/exp/` (未来) 或保持根目录风格（当前过渡期）
- **新数据准备脚本** → `scripts/data/`
- **新评估脚本** → `scripts/eval/`
- **新工具脚本** → `scripts/utils/`
- **新实验日志** → `logs/<EXPERIMENT_NAME>/`（不要散落在 logs/ 根目录）
- **新文档/设计文档** → `docs/claude/` 或 `docs/`（视是否需要跨对话加载）
- **不要**创建命名不清晰的文件（`test.sh`, `new.sh`, `final2.sh`, `临时说明.txt`）
- **不要**直接删除未登记的文件（先在 legacy-assets.md 中登记）
- **不要**覆盖 `data_pd_exp_final_2000/` 下的现有结果

## Protected Paths (不可删除)

```
data/                                     # 原始数据集
data_extra/                               # 处理后数据集
aux_outputs/                              # 深度先验
data_pd_exp_final_2000/                   # 最终实验产物
configs/                                  # 所有配置文件
adaptation/                              # UDA 核心代码
model/                                   # 模型代码
utils/                                   # 工具代码
dataset/                                 # 数据集 loader
tools/                                   # 工具脚本
sam/                                     # SAM2 backbone
assets/                                  # 文本原型等
docs/claude/                             # 本知识库
markdown_archive/                        # 已归档文档（git tracked）
logs/omnisam_final_2000/                 # 最终实验日志
logs/omnisam_time_ablation*/             # 速度消融实验日志
logs/validate_densepass_baseline*.log    # 官方 baseline 验证日志
logs/uda_densepass_20260408_143548.log   # 首轮 UDA 日志
logs/compare_omnisam_vs_60_36.*          # baseline vs 60.36 对比日志
logs/abcd_summary.md / abcd_summary.csv  # ABCD 实验汇总
/root/autodl-tmp/omnisam_s.pth           # 官方 source checkpoint
```

## Experiment ID Convention

实验目录命名规范：

```
<experiment_name>_<YYYYMMDD_HHMMSS>
```

例如：
- `base_last_20260425_232941`
- `A_Dtrain_Etrain_20260425_232941`

## Changelog

| Date | Action | Operator |
|---|---|---|
| 2026-06-03 | 初次建立 docs/claude/ 目录，创建规范文档 | Claude Code |
| 2026-06-03 | 执行仓库资产清理：删除零字节和冗余启动日志 | Claude Code |
