# Document Index

## Canonical Documents

| Document | Purpose | Update Rule |
|---|---|---|
| `CLAUDE.md` (本地) | 项目总览，每次进入仓库自动加载 | 实验结论确认后更新，勿频繁改动 |
| `docs/claude/script-index.md` | Shell 脚本分类索引 | 新增/删除脚本时同步更新 |
| `docs/claude/document-index.md` | 文档分类索引（本文件） | 新增/删除文档时同步更新 |
| `docs/claude/log-index.md` | 实验日志索引 | 新实验完成后追加 |
| `docs/claude/legacy-assets.md` | legacy 资产登记 | 移动或归档时同步更新 |
| `docs/claude/repository-organization.md` | 仓库目录结构规范 | 规范变更时更新 |
| `docs/claude/experiment-conventions.md` | 实验命名与配置规范 | 待补充 |
| `docs/claude/experiment-operations.md` | 实验操作 SOP | 待补充 |
| `docs/plugin_ablation_readme.md` | 插件系统说明（A/D/E），含入口和配置说明 | 插件变更时更新 |
| `docs/efficient_sam2_style_sparse_memory_retrieval_zh.md` | E 模块 (SMR) 详细设计文档 | E 模块修改时更新 |
| `docs/text_modality_roadmap_zh.md` | 文本模态 V1→V3 路线图 | 文本模态有新结论时更新 |
| `docs/text_modality_v1_logit_add_zh.md` | V1 logit_add 详细方案 | 结论确认后封存 |
| `docs/text_modality_v3_grounding_zh.md` | V3 text_to_image_grounding 方案 | 结论确认后封存 |
| `logs/实验脚本编写与执行规范.md` | 实验脚本统一规范（命名/日志/断点） | 规范变更时更新（建议迁移到 docs/claude/） |
| `logs/实验结果记录汇报_截至2026-04-10.md` | 早期实验结果汇总（截至4月10日） | 历史文档，不再更新 |
| `logs/伪标签生成及训练推理加速的算法优化.md` | 加速算法分析文档（D/E 模块依据） | 已封存，结论已在 CLAUDE.md 体现 |
| `logs/abcd_summary.md` | 文本 ABCD 实验对比汇总 | 文本实验最终结论汇总 |

## Reports

| Document | Purpose | Status |
|---|---|---|
| `docs/reports/` | 正式实验报告目录 | 待创建 |

## Legacy Documents (已归档在 markdown_archive/)

| Document | Reason Kept | Canonical Replacement |
|---|---|---|
| `markdown_archive/IMPROVEMENTS.md` | git tracked（已从工作树删除，仅存档） | CLAUDE.md |
| `markdown_archive/README.md` | git tracked（原始 OmniSAM README） | docs/claude/* |
| `markdown_archive/experiment.md` | git tracked（早期实验记录） | docs/claude/experiment-conventions.md |
| `markdown_archive/技术方案_RGBText_OPS路线.md` | 文本多模态 RGBText 完整方案（40KB，含调参/诊断/结论） | 保留原位，待论文阶段参考 |
| `markdown_archive/docs/efficient_sam2_style_sparse_memory_retrieval.md` | 空文件（0 bytes），原路归档 | docs/efficient_sam2_style_sparse_memory_retrieval_zh.md |
| `markdown_archive/docs/efficienttam_memory_compression.md` | EfficientTAM Memory Compression 原始设计（英文） | docs/plugin_ablation_readme.md |
| `markdown_archive/docs/efficienttam_memory_compression_zh.md` | EfficientTAM Memory Compression 中文设计 | docs/plugin_ablation_readme.md |
| `markdown_archive/scripts/README_train.md` | 训练脚本说明（旧版） | docs/claude/script-index.md |
| `markdown_archive/scripts/time_ablation/README.md` | 速度消融说明（旧版） | docs/claude/script-index.md |
| `markdown_archive/sam/` | SAM2 库原始 README 集合 | 无需替代，纯上游文档 |

## Deleted Documents

| Document | Reason | Content Merged Into | Deleted At |
|---|---|---|---|
| `docs/efficient_sam2_style_sparse_memory_retrieval.md` | 0 bytes 空文件 | N/A（无内容） | 2026-06-03 |
