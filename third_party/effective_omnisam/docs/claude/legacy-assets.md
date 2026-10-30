# Legacy Assets

## Legacy Scripts (已移入 scripts/legacy/ 或标记为 legacy)

| Path | Purpose | Status | Replacement | Safe to Delete Later |
|---|---|---|---|---|
| `scripts/auto_iterative_pl_train_overnight.sh` | 迭代式 PL 精化方案（7轮），最终未采用 | git tracked，保留原路径 | `scripts/run_plugin_final_2000_base_vs_ADEtrain.sh` | 是（git commit 后可删） |
| `scripts/run_teacher60_36_newpd_overnight.sh` | 从 60.36 teacher 重生成 PL 再训练 | 历史方案，已超越 | `scripts/run_plugin_final_2000_base_vs_ADEtrain.sh` | 是 |
| `scripts/run_adaptation_reuse_epoch1_optimized.sh` | 复用 epoch1 PL 的早期优化方案 | 已被 uda_densepass.sh 替代 | `scripts/run_adaptation_uda_densepass.sh` | 是 |
| `scripts/uda_densepass_originalish_from_60_36.sh` | 对齐早期风格 UDA（conf_thr=-1） | git tracked，有历史对照价值 | `scripts/run_adaptation_uda_densepass.sh` | 否（git tracked） |
| `scripts/time_ablation/run_time_ablation_smoke.sh` | 速度消融 smoke 版 | 已被 main4 取代 | `scripts/time_ablation/run_time_ablation_main4.sh` | 是 |
| `scripts/time_ablation/run_time_ablation_one.sh` | 速度消融单组版 | 已被 main4 取代 | `scripts/time_ablation/run_time_ablation_main4.sh` | 是 |
| `scripts/time_ablation/run_time_ablation_staged.sh` | 速度消融分阶段版 | 已被 formal 版取代 | `scripts/time_ablation/run_formal_paper_align_background.sh` | 是 |
| `scripts/time_ablation/run_time_ablation_patch_internal.sh` | patch 内部计时消融 | 调试用，已无需重跑 | `scripts/time_ablation/run_time_ablation_main4.sh` | 是 |
| `scripts/time_ablation/run_time_ablation_memory_internal.sh` | memory 内部计时消融 | 调试用，已无需重跑 | `scripts/time_ablation/run_time_ablation_main4.sh` | 是 |

## Legacy Documents (已在 markdown_archive/ 归档)

| Path | Purpose | Status | Canonical Replacement | Safe to Delete Later |
|---|---|---|---|---|
| `markdown_archive/IMPROVEMENTS.md` | 原始 OmniSAM 改进文档 | git tracked（已从工作树删除） | CLAUDE.md | 否（git tracked） |
| `markdown_archive/README.md` | 原始 README | git tracked | docs/claude/* | 否（git tracked） |
| `markdown_archive/experiment.md` | 早期实验记录 | git tracked | docs/claude/experiment-conventions.md | 否（git tracked） |
| `markdown_archive/技术方案_RGBText_OPS路线.md` | 文本多模态完整方案（含调参记录） | 保留，论文参考 | 无需替代 | 否 |
| `markdown_archive/docs/efficient_sam2_style_sparse_memory_retrieval.md` | 空文件 | 0 bytes | docs/efficient_sam2_style_sparse_memory_retrieval_zh.md | 是 |
| `markdown_archive/docs/efficienttam_memory_compression.md` | D 模块英文设计 | 已由 plugin_ablation_readme.md 覆盖 | docs/plugin_ablation_readme.md | 是 |
| `markdown_archive/docs/efficienttam_memory_compression_zh.md` | D 模块中文设计 | 已由 plugin_ablation_readme.md 覆盖 | docs/plugin_ablation_readme.md | 是 |
| `markdown_archive/scripts/README_train.md` | 旧版训练脚本说明 | 已由 script-index.md 覆盖 | docs/claude/script-index.md | 是 |
| `markdown_archive/scripts/time_ablation/README.md` | 旧版速度消融说明 | 已由 script-index.md 覆盖 | docs/claude/script-index.md | 是 |
| `markdown_archive/sam/` | SAM2 上游文档 | git tracked，上游库文档 | 无需替代 | 否（git tracked） |

## Failed Experiment Logs (已记录，可后续清理原始大日志)

| Path | Experiment | Error Summary | Config | Need to Keep |
|---|---|---|---|---|
| `logs/uda_densepass_reuse_epoch1_opt_20260409_194826.log` (4KB) | reuse epoch1 优化方案 #1 | 仅启动信息，进程在后台启动后实际训练在 195415 完成 | `run_adaptation_reuse_epoch1_optimized.sh` | 否 |
| `logs/uda_densepass_reuse_epoch1_opt_20260409_195036.log` (5KB) | reuse epoch1 优化方案 #2 | 仅启动信息，重复启动，实际训练在 195415 完成 | `run_adaptation_reuse_epoch1_optimized.sh` | 否 |
| `logs/multimodal_cdsweep_only_p1_r5_tt0.08_gmhard_wa0.02_D_text_gate_hard_with_adapter.log` (0B) | cdsweep_only r5 | 零字节，进程启动即失败 | cdsweep_only 系列 | 否 |
| `logs/multimodal_abcd_20260422_144235.master.log` (135KB) | 文本 ABCD 第一次启动 | 中途中断，被 200620 和 212119 取代，但含有独立 A_generate_only 数据 | run_multimodal_stage1_stage2_abcd.sh | 是（含 A_generate_only 信息） |
| `logs/multimodal_abcd_20260422_200620.master.log` (94KB) | 文本 ABCD 第二次启动 | 中途中断（B/C/D 未完成），被 212119 取代 | run_multimodal_stage1_stage2_abcd.sh | 参考保留 |
| `logs/omnisam_time_ablation/plugin_ablation_small_20260425_150116.master.log` 等 5 个早期 master | 小样本消融多次重启 | 早期配置迭代过程，被 153453 正式版取代 | 各早期脚本 | 是（参考） |

## Manual Review Required

| Path | Type | Reason | Suggested Action |
|---|---|---|---|
| `wandb/offline-run-*/` | wandb offline runs (~90MB) | 不确定是否已同步或有独立价值，仅包含训练 metrics | 人工确认后整体删除（metrics 已在 metrics.jsonl） |
| `runs/sam2_s_Cityscapes_13cls/events.out.tfevents.*` (~13MB) | TensorBoard events | 早期 UDA 实验 events，不确定是否已被 wandb 覆盖 | 人工确认后删除 |
| `logs/cdsweep_only_p1_r5_tt0.08_gmhard_wa0.02.log` (522B) | 疑似失败启动日志 | 极小体积，cdsweep_only 系列其余 r1-r4 都有完整内容 | 人工确认后删除 |
| `logs/multimodal_abcd_A_baseline.log` (431B) | 疑似失败启动 | 极小体积，ABCD 实验组成员 | 人工确认后删除 |
| `data_pd_cg_src_2000_01_ori/` | 空目录（0 bytes） | 目录存在但内容为空 | 确认无内容后删除目录 |
| `ablation_outputs/` | 空目录（0 bytes） | 目录存在但内容为空 | 确认无内容后删除目录 |
| `exp2/` | 空目录（0 bytes） | 目录存在但内容为空 | 确认后删除目录 |
| `sam2/` | 空目录（8KB，仅 __pycache__） | 似乎是废弃的 SAM2 新版实验目录 | 人工确认后删除 |
