# Repository Assets Cleanup Audit

**执行时间**: 2026-06-03  
**执行者**: Claude Code  
**磁盘状态（清理前）**: 可用 8.5G / 总 30G（overlay），仓库 42G（含数据集/ckpt）

---

## 关键实验保护路径 (PROTECTED_PATHS)

以下路径绝对不得删除：

```
/root/autodl-tmp/OmniSAM-main/data_pd_exp_final_2000/
/root/autodl-tmp/OmniSAM-main/configs/
/root/autodl-tmp/OmniSAM-main/docs/claude/
/root/autodl-tmp/OmniSAM-main/adaptation/
/root/autodl-tmp/OmniSAM-main/model/
/root/autodl-tmp/OmniSAM-main/dataset/
/root/autodl-tmp/OmniSAM-main/utils/
/root/autodl-tmp/OmniSAM-main/tools/
/root/autodl-tmp/OmniSAM-main/sam/
/root/autodl-tmp/OmniSAM-main/assets/
/root/autodl-tmp/OmniSAM-main/data/
/root/autodl-tmp/OmniSAM-main/data_extra/
/root/autodl-tmp/OmniSAM-main/aux_outputs/
/root/autodl-tmp/OmniSAM-main/markdown_archive/
/root/autodl-tmp/OmniSAM-main/logs/omnisam_final_2000/
/root/autodl-tmp/OmniSAM-main/logs/omnisam_time_ablation/
/root/autodl-tmp/OmniSAM-main/logs/omnisam_time_ablation_seed1_triplet/
/root/autodl-tmp/OmniSAM-main/logs/wildpass_results/
/root/autodl-tmp/OmniSAM-main/logs/Standford2d3d_results/
/root/autodl-tmp/OmniSAM-main/logs/abcd_summary.md
/root/autodl-tmp/OmniSAM-main/logs/abcd_summary.csv
/root/autodl-tmp/OmniSAM-main/logs/validate_densepass_baseline_20260407_131045.log
/root/autodl-tmp/OmniSAM-main/logs/validate_wildpass_baseline_20260407_131821.log
/root/autodl-tmp/OmniSAM-main/logs/uda_densepass_20260408_143548.log
/root/autodl-tmp/OmniSAM-main/logs/compare_omnisam_vs_60_36.log
/root/autodl-tmp/OmniSAM-main/logs/compare_omnisam_vs_60_36.json
/root/autodl-tmp/OmniSAM-main/logs/multimodal_abcd_A_generate_only.log
/root/autodl-tmp/OmniSAM-main/logs/multimodal_abcd_B_visual_gate_only.log
/root/autodl-tmp/OmniSAM-main/logs/multimodal_abcd_C_text_gate_hard_no_adapter.log
/root/autodl-tmp/OmniSAM-main/logs/multimodal_abcd_D_text_gate_hard_with_adapter.log
/root/autodl-tmp/OmniSAM-main/logs/cdsweep_20260422_235848.master.log
/root/autodl-tmp/OmniSAM-main/logs/cdsweep_20260422_235848.topk.csv
/root/autodl-tmp/OmniSAM-main/logs/cdsweep_20260422_235848.phase1_results.csv
/root/autodl-tmp/OmniSAM-main/ckpt_zoo/
/root/autodl-tmp/omnisam_s.pth  (仓库外)
```

---

## 分类汇总

### ACTIVE（当前使用，原路径保留）
- 所有正式实验入口脚本（详见 script-index.md）
- `configs/` 下所有 YAML
- `docs/claude/` 所有知识库文档
- `logs/omnisam_final_2000/`，`logs/omnisam_time_ablation*/`
- 关键日志（见 log-index.md）

### CANONICAL（唯一标准版本，原路径保留）
- `docs/plugin_ablation_readme.md`（A/D/E 插件系统说明）
- `docs/efficient_sam2_style_sparse_memory_retrieval_zh.md`（E 模块设计）
- `docs/text_modality_roadmap_zh.md`（文本路线图）
- `logs/实验脚本编写与执行规范.md`（规范文档）
- `logs/实验结果记录汇报_截至2026-04-10.md`（早期结果汇报）
- `logs/伪标签生成及训练推理加速的算法优化.md`（加速分析）
- `markdown_archive/技术方案_RGBText_OPS路线.md`（文本方案完整记录）

### SAFE_DELETE（本轮自动删除）

见 `docs/claude/repository-assets-safe-delete.tsv`

| 文件 | 原因 |
|---|---|
| `logs/multimodal_cdsweep_only_p1_r5_tt0.08_gmhard_wa0.02_D_text_gate_hard_with_adapter.log` | 零字节文件 |
| `logs/uda_densepass_reuse_epoch1_opt_20260409_194826.log` | 仅含启动PID，无有效训练内容，被195415取代 |
| `logs/uda_densepass_reuse_epoch1_opt_20260409_195036.log` | 仅含启动PID，无有效训练内容，被195415取代 |
| `docs/efficient_sam2_style_sparse_memory_retrieval.md` | 0 bytes 空文件（已有 _zh.md 版本） |
| `logs/cdsweep.pid` / `logs/cdsweep_only.pid` / `logs/cdsweep_only_clean.pid` / `logs/multimodal_abcd.pid` | 历史 PID 文件，进程早已结束 |
| `logs/uda_densepass_originalish_20260410_133124.pid` / `logs/uda_densepass_originalish_20260410_154811.pid` | 历史 PID 文件（git tracked，需确认） |

### MANUAL_REVIEW（人工确认）

见 `docs/claude/repository-assets-manual-review.tsv`

| 文件 | 原因 |
|---|---|
| `wandb/` (~90MB) | 不确定哪些 run 已同步，需确认后删除 |
| `runs/` (~13MB TensorBoard events) | 早期实验，不确定是否仍需要 |
| `logs/cdsweep_only_p1_r5_tt0.08_gmhard_wa0.02.log` (522B) | 疑似失败启动 |
| `logs/multimodal_abcd_A_baseline.log` (431B) | 疑似失败启动 |
| `logs/multimodal_abcd_20260422_144235.master.log` (135KB) | 中途中断版本，含 A_generate_only 数据 |
| `logs/multimodal_abcd_20260422_200620.master.log` (94KB) | 中途中断版本，被 212119 取代 |
| `data_pd_cg_src_2000_01_ori/` | 空目录 |
| `ablation_outputs/` | 空目录 |
| `exp2/` | 空目录 |
| `sam2/` | 仅含 __pycache__，疑似废弃 |

### MERGE_THEN_ARCHIVE（本轮不移动，已登记）

以下文档内容已在 docs/claude/ 中被新文档覆盖，但本轮保留原路径：

| 文件 | 独立内容 | 已合并到 |
|---|---|---|
| `markdown_archive/docs/efficienttam_memory_compression_zh.md` | D 模块中文设计细节 | docs/plugin_ablation_readme.md（已覆盖） |
| `markdown_archive/scripts/README_train.md` | 旧版训练脚本说明 | docs/claude/script-index.md（已覆盖） |

---

## 本轮未处理内容

- checkpoint：不处理（data_pd_exp_final_2000/，ckpt_zoo/，logs/time_ablation/中的ckpt）
- 伪标签：不处理（data_pd_exp_final_2000/ 下的 PL）
- 大型日志压缩：wandb/runs/ 待人工确认后处理
- 算法代码：不修改
