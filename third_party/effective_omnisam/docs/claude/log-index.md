# Log Index

## Key Experiment Logs

| Experiment | Log Path | Metrics Path | Config Path | Checkpoint Path | Status |
|---|---|---|---|---|---|
| **baseline (最终 2000 张)** | `logs/omnisam_final_2000/base_last_20260425_232941.log` | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | `configs/ablations/omnisam_base.yaml` | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/` | CANONICAL |
| **A_Dtrain_Etrain (最终 2000 张)** | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | `configs/ablations/omnisam_A_plus4train_plus5train.yaml` | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/` | CANONICAL |
| baseline (小样本消融) | `logs/omnisam_time_ablation/plugin_ablation_base_20260425_170541.log` | — | `configs/ablations/omnisam_base.yaml` | — | ACTIVE |
| E(train-only) 小样本 | `logs/omnisam_time_ablation/plugin_ablation_base_plus5_train_only_20260425_181335.log` | — | — | — | ACTIVE |
| seed=1 三组消融 | `logs/omnisam_time_ablation_seed1_triplet/plugin_ablation_seed1_triplet_20260425_185447.master.log` | — | — | — | ACTIVE |
| DensePASS 官方 baseline 复现 | `logs/validate_densepass_baseline_20260407_131045.log` | — | — | `/root/autodl-tmp/omnisam_s.pth` | CANONICAL |
| WildPASS 官方 baseline 复现 | `logs/validate_wildpass_baseline_20260407_131821.log` | — | — | `/root/autodl-tmp/omnisam_s.pth` | CANONICAL |
| UDA 首轮（60.36 ckpt） | `logs/uda_densepass_20260408_143548.log` | — | — | `pd_uda_densepass/checkpoints/...` | CANONICAL |
| baseline vs 60.36 对比验证 | `logs/compare_omnisam_vs_60_36.log` + `.json` | — | — | — | CANONICAL |
| originalish UDA（无 conf_mask） | `logs/uda_densepass_originalish_20260410_154811.log` | — | — | — | CANONICAL |
| 文本 ABCD 消融（master） | `logs/multimodal_abcd_20260422_212119.master.log` | — | — | — | CANONICAL |
| 文本 ABCD 个组日志 A-D | `logs/multimodal_abcd_A_generate_only.log` 等 4 个 | — | — | — | CANONICAL |
| 文本 CD sweep（master） | `logs/cdsweep_20260422_235848.master.log` | `logs/cdsweep_20260422_235848.topk.csv` | — | — | CANONICAL |
| 文本 CD sweep 各 run（r1-r12） | `logs/cdsweep_p1_r1_tt0.05_gmhard_wa0.02.log` ~ r12 | — | — | — | CANONICAL |
| WildPASS 补充实验 | `logs/wildpass_results/` | `logs/wildpass_results/summaries/` | `configs/extra_exp/wildpass/` | — | ACTIVE |
| Stanford360 补充实验 | `logs/Standford2d3d_results/` | `logs/Standford2d3d_results/summaries/` | `configs/extra_exp/standford2d3d/` | — | ACTIVE |
| 深度先验消融 | `logs/depth_prior/` | — | `configs/depth_prior/` | — | ACTIVE |
| 深度+多模态联合消融 | `logs/depth_mm_ablation/` | — | `configs/experiments/depth_prior/` | — | ACTIVE |
| V1 文本 200张固定实验 | `logs/text_fusion_fixed200/semantic_fixed200_e3_alpha02_0428_110637/` | — | `configs/experiments/text_modality/` | — | ACTIVE |
| V2 cross-attn smoke50 | `logs/text_crossattn_smoke50/` | — | `configs/experiments/text_modality_crossattn/` | — | ACTIVE |
| V2 cross-attn diag_matrix50 | `logs/text_crossattn_diag_matrix50/diag_matrix50_0428_161026/` | — | — | — | ACTIVE |
| V3 grounding 诊断 | `logs/text_grounding_diagnose/` | — | `configs/experiments/text_modality_grounding/` | — | ACTIVE |

## Archived Logs

| Original Path | Current Path | Reason Kept | Compressed |
|---|---|---|---|
| `logs/auto_iterative_pl_train_20260409_233022.log` | 原路径 | git tracked，迭代式PL方案历史记录 | No |
| `logs/uda_densepass_20260408_094541.log` | 原路径 | git tracked，首次UDA训练（无指标，记录过程） | No |
| `logs/uda_densepass_reuse_epoch1_opt_20260409_195415.log` | 原路径 | git tracked，reuse方案实验记录 | No |

## Deleted Logs

| Path | Size | Reason | Deleted At |
|---|---:|---|---|
| `logs/multimodal_cdsweep_only_p1_r5_tt0.08_gmhard_wa0.02_D_text_gate_hard_with_adapter.log` | 0 B | 零字节文件 | 2026-06-03 |
| `logs/uda_densepass_reuse_epoch1_opt_20260409_194826.log` | 4 KB | 仅包含启动信息（PID），未产生有效训练指标，被 195036 和 195415 取代 | 2026-06-03 |
| `logs/uda_densepass_reuse_epoch1_opt_20260409_195036.log` | 5 KB | 仅包含启动信息（PID），实际训练在 195415 完成 | 2026-06-03 |

## Manual Review

| Path | Size | Question |
|---|---:|---|
| `logs/multimodal_abcd_A_baseline.log` | 431 B | 仅 431 字节，可能是失败启动；但属于 abcd 实验组，先保留 |
| `logs/multimodal_cdsweep_p1_r1_tt0.05_gmhard_wa0.02_A_generate_only.log` | 1.7 KB | PL 生成中途日志，有参考价值 |
| `logs/cdsweep_only_p1_r5_tt0.08_gmhard_wa0.02.log` | 522 B | 疑似失败启动；cdsweep_only 系列 r5 只有此文件很小 |
| `runs/sam2_s_Cityscapes_13cls/events.out.tfevents.*` | ~13MB | TensorBoard events，来源为早期 UDA 实验，不确定是否仍需要 |
| `wandb/` | ~90MB | wandb offline runs，不确定哪些已同步/哪些有独立价值 |
