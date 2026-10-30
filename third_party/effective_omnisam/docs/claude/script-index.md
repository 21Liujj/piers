# Script Index

## Active Entrypoints

| Script | Purpose | Stage | Dataset | Called By | Status |
|---|---|---|---|---|---|
| `scripts/run_plugin_final_2000_base_vs_ADEtrain.sh` | **最终对比实验**：baseline vs A+D_train+E_train，2000张，seed=1 | PL+Train+Val | DensePASS | 手动启动 | ACTIVE |
| `scripts/run_ablation_all5.sh` | 全插件消融（A+D+E full），直接调 run_adaptation.py | PL+Train | Cityscapes→DensePASS | 手动启动 | ACTIVE |
| `scripts/run_ablation_base.sh` | 基线消融，调 omnisam_base.yaml | PL+Train | Cityscapes→DensePASS | 手动启动 | ACTIVE |
| `scripts/run_ablation_plus1_confmask.sh` | A 模块消融 | PL+Train | Cityscapes→DensePASS | 手动启动 | ACTIVE |
| `scripts/run_ablation_plus4_efficienttam.sh` | D 模块消融 | PL+Train | Cityscapes→DensePASS | 手动启动 | ACTIVE |
| `scripts/run_ablation_plus5_efficientsam2_sparse.sh` | E 模块消融 | PL+Train | Cityscapes→DensePASS | 手动启动 | ACTIVE |
| `scripts/run_adaptation_uda_densepass.sh` | UDA 智能续跑（DensePASS），含断点复用逻辑 | PL+Train | Cityscapes→DensePASS | 手动启动 | ACTIVE |
| `scripts/resume_adaptation_uda_densepass.sh` | UDA 恢复/校验（DensePASS） | Resume | Cityscapes→DensePASS | 手动启动 | ACTIVE |
| `scripts/validate_densepass.sh` | DensePASS 独立验证 | Val | DensePASS | 手动启动 | ACTIVE |
| `scripts/validate_wildpass.sh` | WildPASS 独立验证 | Val | WildPASS | 手动启动 | ACTIVE |
| `scripts/reproduce_densepass.sh` | 复现官方 omnisam_s baseline | Val | DensePASS | 手动启动 | ACTIVE |
| `scripts/compare_val_two_ckpts.sh` | 两个 ckpt 对比验证 | Val | DensePASS | 手动启动 | ACTIVE |
| `scripts/val_from_ckpt_zoo.sh` | 通过 ckpt_zoo 选ckpt跑验证 | Val | DensePASS/WildPASS | 手动启动 | ACTIVE |
| `scripts/run_adaptation_uda_wildpass.sh` | UDA (WildPASS)，输出目录独立 | PL+Train | Cityscapes→WildPASS | 手动启动 | ACTIVE |
| `scripts/extra_exp/wildpass/run_wildpass_exp.sh` | WildPASS 补充实验总入口 | Val+Train | WildPASS | 手动启动 | ACTIVE |
| `scripts/extra_exp/standford2d3d/run_standford2d3d_exp.sh` | Stanford360_SPAN8 补充实验总入口 | Val+Train | Stanford360 | 手动启动 | ACTIVE |

## Utilities

| Script | Purpose | Called By | Status |
|---|---|---|---|
| `scripts/run_adaptation_nohup.sh` | 后台启动 run_adaptation.py 通用封装 | 手动 | ACTIVE |
| `scripts/run_plugin_ablation_seed1_triplet.sh` | seed=1 三组消融（最终实验核心前序） | 手动 | ACTIVE |
| `scripts/run_plugin_ablation_base_only.sh` | 单独 baseline 小样本消融 | 手动 | ACTIVE |
| `scripts/run_plugin_ablation_base_plus5_train_only.sh` | 单独 E(train-only) 小样本消融 | 手动 | ACTIVE |
| `scripts/run_plugin_ablation_small_sample.sh` | 全插件小样本消融（A/D/E） | 手动 | ACTIVE |
| `scripts/run_ablation_depth_prior_50.sh` | 深度先验三路消融（50张） | 手动 | ACTIVE |
| `scripts/time_ablation/run_time_ablation_main4.sh` | 速度消融主脚本（4组 baseline/mem_only/aps_only/both） | 手动 | ACTIVE |
| `scripts/time_ablation/run_formal_paper_align_background.sh` | 论文对齐 Layer-2 速度消融，后台版 | 手动 | ACTIVE |
| `scripts/time_ablation/run_quick_layer1_background.sh` | 速度消融 Layer-1 快速后台版 | 手动 | ACTIVE |
| `scripts/time_ablation/run_quick_layer1_recheck_once_background.sh` | 速度消融 Layer-1 重新检查后台版 | 手动 | ACTIVE |
| `scripts/time_ablation/run_formal_layer2_background.sh` | 速度消融 Layer-2 正式后台版 | 手动 | ACTIVE |
| `scripts/time_ablation/cleanup_unused_pseudo_labels.sh` | 清理旧伪标签目录（dry-run模式） | 手动 | ACTIVE |
| `scripts/watch_latest_grounding_log.sh` | tail 最新 grounding 日志 | 手动 | ACTIVE |
| `scripts/extra_exp/wildpass/wildpass_default_ckpts.inc.sh` | WildPASS 默认 ckpt 配置（被 wildpass 脚本 source） | wildpass 脚本 | ACTIVE |

## Text/Multimodal Experiment Scripts

| Script | Purpose | Called By | Status |
|---|---|---|---|
| `scripts/run_text_ablation_two_stage.sh` | 文本模态两阶段消融（V1 logit_add A0-A5） | 手动 | CANONICAL |
| `scripts/run_multimodal_ablation_a0_a4.sh` | 文本 A0-A5 消融 launcher | 手动 | CANONICAL |
| `scripts/run_multimodal_stage1_stage2_abcd.sh` | 文本多模态 A/B/C/D 两阶段实验 | 手动 | CANONICAL |
| `scripts/run_multimodal_cd_sweep.sh` | C/D 参数 sweep（tau_t/gate_mode） | 手动 | CANONICAL |
| `scripts/run_depth_multimodal_ablation.sh` | 深度+多模态联合消融 | 手动 | CANONICAL |
| `scripts/run_text_grounding.sh` | V3 grounding 实验统一入口 | 手动 | CANONICAL |
| `scripts/run_text_grounding_diagnose.sh` | V3 grounding 诊断实验 | 手动 | CANONICAL |
| `scripts/run_text_crossattn_smoke50.sh` | V2 cross-attn 50张smoke实验 | 手动 | CANONICAL |
| `scripts/run_text_crossattn_fixed200.sh` | V2 cross-attn 200张固定实验 | 手动 | CANONICAL |
| `scripts/run_text_crossattn_diagnose50.sh` | V2 cross-attn 50张诊断实验 | 手动 | CANONICAL |
| `scripts/run_text_crossattn_diag_matrix50.sh` | V2 cross-attn 矩阵诊断50张 | 手动 | CANONICAL |
| `scripts/run_text_semantic_fixed200.sh` | V1 语义 200张固定实验 | 手动 | CANONICAL |
| `scripts/run_text_semantic_alpha_grid.sh` | V1 alpha grid search | 手动 | CANONICAL |
| `scripts/run_text_smoke_fixed50.sh` | 文本 smoke 50张实验 | 手动 | CANONICAL |
| `scripts/run_text_grounding_fixed20_all.sh` | V3 grounding fixed20 all variants | 手动 | CANONICAL |
| `scripts/run_text_grounding_fixed20_c1.sh` | V3 grounding fixed20 C1 real only | 手动 | CANONICAL |
| `scripts/run_text_grounding_fixed50_all.sh` | V3 grounding fixed50 all variants | 手动 | CANONICAL |

## Legacy Scripts

| Script | Purpose | Replacement | Reason Kept |
|---|---|---|---|
| `scripts/auto_iterative_pl_train_overnight.sh` | 迭代式 PL 精化+训练（旧方案，7轮迭代，未收敛） | run_plugin_final_2000... | git tracked，实验有日志记录 |
| `scripts/run_teacher60_36_newpd_overnight.sh` | 从 60.36 teacher 生成新 PL 再训练（旧方案） | run_plugin_final_2000... | 有独立历史价值 |
| `scripts/run_adaptation_reuse_epoch1_optimized.sh` | 复用 epoch1 PL 优化训练（早期方案） | run_adaptation_uda_densepass.sh | 探索阶段历史脚本 |
| `scripts/uda_densepass_originalish_from_60_36.sh` | 对齐早期风格 UDA（无 IMPROVEMENTS，conf_thr=-1） | run_adaptation_uda_densepass.sh | git tracked |
| `scripts/time_ablation/run_time_ablation_smoke.sh` | 速度消融 smoke 版（已被 main4 替代） | run_time_ablation_main4.sh | 有参考价值 |
| `scripts/time_ablation/run_time_ablation_one.sh` | 速度消融单组版 | run_time_ablation_main4.sh | 有参考价值 |
| `scripts/time_ablation/run_time_ablation_staged.sh` | 速度消融分阶段版 | run_time_ablation_main4.sh | 有参考价值 |
| `scripts/time_ablation/run_time_ablation_patch_internal.sh` | patch 内部计时消融 | run_time_ablation_main4.sh | 有参考价值 |
| `scripts/time_ablation/run_time_ablation_memory_internal.sh` | memory 内部计时消融 | run_time_ablation_main4.sh | 有参考价值 |

## Deleted Scripts

| Script | Reason | Replacement | Deleted At |
|---|---|---|---|
| — | — | — | — |
