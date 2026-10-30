# Experiment Operations

## 1. 正式实验启动流程

### 最终配置对比实验（当前标准）

```bash
cd /root/autodl-tmp/OmniSAM-main
conda activate omnisam

# 最终 2000 张对比实验（A_Dtrain_Etrain vs baseline）
RUN_IN_BACKGROUND=1 \
SRC_CKPT=/root/autodl-tmp/omnisam_s.pth \
TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
bash scripts/run_plugin_final_2000_base_vs_ADEtrain.sh
```

日志：`logs/omnisam_final_2000/final_2000_base_vs_ADEtrain_<STAMP>.master.log`

### 单模块消融实验

```bash
# 消融某一模块
SRC_CKPT=... TGT_CKPT=... bash scripts/run_ablation_base.sh
SRC_CKPT=... TGT_CKPT=... bash scripts/run_ablation_plus1_confmask.sh
SRC_CKPT=... TGT_CKPT=... bash scripts/run_ablation_plus4_efficienttam.sh
SRC_CKPT=... TGT_CKPT=... bash scripts/run_ablation_plus5_efficientsam2_sparse.sh
```

### 验证

```bash
# DensePASS 验证
CKPT=/path/to/best.pth bash scripts/validate_densepass.sh

# WildPASS 验证
CKPT=/path/to/best.pth bash scripts/validate_wildpass.sh
```

---

## 2. 伪标签生成

伪标签由 `adaptation/engine.py` → `PseudoLabelGenerator.run()` 自动触发（run_adaptation.py 调用），
不需要单独运行伪标签生成脚本。

如需复用已有伪标签，设置：
```bash
REUSE_PL_DIR=<已有伪标签目录>
```

---

## 3. 实验结果确认流程

1. 查看 `<pd_root>/checkpoints/.../metrics.jsonl` 中的指标
2. 比较 `data_pd_exp_final_2000/*/metrics.jsonl` 确认 mIoU/mAcc 等
3. 在 `docs/claude/experiment-conventions.md` 的结论表更新
4. 更新 `CLAUDE.md` 第 5 节实验结果表

---

## 4. checkpoint 管理

| 文件 | 说明 |
|---|---|
| `best_sam2_s_iou<XX.XX>.pth` | 最佳 checkpoint |
| `last_sam2_s_13cls.pth` | 最后一轮 checkpoint |
| `resume_sam2_s.pth` | 续跑 checkpoint |
| `metrics.jsonl` | 每 epoch 指标记录 |

关键 checkpoints（不可删除）：
- `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/.../best_iou60.26.pth`
- `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/.../best_iou60.16.pth`
- `/root/autodl-tmp/omnisam_s.pth`（官方 source checkpoint）

---

## 5. 新实验注册

每次新实验完成后：

1. 在 `docs/claude/log-index.md` 的 Key Experiment Logs 表中追加
2. 若有新配置文件，在 `docs/claude/script-index.md` 追加
3. 若结论有更新，更新 `CLAUDE.md` 第 5 节
