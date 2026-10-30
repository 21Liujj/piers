# V3：Text-to-Image Query Grounding（`text_to_image_grounding`）方案说明

本文档描述 OmniSAM 的 V3 文本分支：让**类别文本 query 主动 attend 图像特征**（Q=text，K/V=image），生成 dense `text_grounding_logits`，再通过可学习 `gamma` 与 `visual_logits` 融合，用 real/shuffle/random 负控验证“真实语义是否进入分割决策”。

---

## 1. 核心结构（最小实现）

输入：
- `visual_feature = feat_de`：`[B,C,H',W']`（默认 `C=256`）
- `text_feature`：投影后的类别文本原型 `projected_text_proto`，形状 `[K,C]` 或 `[B,K,C]`

流程：
1) 图像 token 展平：`image_tokens = [B, H'W', C]`（可选 `kv_stride` 池化用于 K/V 降采样）
2) 构造 class queries：默认 `num_queries_per_class=1`（预留 tie-breaker / query-delta）
3) SoftPromptingTransformer：`SelfAttn (可选) → CrossAttn(Q=text, K/V=image) → FFN (可选)`
4) dense grounding logits：
   - `l_ground = logit_scale * cosine(image_tokens, grounded_queries)` → reshape `[B,K,H',W']`
5) 最终融合：
   - `final = l_vis + gamma * upsample(l_ground)`（到 `[B,K,H,W]`）

---

## 2. 关键参数（TrainConfig / YAML）

### 2.1 打开 V3
- `text_fusion.enabled: true`
- `text_fusion.type: text_to_image_grounding`

### 2.2 Grounding Transformer
- `text_ground_num_heads`
- `text_ground_num_layers`
- `text_ground_dropout`
- `text_ground_use_query_self_attn`
- `text_ground_use_ffn`
- `text_ground_ffn_ratio`

### 2.3 计算与稳定性
- `text_ground_kv_stride`：对 image K/V 下采样（节省算力/显存）
- `text_ground_attn_temperature`：对 text query 的温度（越小越尖）

### 2.4 融合与尺度
- `text_ground_gamma_init` / `text_ground_gamma_learnable`
- `text_ground_logit_scale_init` / `text_ground_logit_scale_learnable`

### 2.5 预留（默认不重要）
- `text_ground_num_queries_per_class`（默认 1）
- `text_ground_use_tie_breaker`
- `text_ground_query_delta_init`
- `text_ground_use_learnable_query_delta`
- `text_ground_return_attn`（默认 true，便于诊断/可视化）

---

## 3. Grounding 辅助损失（CE）

用途：让 `l_ground` 本身对 label 有可学习的监督信号，避免 grounding 退化成“无语义的类锚点”。

配置（YAML 的 `loss_text_grounding`）：
- `loss_text_grounding_enabled`（建议 fixed20/50 打开）
- `loss_text_grounding_weight`（默认 0.05）
- `loss_text_grounding_apply_source: true`（默认打开）
- `loss_text_grounding_apply_target: false`（默认关闭，预留接口）

---

## 4. 诊断日志（你需要看的字段）

### 4.1 “分支真的在动”
iter 级（`metrics/text_dynamics.jsonl`）：
- `grounding_gamma`、`grounding_logit_scale`
- `grounding_gamma_grad_norm`、`grounding_logit_scale_grad_norm`
- `query_cross_attn_grad_norm`（必须非 0）
- `text_grounding_logits_mean/std`
- `loss_text_grounding_ce`

### 4.2 “attention 是否有结构”
- `text_to_image_attn_entropy`（越低越尖锐；接近 `log(HW)` 表示近似均匀）
- `text_to_image_attn_max`

### 4.3 “text-only 是否有意义”
- `text_only_prediction_change_ratio`：`argmax(l_ground) != argmax(l_vis)` 的像素占比  
  - 注意：这只是“不同”，不等价于“更好”；仍需靠 real/shuffle/random 或 IoU 分析判读。

---

## 5. Debug 可视化（可选）

由以下开关控制（脚本会透传到 args）：
- `text_fusion.save_debug_vis: true`
- `text_fusion.save_vis_num: N`

输出目录（训练过程中写到 `pd_root` 下）：
- `vis_text_grounding_debug/`
  - `*_rgb.png`
  - `*_visual_pred.png` / `*_final_pred.png` / `*_text_pred.png`
  - `*_changed.png`
  - `*_attn_clsX.png`（按类别输出的注意力图）

---

## 6. 负控实验（必须）

同配置、同 seed、同 fixed 子集，仅切换 `text_proto.mode`：
- `grounding_C1_real_proto.yaml`
- `grounding_C4_shuffle_proto.yaml`
- `grounding_C5_random_proto.yaml`

判读：
- **强正向**：`real` 在最终指标或 `loss_ground` 收敛、attention map 结构上明显优于 `shuffle/random`
- **无效**：`real/shuffle/random` 完全重合，且 attention 近似均匀

---

## 7. 运行脚本（fixed20/50）

脚本：`scripts/run_text_grounding_diagnose.sh`

示例（fixed20，单跑 C1）：

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=0 \
CASE_SET=c1 \
WARMUP_PL=20 ITERS_PL=20 \
BATCH_SIZE=2 PL_BATCH_SIZE=2 \
GROUNDING_GAMMA_INIT=0.05 \
GROUNDING_NUM_LAYERS=2 \
GROUNDING_NUM_HEADS=4 \
GROUNDING_LOSS_WEIGHT=0.05 \
TEXT_ADAPTER_PROJECTOR_TYPE=linear \
MEM_COMPRESS_TRAIN_ONLY=1 \
bash scripts/run_text_grounding_diagnose.sh
```

示例（fixed20，跑 C1/C4/C5）：

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=0 \
CASE_SET=all \
WARMUP_PL=20 ITERS_PL=20 \
BATCH_SIZE=2 PL_BATCH_SIZE=2 \
GROUNDING_GAMMA_INIT=0.05 \
GROUNDING_NUM_LAYERS=2 \
GROUNDING_NUM_HEADS=4 \
GROUNDING_LOSS_WEIGHT=0.05 \
TEXT_ADAPTER_PROJECTOR_TYPE=linear \
MEM_COMPRESS_TRAIN_ONLY=1 \
bash scripts/run_text_grounding_diagnose.sh
```

---

## 8. 代码入口（定位实现）
- 模块：`model/text_align.py`（`TextToImageGroundingFusion`）
- 前向接线：`model/model.py`（`text_fusion_type == text_to_image_grounding`）
- loss/诊断：`adaptation/strategies.py`
- iter/epoch 落盘：`adaptation/trainer.py`、`adaptation/engine.py`

