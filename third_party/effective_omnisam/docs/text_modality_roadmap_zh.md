# OmniSAM 文本模态路线图（V1 → V2 → V3）

本文档总结 OmniSAM 中文本模态从 **V1（logit_add）**、**V2（image_to_text_cross_attn）**到 **V3（text_to_image_grounding）** 的演进动机、结构差异、关键配置与诊断指标，便于论文/组会表述与复现实验。

> 术语约定  
> - `K`：类别数（如 Cityscapes-13：13）  
> - `feat_de`：语义解码器输出特征（典型形状 `[B,256,H',W']`）  
> - `l_vis`：视觉分割 logits（上采样后 `[B,K,H,W]`）  
> - `l_txt`：文本原型产生的 logits（上采样后 `[B,K,H,W]`）  
> - `l_ground`：V3 grounding logits（上采样后 `[B,K,H,W]`）  

---

## 1. 路线对比（核心差异）

### V1：`logit_add`（晚期 logit 级融合）
- **融合方式**：`final = l_vis + alpha * (lambda_txt * l_txt)`  
- **优点**：实现简单、对主干侵入小、方便做文本负控（real/shuffle/random）。  
- **典型失败模式**：`alpha_init=0` 会导致文本分支长期“静默”（梯度极弱/不动），或模型把 `l_txt` 当成可学习类锚点而非利用语言语义。

### V2：`image_to_text_cross_attn`（图像→文本 cross-attn 特征级融合）
- **融合方式**：图像 token 做 Query，文本原型做 K/V，cross-attn 输出作为残差加到图像特征上，再走视觉 head。  
- **优点**：文本对特征有直接影响（可通过 `beta`、feature delta、attention stats 证明分支“真的在动”）。  
- **诊断重点**：即使分支活跃，也可能出现 real/shuffle/random 表现几乎一致（语义未被利用）。

### V3：`text_to_image_grounding`（文本→图像主动 grounding）
- **融合方式**：类别文本 query 主动 attend 图像 token（Q=text，K/V=image），得到 grounded queries，再与图像 token 做 cosine 生成 dense grounding logits：  
  - `l_ground = logit_scale * cosine(image_tokens, grounded_queries)`  
  - `final = l_vis + gamma * l_ground`  
- **目标**：从“像素被动查询文本”转为“文本主动查询图像”，更接近 OpenWorldSAM 思路中的 grounding。

---

## 2. 负控实验设计（必要）

文本原型必须同时提供三种模式对照（同一切其它设置）：
- **real**：真实类别名 embedding  
- **shuffle**：类别与 embedding 随机置换（破坏语义-类别对应）  
- **random**：随机向量（完全无语义）

V3 的成功判据应至少满足其一：
- **强正向**：`real` 在最终指标或 text-only 指标上显著优于 `shuffle/random`  
- **弱正向**：最终 mIoU 接近，但 `real` 的 `text-only prediction`、attention map、或小类 IoU 更“语义化”

---

## 3. 关键诊断指标（iter/epoch 级）

### 通用（已有）
- `prediction_change_ratio`：`argmax(final) != argmax(visual)` 的像素占比  
- `text_logits_mean/std`、`visual_logits_mean/std`、`final_logits_mean/std`

### V2（image→text cross-attn）
- `cross_beta`、`cross_attn_entropy/mean/std/max`  
- `feature_delta_ratio`（融合前后特征变化幅度）

### V3（text→image grounding）
- **标量**：`grounding_gamma`、`grounding_logit_scale`  
- **梯度**：`grounding_gamma_grad_norm`、`grounding_logit_scale_grad_norm`、`query_self_attn_grad_norm`、`query_cross_attn_grad_norm`、`query_ffn_grad_norm`  
- **attention**：`text_to_image_attn_mean/std/entropy/max`  
- **logits**：`text_grounding_logits_mean/std`  
- **text-only**：`text_only_prediction_change_ratio`（`argmax(l_ground) != argmax(l_vis)`）

上述指标会写入：
- `checkpoints/.../metrics/text_dynamics.jsonl`（iter 级）
- `checkpoints/.../metrics/text_fusion_dynamics.jsonl`（epoch 汇总）
- `checkpoints/.../metrics.jsonl`（epoch 级机器可读汇总）

---

## 4. 推荐跑法（从快到慢）

1) **fixed20**：只用小子集做链路与诊断（先 real，再 real/shuffle/random）  
2) **fixed50**：扩大样本，观察负控是否开始分离  

脚本入口：
- V2：`scripts/run_text_crossattn_diagnose50.sh` / `scripts/run_text_crossattn_diag_matrix50.sh`  
- V3：`scripts/run_text_grounding_diagnose.sh`

---

## 5. 代码入口（定位实现）
- 文本模块：`model/text_align.py`  
- 主模型接线：`model/model.py`  
- 配置透传：`run_adaptation.py`（TrainConfig + flatten YAML） / `utils/build.py`  
- loss/日志：`adaptation/strategies.py` / `adaptation/trainer.py` / `adaptation/engine.py`

