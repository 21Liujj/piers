# 路线1（V1）：Text Prototype `logit_add` 融合方案说明

本文档是 OmniSAM 文本模态的 **路线1（V1）**：通过 OpenCLIP 文本原型在 logit 级别对视觉分割做加法融合（`logit_add`），并提供稳定的 real/shuffle/random 负控框架与诊断指标。

---

## 1. 目标与动机

在不改变主干（SAM2/OmniSAM encoder+decoder）结构的前提下，引入类别文本原型（OpenCLIP embedding）作为“语义先验”，对分割 logits 做轻量融合，并通过负控验证“是否真正利用语义”：
- **real**：真实类别名 embedding  
- **shuffle**：类别与 embedding 置换（破坏语义-类别对应）  
- **random**：随机向量（无语义）

---

## 2. 方法定义（公式）

给定：
- `l_vis ∈ R^{B×K×H×W}`：视觉分割 logits
- `l_txt ∈ R^{B×K×H×W}`：文本分支 logits（由图像特征与文本原型 cosine 得到）

融合：
- `final = l_vis + alpha * (lambda_txt * l_txt)`

其中：
- `lambda_txt`：整体文本贡献系数（实验层面/超参）  
- `alpha`：融合系数（可学习或固定初始化）

---

## 3. 关键实现模块与路径

### 3.1 文本原型加载
- 位置：`model/text_align.py` → `load_text_prototypes(...)`
- 输入：`assets/text_prototypes_openclip_text_latest.pt`
- 输出：`text_prototypes [K, D_txt]`（可选 normalize）

### 3.2 文本 logits 头（投影 + cosine）
- 位置：`model/text_align.py` → `TextPrototypeLogitHead`
- 过程：
  - `TextPrototypeProjector`：将 `D_txt → D_model(=256)`（MLP 或 linear）
  - `logits = text_scale * cosine(pixel_feat, projected_text_proto)`

### 3.3 logit 融合
- 位置：`model/text_align.py` → `VisualTextLogitFusion`
- 位置：`model/model.py` → `OmniSAM_adapt.forward(...)`（融合发生在 logits 级）

---

## 4. 必须做的诊断（否则容易“假融合”）

### 4.1 `alpha_init=0` 的静默风险
如果 `alpha_init=0` 且 `alpha` 又是唯一的文本门控，容易导致：
- `final≈l_vis`，`prediction_change_ratio≈0`
- `fusion_alpha_grad_norm` 极小甚至长期为 0

因此建议：
- `alpha_init` 设为小正数（如 0.05~0.2）
- 或使用更强的特征级融合（V2/V3）

### 4.2 负控 sanity check

在固定数据子集（如 fixed20/fixed50）上跑：
- `real` / `shuffle` / `random` 三组
- 观察是否出现：`real` 明显优于 `shuffle/random`

若三者几乎重合，说明**语言语义没有进入决策**（常见原因：文本分支被门控压没、或变成可学习类锚点）。

---

## 5. 关键配置项（YAML/CLI）

### 5.1 文本原型
- `multimodal.use_text_proto`: `true/false`
- `text_proto.path`: 原型文件路径（`.pt`）
- `text_proto.mode`: `real|shuffle|random`
- `text_proto.normalize`: 是否 L2 normalize
- `text_proto.shuffle_seed` / `text_proto.random_seed`

### 5.2 文本 adapter（投影器）
- `text_adapter.projector_type`: `mlp|linear`
- `text_adapter.freeze`: `true/false`（诊断用）

### 5.3 V1 融合（logit_add）
- `text_fusion.enabled`: `true`
- `text_fusion.type`: `logit_add`
- `lambda_txt`: 建议从小到大（如 0.05/0.1/0.2）做扫描
- `text_fusion.alpha_init`: 建议非 0（如 0.05）

---

## 6. 日志与判读（最小闭环）

建议至少关注：
- **融合强度**：`fusion_alpha`、`fusion_alpha_grad_norm`
- **分支是否生效**：`prediction_change_ratio`
- **logits 统计**：`visual_logits_std`、`text_logits_std`、`final_logits_std`

若出现以下现象，需要优先排查 `alpha_init` 或训练参数组：
- `fusion_alpha≈0` 且 `fusion_alpha_grad_norm≈0`
- `prediction_change_ratio≈0`

---

## 7. 推荐实验流程（从快到慢）

1) fixed20：`real` 单跑确认分支活跃（prediction_change_ratio 非 0）  
2) fixed20：跑 `real/shuffle/random` 负控  
3) fixed50：确认趋势是否扩大/稳定

---

## 8. 下一步（何时升级到 V2/V3）

当出现以下任一情况，建议升级到 V2/V3：
- V1 中 `real/shuffle/random` 长期几乎一致
- 文本分支活跃但 attention/feature-level 没有语义差异
- 需要更强的“语义进入决策”约束（例如 grounding）

对应文档：
- `docs/text_modality_roadmap_zh.md`
- `docs/text_modality_v3_grounding_zh.md`
