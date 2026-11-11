# 创新点1成果文档：文本↔语义双向协同（I1 + 开词表，冻结推理发布）

> 状态：已收尾，可写进论文。冻结 release `complete-I1-I2-I3-OV-v3-20261109`。
> 对应代码：`src/piers_ljj/text/`（`grounded_adapter.py`、`pipeline.py`、`feature_cache.py`）、
> `src/piers_ljj/collaboration/`（`reliable_cooperation.py`、`open_vocab.py`）、
> `src/piers_ljj/workers/entry.py::_build_hf_backend`、
> `scripts/piers/full_model_fusion_v3.py::FrozenCooperation`、`full_model_queries_v3.py`。
> 数据口径见 `results/paper20261109/densepass_paired.json`、`complete_table.json`。

## 0. 实际收益（一句话）

- I1 是全景系统**从"只能输出 13 类"到"能输出新类"的唯一来源**：Wild 原生 8 类 baseline 对新类输出恒为零，I1 开词表后 **37.56 → 57.57（+20.01pp）**；Dense13 **+0.56pp（CI 过零，开发集，诚实标注）**；文本 gated **+1.06pp**。
- 性质：冻结推理、零训练，不动权重。

## 1. 解决了什么问题

- 纯语义头（13 类 Cityscapes 预训练）听不懂人话：新类别（Wild 的 curb/crosswalk/truck/bus、Syn 的 road marking/bridge 等）输出恒为零。
- 纯文本分支（Grounding-DINO + SAM）在全景切窗下跨缝断裂、误检多。
- I1 让两者互相改卷子：文本给候选、语义给支撑，双向门控融合，不动权重、不训练。

## 2. 做法（一句话）

- 文本→语义：每 FoV（1024×400，stride 512，共 4 窗）×每 query 调 DINO-tiny（HF）出框 → SAM2.1-Hiera-Small 每框解掩膜 → `box_local_to_global` 拼回 ERP（跨缝拆两段）→ `associate`（共享可见区 IoU≥0.3）并轨。
- 语义→文本（`FrozenCooperation.apply`）：冻住语义 probs，按候选掩膜算支撑 `support=mean P(类)`，`bounded_text_fusion`（α=0.5，门控）软重排 + 硬过滤；`build_novel_evidence` 给新类造证据，老类走条件竞争（`requested_probabilities`），无证据新类输出 255 弃权。
- 守恒：空查询返回 `[]`，绝不整图回退；GT 只读评测，不进推理。

## 3. 成果数字（冻结，只报已验证）

### 3.1 I1 独立贡献（DensePASS100，`results/I1_densepass/`）

- 语义：dense 原版 mIoU **60.2963** → I1 soft **60.5961**（**+0.2998 pp**，image95CI [0.1258, 0.5439]）；I1 hard **60.6952**（+0.3989 pp）。
- 文本：old6 T1 **66.9112** → I1_gated **67.9674**（**+1.0563 pp**）；expanded fine13 T1 **49.9989** → **50.8511**（**+0.8522 pp**）。
- 性质：**冻结推理**发布（`training_gain_established: false`），DensePASS100 是开发集（`new_independent_test_claimed: false`），哈希锁定（`source_fingerprints.json` 14/14 match）。

### 3.2 完整系统中的 I1（含 I2/I3/开词表，`complete_table.json` v3 final）

- DensePASS13：60.2961 → **60.8555**（**+0.5594 pp**，95CI [-0.176, 1.393]；CI 过零，诚实标注）。
- WildPASS原生8类（500 张）：37.5572 → **57.5670**（**+20.01 pp**，95CI [16.39, 23.26]）——新类 curb/crosswalk/truck/bus 从 0 涨到 15–60 IoU，这是开词表的主证据。
- SynPASS原生22类（1690 张）：25.0723 → **25.6163**（**+0.544 pp**，95CI [0.376, 0.731]）。
- SynPASS映射13类：43.5811 → **43.8665**（**+0.285 pp**，95CI [0.160, 0.421]）。

## 4. 诚实局限

- Dense13 的 +0.56 CI 过零，不能单独宣称“显著优于 baseline”；开词表增益（Wild +20pp）才是 I1 的硬证据。
- 附带工程副产品（效率向）：`PreFusionFeatureCache` 整系统串行 -0.2~-1.4s、四场景 sha 全等价（见效率交接），属于实现优化，不算 I1 方法增益。
- 训练增益未做（LoRA/微调是下一步室内场景的事，本次不动权重）。

## 5. 论文写法建议

- 主张：开词表能力（baseline 零输出→有输出）+ Wild +20pp；Dense 写“+0.56pp（CI 过零，开发集）”。
- 消融：T0 vs T1（FoV 切分）+ gated vs ungated（门控压误检），数见 `results/I1_densepass/`。
