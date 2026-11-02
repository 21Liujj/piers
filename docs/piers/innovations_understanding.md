# PIERS 三个创新点理解文档

> 落袋时间：2026-11-02；分支：`innov1-text`（由 `main@d30286e` 切出，另有 `innov2-memory`、`innov3-seam`）；真机：RTX 6000D，`omnisam_s.pth`。
> 本文档只记“理解 + 已验证数”，不编未跑的数。正式数以 `runs/*/metrics.json + summary.json` 为准。

## 0. 一句话总览

- 创新点2：给 SAM 的 Memory 做**按需分配**，省 16 倍、保住难例。
- 创新点3：对接缝做**按需补看**，B=0 严格退化，门控后只融接缝带。
- 创新点1：文本与伪标签**双向协同**，输入一句话就能在全景里找到物体并分割。

---

## 1. 创新点2：全景 Memory 按需分配（已完整实现 + 半正向）

### 1.1 痛点

- OmniSAM 原来全记：`32768` token，`num_maskmem=9` 帧历史全堆。
- 全景 `4096×1024` 按滑窗切（`win 1024、stride 384`），一张图约 18 次前向（正扫 9 + 反扫 9），窗窗重叠，记忆里全是重复像素，又费显存又慢。
- 第一代压缩（`memory_compressor.py` 的 `pool/spatial`，EfficientTAM 式平均池化）是无脑压，好坏一起糊，`M100-pool2048 60.069`，边界和灯牌糊掉。

### 1.2 按需怎么分（两级）

**级一：窗间分钱（PanoSMBAllocator，无参数、不学习）**

总预算锁死（如 2048），按每窗重要性 `importance_from_scores` 按比例分，保底每窗 `min_tokens=256`，整数取整补齐，`budget_error=0`。

```
quota_i = base + remaining * w_i / sum(w)
```

**级二：窗内挑点（SparseMemoryRetriever，同 2048 预算七对照）**

| 模式 | 按需逻辑 | M100（正式100张） | 作用 |
|---|---|---|---|
| uniform | 纯位置均匀抽 | 59.80 | 证伪：光靠位置不够 |
| pool | 整块平均池化（一代） | 60.07 | 一代对照 |
| topk/feature_norm | 只看 L2 范数 | 59.83 | 大类好、小类被淹 |
| random | 随机 | 20张 55.24 | 随机对照 |
| bg（boundary_guarded） | 留 12.5% 给历史伪标签高熵/边界 | 60.27 | 保边界 |
| ca（class_aware） | 留 15% 给 traffic light/sign/pole/rider | 60.30，mAcc 72.07 全场最高 | 保稀有类 |
| cams | 查询亲和度 + 范数自适应加权，query 256维池化到 64 维再算 cos | 20张 55.26 第一 / 100张 59.75 | 小集合灵、大集合过拟合 |

配额：`bg 12.5%`、`ca 15%`，剩余走 norm/cams。

### 1.3 怎么适配全景

1. **历史伪标签空域前传**：`m_eval2.py` 按从左到右扫窗，把前面窗的 `pred.argmax + entropy` 传给后面窗，`bg/ca` 才知道哪里是边界、哪里出过稀有类。普通 SAM2 是时序前传，这里是空域前传 + 回绕。
2. **预算守恒**：不管切几窗，总数永远 2048，与 `dense 32768` 比是严格 16×，与 `pool/topk` 是同预算公平比。
3. **位置去重思想**：窗重叠区重复 token，`uniform/pool` 对照证明“不是靠多记，而是靠记对地方”。

### 1.4 创新在哪、别人做过吗

- EfficientTAM / SAM2Long 做过视频记忆池化剪枝，但都是**时序均匀压**，没有稀有类留配额 + 边界留配额 + 查询亲和度三路，更没有全景回绕 + 窗间守恒。
- OmniSAM 原版就是 dense，DynamicSCM 只管接缝圆填充，不管记忆挑点。
- 本工作是**首个全景 ERP 下同预算记忆分配全对照**（`calls 7200, fallbacks 0`，真机实测）。

### 1.5 正向证据（真机数，不吹超 dense）

- 全局：`dense 60.46` 上限；`ca 60.30 / bg 60.27 / pool 60.07 / topk 59.83 / uniform 59.80 / cams 59.75`。省 16 倍只掉 0.2~0.7。
- 稀有类（13 类中第 6/7 类）：`dense 30.05/26.02` → `ca 30.68/26.69`（+0.6），`pool` 只有 `25.83` → `ca` 涨 4.8。`mAcc 71.73 → 72.07`。
- 耗时实测：`pool 319s / uniform 325s / topk 355s / cams 374s / dense 383s / ca 465s / bg 472s`。`ca/bg` 贵在伪标签 forwarding。
- 写法：**同预算下稀有类/边界不掉反涨**，不是全面超越 dense。
- 待收尾：`WildPASS 500 + SPAN 260` 跨域；`innov2-memory` 分支冻表。

---

## 2. 创新点3：接缝按需补看（保底已成、涨点未成）

### 2.1 痛点

全景左右两边在 ERP 里是断开的，实际是连着的（0° 缝）。滑窗扫到缝附近时，物体被切两半、两边各判各的，接缝带 IoU 低于全局。

### 2.2 按需怎么做（B 预算）

- `B` = 最多允许补几扇**跨缝真窗**（wrap crop，`x_start` 可负/超宽，绝不造全宽假框）。
- 基础 18 窗不动：`B=0` → 18 窗；`B=1` → 19 窗（`model_input_px=(18+B)×1024×1024`，多花算账）；`B=2` → 20 窗。
- 选窗：`score_uda = 熵 + 截断 + 跨 FoV 不一致`（UDA）；`score_text = 贴缝 + (1-质量) + assoc 不确定`（文本）。`select_extra` 取 ≤B 非冗余，同 B 下再比 `adaptive` vs `uniform`，同预算公平。
- 融合：只融接缝带（`BAND_W=256` 左右各一条），可靠性门控 `ent>0.6 & econf>0.7`，`fuse_overlap` 按覆盖×置信加权。`B=0` 严格退化为 base（`fused_px=0`）。

### 2.3 真机数（门控版，不含旧全窗 bug）

- 20 张：`B=0 55.44 / seam 57.74`；`B=1 adapt 55.45 / 57.75（fused 807px）`；`B=1 uniform 55.4487`；`B=2 55.448（fused 2110px）`。
- 100 张：`S100-b0 60.4607 / seam 57.0019`；`S100-b1-adapt 60.4619 / 57.0028（fused 1603px，多花 59s）`；`B=1 uniform` 也是 `60.4619`（选窗没比随机好）。
- 旧版全窗覆盖 `B=1 掉到 48.98` 是 bug，已废弃，不计入正表。

### 2.4 创新在哪

- `fixed_scm`（继承 DensePASS_circular + DynamicSCM）是无脑全缝补，当基线留着，本模块只加 `adaptive_extra` 按需策略。
- 融合只动接缝带 + 置信门控，不碰非缝区；`B=0` 恒等、`covered_fn` 去冗余、同预算对照三件套齐。

### 2.5 还差什么才正向

现在是**保底成、涨点没成**（+0.001 是噪声）。下一步卷接缝难例：只挑熵高/跨缝不一致的图算 `seam_mIoU`，门控放宽一档让 `fused_px` 从 0.15% 涨到 1~2%，看 `B=1/2` 能否涨 0.2~0.3。

---

## 3. 创新点1：文本 + 伪标签双向协同（链路已通、量化待跑）

### 3.1 痛点

纯语义分割听不懂人话（“最左边那辆车”指不出来）；纯文本分割（Grounding-DINO + SAM）在全景切窗下会跨缝断裂、误检多。两者要互相改卷子。

### 3.2 双向怎么做

- **文本→语义（T0/T1）**：输入一句话 → `Grounding-DINO-tiny` 给框（原图像素坐标，不二次缩放）→ `SAM2.1-Hiera-Small` 每框解掩膜（`1 FoV = 1 encode，N 框 = N decode`）→ `box_local_to_global` 拼回 ERP（跨缝拆两段，绝不造全宽框）→ `associate` 按共享可见区 `IoU≥0.3` 并轨 → 同 query 成组。
  - `T0 grounded_direct`：整图一窗直怼（记模型 resize 代价）。
  - `T1 grounded_fov`：`1024×400/stride512` 切窗，专治小目标和缝边车。
  - 空查询返回 `[]`，绝不整图回退。
- **语义→文本（T2，semantic_to_text）**：冻住 E0 语义 `probs.npz` 缓存，按候选掩膜算 `support = mean P(类)`，软重排 `score' = score×(1+α·support)`，硬过滤另做消融。未知类/多义原样保留，只记 reason。
- **文本→语义（U1，text_to_pseudo）**：用可靠文本掩膜修伪标签，只在掩膜内改、不确定边 erode 到 255、冲突回退，非检测区绝不强制改背景。

### 3.3 链路状态（能跑、没数）

- `cpu_grounded_check 12/12` 通过：DINO 本地离线加载、SAM2 权重与 `sam2.1_hiera_s.yaml` 对上、框是原图坐标、掩膜与框对齐（覆盖 0.867）、非整图、空查询零编码。
- `make_manifest.py` 已建 10 图 80 问（`car/person/road/vegetation/sky + the leftmost car + the nearest person + train 负例`），`worker_view` 剥 GT。
- `T0/T1 yaml` 权重路径已填本地，单图探针因模型 ID 未指本地目录挂一次，已改成本地路径重排，队列 `探针 → T0-10 → T1-10` 等 `S100-b2` 让卡。
- 量化：`cli evaluate --task text` 算 `IoU/Dice/correct_reject`；`T2` 需先冻 E0 `probs.npz` 再跑 `α=0.5`。

### 3.4 预期正向

`T1` 打 `person/car` 小目标和缝边实例应明显赢 `T0`；`T2` 重排压误检；`U1` 修伪标签提 UDA。这三个数跑完，I1 才算正向闭环。

---

## 4. 分支管理约定

- `main`：稳定台账 + 公共契约。
- `innov1-text`：`text/`、`collaboration/text_guide.py`、grounded worker、`T*.yaml`、`make_manifest.py`、文本评测。
- `innov2-memory`：`sparse_memory_retriever.py`、`memory/`、`m_eval2.py`、M 系脚本与论文表。
- `innov3-seam`：`seam/adaptive.py + policy.py`、`s_adaptive_eval.py`、S 系表。
- 当前 `push origin` 被 `schannel TLS` 拦，先本地用，有梯子再推，不影响真机跑数。
