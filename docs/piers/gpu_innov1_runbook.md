# GPU 上卡实验手册（创新点 1：文本↔语义双向协同）

适用对象：租到带 CUDA 的单卡机器后，按本文从零跑通 **T0 → T1 →（E0 就绪后）T2**。
反向 **U1** 与双向 **B1** 依赖 omni 主引擎接线，本文只给前置准备，不给运行命令。

状态基线（2026-10-30，CPU 已验证）：
`pytest tests/piers` 52 passed；`cpu_grounded_check.py` 12/12（真权重真推理）；
`text/pipeline.py` + `workers/entry.py::grounded_main` 为真实 HF 后端；
`omni_main` 仍拒绝假加载（主引擎未接线），E0/T2/U1/B1 的 omni 侧跑不了。

---

## 0. 结论先行：租卡后什么能跑、什么不能跑

| 链路 | Suite | 能否跑 | 条件 |
|---|---|---|---|
| 文本直出（整图） | T0 | ✅ 能跑 | 填好 manifest + sam2 ckpt（§2） |
| 文本分 FoV + remap + assoc | T1 | ✅ 能跑 | 同上（§3） |
| 语义→文本（正向） | T2 | ❌ 需 E0 先出语义缓存 | `runs/E0/semantics/probs.npz` 不存在 |
| 文本→伪标签（反向） | U1 | ❌ 需 omni 接线 | `omni_main` 未接线 + 需训练数据 |
| 双向 | B1 | ❌ 同上 | 依赖 T2 + U1 |

**所以租卡当天能产出数字的是 T0/T1**（文本基线 + FoV 切分收益）。
T2/U1/B1 的代码（`text_guide.py` / `revision.py`）已写完并过 CPU 测试，
缺的是上游输入（E0 语义缓存）与下游消费（omni 训练接线）。

---

## 1. 上卡前置（每个 conda env 各做一次，约 15 分钟）

### 1.1 机器检查

```bash
nvidia-smi            # 确认看得到卡，记下 driver/CUDA 版本与显存
nvcc --version        # 若无 nvcc 也可跑，但需确认 torch 的 CUDA 版本与 driver 兼容
```

### 1.2 两个 env（按仓库 requirements 装，不要混装）

```bash
# piers-grounded：文本分支（DINO + SAM2 predictor）
conda create -n piers-grounded python=3.12 -y
conda activate piers-grounded
pip install -e .                       # 装 piers-ljj 本体（含 CLI）
pip install -r requirements-grounded.txt   # torch cu121 + transformers>=4.57,<5 + sam2 运行时依赖

# piers-omni：语义/UDA 分支（T2/U1/B1 用，现在只装好备用）
conda create -n piers-omni python=3.10 -y   # omni 继承代码要求 3.10
conda activate piers-omni
pip install -e .
# omni 侧 torch/依赖按 third_party/effective_omnisam/requirements.txt 另装（E0 未通前可暂缓）
```

### 1.3 权重准备（只做一次，离线）

```bash
conda activate piers-grounded
export PIERS_WEIGHTS=/path/to/weights   # 你放权重的目录

# a) Grounding-DINO 必须先转换（已在你本机做过：D:/piers2027/weights/grounding-dino-tiny-hf）。
#    租卡机上把 converted 目录整体拷过去即可；若只有 raw 下载，则跑：
python scripts/piers/convert_grounding_ckpt.py \
    --src "$PIERS_WEIGHTS/grounding-dino-tiny/pytorch_model.bin" \
    --dst "$PIERS_WEIGHTS/grounding-dino-tiny-hf" --copy-aux
#    检查 conversion_report.json：fused 层应全部分割、无 leftover；
#    若跳过此步，detector 会静默返回 0 候选（无报错），见 gpu_runbook bug 2。

# b) SAM2 ckpt：仓库默认用 small（sam2.1_hiera_small.pt + sam2_hiera_s.yaml）。
#    若你只有 large ckpt，把 config 里 sam2_cfg 改成 sam2_hiera_l.yaml。
sha256sum "$PIERS_WEIGHTS/sam2.1_hiera_small.pt"   # 记下来，填 paths.yaml 备查
```

### 1.4 路径文件（不进 git）

```bash
cp configs/piers/paths.example.yaml configs/piers/paths.yaml
# 按租卡机实际路径填写：
#   grounded_hf_dir: <converted dino 目录>
#   sam2_ckpt:       <sam2.1_hiera_small.pt>
#   sam2_cfg:        sam2_hiera_s.yaml
#   data_root:       <DensePASS 根>
#   manifest:        <§2 生成的 manifest 路径>
```

---

## 2. 生成查询清单 manifest（CPU/GPU 均可，先做）

```bash
# 按 DensePASS 目录结构生成（图片 + GT 路径只记录，不读取像素）
python scripts/piers/make_manifest.py \
    --img-dir "$DATA/DensePASS/val/img" \
    --gt-dir  "$DATA/DensePASS/val/gtFine" \
    --split dev-calib \
    --out configs/piers/manifest.dev-calib.json \
    --limit 20        # 先拿 20 张做 smoke；全量时去掉 --limit

# 校验（manifest 结构 + GT 隔离 + negative 依据；--check-images 会确认图片存在）
piers_ljj validate-data --manifest configs/piers/manifest.dev-calib.json --check-images
# 期望：errors 为空；输出各 subset 计数（category_union / instance_referring / negatives）
```

**negative 注意事项**：模板默认把 `train` 标为 negative（densepass13 无此类），
但 `missing_basis` 必须逐图人工确认（taxonomy-missing ≠ absent），
未确认的 negative 不得计入拒检分数（`binary_scores` 会标 `correct_reject`/`false_positive`，见 §5）。

把 manifest 路径填进 T0/T1 配置：

```yaml
# configs/piers/T1_grounded_fov.yaml（T0 同理）
manifest: configs/piers/manifest.dev-calib.json
grounded:
  grounding_model_id: /path/to/grounding-dino-tiny-hf   # converted 目录
  sam2_ckpt: /path/to/sam2.1_hiera_small.pt
  sam2_cfg: sam2_hiera_s.yaml
  dtype: bf16
```

---

## 3. T0/T1 运行（租卡后第一个能出数字的实验）

### 3.1 dry-run（不触 GPU，验依赖与配置）

```bash
conda activate piers-grounded
piers_ljj run --config configs/piers/T1_grounded_fov.yaml --dry-run
# 期望：dry-run OK。若报 BLOCKED 缺 manifest，回 §2 检查路径。
```

### 3.2 后端冒烟（单图，确认真权重在 CUDA 上工作）

```bash
PIERS_WEIGHTS=/path/to/weights python scripts/piers/cpu_grounded_check.py
# CPU 已验证 12/12；GPU 上重点看 latency 与 n_boxes 是否一致（权重转换正确则一致）。
```

### 3.3 真实运行（T1；T0 把 config 换成 T0_grounded_direct.yaml 即可）

```bash
# 必须 cd 到 vendor 目录跑，避免 repo-root sam2/ 遮蔽（worker 会自检并告警）
cd third_party/grounded_sam2
conda run -n piers-grounded python -m piers_ljj.workers.entry grounded \
    --config ../../configs/piers/T1_grounded_fov.yaml \
    --manifest ../../configs/piers/manifest.dev-calib.json \
    --out ../../runs/T1-grounded-fov/worker --device cuda
cd ../..

# 或经由调度器（会自动写 runs/T1-grounded-fov/status.json + worker.log）：
conda activate piers-grounded
piers_ljj run --config configs/piers/T1_grounded_fov.yaml
```

输出（`runs/T1-grounded-fov/worker/`）：

| 文件 | 内容 |
|---|---|
| `predictions.json` | 每 query 的 ERP 全局 box/候选 + assoc 分组 + `calls`（`fov_windows`、`sam_image_encodes`、`sam_mask_decoder_calls`、`latency_ms`） |
| `collab_log.json` | T1 时 `collab: none`，全为 keep（passthrough） |
| `status.json` | `completed_real` + device + latency；`synthetic: false` |

**成本诚实记录**：T0 整图模式会产生一次大尺寸 `set_image`（resize 代价如实计入 `latency_ms`，
不得与 T1 的 FoV 切分代价直接比“谁更快”——比的是精度，代价只并列报告）。

### 3.4 评测（只读预测 + GT，绝不把 GT 喂回 worker）

```bash
# predictions.json -> per-query mask stack（需先把 box/mask 落成 npz；脚本见附录 A）
# 文本任务：
piers_ljj evaluate --pred-npz runs/T1-grounded-fov/pred.npz \
    --gt-npz runs/T1-grounded-fov/gt.npz --task text \
    --out runs/T1-grounded-fov/metrics.json
# 输出：macro/micro IoU（只在有 GT 的 query 上平均）、correct_reject（双空判拒检，
# 不计满分）、false_positive（空 GT 有预测且面积>=thr 计 0 分）。

# 多 run 汇总（只收 completed_real + synthetic=false）：
piers_ljj summarize --runs-root runs --out runs/summary.json
```

---

## 4. T2/U1/B1 为何现在跑不了（缺口与接线图）

```
E0 (omni baseline) --semantics--> runs/E0/semantics/probs.npz --+--> T2 正向重排
      ↑ omni_main 未接线（entry.py:80-84 return 2）               |
T1 文本候选 ------------------------------------------------------+
T1 可靠 mask --> revision NPZ --> U1 训练读回 --------------------→ 需 omni 训练消费 revision.py 补丁
B1 = T2 + U1 各一轮（rounds: 1），依赖前两者
```

缺口清单（按阻塞强度排序）：

1. **`omni_main` 未接线**（最硬）：`workers/entry.py:80-84` 主动拒绝加载，
   继承引擎（`third_party/effective_omnisam/{model,dataset,adaptation,train.py,…}`）
   已迁入但未接到 worker。需写 `omni worker` 的 E0 推理 + U1 训练两条调用，
   并做 ckpt-gated parity（与继承工程同 ckpt 同数据对齐，声明容差）。
2. **E0 基线未跑**：`effective_start.yaml` 的 `checkpoint.path` 为 null；
   `ckpt_zoo/` 只有索引无权重文件；Cityscapes/DensePASS 未就位。
3. **U1 训练消费 revision 补丁的读取位未定**：`revision.py` 输出格式已冻结
   （`revise_mask/ignore_mask/target_cls`），但 omni 训练侧哪一步读入、
   与 aug 对齐的 warp 点还没写。

T2/U1/B1 的 **piers 侧代码已完成并过 CPU 测试**（`text_guide.py` / `revision.py` /
`manifest.py` + `test_innov1.py` 8 项），GPU 上只需补上游输入与下游消费，
不需重写策略。

---

## 5. 评测口径（防刷分规则，先看懂再跑全量）

- UDA 语义：dataset-level 混淆矩阵 → mIoU（`scores_from_cm`），**不是**每图 mIoU 再平均；
  19→13 映射用 `MAP19_TO_13`（train→255 忽略），`validate-data` 会断言防漂移。
- 文本：per-query IoU/Dice；空预测对非空 GT 计 0；**双空只标 `correct_reject`，不计 1.0**；
  空 GT 有预测按 `area_thr`（默认 16px）判 `false_positive`。
- `category_union` 与 `instance_referring` 分开报表；negative 单列拒检率。
- `summarize` 只收 `status == completed_real` 且 `synthetic == false` 的 run；
  dry-run / pending_gpu / blocked 一律不聚合。
- FoV 切分/跨缝补看的代价（`windows/dino_calls/sam_enc/sam_dec/latency_ms`）
  与精度并列报告，不折算成“加速比”自我美化（`nocompress` 只做精度参照，
  token 不可比）。

---

## 附录 A：predictions.json → 评测 npz（模板，按实际 mask 落盘改）

`predictions.json` 存的是 box 与候选元数据；评测需要逐 query 的二值 mask stack。
若你的 worker 已把 mask 存成 PNG/npz，把下面两行换成真实读取即可；
**不要**为评测重跑推理（`evaluate` 只读预测与 GT）。

```bash
# 伪代码，按你的 mask 落盘格式实现 scripts/piers/pred_to_npz.py 后：
python scripts/piers/pred_to_npz.py \
    --predictions runs/T1-grounded-fov/worker/predictions.json \
    --manifest configs/piers/manifest.dev-calib.json \
    --out runs/T1-grounded-fov/pred.npz
```

## 附录 B：故障排查速查

| 现象 | 最可能原因 | 查哪里 |
|---|---|---|
| detector 0 候选、无报错 | 权重未转换（fused layout） | `conversion_report.json`；load 日志 “newly initialized” |
| `TypeError: threshold` | transformers 版本差异 | worker 已做签名自适应；若仍错，确认 `transformers>=4.57,<5` |
| `sam2` 路径告警 | cwd 不对 | 必须 `cd third_party/grounded_sam2` 再起 worker |
| T2 BLOCKED 缺 E0 | 正常（E0 未跑） | `sweep --suite T2 --check-only` 看依赖 |
| `manifest: null` BLOCKED | 配置没填 manifest | 回 §2 |
| 0 候选但应有目标 | 阈值/文本大小写 | 先查权重，再调 `thresholds.det/text`（T1 默认 0.35/0.25） |
