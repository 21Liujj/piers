# 边缘端部署与实时性评估方案（Edge Deployment & Real-Time Evaluation）

状态：草案 v0.1（开题答辩后新增章节，待方案细化）
适用对象：刘嘉俊（2532009）
依据：开题答辩老师意见 —— "全景图像在自动驾驶与机器人导航中有广泛应用，需考虑边缘侧实时部署，并增加对应测试指标与实验创新"

---

## 0. 一句话结论

**端侧部署可行，8GB 显存够用，但当前架构距实时差 10~100 倍，因此这本身就是一项创新（而非附加实验）。**

工作定位：
- **Jetson（Orin 系列，8GB）** = 主力实时性验证平台，证明"能跑 + 能跑快"
- **RK3588（8GB）** = 低功耗异构对比平台，证明"省电 + 精度可保持"
- **RTX 6000D（远端）** = 服务端基线参照点，构成"云-边协同"叙事闭环

---

## 1. 现状基线（远端实测，2026-11-02）

数据来源：`/root/autodl-tmp/piers-main` 下的 `formal100.log` / `i1_100.log` / `runs/*/metrics.json`
硬件：RTX 6000D 85GB，torch 2.8.0+cu128，transformers 4.57.6

| 链路 | Suite | 数据量 | 总耗时 | 折算单帧/单查询 | 精度 |
|---|---|---|---|---|---|
| 文本直出（整图） | T0 | 800 queries | 232.6 s | **≈ 290 ms/query** | 见 `runs/T0-100` |
| 语义（记忆 CAMS-2048） | M100-cams2048 | 100 img | 374.25 s | **≈ 3.74 s/img** | mIoU 59.75 / mAcc 70.29 / PixAcc 88.86 / FWIoU 81.09 / mDice 72.10 |
| 语义 + 接缝 B=0 | S100-b0 | 100 img | 584.13 s | **≈ 5.84 s/img** | mIoU 60.46 / seam_mIoU 57.00 |
| 语义 + 接缝 B=1(自适应) | S100-b1-adapt | 100 img | 673.18 s | **≈ 6.73 s/img** | mIoU 60.46 / extra 59.29 s |
| 语义 + 接缝 B=1(均匀) | S100-b1-uniform | 100 img | 672.21 s | **≈ 6.71 s/img** | mIoU 60.46 |
| 语义 + 接缝 B=2 | S100-b2-adapt | 100 img | 642.84 s | **≈ 6.43 s/img** | mIoU 60.46 / extra 77.66 s |

T0 单查询内的分模块开销（`runs/T0-100/predictions.json` → `calls` 字段）：
- `latency_ms` 首个查询 ≈ 1907.9 ms（含模型冷启动/warmup，不可直接当稳态值）
- `sam_image_encodes` = 1、`sam_mask_decoder_calls` = 2、FoV 窗数 = 1

### 1.1 与实时要求的差距

| 目标 | 要求 | 当前（服务端最优） | 差距 |
|---|---|---|---|
| 自动驾驶语义分割 | 30 FPS → 33 ms/帧 | 3740 ms/帧 | **≈ 113×** |
| 机器人导航（可放宽） | 5 FPS → 200 ms/帧 | 3740 ms/帧 | **≈ 19×** |
| 文本查询响应 | 交互级 ≤ 500 ms | 290 ms/query | 已接近 |

**结论**：服务端远未实时 → 端侧优化空间大 → 端侧实时化可作为独立创新点，且与"效率"主线（R5 eval+cost）天然衔接。

### 1.2 权重与显存账

| 权重 | 大小 (FP32) |
|---|---|
| `sam2.1_hiera_small.pt` | 176 MB |
| `grounding-dino-tiny` | ≈ 170 MB |
| `omnisam_s.pth` | 178 MB |
| 合计 | ≈ 524 MB |

- FP16 后 ≈ 262 MB；INT8 后 ≈ 131 MB
- 8GB 统一内存板上：**权重不是瓶颈**（Jetson 还需预留 CUDA context ~500MB + 激活缓存）
- 真实瓶颈是 **激活显存 + SAM2 image encoder 的算力**（1024×1024 输入下 encoder 单次前向占主要耗时）

---

## 2. 硬件平台与分工

### 2.1 Jetson（主力）

- 定位：**实时性主战场**
- 前提：JetPack 5.x，CUDA 12.x，TensorRT 8.6+
- 路线：
  1. **FP16 PyTorch 基线**（先不改模型，只开 `torch.autocast(dtype=torch.float16)`）→ 得到"裸 PyTorch 端侧数字"
  2. **ONNX 导出**（SAM2 image encoder / decoder、DINO backbone 分开导出）
  3. **TensorRT FP16 引擎**（FP32 engine 作为速度对照，FP16 engine 作为部署形态）
  4. **INT8 校准 + 量化**（100 张 DensePASS val 做校准集）→ 仅在精度保持达标时启用
- 预期量级（待实测）：
  - T0 单 FoV：290 ms → 80~120 ms（FP16+TensorRT）
  - T1 四 FoV 窗：→ 300~500 ms
  - 仍达不到 30 FPS → **必须配合 M/S 策略做预算自适应**，见 §5

### 2.2 RK3588（对比 / 低功耗）

- 定位：**低功耗异构验证**，不与 Jetson 拼 FPS
- 关键约束：6 TOPS NPU 只支持 INT8 / 特定算子；Transformer attention、SAM2 decoder 转 RKNN 会大量 fallback 到 CPU
- 可行路线（择一或都做）：
  - 路线 A：DINO backbone → RKNN INT8 放 NPU，SAM2 留 CPU
  - 路线 B：只跑语义分支（OmniSAM + 轻量 FoV），输出 mIoU + 功耗
- 论文写法定位为："同等语义精度下的整机功耗对比"，而非 FPS 竞赛

### 2.3 功耗与温控（必做，否则会被质疑）

| 平台 | 采集手段 |
|---|---|
| Jetson | `tegrastats`（读取 `POM_5V_IN` / `POM_5V_GPU`），1 Hz 采样 |
| RK3588 | 串口功率计 或 USB 电流表，1 Hz 采样 |
| RTX 6000D | `nvidia-smi --query-gpu=power.draw`，作为"高功耗高吞吐"参照点 |

统一换算：**单帧能耗 J/frame = 平均功率(W) / FPS**

---

## 3. 指标体系（开题需写进"实验指标"一节）

### 3.1 精度保持（不下降即可接受）

- `mIoU` / `mAcc` / `PixAcc` / `FWIoU` / `mDice`（语义分支，沿用 `evaluation/metrics.py`）
- 文本分支：macro/micro IoU、Dice、`correct_reject`、`false_positive`（沿用 `m_eval_text.py`）
- **精度保持率** `= 端侧 mIoU / 服务端 mIoU`
- **量化准入门槛**：FP16 掉点 ≤ 0.5 mIoU；INT8 掉点 ≤ 1.5 mIoU，超标则不采用 INT8 结论

### 3.2 速度

- 端到端延迟 **P50 / P90 / P95 / P99**（warmup ≥ 20 次后统计，至少 100 张图）
- **FPS**（≥ 阈值判为实时）
- 分模块耗时拆解（必须拆，否则无法说明瓶颈与优化依据）：
  `DINO 检测` / `SAM image encode` / `SAM mask decode` / `ERP remap` / `跨 FoV 关联 assoc` / `后处理`
- 端到端 vs 端到端+后处理 两个口径都报（部署关心的是后者）

### 3.3 资源开销

- Params (M)、FLOPs (G)（`thop` 或 `fvcore`，标注输入分辨率）
- 峰值显存 / 峰值内存（`torch.cuda.max_memory_allocated` / `max_memory_reserved`；RKNN 用 `rknn_query`）
- 模型体积（FP32 / FP16 / INT8 三档）
- 平均功耗 (W)、单帧能耗 (J/frame)、峰值温度（热稳态）

### 3.4 鲁棒性

- 连续运行 30 min 的温度墙降频曲线（体现"长时间在线可用"，机器人/自动驾驶真实场景）
- 连续推理无 OOM、无内存泄漏（多轮后显存不增长）

---

## 4. 技术实现：需要打点的代码位置

### 4.1 服务端（远端，先做，用于建立"非实时"证据）

| 文件 | 改动 |
|---|---|
| `src/piers_ljj/workers/entry.py` | 已有 `time.perf_counter()`，扩展为分模块计时器（`t_detect / t_encode / t_decode / t_remap / t_assoc`），写入 `predictions.json` 的 `calls` 字段（现已含 `latency_ms`, `sam_image_encodes`, `sam_mask_decoder_calls`） |
| `src/piers_ljj/workers/entry.py` | 增加 `torch.cuda.max_memory_allocated()` 峰值显存上报 |
| `src/piers_ljj/evaluation/metrics.py` | 新增 `edge_cost` 统计聚合：延迟分位数、峰值显存、功耗、能耗/帧、精度保持率 |
| 新增 `scripts/piers/edge_bench.py` | 统一跑批入口：遍历 manifest，warmup N 次，统计上述全部指标，输出 `runs/<tag>/edge_metrics.json` |

### 4.2 端侧

- 导出脚本 `scripts/piers/export_onnx.py`：`dino_backbone` / `sam_image_encoder` / `sam_mask_decoder` 三件套
- 推理脚本 `scripts/piers/edge_infer_trt.py`：Jetson 上 TensorRT 推理 + 与 PyTorch 输出做数值一致性校验（max abs diff < 1e-2）
- RK3588：`scripts/piers/export_rknn.py`（DINO backbone INT8）+ CPU fallback 段

### 4.3 一致性校验（防止"端侧数字不可信"）

端侧推理结果必须与远端 PyTorch 结果做**逐像素比对**：
- 语义分支：mIoU(端侧输出 vs 服务端输出) ≥ 99%，否则视为导出错误
- 文本分支：box IoU ≥ 0.99、mask IoU ≥ 0.98

---

## 5. 实验矩阵（论文最终要出的表）

### E-Edge-1：精度保持率

| 平台 | 精度设置 | 语义 mIoU | 保持率 | 文本 macro IoU | 保持率 |
|---|---|---|---|---|---|
| RTX 6000D | FP32 (基线) | 59.75 | 1.000 | — | 1.000 |
| Jetson | FP16 | | | | |
| Jetson | FP16 + TensorRT | | | | |
| Jetson | INT8 | | | | |
| RK3588 | INT8(NPU)+FP32(CPU) | | | | |

### E-Edge-2：精度-速度-能耗 trade-off（核心创新表）

横轴 FPS，纵轴 mIoU，每个点一个配置；画三平台曲线。配置维度正好复用现有 M/S 预算：

| 配置维度 | 取值 | 已有产物 |
|---|---|---|
| 记忆预算 `eff_len` | 1024 / 2048 | `M100-{topk,bg,pool,ca,cams}-2048` |
| CAMS 采样 | topk / background / uniform / dense | `M100-*-2048` 系列 |
| 接缝额外窗 B | 0 / 1 / 2 | `S100-b0/b1-adapt/b1-uniform/b2-adapt` |
| FoV 窗数 | 1 / 4 / 更多 | T0 (1 窗) vs T1 (stride 512) |

> 这一组是**最强的创新点**：把"语义精度"与"端侧实时性"用同一套预算策略统一起来，证明 PIERS 的自适应预算机制在边缘端同样有效 —— 单一方法在两个维度上同时取得 Pareto 前沿。

### E-Edge-3：分模块延迟拆解（瓶颈分析表）

平台 × 配置 → DINO / SAM-enc / SAM-dec / remap / assoc 各自的 ms 与占比。用于论证"瓶颈在 SAM2 image encoder"，从而支撑后续优化选择。

### E-Edge-4：三平台横向对比

| 平台 | 功耗 | FPS | mIoU | J/frame | 成本 | 结论 |
|---|---|---|---|---|---|---|
| RTX 6000D | | | | | 高 | 服务端 |
| Jetson | | | | | 中 | 实时主力 |
| RK3588 | | | | | 低 | 低功耗 |

### E-Edge-5（可选加分）：热稳态与降频

Jetson 连续 30 min，每 5 min 记录 FPS / 功耗 / 温度，画曲线；对比有/无主动散热。

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| Jetson 上 SAM2 encoder 不够快，达不到实时 | 高 | 本来就要靠 M/S 预算自适应达成；trade-off 表本身就是结论；可用 TensorRT + 降输入分辨率（1024→768）再优化 |
| RK3588 NPU 算子 fallback 严重 | 中 | 提前把 NPU 部分降级为"DINO backbone 验证"，不硬刚全模型 FPS |
| TensorRT 导出踩 sam2 / transformers 的坑 | 中 | 沿用已有经验：权重布局/参数改名问题要显式校验；导出后必须做数值一致性校验（§4.3） |
| 端侧 mIoU 与服务端不一致，数字不可信 | 高（致命） | §4.3 的逐像素比对作为门禁，不通过不进入正式实验 |
| 板子供电/散热导致频率波动，延迟数据不可比 | 中 | 固定功耗模式（`nvpmodel -m 0` / `jetson_clocks`）、固定环境温度、每组实验前 warmup 20 次 |
| 两块板子当前无 SSH / 无环境 | 高（阻塞） | 见 §7 待办，需用户提供型号与系统版本 |
| 8GB 板子 OOM（尤其是 RK3588 的 CPU 内存） | 中 | 统一内存板优先 FP16；分模块加载-卸载；显式 `torch.cuda.empty_cache()` |

---

## 7. 待办（阻塞项，需用户/导师确认）

1. **Jetson 具体型号** + JetPack 版本 + `jetson_release` 输出（决定 CUDA/TensorRT 版本与可用算子）
2. **RK3588 板子型号**（OrangePi 5 / Firefly / 友善 NanoPi R6S 等）+ 系统（Linux / Android）+ 是否可 SSH
3. 两块板子的**供电方式**与是否支持外接功率计（若不支持，功耗退化为 `tegrastats` / `sysfs` 估算，论文中需说明为"估算值"）
4. 是否需要在开题 PPT 中新增"实时性"页（建议加，含一张"服务端 3.74 s/帧 → 端侧 XX ms/帧"的对比图）
5. 是否将 §3 指标体系并入开题报告的"实验指标"章节作为新增小节（建议并入）

---

## 8. 里程碑（建议排期）

| 阶段 | 内容 | 产出 |
|---|---|---|
| M0（开题后立即） | 服务端分模块计时 + 峰值显存打点；`edge_bench.py` 跑出基线 | "非实时"证据表 + 瓶颈拆解图（PPT 用） |
| M1 | Jetson 环境搭建（JetPack + TensorRT + PyTorch 轮子） | Jetson 上 PyTorch FP16 能跑通 T0 |
| M2 | ONNX 导出 + TensorRT 引擎 + 数值一致性校验 | Jetson TensorRT 数字，服务端交叉验证通过 |
| M3 | RK3588 环境搭建 + DINO INT8 落地 | RK3588 能跑通（至少 DINO 段） |
| M4 | 三平台统一 `edge_bench` 全量跑通（E-Edge-1/3/4） | 主表 |
| M5 | trade-off 曲线（E-Edge-2）+ 热稳态（E-Edge-5） | 核心创新图 |
| M6 | 论文写作整合 | 第四章「边缘端实时部署」 |

---

## 附录 A：相关文件索引

| 路径 | 作用 |
|---|---|
| `src/piers_ljj/workers/entry.py` | 推理主入口，打点位置 |
| `src/piers_ljj/evaluation/metrics.py` | 精度指标，`edge_cost` 新增位置 |
| `m_eval_text.py` | 文本分支指标（远端 `piers-main` 根目录） |
| `runs/M100-*/metrics.json` | 语义分支精度基线 |
| `runs/T0-100/predictions.json` | 文本分支逐查询结果与 `calls` 计时 |
| `configs/piers/M_memory.yaml` / `S_seam.yaml` | 预算与接缝配置，trade-off 实验的配置来源 |
| `docs/piers/gpu_innov1_runbook.md` | 现有 GPU 实验手册风格参考 |

## 附录 B：数据来源

- 远端路径：`/root/autodl-tmp/piers-main`
- 硬件：RTX 6000D 85GB / CUDA 13.2 driver / torch 2.8.0+cu128
- 采集时间：2026-11-02 ~ 2026-11-06
