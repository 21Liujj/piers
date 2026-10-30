# Experiment Conventions

## 1. 实验命名规范

### 实验目录命名
```
<experiment_name>_<YYYYMMDD_HHMMSS>
```
例如：`base_last_20260425_232941`，`A_Dtrain_Etrain_20260425_232941`

### 日志文件命名
```
<experiment_name>_<YYYYMMDD_HHMMSS>.log
<experiment_name>_<YYYYMMDD_HHMMSS>.master.log   # 批量实验主日志
```

### PID 文件命名
```
<experiment_name>.pid
```

### 配置文件命名
```
configs/ablations/omnisam_<variant>.yaml
configs/experiments/<domain>/<ablation_id>.yaml
configs/extra_exp/<dataset>/<purpose>.yaml
```

---

## 2. 强制记录项

每个实验必须记录：

| 项目 | 来源 |
|---|---|
| seed | YAML 或命令行参数 |
| GPU | 脚本中 CUDA_VISIBLE_DEVICES |
| commit | `git rev-parse HEAD` 输出 |
| 启动命令 | 脚本内打印 |
| 配置文件路径 | YAML --config 参数 |
| pd_root | 数据目录路径 |
| src_ckpt / tgt_ckpt | checkpoint 路径 |

---

## 3. 结构化指标要求

每次训练必须产出：

```
<pd_root>/checkpoints/<dataset>/<backbone>/metrics.jsonl
<pd_root>/metrics/pl_timing.jsonl
<pd_root>/metrics/train_timing.jsonl
```

字段统一，禁止同义混用（`pl_total_time` vs `pseudo_time`）。

---

## 4. 速度实验规范

- 必须包含 warm-up（使用 `base_first` 热身后的状态对比）
- 必须区分报告：Train / PL / Val / E2E 时间
- 不能混合报告总时间

---

## 5. 断点续跑规范

- 若 `metrics.jsonl` 已存在，跳过该组（已完成）
- 支持环境变量覆盖：`SKIP_COMPLETED=1`
- 批量脚本单组失败不影响后续组

---

## 6. 插件规范

- 新插件默认 `enabled: false`
- 配置通过 YAML `plugins:` 字段控制
- 每个插件必须有 `apply_in_train` / `apply_in_pl` 独立开关（D/E 模块）
- 新插件必须在 `docs/claude/module-map.md`（待补充）中登记
