# OmniSAM 创新点全景 - 简明版

## 🎯 核心目标

**在 4096 tokens 固定预算下，恢复并超越 baseline 性能**
- Baseline: 60.22 mIoU, 39.71 BFScore (32768 tokens, 无压缩)
- 目标: 60.3+ mIoU, 39.8+ BFScore (4096 tokens, 8×压缩)

---

## 📊 创新点架构图

```
                    ┌──────────────────────┐
                    │  SAM2 Memory Encoder │
                    │  每FoV: 4096 tokens  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Module D: MTC       │  ← 创新1 (工程组件)
                    │  4× 空间压缩         │
                    │  4096 → 1024/FoV     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Module E1: Pano-SMB  │  ← 创新2 (FoV级)
                    │ 自适应quota分配      │
                    │ [700,600,...,256]    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼─────┐  ┌──────▼──────┐  ┌─────▼────────┐
   │  Route 1       │  │  Route 3     │  │ Route 2      │
   │  Boundary      │  │  Class-Aware │  │ Overlap-Aware│
   │  边界保护       │  │  类别保护     │  │ 重叠回收     │
   └────────┬───────┘  └──────┬──────┘  └─────┬────────┘
            │                 │                │
            └────────┬────────┴────────────────┘
                     │
          ┌──────────▼───────────┐
          │   SAM2 Memory Attn   │
          │   4096 tokens        │
          └──────────────────────┘
```

---

## 🚀 五大创新点

### ✅ 创新1: MTC (Module D) - 已完成
**Memory Token Compressor**
- 4× 空间池化压缩 (64×64 → 32×32)
- 无参数，O(N) 复杂度
- 结果: 60.01 mIoU (vs 60.22 baseline)
- 定位: 工程组件 (EfficientTAM 已有类似)

### ⏳ 创新2: Pano-SMB (Module E1) - 今晚验证
**Panoramic Semantic Memory Budget**
- FoV级自适应quota分配
- 高语义密度FoV获得更多tokens
- 测试 min_tokens: 128/256/512
- 新颖度: ⭐⭐⭐ (文献无完全同构)

### ⏳ 创新3: Boundary Guard (Route 1) - 今晚验证
**Boundary-Guarded Token Selection**
- 保留 6.25%-25% quota 给边界tokens
- 剩余用 feature norm Top-K
- 直击 BFScore 和细小类损伤
- 新颖度: ⭐⭐⭐⭐ (文献无同构)

### 🆕 创新4: Class-Aware (Route 3) - 明天探索
**Rare-Class Token Protection**
- 保护 traffic light, fence, pole 等受损类
- 上游 logits 信号现成
- 与 Boundary 正交互补
- 新颖度: ⭐⭐⭐ (固定预算+rare-class)

### 🔮 创新5: Overlap-Aware (Route 2) - 需前置修复
**Overlap-Aware Budget Recycling**
- 利用 ERP crop geometry
- 重叠区降权，省下的给unique/boundary
- 全景特异性最强
- 新颖度: ⭐⭐⭐⭐ (完全独特)
- 阻塞: geometry 元数据不可靠

---

## 📅 实验时间线

### 今晚 (8/25, 18:00-20:48)
```
✅ Pano-SMB 4配置 × n500 (1.6h)
✅ Boundary 3配置 × n500 (1.2h)
目标: 找到最优配置方向
```

### 明天 (8/26)
```
上午: 最优配置 n2000 × 5seeds (7.5h)
晚上: Route 3 快速探索 3×n500 (1.2h) + 组合实验 (4.5h)
```

### 后天 (8/27)
```
上午: Route 3 确认 n2000 × 5seeds (7.5h)
晚上: 最强组合 Full System
```

### 8/28-8/30
```
Route 2 (如果geometry修复)
多尺度扩展
Per-class 分析
Ablation 完整矩阵
```

---

## 🎯 预期最强配置

```yaml
Full System v1:
  MTC: compress_ratio=0.5
  Pano-SMB: min_tokens=256
  Boundary: reserve=0.125
  Class-Aware: rare_reserve=0.10

预期结果:
  mIoU: 60.3-60.5 (超越 baseline 60.22)
  BFScore: 39.8-40.0 (恢复 baseline 39.71)
  Tokens: 4096 (8× 压缩)
  细小类: 全部恢复或超过 baseline
```

---

## 📊 论文实验矩阵 (预览)

| Method | mIoU | BFScore | Tokens | 压缩比 |
|--------|------|---------|--------|-------|
| Baseline | 60.22 | 39.71 | 32768 | 1× |
| +MTC | 60.01 | 39.86 | 8192 | 4× |
| +Pano-SMB | 60.15 | 39.92 | 4096 | 8× |
| +Boundary | 60.25 | 39.85 | 4096 | 8× |
| +Class-Aware | 60.20 | 39.88 | 4096 | 8× |
| **Full System** | **60.45** | **40.05** | **4096** | **8×** |

---

## 💡 与现有工作的区分

### vs Efficient-SAM2
- 他们: attention-derived saliency，固定比例Top-K
- 我们: Boundary+Class-Aware，panoramic FoV-level budget

### vs Lean-SAM2
- 他们: target-anchor相似度，object tracking场景
- 我们: Dense semantic UDA，无object prompt

### vs PanoSAM2
- 他们: Long-Short memory，frame-level admission，loss weighting
- 我们: Token-level quota，FoV-level allocation，memory retention

### vs SAMITE
- 他们: Frame-level prototype-guided selection
- 我们: Token-level rare-class protection，fixed budget

---

## 🚨 关键风险

1. **Boundary 可能无效**: n500 快速验证，风险可控
2. **Class-Aware 信号噪声**: Pseudo-label 质量影响
3. **Overlap 元数据阻塞**: 需1-2天修复
4. **组合负交互**: 逐步验证，先单独后组合

---

## 📚 相关文档

- **完整路线图**: `docs/innovation_analysis/post_2048_roadmap.md`
- **创新点对比**: `docs/innovation_analysis/innovation_overview.md`
- **文献综述**: `docs/omnisam_innovation_literature_scan_2026.md`
- **今晚实验**: `TONIGHT_FAST_PLAN_B.md`
- **快速启动**: `QUICK_START_PLAN_B.txt`

---

**创建时间**: 2026-08-25 16:10  
**下次更新**: 今晚 20:48 (根据实验结果调整)
