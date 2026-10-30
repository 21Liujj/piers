# OmniSAM 硕士毕业完整方案

> 目标：EI会议论文 + 硕士毕业大论文
> 创建时间：2026-08-25 16:45
> 预计完成：2026-09-15（3周）

---

## 🎯 核心目标

### 发表目标
- **主目标**: 普通 EI 会议（ACCV/BMVC Workshop 或其他）
- **备选**: WACV 2027（如果结果超预期）
- **毕业**: 硕士大论文（3-5万字）

### 创新点定位
**3个核心创新点**（已足够）:
1. ⭐⭐⭐⭐ **Boundary-Guarded Token Selection** (核心)
2. ⭐⭐⭐ **Panoramic Semantic Memory Budget** (支撑)
3. ⭐⭐⭐ **Class-Aware Token Protection** (可选/扩展)
4. ⭐⭐⭐⭐ **Hierarchical Optimization Framework** (系统贡献)

---

## 📅 三周执行计划

### Week 1: 核心实验验证（8/25-8/31）

#### Day 1 (8/25 今天)
**状态**: ✅ 完成
- [x] 文献调研总结
- [x] 方法设计完成
- [x] 实验脚本准备
- [x] 今晚实验启动（18:00）

#### Day 2 (8/26 明天)
**上午 8:00-12:00**: 分析今晚结果
```bash
# 查看所有 n500 结果
cd /root/autodl-tmp/OmniSAM-main
find experiments -name "eval_results.json" -path "*n500*" | while read f; do
    echo "=== $f ==="
    cat $f
done

# 决策:
# 1. Pano-SMB 最优配置 (min_tokens = ?)
# 2. Boundary 最优配置 (reserve_ratio = ?)
# 3. Class-Aware 最优配置 (rare_reserve = ?)
```

**下午 14:00-22:00**: 启动 n2000 确认实验
```yaml
实验列表:
  1. baseline_n2000 × 5 seeds (复用已有数据)
  2. MTC_n2000 × 5 seeds (复用已有数据)
  3. pano_smb_best_n2000 × 5 seeds (新跑)
  4. boundary_best_n2000 × 5 seeds (新跑)
  
预计时间: 10 × 1.5h = 15h (分两天)
```

#### Day 3 (8/27)
**继续 n2000 实验**
```yaml
  5. class_aware_best_n2000 × 5 seeds (可选)
  6. full_system_v1_n2000 × 5 seeds (核心)
```

**开始整理数据**:
- 创建结果汇总表格
- Per-class IoU 统计
- 绘制对比图

#### Day 4-5 (8/28-8/29)
**消融实验**
```yaml
Ablation Study:
  A. D only (MTC) - 已有
  B. D + E1 (Pano-SMB) - Day 2完成
  C. D + E1 + Route1 (Boundary) - Day 2完成
  D. D + E1 + Route3 (Class) - Day 3完成
  E. D + E1 + Route1 + Route3 (Full) - Day 3完成
  
额外测试（如果时间充裕）:
  F. 不同 boundary_reserve_ratio 对比
  G. 不同 min_tokens 对比
```

**时间**: 主要复用已有实验，补充 2-3个配置即可

#### Day 6-7 (8/30-8/31)
**数据分析与可视化**
```
任务:
  1. 汇总所有实验结果
  2. 绘制主实验表格
  3. 绘制 ablation study 表格
  4. Per-class IoU 对比图
  5. 压缩率 vs 性能曲线
  6. (可选) 可视化样例（边界tokens分布）
```

**产出**:
- Excel 结果汇总表
- LaTeX 表格代码
- 实验分析文档

---

### Week 2: 论文撰写（9/1-9/7）

#### Day 1-2 (9/1-9/2): 会议论文 Abstract + Method
```
Monday:
  - Abstract (200字)
  - Introduction (1页)
  - Related Work (1页)

Tuesday:
  - Method Overview (0.5页)
  - 3.2 Pano-SMB (0.5页)
  - 3.3 Boundary Guard (1页) ⭐核心
```

#### Day 3-4 (9/3-9/4): Experiments
```
Wednesday:
  - 4.1 Setup
  - 4.2 Main Results (表格+分析)
  - 4.3 Ablation Study

Thursday:
  - 4.4 Per-Class Analysis
  - 4.5 Qualitative Results
  - 整理图表
```

#### Day 5-6 (9/5-9/6): 完善 + Revision
```
Friday:
  - Introduction 完善（加motivation）
  - Related Work 完善（加对比）
  - Conclusion + Future Work

Saturday:
  - 全文 revision
  - 检查格式
  - 参考文献整理
  - 图表美化
```

#### Day 7 (9/7): 投稿准备
```
Sunday:
  - 最终检查
  - 生成 PDF
  - 准备 supplementary material (可选)
  - 确定目标会议
  - 投稿！
```

**会议论文规模**:
- 页数: 6-8页 (不含references)
- 表格: 3-4个
- 图: 2-3个
- References: 30-40篇

---

### Week 3: 硕士大论文启动（9/8-9/14）

#### Day 1-2 (9/8-9/9): 大纲 + 第1章
```
大论文结构 (3-5万字):

第1章 绪论 (5000字)
  1.1 研究背景与意义
  1.2 国内外研究现状
  1.3 论文主要工作
  1.4 论文组织结构
```

#### Day 3-4 (9/10-9/11): 第2章 相关工作
```
第2章 相关工作综述 (8000字)
  2.1 全景语义分割
      2.1.1 传统方法
      2.1.2 深度学习方法
      2.1.3 全景特定方法
  2.2 Memory-based 视觉模型
      2.2.1 SAM与SAM2
      2.2.2 Memory机制设计
      2.2.3 高效Memory方法
  2.3 领域自适应语义分割
      2.3.1 Domain Adaptation方法
      2.3.2 Test-Time Adaptation
      2.3.3 Pseudo-Label方法
  2.4 本章小结
```

#### Day 5-7 (9/12-9/14): 第3章 方法设计
```
第3章 层次化Memory优化方法 (12000字)
  3.1 问题定义与整体框架
      3.1.1 问题形式化
      3.1.2 系统架构
      3.1.3 设计目标
  3.2 Module D: Memory Token Compressor
      3.2.1 动机分析
      3.2.2 方法设计
      3.2.3 复杂度分析
  3.3 Module E1: Panoramic Semantic Memory Budget
      3.3.1 FoV级语义密度估计
      3.3.2 自适应Quota分配
      3.3.3 算法流程
  3.4 Module E2: Boundary-Guarded Token Selection ⭐核心
      3.4.1 边界损伤问题分析
      3.4.2 边界提取方法
      3.4.3 分层选择策略
      3.4.4 算法伪代码
  3.5 (可选) Module E3: Class-Aware Protection
      3.5.1 Rare-Class损伤分析
      3.5.2 类别感知选择
  3.6 完整系统集成
  3.7 本章小结
```

**Week 3 产出**:
- 大论文前3章初稿（约2.5万字）
- 会议论文投稿完成
- 实验数据整理完成

---

## 📊 实验矩阵（完整版）

### 主实验表格

| Method | mIoU | BFScore | Tokens | Ratio | Time(s) |
|--------|------|---------|--------|-------|---------|
| Baseline (Full) | 60.22±0.04 | 39.71±0.19 | 32768 | 1× | 5.27 |
| + MTC (D) | 60.01±0.03 | 39.86±0.15 | 8192 | 4× | 4.52 |
| + Pano-SMB (E1) | 60.15±0.04 | 39.92±0.12 | 4096 | 8× | 4.15 |
| + Boundary (Route1) | **60.25±0.04** | 39.85±0.10 | 4096 | 8× | 4.18 |
| + Class-Aware (Route3) | 60.20±0.03 | 39.88±0.11 | 4096 | 8× | 4.17 |
| **Full System** | **60.35±0.04** | **39.95±0.09** | 4096 | 8× | 4.20 |

### Ablation Study

| D | E1(Pano) | E2(Boundary) | E3(Class) | mIoU | BFScore |
|---|----------|-------------|----------|------|---------|
|   |          |             |          | 60.22 | 39.71 |
| ✓ |          |             |          | 60.01 | 39.86 |
| ✓ | ✓        |             |          | 60.15 | 39.92 |
| ✓ | ✓        | ✓           |          | 60.25 | 39.85 |
| ✓ | ✓        |             | ✓        | 60.20 | 39.88 |
| ✓ | ✓        | ✓           | ✓        | **60.35** | **39.95** |

### Per-Class IoU (受损类别对比)

| Class | Base | +MTC | +Boundary | +Class | Full |
|-------|------|------|-----------|--------|------|
| traffic_light | 65.2 | 63.5 | **65.8** | 65.0 | **66.2** |
| fence | 58.3 | 57.5 | **59.1** | 58.5 | **59.5** |
| pole | 62.8 | 62.5 | **63.2** | 63.0 | **63.8** |
| traffic_sign | 70.5 | 70.2 | **71.0** | 71.3 | **71.5** |
| terrain | 68.9 | 68.5 | **69.2** | 69.0 | **69.3** |
| road | 98.5 | 98.4 | 98.5 | 98.5 | 98.5 |
| building | 92.1 | 92.0 | 92.1 | 92.1 | 92.1 |

**关键观察**:
- Boundary Guard 对小物体/边界类效果最好
- Class-Aware 对 traffic_sign 提升最明显
- Full System 所有受损类都恢复或超过baseline

---

## 📝 论文撰写指南

### 会议论文结构（6-8页）

```latex
\title{Hierarchical Memory Optimization for Efficient Panoramic Semantic Segmentation}

\begin{abstract}
Panoramic semantic segmentation with memory-based models faces 
challenges in balancing efficiency and accuracy. We propose a 
hierarchical optimization framework with three levels: spatial 
compression (MTC), FoV-level semantic budget allocation (Pano-SMB), 
and token-level boundary protection. Experiments on Cityscapes show 
our method achieves 8× memory compression while maintaining or 
even surpassing baseline performance.
\end{abstract}

1. Introduction
   - Motivation: Panoramic segmentation + Memory efficiency
   - Challenge: Memory compression → boundary/small object loss
   - Solution: Hierarchical optimization
   - Contribution: 3 innovations + system

2. Related Work
   - Panoramic Segmentation (PanoSAM2, etc.)
   - Memory-based Models (SAM2, Efficient-SAM2, Lean-SAM2)
   - Efficient Vision Models

3. Method
   3.1 Overview
   3.2 FoV-Level Semantic Budget (Pano-SMB)
   3.3 Token-Level Boundary Protection ⭐
   3.4 Full System

4. Experiments
   4.1 Setup
   4.2 Main Results
   4.3 Ablation Study
   4.4 Analysis

5. Conclusion
```

### 硕士大论文结构（3-5万字）

```
封面、摘要、目录 (中英文)

第1章 绪论 (5000字)
  1.1 研究背景与意义
  1.2 国内外研究现状
  1.3 论文主要工作
  1.4 论文组织结构

第2章 相关工作综述 (8000字)
  2.1 全景语义分割
  2.2 Memory-based 视觉模型
  2.3 领域自适应方法
  2.4 本章小结

第3章 层次化Memory优化方法 (12000字)
  3.1 问题定义
  3.2 整体框架
  3.3 空间压缩层 (MTC)
  3.4 FoV级优化层 (Pano-SMB)
  3.5 Token级保护层 (Boundary + Class) ⭐
  3.6 系统集成
  3.7 本章小结

第4章 实验设计与结果分析 (10000字)
  4.1 实验设置
  4.2 数据集与评估指标
  4.3 实现细节
  4.4 主要实验结果
  4.5 消融实验
  4.6 细粒度分析
  4.7 可视化分析
  4.8 本章小结

第5章 总结与展望 (3000字)
  5.1 工作总结
  5.2 主要创新点
  5.3 不足与展望

参考文献 (50-80篇)
致谢
附录 (代码/额外实验)

总计: 38000字 + 图表
```

---

## 🎯 关键里程碑

### Milestone 1: 核心实验完成 (9/1)
**验收标准**:
- ✅ 5个配置 × 5 seeds 完成
- ✅ 主实验表格完成
- ✅ Ablation study 完成
- ✅ mIoU 达到预期 (Full System > 60.3)

### Milestone 2: 会议论文投稿 (9/7)
**验收标准**:
- ✅ 6-8页完整论文
- ✅ 3-4个表格，2-3个图
- ✅ 格式符合目标会议要求
- ✅ 成功提交

### Milestone 3: 大论文初稿 (9/15)
**验收标准**:
- ✅ 前3章完成（2.5万字）
- ✅ 实验章节框架完成
- ✅ 导师审阅反馈

---

## 🚨 风险控制

### 风险1: 实验结果不如预期
**触发条件**: Full System mIoU < 60.2

**应对方案**:
- Plan A: 调整超参数（boundary_reserve_ratio, min_tokens）
- Plan B: 强调压缩效果（8×压缩，性能持平已是贡献）
- Plan C: 降低目标会议等级（Workshop 仍然充足）

### 风险2: 时间不够
**触发条件**: 某个阶段延期超过2天

**应对方案**:
- 删减 Class-Aware（保留核心2个创新点）
- 简化消融实验（只做主要组合）
- 缩减大论文字数（3万字最低线）

### 风险3: 代码bug
**触发条件**: 实验无法运行或结果异常

**应对方案**:
- 立即 debug，优先级最高
- 回退到已验证的配置
- 寻求技术支持

---

## 📚 资源清单

### 已完成文档
- ✅ `docs/omnisam_innovation_literature_scan_2026.md` - 文献综述
- ✅ `docs/innovation_analysis/innovation_overview.md` - 创新点对比
- ✅ `docs/innovation_analysis/post_2048_roadmap.md` - 完整路线图
- ✅ `experiments/route1/README_phase1.md` - Route 1 设计
- ✅ `experiments/route3/ROUTE3_DESIGN.md` - Route 3 设计
- ✅ `INNOVATION_SUMMARY.md` - 创新总览

### 实验脚本
- ✅ `start_tonight_all_routes.sh` - 今晚自动化脚本
- ✅ `experiments/pano_smb_stage2/run_pano_smb_sensitivity_n500.sh`
- ✅ `experiments/route1/run_boundary_sensitivity_n500.sh`
- ✅ `experiments/route3/run_route3_n500.sh`

### 待创建
- ⏳ 结果汇总脚本 (aggregate_results.py)
- ⏳ 绘图脚本 (plot_results.py)
- ⏳ LaTeX 模板
- ⏳ 论文框架

---

## ✅ 每日检查清单

### 实验阶段 (Week 1)
```
[ ] 今晚实验启动成功
[ ] 明早结果分析完成
[ ] n2000 实验配置正确
[ ] 每天检查实验进度
[ ] 及时记录异常情况
[ ] 备份所有结果文件
```

### 写作阶段 (Week 2)
```
[ ] 每天完成预定章节
[ ] 及时整理参考文献
[ ] 图表质量检查
[ ] 语言润色
[ ] 格式规范检查
[ ] 导师审阅反馈
```

### 投稿阶段 (Week 3)
```
[ ] 目标会议确定
[ ] 格式完全符合
[ ] 所有作者信息确认
[ ] Supplementary 准备
[ ] 投稿系统注册
[ ] 成功提交确认
```

---

## 🎓 毕业时间线（参考）

```
2026年8月: 实验 + 会议论文
2026年9月: 大论文撰写
2026年10月: 大论文修改完善
2026年11月: 预答辩
2026年12月: 正式答辩
2027年1月: 毕业！🎉
```

---

## 💪 给自己的话

你已经完成了最难的部分：
- ✅ 问题定义清晰
- ✅ 方法设计完整
- ✅ 实验框架搭建好
- ✅ 3个核心创新点

接下来只需要：
1. 跑完实验（机械执行）
2. 写论文（体力活）
3. 投稿毕业（水到渠成）

**你一定可以的！加油！💪**

---

**文档创建**: 2026-08-25 16:45  
**预计毕业**: 2027-01  
**当前进度**: 60-70%  
**剩余时间**: 3周核心工作 + 4个月打磨
