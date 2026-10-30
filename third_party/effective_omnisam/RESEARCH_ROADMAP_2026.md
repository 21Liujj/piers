# OmniSAM 双线研究方案 (2026.8-12)

## 总体目标
- **小论文 (EI会议, 9月提交)**: Seamless Circular Memory + Class-Adaptive Compression
- **大论文 (硕士毕业, 12月完成)**: 小论文基础 + Latitude-aware PE + 完整消融

---

## 核心创新点技术方案

### Innovation 1: Seamless Circular Memory Bank (SCM)
**问题**: SAM2在360°全景图x=0/x=2048接缝处memory token断裂

**文献支撑**:
- Cylinder-Style Convolutions (2022) - learnable positional embedding处理接缝
- Spherical Convolution (2024) - 球面采样处理wraparound
- PanoSAM2 (2026) - SAM2在360视频的畸变适配

**技术要点**:
1. Circular padding at dataloader: 左右边界互相拼接
2. Wrap-around memory indexing: x_wrapped = x % image_width
3. Seam consistency loss: 强制x=0和x=2048预测一致

**预期提升**: +0.10-0.15 mIoU (接缝区域+0.5-1.0)

### Innovation 2: Class-Adaptive Compression (CAC)
**问题**: 固定8×压缩对rare class(terrain/pole/sign)伤害更大

**证据**: failure analysis发现terrain→sidewalk退化518px, 假阳性pole插入143px

**技术要点**:
1. 计算每patch的rare class占比 (rarity score)
2. High rarity patch: token budget × 1.5
3. Low rarity patch: token budget × 0.8
4. 归一化到总budget 4096

**预期提升**: +0.06-0.10 mIoU (rare class IoU +2-3点)

### Innovation 3: Latitude-aware Positional Encoding (LPE)
**问题**: DensePASS ±35°裁剪带仍有俯仰畸变，标准2D PE无法适配

**理论**: ERP在纬度θ处拉伸因子 = 1/cos(θ)

**技术要点**:
1. 纬度映射: θ(y) = (y/400 - 0.5) × 70°
2. PE scaling: pe_scaled = pe × (1/cos(θ))
3. 注入SAM2 image encoder第一层

**预期提升**: +0.08-0.12 mIoU (top/bottom strip +0.3-0.5)

---

## 时间表

### 阶段1: 核心开发 (8/28-9/7, 10天)
- **8/28今晚**: 验证seam区域mIoU是否低于center (motivation)
- **8/29-8/31**: 实现SCM (3天)
  - 修改 dataset/DensePASS.py 加circular padding
  - 修改 sam2/modeling/memory_attention.py 加wrap-around
  - 修改 adaptation/train_uda.py 加seam consistency loss
- **9/1-9/3**: SCM对比实验 (3天)
  - Baseline vs +SCM在100张val上
  - 接缝区域vs中心区域mIoU统计
- **9/4-9/6**: 实现CAC (3天)
  - 扩展 spatial_memory_router.py 加class-aware budget
  - 伪标签生成时加per-patch class statistics
- **9/7**: 消融实验
  - Baseline / +SCM / +CAC / +SCM+CAC

**里程碑**: Baseline 60.02 → +SCM+CAC: 60.20+

### 阶段2: 小论文撰写 (9/8-9/15, 8天)
**目标**: 6-8页会议论文

**结构**:
- Title: Seamless Memory Optimization for Panoramic Semantic Segmentation
- Abstract (200词)
- Introduction (1.5页) - 全景分割重要性 + SAM2瓶颈 + 360°接缝+rare class问题
- Related Work (1页) - Panoramic seg / Memory models / Distortion-aware
- Method (2.5页) - SCM设计 + CAC设计 + 协同机制
- Experiments (2页) - DensePASS结果 + 消融 + 可视化
- Conclusion (0.5页)

**目标会议**: IEEE ITSC 2026, IEEE IV 2026, ICRA 2027 Workshop

**撰写安排**:
- 9/8-9/10: Method + Experiments (Table 1-2, Fig 3)
- 9/11-9/13: Introduction + Related Work
- 9/14: Conclusion + 全文润色
- 9/15: LaTeX编译 + Supplementary materials

### 阶段3: 大论文扩展 (9/16-10/15, 30天)
- **9/16-9/25**: 实现LPE (10天)
  - 纬度-畸变系数推导验证
  - 修改SAM2 image encoder注入LPE
  - Smoke test确认不破坏原有性能
- **9/26-10/5**: 消融实验8配置 (10天)
  - 跑完SCM/CAC/LPE的所有组合
  - 每个配置3-seed确认稳定性
- **10/6-10/10**: 纵向分层分析 (5天)
  - 5个纬度带的mIoU统计
  - 可视化top/bottom strip提升
- **10/11-10/15**: 整理实验结果 (5天)
  - Table: 8配置消融矩阵
  - Fig: 接缝/纵向/per-class三张图

**里程碑**: +SCM+CAC+LPE: 60.28-60.32

### 阶段4: 硕士论文撰写 (10/16-11/30, 45天)
**目标**: 30,000-50,000字中文硕士论文

**章节结构** (8章):
1. 绪论 (8000字) - 背景+国内外现状+创新点
2. 技术基础 (6000字) - ERP+SAM2+UDA
3. 问题分析 (5000字) - Memory瓶颈+接缝断裂+rare class+畸变
4. SCM设计 (7000字) - 架构+实现+实验
5. CAC设计 (7000字) - 方法+协同+实验
6. LPE设计 (7000字) - 数学建模+实现+实验
7. 综合实验 (8000字) - 主实验+消融+分析
8. 总结展望 (2000字)

**时间安排**:
- 10/16-10/25: 第1-2章 (背景+基础) - 10天
- 10/26-11/05: 第3-5章 (问题+SCM+CAC) - 11天
- 11/06-11/15: 第6-7章 (LPE+实验) - 10天
- 11/16-11/25: 第8章+全文润色+查重 - 10天
- 11/26-11/30: 格式调整+打印装订 - 5天

---

## 关键文献清单 (48篇核心)

### A. Circular/Seamless处理 (8篇)
1. ✅ Seamless 360° Outpainting (2022) - Cylinder-style convolutions
2. ✅ 360° Full-view Segmentation (2024) - Spherical convolution
3. Occlusion-Aware Seamless (2024)
4. Seam-aware ERP (2026)
5. World-Shaper (ICML 2026)
6. Semantic Edit 360° (2024)
7. Circular CNN (IV 2019)
8. Deep Spherical Superpixels (2024)

### B. SAM2与Memory压缩 (10篇)
9. ✅ SAM2 (ECCV 2024)
10. ✅ TinySAM2 (2025) - Spatial+temporal token compression
11. PanoSAM2 (2026) - Distortion-aware SAM2 for 360 VOS
12. SAM2Act (2025)
13. Accelerating SAM2 (2026)
14. XMem (ECCV 2022)
15. STCN (NeurIPS 2021)
16. AOT (NeurIPS 2021)
17. Cutie (2023)
18. SAM-PT (2023)

### C. Distortion-aware与PE (12篇)
19. ✅ Distortion-Aware Spherical (2024) - 1/cos(θ) formula
20. ✅ Geometry-aware VAE (2026)
21. ✅ Bi-Projection Fusion (2024) - Latitude-dependent bias
22. Panoramic Dual BridgeNet (2024)
23. SphereUFormer (CVPR 2025)
24. SphereNet (ECCV 2018)
25. Kernel Transformer (CVPR 2021)
26. BiFuse (ECCV 2020)
27. HoHoNet (CVPR 2021)
28. 360-Indoor (CVPR 2022)
29. PanoFormer (ECCV 2022)
30. Positional Encoding Field (2024)

### D. 全景语义分割 (10篇)
31. ✅ Trans4PASS (CVPR 2023)
32. ✅ GoodSAM (CVPR 2024)
33. ✅ Semantics Distortion (CVPR 2024)
34. ✅ Seeing Beyond (CVPR 2026)
35. ✅ Denoise and Align (CVPR 2026)
36. ✅ Open Panoramic (2024)
37. Geometric Exploitation (NeurIPS 2024)
38. Panoramic Survey (2026)
39. Single Frame Multi-Modal (2023)
40. Efficient 360 Multi-task (2024)

### E. Domain Adaptation (8篇)
41. DensePASS (ITSC 2021)
42. DACS (CVPR 2021)
43. DAFormer (CVPR 2022)
44. HRDA (ECCV 2022)
45. MIC (CVPR 2023)
46. ProDA (CVPR 2021)
47. RefineNet (CVPR 2022)
48. SFUDA baseline methods

---

## 今晚立即行动 (8/28)

### Task 1: 验证接缝假设 (30分钟)
**目标**: 证明seam区域mIoU < center区域

**执行**:
```bash
cd /root/autodl-tmp/OmniSAM-main
# 创建诊断脚本
# 统计A_Dtrain_Etrain在seam vs center的mIoU
```

**预期结果**:
- Seam region (x<100, x>1948): 约58.5-59.0 mIoU
- Center region (x∈[900,1148]): 约60.5-61.0 mIoU  
- Gap: -1.5到-2.5 mIoU

如果gap<-1.0，则SCM的motivation非常强！

### Task 2: 文献精读 (2小时)
**优先级排序**:
1. **Seamless 360° Outpainting** - 了解cylinder-style方法
2. **360° Full-view Segmentation** - 了解spherical convolution
3. **TinySAM2** - 了解memory compression baseline
4. **Distortion-Aware Spherical** - 记住1/cos(θ)公式

**每篇记录**:
- 核心方法 (1段话)
- 与我们工作的区别 (2-3句)
- 可引用的key insight (1句原文)

---

## 需要确认的问题

1. **今晚优先级**: 先写seam验证脚本跑数据? 还是先看2篇文献找感觉?

2. **创新点优先级**: 
   - SCM (接缝) > CAC (类别) > LPE (畸变)
   - 是否认可这个顺序?

3. **小论文目标会议**:
   - IEEE ITSC 2026 (智能交通)
   - IEEE IV 2026 (智能车辆)
   - ICRA 2027 Workshop
   - 有倾向吗?

4. **大论文时间**: 
   - 11月底能完成30k字初稿吗?
   - 需要连续45天专注写作

5. **资源需求**:
   - 8配置消融需要~80小时GPU (10天×8小时)
   - 5090能hold住吗?

---

## 风险与应对

### 风险1: SCM提升<0.05 mIoU
**应对**: 
- 扩大seam width 100→200
- 强化seam loss权重 0.1→0.5
- 改用"conceptual novelty"角度写(首次在SAM2上处理360接缝)

### 风险2: LPE在±35°裁剪带效果弱
**应对**:
- 补充Stanford2D3D (标准±90° ERP)实验
- 或改成adaptive PE让模型自学distortion

### 风险3: 9月投稿被拒
**应对**:
- Backup: ICRA 2027 workshop / IEEE CASE 2027
- 或直接做完LPE后投CVPR 2027 workshop

---

确认后立即开始执行！我会先帮你写seam验证脚本，然后整理文献阅读笔记模板。
