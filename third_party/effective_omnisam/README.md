# OmniSAM: Memory-Efficient Panoramic Segmentation with Boundary-Aware Reasoning

[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**高效全景语义分割框架，基于记忆优化和边界感知推理**

> 🎯 **核心成就**: 在仅损失0.24% mIoU的情况下，实现**55.8%注意力时间降低**和**1.2GB显存节省**，同时通过循环记忆使交通标志类别提升**+5.04%**

## 📋 目录
- [概述](#概述)
- [核心创新](#核心创新)
- [架构设计](#架构设计)
- [实验结果](#实验结果)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [实验复现](#实验复现)
- [项目结构](#项目结构)
- [引用](#引用)

## 🌟 概述

OmniSAM是一个基于SAM2的全景语义分割研究项目，通过以下三大创新实现内存优化和边界感知推理：

1. **MTC (Memory Token Compressor)** - 8×记忆Token压缩
2. **SMR (Sparse Memory Retrieval)** - 稀疏记忆检索，50%稀疏率
3. **SCM (Seamless Circular Memory)** - 无缝循环记忆填充

### 核心结果

| 方法 | mIoU (%) | 注意力时间(s) | 显存(GB) | Token数 |
|------|----------|---------------|----------|---------|
| OmniSAM基线 | 60.22±0.02 | 1.13 | 6.8 | 32768 |
| **+ MTC+SMR** | **60.00±0.05** | **0.50** | **5.6** | **2048** |
| 性能变化 | -0.24% | **-55.8%** | **-1.2GB** | **16×压缩** |

**SCM循环记忆结果** (在MTC+SMR基础上):

| Pad Width | mIoU (%) | Δ | 交通标志IoU | Δ |
|-----------|----------|---|-------------|---|
| 基线(推理) | 60.51 | - | 26.07 | - |
| 1024 | **60.84** | **+0.33** | **31.11** | **+5.04 (+19.3%)** |

> 📊 **关键发现**: 循环记忆填充使边界敏感类别（交通标志）获得显著提升，验证了跨边界上下文的重要性

## 🚀 核心创新

### 1. Memory Token Compressor (MTC)
- **8×空间池化压缩** (64×64 → 32×32 → 8×8)
- 无参数，O(N)复杂度
- Token数: 32768 → 4096
- 性能损失: <0.4% mIoU

### 2. Sparse Memory Retrieval (SMR)
- **50%稀疏率**的记忆检索
- 基于特征重要性的Top-K选择
- 注意力时间降低: **55.8%** (1.13s → 0.50s)
- 显存节省: **1.2GB** (6.8GB → 5.6GB)
- 有效压缩率: **16×** (8× MTC + 50% SMR)

### 3. Seamless Circular Memory (SCM)
- **循环填充**机制跨越全景边界
- Padding width: 256/512/768/1024 pixels
- 交通标志类别提升: **+5.04%** (相对增益19.3%)
- 整体mIoU提升: **+0.33%**

### 多随机种子验证
- **5个种子** (1, 2, 3, 42, 100) 用于基线
- **4个种子** (1, 2, 3, 42) 用于MTC+SMR
- 标准差: **<0.05%**，统计显著性良好

## 🏗️ 架构设计

```
┌──────────────────────┐
│  SAM2 Memory Encoder │
│  每FoV: 4096 tokens  │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  MTC (Module D)      │  ← 创新1: 8×压缩
│  64×64 → 32×32 → 8×8 │
│  32768 → 4096 tokens │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  SMR (Module E)      │  ← 创新2: 稀疏检索
│  Top-50% Selection  │
│  4096 → 2048 tokens  │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  SCM (Circular Pad) │  ← 创新3: 循环记忆
│  跨边界上下文融合    │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│   SAM2 Memory Attn   │
│   高效注意力计算     │
└──────────────────────┘
```

## 📊 实验结果

### 表1: 主要结果（多随机种子验证）
| 方法 | Seeds | mIoU (%) | 注意力时间(s) | 显存(GB) |
|------|-------|----------|---------------|----------|
| OmniSAM基线 | 5 | 60.22±0.02 | 1.13 | 6.8 |
| + MTC+SMR (A+D+E) | 4 | 60.00±0.05 | 0.50 | 5.6 |
| **性能变化** | - | **-0.24%** | **-55.8%** | **-17.6%** |

### 表2: 效率分析
| 配置 | Tokens/Memory | 注意力时间(s) | mIoU (%) |
|------|---------------|---------------|----------|
| 基线 | 32768 | 1.13 | 60.22 |
| + MTC (8×) | 4096 | 0.65 | 60.08 |
| + MTC+SMR (50%) | **2048** | **0.50** | **60.00** |

### 表3: SCM循环记忆消融
| Pad Width | mIoU (%) | Δ vs Baseline |
|-----------|----------|---------------|
| 基线(推理) | 60.51 | - |
| 256 | 60.60 | +0.09 |
| 512 | 60.72 | +0.21 |
| 768 | 60.80 | +0.29 |
| **1024** | **60.84** | **+0.33** |

> **趋势**: 单调递增，验证"更多跨边界上下文→更好性能"假设

### 表4: SCM类别级别分析
| 类别 | 基线 (%) | SCM (%) | Δ (%) | 相对增益 | 类型 |
|------|----------|---------|-------|----------|------|
| **traffic sign** | 26.07 | **31.11** | **+5.04** | **+19.3%** | 边界敏感 |
| wall | 45.19 | 46.07 | +0.88 | +1.9% | 边界类别 |
| fence | 49.43 | 49.69 | +0.26 | +0.5% | 边界类别 |
| road | 81.00 | 80.91 | -0.09 | -0.1% | 大面积 |
| building | 89.96 | 89.76 | -0.20 | -0.2% | 大面积 |
| sky | 94.60 | 94.58 | -0.02 | -0.0% | 大面积 |

> **关键发现**: 交通标志等边界敏感类别获得显著提升，大面积类别保持稳定

## 🔧 安装说明

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (GPU训练)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/21Liujj/OmniSAM.git
cd OmniSAM

# 创建conda环境
conda create -n omnisam python=3.8
conda activate omnisam

# 安装PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# 安装依赖
pip install -r requirements.txt

# 安装SAM2
cd sam
pip install -e .
cd ..
```

### 数据集准备

详细的数据集准备说明请参考 [docs/REPRODUCE.md](docs/REPRODUCE.md)

## 🚀 快速开始

### 1. 下载预训练模型

```bash
# 下载OmniSAM模型权重
# wget <URL>/omnisam_s.pth -P ckpt_zoo/

# 下载SAM2基础模型
# wget <URL>/sam2_base.pth -P ckpt_zoo/
```

### 2. 运行推理

```bash
# 在DensePASS验证集上运行推理
python validation.py \
    --config configs/omnisam_densepass.yaml \
    --checkpoint ckpt_zoo/omnisam_s.pth
```

### 3. 从头训练

```bash
# 在Cityscapes上训练OmniSAM
python train.py \
    --config configs/train_cityscapes.yaml \
    --output_dir outputs/omnisam_train
```

## 🔬 实验复现

详细的实验复现指南请参考 [docs/REPRODUCE.md](docs/REPRODUCE.md)

### 核心实验

```bash
# 1. 基线实验 (5个随机种子)
bash scripts/run_baseline_multiseed.sh

# 2. MTC+SMR实验 (4个随机种子)
bash scripts/run_mtc_smr_multiseed.sh

# 3. SCM循环记忆消融
bash scripts/run_scm_ablation.sh

# 4. 完整系统评估
bash scripts/run_full_evaluation.sh
```

## 📁 项目结构

```
OmniSAM/
├── model/              # 模型实现
│   ├── model.py       # 主OmniSAM模型
│   └── modules/
│       ├── memory_token_compressor.py  # MTC模块
│       ├── sparse_memory_retrieval.py  # SMR模块
│       └── circular_memory.py          # SCM模块
├── configs/            # 配置文件
├── dataset/            # 数据集加载器
├── docs/               # 文档
│   └── REPRODUCE.md   # 实验复现指南
├── experiments/        # 实验脚本和结果
│   └── tecma/         # 记忆实验
├── paper/              # 论文稿件和图表
│   └── EI_Conference_Paper_v2_CN/  # EI会议论文
├── sam/                # SAM2集成
├── scripts/            # 实用脚本
├── train/              # 训练模块
├── utils/              # 工具函数
├── train.py           # 训练入口
└── validation.py      # 验证入口
```

## 📖 文档

- [安装指南](docs/INSTALL.md)
- [数据集准备](docs/DATASET.md)
- [实验复现](docs/REPRODUCE.md)
- [创新分析](INNOVATION_SUMMARY.md)
- [技术细节](INNOVATION_DETAILED_EXPLANATION.md)

## 📝 引用

如果这项工作对您有帮助，请引用：

```bibtex
@inproceedings{omnisam2026,
  title={OmniSAM: Memory-Efficient Panoramic Segmentation with Boundary-Aware Reasoning},
  author={Your Name},
  booktitle={Conference Name},
  year={2026}
}
```

## 🙏 致谢

本项目基于：
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Meta的Segment Anything Model 2
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [DensePASS Dataset](https://github.com/DENSE-PASS/DENSE-PASS)

## 📄 许可证

本项目仅供研究使用。详见 [LICENSE](LICENSE)

## 📧 联系方式

问题和讨论:
- GitHub Issues: [创建issue](https://github.com/21Liujj/OmniSAM/issues)

---

**状态**: 🚧 活跃研究项目 | 最后更新: 2026年9月
