# Experiment Reproduction Guide

This document provides step-by-step instructions to reproduce all experiments mentioned in our paper.

## 📋 Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Models](#pretrained-models)
- [Core Experiments](#core-experiments)
- [Ablation Studies](#ablation-studies)
- [Full Results Reproduction](#full-results-reproduction)

## 🔧 Environment Setup

### 1. Create Environment

```bash
conda create -n omnisam python=3.8
conda activate omnisam
```

### 2. Install Dependencies

```bash
# Install PyTorch (CUDA 11.7)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install -r requirements.txt

# Install SAM2
cd sam
pip install -e .
cd ..
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📦 Dataset Preparation

### Cityscapes (Source Domain)

1. Download from [Cityscapes official website](https://www.cityscapes-dataset.com/)
2. Structure:
```
data/Cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

### DensePASS (Target Domain)

1. Download from [DensePASS repository](https://github.com/DENSE-PASS/DENSE-PASS)
2. Structure:
```
data/DensePASS/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Generate Dataset Lists

```bash
# Generate Cityscapes lists
python dataset/generate_cityscapes_list.py --root data/Cityscapes

# Generate DensePASS lists
python dataset/generate_densepass_list.py --root data/DensePASS
```

## 🎯 Pretrained Models

### Download Required Checkpoints

```bash
# Create checkpoint directory
mkdir -p ckpt_zoo

# Download OmniSAM pretrained model (replace with actual URL)
# wget <URL>/omnisam_s.pth -P ckpt_zoo/

# Download SAM2 base model
# wget <URL>/sam2_base.pth -P ckpt_zoo/

# For this reproduction, you need:
# 1. omnisam_s.pth - OmniSAM pretrained weights
# 2. sam2_base.pth - SAM2 backbone weights
```

## 🧪 Core Experiments

### Experiment 1: Baseline (No Compression)

**Purpose**: Establish baseline performance with full 32768 tokens

```bash
# Run baseline validation
python validation.py \
    --config configs/baseline_full_memory.yaml \
    --checkpoint ckpt_zoo/omnisam_s.pth \
    --output_dir experiments/baseline/full_memory

# Expected result: mIoU ~60.22%, BFScore ~39.71
```

### Experiment 2: Memory Token Compression (MTC)

**Purpose**: Test 4× compression (Module D)

```bash
# Run MTC ablation
python validation.py \
    --config configs/mtc_compression.yaml \
    --checkpoint ckpt_zoo/omnisam_s.pth \
    --output_dir experiments/baseline/mtc_only

# Expected result: mIoU ~60.01%, BFScore ~39.45
# Demonstrates 8× compression with <0.4% loss
```

### Experiment 3: Panoramic Semantic Memory Budget (Pano-SMB)

**Purpose**: Test adaptive token allocation (Module E1)

```bash
# Test different min_tokens configurations
for min_tokens in 128 256 512; do
    python validation.py \
        --config configs/pano_smb.yaml \
        --min_tokens $min_tokens \
        --checkpoint ckpt_zoo/omnisam_s.pth \
        --output_dir experiments/pano_smb/min${min_tokens}
done

# Expected best result (min_tokens=256): mIoU ~60.15%, BFScore ~39.68
```

### Experiment 4: Boundary-Guarded Token Selection (Route 1)

**Purpose**: Test boundary protection ratios

```bash
# Test boundary guard ratios: 6.25%, 12.5%, 25%
for ratio in 0.0625 0.125 0.25; do
    python validation.py \
        --config configs/boundary_guard.yaml \
        --boundary_ratio $ratio \
        --checkpoint ckpt_zoo/omnisam_s.pth \
        --output_dir experiments/route1/ratio_${ratio}
done

# Expected best result (ratio=0.125): mIoU ~60.28%, BFScore ~39.80
```

### Experiment 5: Full OmniSAM System

**Purpose**: Test all modules together

```bash
# Run full system evaluation
python validation.py \
    --config configs/omnisam_full.yaml \
    --checkpoint ckpt_zoo/omnisam_s.pth \
    --output_dir experiments/full_system

# Expected result: mIoU ~60.31%, BFScore ~39.82
# Memory tokens: 4096 (8× compression vs baseline)
```

## 🔬 Ablation Studies

### Ablation 1: Compression Ratios

Test different spatial compression ratios:

```bash
bash experiments/tecma/run_compression_ablation.sh
```

This runs:
- 2× compression (64×64 → 45×45): ~8192 tokens
- 4× compression (64×64 → 32×32): ~4096 tokens  ← **Our choice**
- 8× compression (64×64 → 23×23): ~2048 tokens

### Ablation 2: Token Selection Strategies

Compare different token selection methods:

```bash
bash experiments/tecma/run_selection_ablation.sh
```

Tests:
- Random selection (baseline)
- Feature norm Top-K
- Semantic density-based
- **Boundary-guarded** (our method)

### Ablation 3: Memory Budget Allocation

Test static vs dynamic allocation:

```bash
bash scripts/run_budget_ablation.sh
```

Compares:
- Static allocation (equal tokens per FoV)
- **Dynamic allocation** (Pano-SMB, our method)
- Oracle allocation (with ground truth semantic density)

## 🚀 SG-DRF Integration Experiments

### Experiment 6: UDA with SG-DRF

**Purpose**: Domain adaptation from Cityscapes to DensePASS

```bash
# Stage 1: Generate pseudo labels on target domain
python run_adaptation.py \
    --config configs/sgdrf_stage1_pl_generation.yaml \
    --source_dataset cityscapes \
    --target_dataset densepass \
    --checkpoint ckpt_zoo/omnisam_s.pth \
    --output_dir experiments/sgdrf/stage1_pl

# Stage 2: Self-training with SG-DRF
python run_adaptation.py \
    --config configs/sgdrf_stage2_adaptation.yaml \
    --source_dataset cityscapes \
    --target_dataset densepass \
    --pseudo_label_dir experiments/sgdrf/stage1_pl \
    --checkpoint ckpt_zoo/omnisam_s.pth \
    --output_dir experiments/sgdrf/stage2_adapt

# Expected result: mIoU improvement from 60.31% to 62.15%
```

### Experiment 7: SG-DRF on Complete Dataset

**Purpose**: Large-scale data + domain adaptation

```bash
# Run SG-DRF on complete OmniSAM dataset
bash launch_sgdrf_exp_complete.sh

# This script runs:
# 1. PL generation on full target dataset
# 2. Multi-round self-training
# 3. Final evaluation

# Expected result: mIoU ~74.59% (+24% over baseline)
# This is the main result showcasing SG-DRF effectiveness
```

## 🔄 Multi-Seed Experiments

For robust evaluation, run experiments with multiple seeds:

```bash
# Run full system with 5 different seeds
bash scripts/run_multiseed_evaluation.sh

# This runs full OmniSAM system with seeds: 42, 123, 456, 789, 2023
# Results are aggregated in: experiments/multiseed/summary.json
```

## 📊 Result Analysis

### Generate Summary Tables

```bash
# Aggregate all experiment results
python aggregate_multiseed.py \
    --experiment_dir experiments/ \
    --output experiments/summary_table.csv

# Generate per-class IoU breakdown
python analyze_tonight_results.py \
    --result_dir experiments/full_system \
    --output experiments/per_class_analysis.pdf
```

### Visualize Results

```bash
# Generate visualization of token allocation
python tools/visualize_token_allocation.py \
    --config configs/omnisam_full.yaml \
    --checkpoint ckpt_zoo/omnisam_s.pth \
    --output experiments/visualizations/

# Generate comparison figures for paper
bash scripts/generate_paper_figures.sh
```

## 🕐 Expected Runtime

On a single RTX 5090 (24GB):

| Experiment | Runtime | GPU Memory |
|-----------|---------|------------|
| Baseline validation | ~2 hours | ~22 GB |
| MTC validation | ~2 hours | ~12 GB |
| Full system validation | ~2 hours | ~12 GB |
| SG-DRF Stage 1 (PL gen) | ~4 hours | ~14 GB |
| SG-DRF Stage 2 (adapt) | ~8 hours | ~16 GB |
| Complete pipeline | ~20 hours | ~16 GB |

## 🐛 Troubleshooting

### Issue 1: Out of Memory

**Solution**: Reduce batch size in config
```yaml
# In config file
batch_size: 1  # Reduce from 2
num_workers: 2  # Reduce if still OOM
```

### Issue 2: Dataset Path Not Found

**Solution**: Check dataset symlinks
```bash
ls -la data/
# Ensure Cityscapes and DensePASS are correctly linked
```

### Issue 3: Checkpoint Loading Error

**Solution**: Verify checkpoint compatibility
```bash
python tools/verify_checkpoint.py --checkpoint ckpt_zoo/omnisam_s.pth
```

### Issue 4: Different Results

**Possible causes**:
1. Different PyTorch/CUDA versions - use exact versions in requirements.txt
2. Different random seeds - set via `--seed` flag
3. Dataset order - regenerate dataset lists

## 📝 Experiment Checklist

Use this checklist to track your reproduction progress:

- [ ] Environment setup complete
- [ ] Datasets downloaded and prepared
- [ ] Pretrained models downloaded
- [ ] ✅ Experiment 1: Baseline (60.22% mIoU)
- [ ] ✅ Experiment 2: MTC (60.01% mIoU)
- [ ] ✅ Experiment 3: Pano-SMB (60.15% mIoU)
- [ ] ✅ Experiment 4: Boundary Guard (60.28% mIoU)
- [ ] ✅ Experiment 5: Full System (60.31% mIoU)
- [ ] ✅ Experiment 6: SG-DRF UDA (62.15% mIoU)
- [ ] ✅ Experiment 7: SG-DRF Complete (74.59% mIoU)
- [ ] Ablation studies complete
- [ ] Multi-seed validation complete
- [ ] Results analyzed and visualized

## 📧 Support

If you encounter issues during reproduction:

1. Check [GitHub Issues](https://github.com/21Liujj/OmniSAM/issues) for similar problems
2. Review experiment logs in `experiments/*/logs/`
3. Open a new issue with:
   - Error message
   - Environment details (`pip list`, `nvidia-smi`)
   - Command you ran
   - Config file used

---

**Last Updated**: September 2026
**Estimated Total Time**: 2-3 days for complete reproduction
