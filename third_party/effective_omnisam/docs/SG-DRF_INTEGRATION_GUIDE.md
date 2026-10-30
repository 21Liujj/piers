# SG-DRF Integration Guide for OmniSAM
# =====================================
# 
# This guide explains how to enable and configure SG-DRF (Semantic-Guided Dynamic Receptive Field)
# in OmniSAM for panoramic semantic segmentation.

## What is SG-DRF?

SG-DRF is the third innovation that complements OmniSAM's existing memory optimizations:
- **Innovation #1 (Pano-SMB)**: FoV-level adaptive memory allocation
- **Innovation #2 (Boundary-Guard)**: Token-level boundary protection  
- **Innovation #3 (SG-DRF)**: Feature-level adaptive receptive field

SG-DRF consists of three components:
1. **Semantic Density Estimator**: Predicts semantic complexity map (0-1 scale)
2. **Dynamic FSMC**: Adaptive memory compression (4×-16× based on density)
3. **Dynamic SCM**: Adaptive circular padding (512-1536px based on boundary density)

## How to Enable SG-DRF

### Option 1: Add to training script (train.py)

When building the model, pass SG-DRF parameters to build_model():

```python
# In utils/build_ddp.py or wherever build_model is defined
model = build_model(
    args,
    num_classes=num_classes,
    num_maskmem=args.num_maskmem,
    device=device,
    local_rank=local_rank,
    # NEW: SG-DRF parameters
    enable_sgdrf=True,
    sgdrf_backbone_channels=256,
    sgdrf_enable_dynamic_fsmc=True,
    sgdrf_enable_dynamic_scm=True,
    sgdrf_fsmc_min_ratio=4.0,
    sgdrf_fsmc_max_ratio=16.0,
    sgdrf_scm_base_pad=1024,
    sgdrf_scm_min_pad=512,
    sgdrf_scm_max_pad=1536,
    sgdrf_training_stage='joint',  # or 'cold_start' for phase 1
)
```

### Option 2: Add arguments to TrainConfig dataclass

```python
@dataclass
class TrainConfig:
    # ... existing fields ...
    
    # SG-DRF parameters
    enable_sgdrf: bool = field(
        default=False,
        metadata={"help": "Enable SG-DRF (Semantic-Guided Dynamic Receptive Field)"}
    )
    sgdrf_training_stage: str = field(
        default="joint",
        metadata={
            "help": "SG-DRF training stage: 'cold_start' or 'joint'",
            "choices": ["cold_start", "joint"]
        }
    )
    sgdrf_enable_dynamic_fsmc: bool = field(
        default=True,
        metadata={"help": "Enable Dynamic FSMC compression"}
    )
    sgdrf_enable_dynamic_scm: bool = field(
        default=True,
        metadata={"help": "Enable Dynamic SCM padding"}
    )
    sgdrf_fsmc_min_ratio: float = field(
        default=4.0,
        metadata={"help": "Min compression ratio for Dynamic FSMC (high density)"}
    )
    sgdrf_fsmc_max_ratio: float = field(
        default=16.0,
        metadata={"help": "Max compression ratio for Dynamic FSMC (low density)"}
    )
    sgdrf_scm_base_pad: int = field(
        default=1024,
        metadata={"help": "Base padding width for Dynamic SCM"}
    )
    sgdrf_scm_min_pad: int = field(
        default=512,
        metadata={"help": "Min padding width for Dynamic SCM"}
    )
    sgdrf_scm_max_pad: int = field(
        default=1536,
        metadata={"help": "Max padding width for Dynamic SCM"}
    )
    sgdrf_density_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for density pseudo-label loss"}
    )
    sgdrf_consistency_loss_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for density-segmentation consistency loss"}
    )
```

Then pass args to build_model():

```python
model = build_model(
    args,
    num_classes=num_classes,
    num_maskmem=args.num_maskmem,
    device=device,
    local_rank=local_rank,
    # SG-DRF parameters from args
    enable_sgdrf=args.enable_sgdrf,
    sgdrf_backbone_channels=256,  # Fixed based on SAM2 architecture
    sgdrf_enable_dynamic_fsmc=args.sgdrf_enable_dynamic_fsmc,
    sgdrf_enable_dynamic_scm=args.sgdrf_enable_dynamic_scm,
    sgdrf_fsmc_min_ratio=args.sgdrf_fsmc_min_ratio,
    sgdrf_fsmc_max_ratio=args.sgdrf_fsmc_max_ratio,
    sgdrf_scm_base_pad=args.sgdrf_scm_base_pad,
    sgdrf_scm_min_pad=args.sgdrf_scm_min_pad,
    sgdrf_scm_max_pad=args.sgdrf_scm_max_pad,
    sgdrf_training_stage=args.sgdrf_training_stage,
)

# Set loss weights as model attributes
if args.enable_sgdrf:
    model.module.sgdrf_density_loss_weight = args.sgdrf_density_loss_weight
    model.module.sgdrf_consistency_loss_weight = args.sgdrf_consistency_loss_weight
```

## Two-Stage Training Strategy

### Stage 1: Cold-Start (2 epochs recommended)

Train only the Semantic Density Estimator with pseudo-labels to initialize it.

```bash
python train.py \
    --enable_sgdrf \
    --sgdrf_training_stage cold_start \
    --num_epochs 2 \
    --lr 5e-5 \
    --save_ckpt_dir ./checkpoints/sgdrf_coldstart
```

What happens:
- All parameters frozen **except** SemanticDensityEstimator
- Loss = density_loss (MSE against Sobel-based pseudo-labels)
- Outputs stable density maps after 2 epochs

### Stage 2: Joint Training (30+ epochs)

Unfreeze all parameters and train end-to-end with combined losses.

```bash
python train.py \
    --enable_sgdrf \
    --sgdrf_training_stage joint \
    --num_epochs 30 \
    --lr 1e-4 \
    --load_ckpt ./checkpoints/sgdrf_coldstart/best_model.pth \
    --save_ckpt_dir ./checkpoints/sgdrf_joint
```

What happens:
- All parameters trainable
- Loss = seg_loss + 0.1 * density_loss + 0.05 * consistency_loss
- Dynamic FSMC and Dynamic SCM operate during forward pass
- Density estimator fine-tunes based on segmentation feedback

### Switching Stages Mid-Training

You can also switch stages programmatically:

```python
# After loading checkpoint from cold-start
if args.sgdrf_training_stage == 'joint':
    model.module.set_sgdrf_training_stage('joint')
    print("[SG-DRF] Switched to joint training mode")
```

## Expected Performance Improvements

Based on the design document (SG-DRF技术设计文档.md):

| Metric | Baseline | With SG-DRF | Improvement |
|--------|----------|-------------|-------------|
| Overall mIoU | 65.3 | 66.3 | +1.0% |
| Small Objects mIoU | 42.1 | 45.1 | +3.0% |
| GFLOPs | 180 | 170 | -5.6% |

Key improvements:
- **Traffic light**: +2.3% IoU
- **Pole**: +0.7% IoU
- **Fence**: +1.6% IoU
- **Reduced computation**: ~10 GFLOPs saved

## Ablation Study Configurations

To reproduce the ablation experiments from the design doc:

### E0: Baseline (No SG-DRF)
```bash
python train.py --enable_sgdrf False
```

### E1: Dynamic FSMC only
```bash
python train.py \
    --enable_sgdrf \
    --sgdrf_enable_dynamic_fsmc \
    --sgdrf_enable_dynamic_scm False
```

### E2: Dynamic SCM only
```bash
python train.py \
    --enable_sgdrf \
    --sgdrf_enable_dynamic_fsmc False \
    --sgdrf_enable_dynamic_scm
```

### E3: Full SG-DRF without cold-start
```bash
python train.py \
    --enable_sgdrf \
    --sgdrf_training_stage joint \
    # Skip cold-start phase
```

### E4: Full SG-DRF without consistency loss
```bash
python train.py \
    --enable_sgdrf \
    --sgdrf_training_stage joint \
    --sgdrf_consistency_loss_weight 0.0
```

### E5: Full SG-DRF (recommended)
```bash
# Stage 1
python train.py \
    --enable_sgdrf \
    --sgdrf_training_stage cold_start \
    --num_epochs 2

# Stage 2
python train.py \
    --enable_sgdrf \
    --sgdrf_training_stage joint \
    --num_epochs 30 \
    --load_ckpt <cold_start_ckpt>
```

## Hyperparameter Tuning Guide

### Compression Ratios (FSMC)
- **Conservative**: min=8, max=16 (less aggressive)
- **Recommended**: min=4, max=16 (balanced)
- **Aggressive**: min=2, max=16 (maximum efficiency)

### Padding Widths (SCM)
- **Conservative**: base=1024, min=768, max=1280
- **Recommended**: base=1024, min=512, max=1536
- **Aggressive**: base=1024, min=256, max=2048

### Loss Weights
- **density_loss_weight**: 0.1 (default), range [0.05, 0.2]
- **consistency_loss_weight**: 0.05 (default), range [0.01, 0.1]

## Troubleshooting

### Issue: Density loss not decreasing in cold-start
**Solution**: 
- Check pseudo-label generation is working (add debug prints)
- Increase learning rate to 1e-4
- Increase cold-start epochs to 3-4

### Issue: Segmentation performance degrades with SG-DRF
**Solution**:
- Reduce density_loss_weight to 0.05
- Reduce consistency_loss_weight to 0.01
- Ensure cold-start was run properly

### Issue: OOM (Out of Memory) errors
**Solution**:
- Disable Dynamic SCM (padding increases memory)
- Reduce max_pad_width to 1280
- Reduce batch size

### Issue: Training slower with SG-DRF
**Solution**:
- Expected ~5% slowdown due to density estimation
- Disable Dynamic SCM if speed is critical (SCM has higher overhead)
- Use mixed precision training (if not already)

## Visualization and Analysis

To visualize density maps during validation:

```python
# In validation loop
if args.enable_sgdrf and idx % 100 == 0:
    with torch.no_grad():
        pred, _, _, density_map, stats = model(
            0, images[:, 0],
            return_sgdrf_aux=True
        )
    
    # Save density map visualization
    import matplotlib.pyplot as plt
    plt.imsave(
        f'density_map_{idx}.png',
        density_map[0, 0].cpu().numpy(),
        cmap='hot'
    )
```

## Integration Checklist

- [x] model/model.py: Added SG-DRF initialization in OmniSAM_adapt.__init__
- [x] model/model.py: Added SG-DRF forward pass logic
- [x] model/model.py: Added compute_sgdrf_consistency_loss method
- [x] model/model.py: Added set_sgdrf_training_stage method
- [x] train/strategies.py: Updated WithMemStrategy for SG-DRF losses
- [x] train/strategies.py: Updated NoMemStrategy for SG-DRF losses
- [x] utils/generate_density_pseudo_labels.py: Created pseudo-label generator
- [ ] train.py: Add SG-DRF arguments to TrainConfig (see Option 2 above)
- [ ] utils/build_ddp.py: Pass SG-DRF parameters to model creation

## Next Steps

1. **Test basic integration**: Run a quick test with --num_epochs 1 to ensure no errors
2. **Run cold-start**: Train for 2 epochs with sgdrf_training_stage=cold_start
3. **Run joint training**: Load cold-start checkpoint and train with joint stage
4. **Evaluate**: Compare mIoU with baseline (enable_sgdrf=False)
5. **Ablation study**: Run E0-E5 configurations to verify each component

## Reference Files

- **Design Document**: /home/lightwheel/Desktop/TECMA-PASS/docs/SG-DRF技术设计文档.md
- **Core Modules**: 
  - model/modules/semantic_density_estimator.py
  - model/modules/dynamic_fsmc.py
  - model/modules/dynamic_scm.py
- **Modified Files**:
  - model/model.py
  - train/strategies.py
  - utils/generate_density_pseudo_labels.py
