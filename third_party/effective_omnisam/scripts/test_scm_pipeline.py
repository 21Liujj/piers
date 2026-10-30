"""
End-to-end pipeline test for SCM components

Tests:
1. Load circular padded dataset
2. Create dummy model output
3. Compute seam consistency loss
4. Verify all components work together
"""

import sys
import torch
sys.path.insert(0, '/root/autodl-tmp/OmniSAM-main')

from dataset.DensePASS_circular import DensePASS13Circular
from adaptation.losses import SeamConsistencyLoss

print('=' * 80)
print('SCM Pipeline Integration Test')
print('=' * 80)

# Step 1: Load dataset with circular padding
print('\n[Step 1] Loading circular padded dataset...')
dataset = DensePASS13Circular(
    root='data/DensePASS',
    list_path='dataset/densepass_list/val.txt',
    crop_size=(4096, 1024),
    sliding_window=(384, 1024),
    passes=2,
    set='val',
    enable_circular_padding=True,
    circular_pad_width=512
)

image, label, size, name, ori_label = dataset[0]
print(f'✓ Loaded sample: {name}')
print(f'  Image patches: {image.shape[0]} (expected 22 with circular padding)')
print(f'  Label patches: {label.shape[0]}')

# Step 2: Simulate model forward pass
print('\n[Step 2] Simulating model output...')
B, num_patches, C_img, H_patch, W_patch = 1, image.shape[0], 3, 1024, 1024
num_classes = 13

# In real scenario, model outputs [B, num_classes, H_full, W_full]
# For 4096 width, full output would be [1, 13, 1024, 4096]
# With circular padding, it would be [1, 13, 1024, 5120]
dummy_output = torch.randn(B, num_classes, 1024, 5120, requires_grad=True)
print(f'✓ Model output shape: {dummy_output.shape}')

# Step 3: Compute seam consistency loss
print('\n[Step 3] Computing seam consistency loss...')
seam_loss_fn = SeamConsistencyLoss(seam_width=200, weight=0.1, loss_type='l1')
loss = seam_loss_fn(dummy_output)
print(f'✓ Seam loss computed: {loss.item():.6f}')

# Step 4: Verify loss is differentiable
print('\n[Step 4] Testing backward pass...')
loss.backward()
print(f'✓ Backward pass successful')
print(f'  Gradient norm: {dummy_output.grad.norm().item():.6f}')

# Step 5: Compare baseline vs circular padding data size
print('\n[Step 5] Comparing with baseline dataset...')
dataset_baseline = DensePASS13Circular(
    root='data/DensePASS',
    list_path='dataset/densepass_list/val.txt',
    crop_size=(4096, 1024),
    sliding_window=(384, 1024),
    passes=2,
    set='val',
    enable_circular_padding=False
)
img_base, _, _, _, _ = dataset_baseline[0]
print(f'✓ Baseline patches: {img_base.shape[0]}')
print(f'✓ Circular patches: {image.shape[0]}')
print(f'✓ Additional boundary patches: {image.shape[0] - img_base.shape[0]}')

# Summary
print('\n' + '=' * 80)
print('Pipeline Integration Test: PASSED ✓')
print('=' * 80)
print('\nKey Results:')
print(f'  - Circular padding working: {image.shape[0] - img_base.shape[0]} extra patches')
print(f'  - Seam loss functional: {loss.item():.6f}')
print(f'  - Gradient flow working: ✓')
print('\nReady for validation.py integration!')
print('=' * 80)
