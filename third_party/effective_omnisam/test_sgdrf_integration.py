#!/usr/bin/env python3
"""
Test script to verify SG-DRF integration in OmniSAM
测试脚本：验证SG-DRF集成是否成功

Usage:
    python test_sgdrf_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

print("=" * 70)
print("SG-DRF Integration Test")
print("=" * 70)

# Test 1: Import SG-DRF modules
print("\n[Test 1] Importing SG-DRF modules...")
try:
    from model.modules.semantic_density_estimator import SemanticDensityEstimator
    from model.modules.dynamic_fsmc import DynamicFSMC
    from model.modules.dynamic_scm import DynamicSCM
    print("✓ All SG-DRF modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SG-DRF modules: {e}")
    sys.exit(1)

# Test 2: Import modified model
print("\n[Test 2] Importing modified OmniSAM model...")
try:
    from model.model import OmniSAM_adapt
    print("✓ OmniSAM_adapt imported successfully")
except ImportError as e:
    print(f"✗ Failed to import OmniSAM_adapt: {e}")
    sys.exit(1)

# Test 3: Import pseudo-label generator
print("\n[Test 3] Importing pseudo-label generator...")
try:
    from utils.generate_density_pseudo_labels import generate_density_pseudo_label_batch
    print("✓ Pseudo-label generator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pseudo-label generator: {e}")
    sys.exit(1)

# Test 4: Import updated strategies
print("\n[Test 4] Importing updated training strategies...")
try:
    from train.strategies import WithMemStrategy, NoMemStrategy
    print("✓ Training strategies imported successfully")
except ImportError as e:
    print(f"✗ Failed to import training strategies: {e}")
    sys.exit(1)

# Test 5: Create SG-DRF modules
print("\n[Test 5] Creating SG-DRF modules...")
try:
    density_estimator = SemanticDensityEstimator(in_channels=256)
    dynamic_fsmc = DynamicFSMC(in_channels=256, min_ratio=4.0, max_ratio=16.0)
    dynamic_scm = DynamicSCM(base_pad_width=1024, min_pad_width=512, max_pad_width=1536)

    print(f"  - SemanticDensityEstimator: {sum(p.numel() for p in density_estimator.parameters()):,} params")
    print(f"  - DynamicFSMC: {sum(p.numel() for p in dynamic_fsmc.parameters()):,} params")
    print(f"  - DynamicSCM: {sum(p.numel() for p in dynamic_scm.parameters()):,} params")
    print("✓ All SG-DRF modules created successfully")
except Exception as e:
    print(f"✗ Failed to create SG-DRF modules: {e}")
    sys.exit(1)

# Test 6: Test density estimator forward pass
print("\n[Test 6] Testing density estimator forward pass...")
try:
    batch_size = 2
    H, W = 128, 256
    features = torch.randn(batch_size, 256, H, W)

    with torch.no_grad():
        density_map = density_estimator(features)

    assert density_map.shape == (batch_size, 1, H, W), \
        f"Wrong output shape: {density_map.shape}"
    assert density_map.min() >= 0 and density_map.max() <= 1, \
        f"Density out of range [0,1]: [{density_map.min():.4f}, {density_map.max():.4f}]"

    print(f"  - Input shape: {features.shape}")
    print(f"  - Output shape: {density_map.shape}")
    print(f"  - Density range: [{density_map.min():.4f}, {density_map.max():.4f}]")
    print("✓ Density estimator forward pass successful")
except Exception as e:
    print(f"✗ Density estimator forward pass failed: {e}")
    sys.exit(1)

# Test 7: Test Dynamic FSMC
print("\n[Test 7] Testing Dynamic FSMC...")
try:
    features = torch.randn(batch_size, 256, H, W)
    density_map = torch.rand(batch_size, 1, H, W)

    with torch.no_grad():
        compressed_features, stats = dynamic_fsmc(features, density_map)

    assert compressed_features.shape == features.shape, \
        f"Output shape mismatch: {compressed_features.shape} vs {features.shape}"

    print(f"  - Compression ratio: {stats['compression_ratio']:.2f}×")
    print(f"  - Original width: {stats['original_width']}")
    print(f"  - Compressed width: {stats['compressed_width']}")
    print("✓ Dynamic FSMC test successful")
except Exception as e:
    print(f"✗ Dynamic FSMC test failed: {e}")
    sys.exit(1)

# Test 8: Test Dynamic SCM
print("\n[Test 8] Testing Dynamic SCM...")
try:
    features = torch.randn(batch_size, 256, H, W)
    density_map = torch.rand(batch_size, 1, H, W)

    with torch.no_grad():
        padded_features, pad_width = dynamic_scm(features, density_map, return_pad_width=True)

    expected_width = W + 2 * pad_width
    assert padded_features.shape[-1] == expected_width, \
        f"Padded width mismatch: {padded_features.shape[-1]} vs {expected_width}"

    # Test removal
    unpadded_features = dynamic_scm.remove_circular_padding(padded_features, pad_width)
    assert unpadded_features.shape == features.shape, \
        f"Unpadded shape mismatch: {unpadded_features.shape} vs {features.shape}"

    print(f"  - Original width: {W}")
    print(f"  - Padded width: {padded_features.shape[-1]} (pad_width={pad_width})")
    print(f"  - After removal: {unpadded_features.shape[-1]}")
    print("✓ Dynamic SCM test successful")
except Exception as e:
    print(f"✗ Dynamic SCM test failed: {e}")
    sys.exit(1)

# Test 9: Test pseudo-label generation
print("\n[Test 9] Testing pseudo-label generation...")
try:
    seg_masks = torch.randint(0, 13, (batch_size, 512, 1024))

    with torch.no_grad():
        density_pseudo_labels = generate_density_pseudo_label_batch(seg_masks)

    assert density_pseudo_labels.shape == (batch_size, 1, 512, 1024), \
        f"Wrong output shape: {density_pseudo_labels.shape}"
    assert density_pseudo_labels.min() >= 0 and density_pseudo_labels.max() <= 1, \
        f"Density out of range: [{density_pseudo_labels.min():.4f}, {density_pseudo_labels.max():.4f}]"

    print(f"  - Input shape: {seg_masks.shape}")
    print(f"  - Output shape: {density_pseudo_labels.shape}")
    print(f"  - Density range: [{density_pseudo_labels.min():.4f}, {density_pseudo_labels.max():.4f}]")
    print("✓ Pseudo-label generation successful")
except Exception as e:
    print(f"✗ Pseudo-label generation failed: {e}")
    sys.exit(1)

# Test 10: Test OmniSAM_adapt with SG-DRF disabled
print("\n[Test 10] Testing OmniSAM_adapt with SG-DRF disabled...")
try:
    # Create a minimal mock SAM model
    class MockSAM(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = nn.Module()
            self.image_encoder.neck = nn.Module()
            self.image_encoder.neck.d_model = 256
            self.image_encoder.trunk = nn.Module()
            self.image_encoder.trunk.blocks = nn.ModuleList([])
            self.memory_encoder = nn.Module()
            self.memory_encoder.out_proj = nn.Linear(256, 64)

    mock_sam = MockSAM()

    model = OmniSAM_adapt(
        sam_model=mock_sam,
        num_classes=13,
        num_maskmem=3,
        enable_sgdrf=False  # Disabled for this test
    )

    assert model.enable_sgdrf == False
    assert model.density_estimator is None
    assert model.dynamic_fsmc is None
    assert model.dynamic_scm is None

    print("✓ OmniSAM_adapt with SG-DRF disabled works correctly")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 11: Test OmniSAM_adapt with SG-DRF enabled
print("\n[Test 11] Testing OmniSAM_adapt with SG-DRF enabled...")
try:
    model_sgdrf = OmniSAM_adapt(
        sam_model=mock_sam,
        num_classes=13,
        num_maskmem=3,
        enable_sgdrf=True,
        sgdrf_backbone_channels=256,
        sgdrf_enable_dynamic_fsmc=True,
        sgdrf_enable_dynamic_scm=True,
        sgdrf_training_stage='joint'
    )

    assert model_sgdrf.enable_sgdrf == True
    assert model_sgdrf.density_estimator is not None
    assert model_sgdrf.dynamic_fsmc is not None
    assert model_sgdrf.dynamic_scm is not None
    assert model_sgdrf.sgdrf_training_stage == 'joint'

    print("  - Density Estimator: ✓")
    print("  - Dynamic FSMC: ✓")
    print("  - Dynamic SCM: ✓")
    print("  - Training Stage: joint")
    print("✓ OmniSAM_adapt with SG-DRF enabled works correctly")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 12: Test training stage switching
print("\n[Test 12] Testing training stage switching...")
try:
    # Check cold-start stage
    model_sgdrf.set_sgdrf_training_stage('cold_start')
    assert model_sgdrf.sgdrf_training_stage == 'cold_start'

    # Verify density_estimator is trainable
    density_trainable = any(p.requires_grad for p in model_sgdrf.density_estimator.parameters())
    assert density_trainable, "Density estimator should be trainable in cold-start"

    # Switch back to joint
    model_sgdrf.set_sgdrf_training_stage('joint')
    assert model_sgdrf.sgdrf_training_stage == 'joint'

    print("  - Switch to cold_start: ✓")
    print("  - Switch to joint: ✓")
    print("✓ Training stage switching works correctly")
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)

# Test 13: Test consistency loss computation
print("\n[Test 13] Testing consistency loss computation...")
try:
    batch_size = 2
    num_classes = 13
    H, W = 128, 256

    density_map = torch.rand(batch_size, 1, H, W)
    seg_logits = torch.randn(batch_size, num_classes, H, W)

    consistency_loss = model_sgdrf.compute_sgdrf_consistency_loss(density_map, seg_logits)

    assert consistency_loss.item() >= 0, f"Loss should be non-negative: {consistency_loss.item()}"
    assert not torch.isnan(consistency_loss), "Loss should not be NaN"

    print(f"  - Consistency loss: {consistency_loss.item():.6f}")
    print("✓ Consistency loss computation successful")
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nSG-DRF Integration Summary:")
print("  - Core modules: ✓ Imported and functional")
print("  - Model integration: ✓ OmniSAM_adapt supports SG-DRF")
print("  - Training support: ✓ Strategies updated for SG-DRF losses")
print("  - Pseudo-labels: ✓ Generator working correctly")
print("  - Stage switching: ✓ Cold-start and joint modes work")
print("\nNext Steps:")
print("  1. Add SG-DRF arguments to train.py (see SG-DRF_INTEGRATION_GUIDE.md)")
print("  2. Update utils/build_ddp.py to pass SG-DRF parameters")
print("  3. Run a quick training test: python train.py --enable_sgdrf --num_epochs 1")
print("  4. Start cold-start training: 2 epochs with sgdrf_training_stage=cold_start")
print("  5. Continue with joint training: 30+ epochs with sgdrf_training_stage=joint")
print("=" * 70)
