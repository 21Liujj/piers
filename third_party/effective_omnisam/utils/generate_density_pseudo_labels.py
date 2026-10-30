"""
Generate Density Pseudo-labels for SG-DRF Training
根据分割GT生成密度伪标签，用于冷启动训练

Usage:
    python generate_density_pseudo_labels.py --seg_labels <path> --output <path>
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def generate_density_pseudo_label_from_mask(
    seg_mask: np.ndarray,
    sobel_ksize: int = 3,
    gaussian_ksize: int = 11,
    gaussian_sigma: float = 0.0
) -> np.ndarray:
    """
    Generate density pseudo-label from segmentation mask using edge detection

    Args:
        seg_mask: (H, W) segmentation mask with class IDs
        sobel_ksize: Sobel kernel size
        gaussian_ksize: Gaussian blur kernel size
        gaussian_sigma: Gaussian blur sigma (0 = auto)

    Returns:
        density_gt: (H, W) density pseudo-label [0, 1]
    """
    # Convert to uint8 for cv2 operations
    seg_mask = seg_mask.astype(np.uint8)

    # 1. Extract semantic boundaries using Sobel
    edges_x = cv2.Sobel(seg_mask, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    edges_y = cv2.Sobel(seg_mask, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    edges = np.abs(edges_x) + np.abs(edges_y)

    # 2. Binarize edges
    edges_binary = (edges > 0).astype(np.uint8)

    # 3. Distance transform (distance to nearest edge)
    distance = cv2.distanceTransform(1 - edges_binary, cv2.DIST_L2, 5)

    # 4. Normalize and invert (high density at boundaries)
    max_dist = distance.max()
    if max_dist > 0:
        density_gt = 1.0 - np.clip(distance / max_dist, 0, 1)
    else:
        density_gt = np.ones_like(distance, dtype=np.float32)

    # 5. Gaussian smoothing for smooth transitions
    density_gt = cv2.GaussianBlur(
        density_gt.astype(np.float32),
        (gaussian_ksize, gaussian_ksize),
        gaussian_sigma
    )

    return np.clip(density_gt, 0.0, 1.0).astype(np.float32)


def generate_density_pseudo_label_batch(
    seg_masks: torch.Tensor,
    sobel_ksize: int = 3,
    gaussian_ksize: int = 11,
    gaussian_sigma: float = 0.0
) -> torch.Tensor:
    """
    Batch version of density pseudo-label generation

    Args:
        seg_masks: (B, H, W) segmentation masks
        sobel_ksize: Sobel kernel size
        gaussian_ksize: Gaussian blur kernel size
        gaussian_sigma: Gaussian blur sigma

    Returns:
        density_batch: (B, 1, H, W) density pseudo-labels
    """
    B, H, W = seg_masks.shape
    device = seg_masks.device

    density_batch = []

    for i in range(B):
        # Convert to numpy for cv2 processing
        mask_np = seg_masks[i].cpu().numpy()

        # Generate density pseudo-label
        density_np = generate_density_pseudo_label_from_mask(
            mask_np,
            sobel_ksize=sobel_ksize,
            gaussian_ksize=gaussian_ksize,
            gaussian_sigma=gaussian_sigma
        )

        # Convert back to tensor
        density_tensor = torch.from_numpy(density_np).unsqueeze(0)  # (1, H, W)
        density_batch.append(density_tensor)

    # Stack batch
    density_batch = torch.stack(density_batch, dim=0).to(device)  # (B, 1, H, W)

    return density_batch


if __name__ == "__main__":
    """Test the pseudo-label generation"""
    print("=" * 60)
    print("Testing Density Pseudo-label Generation")
    print("=" * 60)

    # Create a synthetic segmentation mask
    H, W = 512, 1024
    seg_mask = np.zeros((H, W), dtype=np.uint8)

    # Add some objects with different classes
    seg_mask[100:200, 100:300] = 1  # Building
    seg_mask[250:350, 400:600] = 2  # Road
    seg_mask[150:180, 700:900] = 3  # Car
    seg_mask[50:80, 850:950] = 4    # Traffic sign (small object)

    print(f"\nInput seg_mask shape: {seg_mask.shape}")
    print(f"Unique classes: {np.unique(seg_mask)}")

    # Generate density pseudo-label
    density_gt = generate_density_pseudo_label_from_mask(seg_mask)

    print(f"\nOutput density_gt shape: {density_gt.shape}")
    print(f"Density range: [{density_gt.min():.4f}, {density_gt.max():.4f}]")
    print(f"Density mean: {density_gt.mean():.4f}")
    print(f"Density std: {density_gt.std():.4f}")

    # Test batch version
    print("\n" + "=" * 60)
    print("Testing Batch Version")
    print("=" * 60)

    batch_size = 4
    seg_masks_batch = torch.from_numpy(
        np.stack([seg_mask] * batch_size)
    )  # (B, H, W)

    density_batch = generate_density_pseudo_label_batch(seg_masks_batch)

    print(f"\nInput batch shape: {seg_masks_batch.shape}")
    print(f"Output density batch shape: {density_batch.shape}")
    print(f"Batch density range: [{density_batch.min():.4f}, {density_batch.max():.4f}]")
    print(f"Batch density mean: {density_batch.mean():.4f}")

    print("\n" + "=" * 60)
    print("Test passed ✓")
    print("=" * 60)
