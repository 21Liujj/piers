"""
Loss functions for Seamless Circular Memory (SCM)

Author: Research Team
Date: 2026-08-28
Branch: feature/scm-circular-memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeamConsistencyLoss(nn.Module):
    """
    Seam Consistency Loss for 360° panoramic segmentation.
    
    Enforces that predictions at left boundary (x=0) and right boundary (x=W)
    should be consistent, since they are physically connected in 360° panorama.
    
    Args:
        seam_width: Width of seam region in pixels (for 4096 width, 128-200 recommended)
        weight: Loss weight coefficient
        loss_type: 'l1' | 'kl' | 'cosine'
    
    Example:
        >>> seam_loss = SeamConsistencyLoss(seam_width=128, weight=0.1, loss_type='l1')
        >>> pred_logits = model(images)  # [B, C, H, W]
        >>> loss = seam_loss(pred_logits)
    """
    
    def __init__(self, seam_width=128, weight=0.1, loss_type='l1'):
        super().__init__()
        self.seam_width = seam_width
        self.weight = weight
        self.loss_type = loss_type
        
        # Validate loss type
        assert loss_type in ['l1', 'kl', 'cosine'],             f"loss_type must be 'l1', 'kl', or 'cosine', got {loss_type}"
        
        print(f"[SeamConsistencyLoss] Initialized:")
        print(f"  - seam_width: {seam_width}px")
        print(f"  - weight: {weight}")
        print(f"  - loss_type: {loss_type}")
    
    def forward(self, pred_logits):
        """
        Compute seam consistency loss.
        
        Args:
            pred_logits: [B, C, H, W] model output logits
        
        Returns:
            loss: scalar tensor
        """
        B, C, H, W = pred_logits.shape
        
        # Extract left and right boundary regions
        pred_left = pred_logits[:, :, :, :self.seam_width]      # [B, C, H, seam_width]
        pred_right = pred_logits[:, :, :, -self.seam_width:]    # [B, C, H, seam_width]
        
        # Compute consistency loss based on type
        if self.loss_type == 'l1':
            # L1 loss on logits
            loss = F.l1_loss(pred_left, pred_right)
        
        elif self.loss_type == 'kl':
            # KL divergence on probability distributions
            pred_left_prob = F.softmax(pred_left, dim=1)
            pred_right_prob = F.softmax(pred_right, dim=1)
            
            # Symmetric KL: 0.5 * [KL(left||right) + KL(right||left)]
            kl_lr = F.kl_div(
                pred_left_prob.log(), 
                pred_right_prob, 
                reduction='batchmean'
            )
            kl_rl = F.kl_div(
                pred_right_prob.log(), 
                pred_left_prob, 
                reduction='batchmean'
            )
            loss = 0.5 * (kl_lr + kl_rl)
        
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            # Flatten spatial dimensions
            left_flat = pred_left.reshape(B, C, -1)    # [B, C, H*seam_width]
            right_flat = pred_right.reshape(B, C, -1)
            
            # Compute cosine similarity per spatial position
            cos_sim = F.cosine_similarity(left_flat, right_flat, dim=1)  # [B, H*seam_width]
            
            # Loss = 1 - average similarity
            loss = 1.0 - cos_sim.mean()
        
        return self.weight * loss


class BoundaryWeightedCELoss(nn.Module):
    """
    Cross-entropy loss with higher weight on boundary regions.
    
    Useful for emphasizing seam regions during training.
    Not required for SCM but can be used as alternative to SeamConsistencyLoss.
    
    Args:
        boundary_width: Width of boundary region
        boundary_weight: Weight multiplier for boundary pixels (>1.0)
        ignore_index: Ignore label (default 255)
    """
    
    def __init__(self, boundary_width=128, boundary_weight=2.0, ignore_index=255):
        super().__init__()
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] labels
        
        Returns:
            loss: scalar
        """
        B, C, H, W = pred.shape
        
        # Compute per-pixel CE loss
        loss_map = self.ce_loss(pred, target)  # [B, H, W]
        
        # Create weight map (higher weight at boundaries)
        weight_map = torch.ones_like(loss_map)
        weight_map[:, :, :self.boundary_width] = self.boundary_weight
        weight_map[:, :, -self.boundary_width:] = self.boundary_weight
        
        # Weighted loss
        weighted_loss = (loss_map * weight_map).mean()
        
        return weighted_loss


# Utility function for testing
def test_seam_loss():
    """Quick test for SeamConsistencyLoss"""
    print("Testing SeamConsistencyLoss...")
    
    # Create dummy prediction
    B, C, H, W = 2, 13, 1024, 4096
    pred = torch.randn(B, C, H, W)
    
    # Test all loss types
    for loss_type in ['l1', 'kl', 'cosine']:
        loss_fn = SeamConsistencyLoss(seam_width=128, weight=0.1, loss_type=loss_type)
        loss = loss_fn(pred)
        print(f"{loss_type}: {loss.item():.6f}")
    
    print("Test passed!")


if __name__ == '__main__':
    test_seam_loss()
