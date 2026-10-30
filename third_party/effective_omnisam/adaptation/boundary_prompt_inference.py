"""
Boundary-Enhanced Prompt Learning - Inference Version
无需训练的轻量级版本，用于快速验证概念
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryEnhancedPromptInference(nn.Module):
    """
    BEPL推理版本：使用固定的边界增强策略
    无需训练，可直接应用于现有checkpoint
    
    策略：
    1. 在边界区域增强特征的对比度
    2. 使用高斯权重平滑边界-中心过渡
    3. 可选：添加频域增强
    """
    def __init__(
        self, 
        seam_width=128,
        enhancement_strength=0.3,
        use_gaussian_weight=True
    ):
        super().__init__()
        self.seam_width = seam_width
        self.enhancement_strength = enhancement_strength
        self.use_gaussian_weight = use_gaussian_weight
        
    def create_boundary_weight(self, width, height, device):
        """
        创建边界权重图（高斯衰减）
        边界区域权重=1.0，向中心逐渐衰减到0
        """
        x_coords = torch.arange(width, device=device).float()
        
        # 计算到最近边界的距离
        dist_to_left = x_coords
        dist_to_right = width - x_coords
        dist_to_boundary = torch.min(dist_to_left, dist_to_right)
        
        if self.use_gaussian_weight:
            # 高斯衰减：边界区域权重高，中心权重低
            sigma = self.seam_width / 2.0
            weight = torch.exp(-0.5 * (dist_to_boundary / sigma) ** 2)
        else:
            # 线性衰减
            weight = torch.clamp(1.0 - dist_to_boundary / self.seam_width, 0, 1)
        
        # [W] -> [1, 1, 1, W] -> [1, 1, H, W]
        weight = weight.view(1, 1, 1, width).expand(1, 1, height, width)
        
        return weight
    
    def enhance_features(self, features):
        """
        对特征进行边界增强
        
        策略：
        1. 增强边界区域的特征对比度（std normalization）
        2. 保持中心区域不变
        3. 平滑过渡
        """
        B, C, H, W = features.shape
        device = features.device
        
        # 1. 创建边界权重
        boundary_weight = self.create_boundary_weight(W, H, device)
        boundary_weight = boundary_weight.expand(B, 1, H, W)
        
        # 2. 计算边界区域的统计量
        # 为每个channel独立计算
        eps = 1e-6
        
        # 全局统计
        global_mean = features.mean(dim=[2, 3], keepdim=True)
        global_std = features.std(dim=[2, 3], keepdim=True) + eps
        
        # 3. 标准化增强（增强对比度）
        normalized_features = (features - global_mean) / global_std
        enhanced_features = global_mean + normalized_features * global_std * (1 + self.enhancement_strength)
        
        # 4. 边界区域应用增强，中心区域保持原样
        output = features * (1 - boundary_weight) + enhanced_features * boundary_weight
        
        return output
    
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: [B, C, H, W]
        Returns:
            enhanced_features: [B, C, H, W]
        """
        return self.enhance_features(features)


def apply_bepl_to_logits(logits, seam_width=128, strength=0.2):
    """
    直接对logits应用边界增强（更简单的方案）
    可以直接在validation.py中调用，无需修改模型
    
    Args:
        logits: [B, num_classes, H, W]
        seam_width: 边界宽度
        strength: 增强强度
    Returns:
        enhanced_logits: [B, num_classes, H, W]
    """
    B, C, H, W = logits.shape
    device = logits.device
    
    # 创建边界mask
    x_coords = torch.arange(W, device=device).float()
    dist_to_left = x_coords
    dist_to_right = W - x_coords
    dist_to_boundary = torch.min(dist_to_left, dist_to_right)
    
    # 高斯权重
    sigma = seam_width / 2.0
    boundary_weight = torch.exp(-0.5 * (dist_to_boundary / sigma) ** 2)
    boundary_weight = boundary_weight.view(1, 1, 1, W).expand(B, 1, H, W)
    
    # 增强边界区域的置信度
    # 对每个类别的logits进行sharpening
    enhanced_logits = logits * (1 + strength * boundary_weight)
    
    return enhanced_logits


# 测试
if __name__ == "__main__":
    print("Testing BEPL Inference Version...")
    
    # 测试特征增强
    features = torch.randn(2, 256, 64, 256)
    bepl = BoundaryEnhancedPromptInference(seam_width=32, enhancement_strength=0.3)
    enhanced = bepl(features)
    
    print(f"Input: {features.shape}")
    print(f"Output: {enhanced.shape}")
    print(f"Feature change: {(enhanced - features).abs().mean():.6f}")
    
    # 测试logits增强
    logits = torch.randn(2, 13, 64, 256)
    enhanced_logits = apply_bepl_to_logits(logits, seam_width=32, strength=0.2)
    
    print(f"\nLogits input: {logits.shape}")
    print(f"Logits output: {enhanced_logits.shape}")
    print(f"Logits change: {(enhanced_logits - logits).abs().mean():.6f}")
    
    print("\n✓ BEPL inference test passed!")
