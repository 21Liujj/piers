"""
Boundary-Enhanced Prompt Learning (BEPL)
针对360°全景图边界区域的可学习prompt增强
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryEnhancedPrompt(nn.Module):
    """
    在360°全景图的seam region添加可学习的boundary-aware prompt
    
    Args:
        embed_dim: 特征维度 (SAM2: 256)
        seam_width: 边界区域宽度（像素）
        num_prompts: prompt向量数量
        prompt_dropout: dropout比例
    """
    def __init__(
        self, 
        embed_dim=256, 
        seam_width=128,
        num_prompts=8,
        prompt_dropout=0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seam_width = seam_width
        self.num_prompts = num_prompts
        
        # 可学习的boundary prompt embeddings
        # [num_prompts, embed_dim]
        self.boundary_prompts = nn.Parameter(
            torch.randn(num_prompts, embed_dim) * 0.02
        )
        
        # Prompt projection layer
        self.prompt_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(prompt_dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Spatial attention for adaptive weighting
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def create_boundary_mask(self, width, height, device):
        """
        创建边界区域mask
        
        Args:
            width: 图像宽度（全景图通常4096）
            height: 图像高度（全景图通常1024）
            device: torch device
            
        Returns:
            boundary_mask: [1, 1, H, W] 边界区域为1，中心为0
        """
        x_coords = torch.arange(width, device=device).view(1, 1, 1, width)
        
        # 左边界: x < seam_width
        left_boundary = (x_coords < self.seam_width).float()
        
        # 右边界: x > (width - seam_width)
        right_boundary = (x_coords > (width - self.seam_width)).float()
        
        # 合并左右边界
        boundary_mask = torch.clamp(left_boundary + right_boundary, 0, 1)
        
        # 扩展到高度维度
        boundary_mask = boundary_mask.expand(1, 1, height, width)
        
        return boundary_mask
    
    def forward(self, features, return_attention=False):
        """
        对特征添加boundary-enhanced prompt
        
        Args:
            features: [B, C, H, W] 输入特征图
            return_attention: 是否返回attention map
            
        Returns:
            enhanced_features: [B, C, H, W] 增强后的特征
            (optional) attention_map: [B, 1, H, W]
        """
        B, C, H, W = features.shape
        device = features.device
        
        # 1. 创建边界mask
        boundary_mask = self.create_boundary_mask(W, H, device)  # [1, 1, H, W]
        boundary_mask = boundary_mask.expand(B, 1, H, W)  # [B, 1, H, W]
        
        # 2. 投影prompt embeddings
        # [num_prompts, C] -> [num_prompts, C]
        projected_prompts = self.prompt_proj(self.boundary_prompts)
        
        # 3. 聚合多个prompt (平均)
        # [num_prompts, C] -> [C]
        aggregated_prompt = projected_prompts.mean(dim=0)
        
        # 4. 扩展到空间维度
        # [C] -> [B, C, H, W]
        prompt_map = aggregated_prompt.view(1, C, 1, 1).expand(B, C, H, W)
        
        # 5. 计算spatial attention (自适应权重)
        attention_map = self.spatial_attn(features)  # [B, 1, H, W]
        
        # 6. 边界区域应用prompt
        # prompt只在边界区域生效，中心区域保持原特征
        prompt_weight = boundary_mask * attention_map  # [B, 1, H, W]
        enhanced_features = features + prompt_map * prompt_weight
        
        if return_attention:
            return enhanced_features, attention_map
        return enhanced_features
    
    def get_boundary_statistics(self, features):
        """
        获取边界区域的统计信息（用于分析）
        
        Returns:
            stats: dict with boundary/center feature statistics
        """
        B, C, H, W = features.shape
        device = features.device
        
        boundary_mask = self.create_boundary_mask(W, H, device)
        boundary_mask = boundary_mask.expand(B, C, H, W)
        
        # 边界区域特征
        boundary_features = features * boundary_mask
        boundary_mean = boundary_features.sum() / boundary_mask.sum()
        
        # 中心区域特征
        center_mask = 1 - boundary_mask
        center_features = features * center_mask
        center_mean = center_features.sum() / center_mask.sum()
        
        return {
            'boundary_mean': boundary_mean.item(),
            'center_mean': center_mean.item(),
            'boundary_ratio': boundary_mask.sum().item() / (B * C * H * W)
        }


def test_boundary_prompt():
    """测试BEPL模块"""
    print("Testing Boundary-Enhanced Prompt Learning...")
    
    # 模拟SAM2特征图
    B, C, H, W = 2, 256, 64, 256  # 全景图缩放到64×256
    features = torch.randn(B, C, H, W)
    
    # 创建BEPL模块
    bepl = BoundaryEnhancedPrompt(
        embed_dim=256,
        seam_width=32,  # 边界宽度32像素（对应原图128）
        num_prompts=8
    )
    
    # 前向传播
    enhanced_features, attention = bepl(features, return_attention=True)
    
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {enhanced_features.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # 统计信息
    stats = bepl.get_boundary_statistics(features)
    print(f"\nBoundary statistics:")
    print(f"  Boundary mean: {stats['boundary_mean']:.4f}")
    print(f"  Center mean: {stats['center_mean']:.4f}")
    print(f"  Boundary ratio: {stats['boundary_ratio']:.4f}")
    
    print("\n✓ BEPL module test passed!")


if __name__ == "__main__":
    test_boundary_prompt()
