"""
Semantic Density Estimator Module
语义密度估计模块 - SG-DRF核心组件

功能: 预测全景图像中每个空间位置的语义复杂度
- 多尺度特征提取 (3x3, 5x5, 7x7卷积)
- 密度图生成 (输出范围[0,1])
- 用于指导动态FSMC压缩和动态SCM padding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticDensityEstimator(nn.Module):
    """
    语义密度估计器

    Architecture:
        - Multi-scale Feature Extraction: 3个不同感受野的卷积分支
        - Density Head: 将多尺度特征融合为单通道密度图

    Input:
        - features: (B, C, H, W) - 来自backbone的特征图, C=256

    Output:
        - density_map: (B, 1, H, W) - 归一化密度图, 范围[0,1]
          高值表示语义复杂区域, 低值表示简单区域
    """

    def __init__(self, in_channels=256):
        """
        Args:
            in_channels: 输入特征通道数, 默认256 (来自Trans4PASS+ backbone)
        """
        super(SemanticDensityEstimator, self).__init__()

        self.in_channels = in_channels

        # 多尺度特征提取 - 3个并行卷积分支
        # Branch 1: 小感受野 (3x3) - 捕捉局部细节
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        # Branch 2: 中等感受野 (5x5) - 捕捉中等范围语义
        self.conv2 = nn.Conv2d(in_channels, 128, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # Branch 3: 大感受野 (7x7) - 捕捉全局语义
        self.conv3 = nn.Conv2d(in_channels, 128, kernel_size=7, padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)

        # Density Head - 将384维多尺度特征映射到1维密度图
        self.density_head = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  # 输出范围[0,1]
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化 - 使用Kaiming初始化保证训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        前向传播

        Args:
            features: (B, C, H, W) - backbone特征图

        Returns:
            density_map: (B, 1, H, W) - 语义密度图, 范围[0,1]
        """
        # 输入验证
        assert features.dim() == 4, f"Expected 4D input (B,C,H,W), got {features.dim()}D"
        assert features.size(1) == self.in_channels, \
            f"Expected {self.in_channels} channels, got {features.size(1)}"

        # 多尺度特征提取
        f1 = self.relu(self.bn1(self.conv1(features)))  # (B, 128, H, W)
        f2 = self.relu(self.bn2(self.conv2(features)))  # (B, 128, H, W)
        f3 = self.relu(self.bn3(self.conv3(features)))  # (B, 128, H, W)

        # 特征拼接
        multi_scale_features = torch.cat([f1, f2, f3], dim=1)  # (B, 384, H, W)

        # 生成密度图
        density_map = self.density_head(multi_scale_features)  # (B, 1, H, W)

        return density_map

    def compute_compression_ratio(self, density_map, min_ratio=4.0, max_ratio=16.0):
        """
        根据密度图计算动态压缩比率

        设计原则:
            - 高密度区域 (density → 1.0) → 低压缩 (ratio → 4×)
            - 低密度区域 (density → 0.0) → 高压缩 (ratio → 16×)

        Args:
            density_map: (B, 1, H, W) - 语义密度图
            min_ratio: 最小压缩比 (高密度区域)
            max_ratio: 最大压缩比 (低密度区域)

        Returns:
            compression_ratio: (B, 1, H, W) - 空间自适应压缩比
        """
        # 线性映射: ratio = max_ratio - density * (max_ratio - min_ratio)
        ratio = max_ratio - density_map * (max_ratio - min_ratio)

        # 确保在有效范围内
        ratio = torch.clamp(ratio, min=min_ratio, max=max_ratio)

        return ratio

    def get_adaptive_pad_width(self, density_map, base_width=1024):
        """
        根据边界密度计算动态padding宽度

        设计原则:
            - 高边界密度 → 宽padding (512-1536px)
            - 低边界密度 → 窄padding (512px)

        Args:
            density_map: (B, 1, H, W) - 语义密度图
            base_width: 基准padding宽度

        Returns:
            pad_width: (B,) - 每个样本的padding宽度 (整数)
        """
        # 提取图像边界区域的平均密度
        left_boundary = density_map[:, :, :, :32].mean(dim=[1, 2, 3])    # 左边界
        right_boundary = density_map[:, :, :, -32:].mean(dim=[1, 2, 3])  # 右边界
        boundary_density = (left_boundary + right_boundary) / 2  # (B,)

        # 根据边界密度调整padding: width = base + (density - 0.5) * 1024
        pad_width = base_width + (boundary_density - 0.5) * 1024

        # 限制范围 [512, 1536]
        pad_width = torch.clamp(pad_width, min=512, max=1536)

        # 转为整数
        pad_width = pad_width.int()

        return pad_width


if __name__ == "__main__":
    """单元测试"""
    print("=" * 60)
    print("SemanticDensityEstimator 单元测试")
    print("=" * 60)

    # 创建模块
    model = SemanticDensityEstimator(in_channels=256)
    model.eval()

    # 模拟输入
    batch_size = 2
    height, width = 128, 256  # 典型全景图下采样尺寸
    features = torch.randn(batch_size, 256, height, width)

    print(f"\n输入特征: {features.shape}")

    # 前向传播
    with torch.no_grad():
        density_map = model(features)

    print(f"输出密度图: {density_map.shape}")
    print(f"密度范围: [{density_map.min().item():.4f}, {density_map.max().item():.4f}]")
    print(f"平均密度: {density_map.mean().item():.4f}")

    # 测试压缩比计算
    compression_ratio = model.compute_compression_ratio(density_map)
    print(f"\n压缩比范围: [{compression_ratio.min().item():.2f}×, {compression_ratio.max().item():.2f}×]")
    print(f"平均压缩比: {compression_ratio.mean().item():.2f}×")

    # 测试padding宽度计算
    pad_width = model.get_adaptive_pad_width(density_map)
    print(f"\nPadding宽度: {pad_width.tolist()}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("测试通过 ✓")
    print("=" * 60)
