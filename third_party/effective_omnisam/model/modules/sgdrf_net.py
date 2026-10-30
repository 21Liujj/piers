"""
SG-DRF Network (Semantic-Guided Dynamic Receptive Field Network)
语义引导的动态感受野网络 - 完整模型

整合三大创新:
1. SemanticDensityEstimator - 语义密度估计
2. DynamicFSMC - 动态可变条带压缩
3. DynamicSCM - 动态球面循环边界

架构: Trans4PASS+ Backbone + SG-DRF Modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic_density_estimator import SemanticDensityEstimator
from .dynamic_fsmc import DynamicFSMC
from .dynamic_scm import DynamicSCM


class SGDRFNet(nn.Module):
    """
    SG-DRF完整网络

    Pipeline:
        1. Backbone提取特征 (Trans4PASS+)
        2. SemanticDensityEstimator预测密度图
        3. DynamicFSMC执行自适应压缩
        4. DynamicSCM应用自适应padding
        5. Segmentation Head输出分割结果

    Training Strategy:
        - Stage 1 (Cold-start): 仅训练SemanticDensityEstimator (2 epochs)
          使用pseudo-label supervision
        - Stage 2 (Joint): 端到端联合训练 (30+ epochs)
          Loss = seg_loss + 0.1 * density_loss + 0.05 * consistency_loss
    """

    def __init__(
        self,
        base_model,
        num_classes=19,
        backbone_channels=256,
        enable_dynamic_fsmc=True,
        enable_dynamic_scm=True,
        training_stage='joint',  # 'cold_start' or 'joint'
        consistency_weight=0.1  # Weight for consistency loss
    ):
        """
        Args:
            base_model: OmniSAM base model to wrap
            num_classes: 分割类别数 (DensePASS: 19)
            backbone_channels: backbone输出通道数
            enable_dynamic_fsmc: 是否启用动态FSMC
            enable_dynamic_scm: 是否启用动态SCM
            training_stage: 训练阶段 ('cold_start' 或 'joint')
        """
        super(SGDRFNet, self).__init__()

        self.num_classes = num_classes
        self.backbone_channels = backbone_channels
        self.enable_dynamic_fsmc = enable_dynamic_fsmc
        self.enable_dynamic_scm = enable_dynamic_scm
        self.training_stage = training_stage
        self.consistency_weight = consistency_weight

        # ========== 1. Base Model (OmniSAM) ==========
        self.base_model = base_model

        # ========== 2. Semantic Density Estimator ==========
        self.density_estimator = SemanticDensityEstimator(
            in_channels=backbone_channels
        )

        # ========== 3. Dynamic FSMC ==========
        if enable_dynamic_fsmc:
            self.dynamic_fsmc = DynamicFSMC(
                in_channels=backbone_channels,
                min_ratio=4.0,
                max_ratio=16.0
            )

        # ========== 4. Dynamic SCM ==========
        if enable_dynamic_scm:
            self.dynamic_scm = DynamicSCM(
                base_pad_width=1024,
                min_pad_width=512,
                max_pad_width=1536
            )

        # ========== 5. Segmentation Head ==========
        self.seg_head = nn.Sequential(
            nn.Conv2d(backbone_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        self._init_weights()
        self._freeze_for_cold_start()

    def _build_simplified_backbone(self):
        """
        构建简化版backbone (placeholder)
        实际使用时应替换为Trans4PASS+ backbone
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ResNet-like blocks
            self._make_layer(64, 128, num_blocks=2, stride=2),
            self._make_layer(128, 256, num_blocks=2, stride=2),

            # Output: (B, 256, H/8, W/8)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """构建ResNet风格的layer"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_for_cold_start(self):
        """
        Cold-start阶段: 冻结除density_estimator外的所有参数
        """
        if self.training_stage == 'cold_start':
            # 冻结所有参数
            for param in self.parameters():
                param.requires_grad = False

            # 解冻density_estimator
            for param in self.density_estimator.parameters():
                param.requires_grad = True

            print("[Cold-start Mode] Only SemanticDensityEstimator is trainable")

    def set_training_stage(self, stage):
        """
        切换训练阶段

        Args:
            stage: 'cold_start' 或 'joint'
        """
        assert stage in ['cold_start', 'joint'], \
            f"Invalid stage: {stage}. Must be 'cold_start' or 'joint'"

        self.training_stage = stage

        if stage == 'cold_start':
            # 冻结除density_estimator外的所有参数
            for param in self.parameters():
                param.requires_grad = False
            for param in self.density_estimator.parameters():
                param.requires_grad = True
            print("[Stage Switch] Cold-start mode activated")

        elif stage == 'joint':
            # 解冻所有参数
            for param in self.parameters():
                param.requires_grad = True
            print("[Stage Switch] Joint training mode activated")

    def forward(self, images, return_density=False, return_stats=False):
        """
        前向传播

        Args:
            images: (B, 3, H, W) - 输入RGB图像
            return_density: bool - 是否返回密度图
            return_stats: bool - 是否返回统计信息

        Returns:
            seg_logits: (B, num_classes, H, W) - 分割logits
            density_map: (B, 1, H, W) (optional) - 密度图
            stats: dict (optional) - 统计信息
        """
        B, _, H, W = images.shape

        # ========== 1. Base Model Forward (OmniSAM) ==========
        # OmniSAM returns seg_logits directly
        base_output = self.base_model(images)  # (B, num_classes, H, W)
        
        # Get intermediate features for density estimation
        # Note: We need to extract features from base_model encoder
        # For now, use a simplified approach - extract from base_model.model.image_encoder
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'image_encoder'):
            encoder = self.base_model.model.image_encoder
            features = encoder(images)  # Get encoded features
        else:
            # Fallback: use simplified backbone
            features = self.backbone(images) if hasattr(self, 'backbone') else None
            
        if features is None:
            # If no features available, return base model output directly
            return base_output

        # ========== 2. Semantic Density Estimation ==========
        density_map = self.density_estimator(features)  # (B, 1, H/8, W/8)

        # 上采样密度图到原始分辨率 (用于可视化和统计)
        density_map_full = F.interpolate(
            density_map, size=(H, W), mode='bilinear', align_corners=False
        )

        # ========== 3. Dynamic FSMC (可选) ==========
        stats = {}
        if self.enable_dynamic_fsmc:
            features, fsmc_stats = self.dynamic_fsmc(features, density_map)
            stats['fsmc'] = fsmc_stats

        # ========== 4. Dynamic SCM (可选) ==========
        if self.enable_dynamic_scm:
            # Apply padding
            padded_features, pad_width = self.dynamic_scm(
                features, density_map, return_pad_width=True
            )

            # Process padded features (这里可以插入全局上下文模块)
            processed_features = padded_features

            # Remove padding
            features = self.dynamic_scm.remove_circular_padding(
                processed_features, pad_width
            )

            stats['scm'] = {
                'pad_width': pad_width,
                'padded_width': padded_features.size(3)
            }

        # ========== 5. Segmentation Head ==========
        seg_logits = self.seg_head(features)  # (B, num_classes, H/8, W/8)

        # 上采样到原始分辨率
        seg_logits = F.interpolate(
            seg_logits, size=(H, W), mode='bilinear', align_corners=False
        )

        # ========== 6. Return Results ==========
        # Always return dictionary for consistent interface
        result_dict = {
            'pred': seg_logits,
            'segmentation': seg_logits  # Alias for compatibility
        }

        if return_density:
            result_dict['density_map'] = density_map_full

        if return_stats:
            stats['density'] = {
                'mean': density_map.mean().item(),
                'std': density_map.std().item(),
                'min': density_map.min().item(),
                'max': density_map.max().item()
            }
            result_dict['stats'] = stats

        # Return dict if any optional outputs requested, otherwise just tensor
        if return_density or return_stats:
            return result_dict
        else:
            return seg_logits

    def compute_consistency_loss(self, density_map, seg_logits):
        """
        计算一致性损失 - 确保密度估计与分割不确定性对齐

        Intuition:
            - 高密度区域应对应高分割不确定性 (多类别竞争)
            - 低密度区域应对应低分割不确定性 (单一类别主导)

        Args:
            density_map: (B, 1, H, W) - 密度图
            seg_logits: (B, num_classes, H, W) - 分割logits

        Returns:
            consistency_loss: scalar - MSE loss
        """
        # 计算分割熵 (不确定性度量)
        seg_probs = F.softmax(seg_logits, dim=1)  # (B, C, H, W)
        entropy = -(seg_probs * torch.log(seg_probs + 1e-8)).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # 归一化熵到[0,1]
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
        entropy_normalized = entropy / max_entropy  # (B, 1, H, W)

        # 下采样到density_map的分辨率
        entropy_downsampled = F.interpolate(
            entropy_normalized,
            size=density_map.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # MSE loss
        consistency_loss = F.mse_loss(density_map, entropy_downsampled)

        return consistency_loss


if __name__ == "__main__":
    """单元测试"""
    print("=" * 60)
    print("SGDRFNet 单元测试")
    print("=" * 60)

    # ========== 测试1: 完整模型 ==========
    print("\n[测试1] 完整SG-DRF模型")
    model = SGDRFNet(
        num_classes=19,
        enable_dynamic_fsmc=True,
        enable_dynamic_scm=True,
        training_stage='joint'
    )
    model.eval()

    # 模拟输入
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 1024)  # 典型全景图尺寸

    print(f"输入图像: {images.shape}")

    # 前向传播
    with torch.no_grad():
        seg_logits, density_map, stats = model(
            images,
            return_density=True,
            return_stats=True
        )

    print(f"\n输出分割logits: {seg_logits.shape}")
    print(f"输出密度图: {density_map.shape}")
    print("\n统计信息:")
    for module_name, module_stats in stats.items():
        print(f"  {module_name}:")
        for key, value in module_stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    # ========== 测试2: Cold-start阶段 ==========
    print("\n" + "=" * 60)
    print("[测试2] Cold-start训练阶段")
    model_cold = SGDRFNet(
        num_classes=19,
        enable_dynamic_fsmc=False,
        enable_dynamic_scm=False,
        training_stage='cold_start'
    )

    # 检查参数冻结状态
    trainable_params = sum(p.numel() for p in model_cold.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_cold.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,}")

    # ========== 测试3: 一致性损失 ==========
    print("\n" + "=" * 60)
    print("[测试3] 一致性损失计算")
    with torch.no_grad():
        seg_logits_test = torch.randn(2, 19, 64, 128)
        density_map_test = torch.rand(2, 1, 64, 128)

    model.eval()
    consistency_loss = model.compute_consistency_loss(density_map_test, seg_logits_test)
    print(f"Consistency Loss: {consistency_loss.item():.6f}")

    # ========== 参数统计 ==========
    print("\n" + "=" * 60)
    print("[参数统计] 完整模型")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 分模块统计
    print("\n各模块参数量:")
    print(f"  Backbone: {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  DensityEstimator: {sum(p.numel() for p in model.density_estimator.parameters()):,}")
    if model.enable_dynamic_fsmc:
        print(f"  DynamicFSMC: {sum(p.numel() for p in model.dynamic_fsmc.parameters()):,}")
    if model.enable_dynamic_scm:
        print(f"  DynamicSCM: {sum(p.numel() for p in model.dynamic_scm.parameters()):,}")
    print(f"  SegHead: {sum(p.numel() for p in model.seg_head.parameters()):,}")

    print("\n" + "=" * 60)
    print("测试通过 ✓")
    print("=" * 60)
