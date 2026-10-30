import torch.nn as nn
import torch
import torch.nn.functional as F

class _ConvBNReLU(nn.Module):
    """与 mmcv ConvModule(kernel=1, norm=BN, act=ReLU) 的参数命名一致，便于加载官方权重。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.activate(x)

class _DepthwiseSeparableBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.activate(x)


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=(32, 64, 256), embedding_dim=256, num_classes=8,
                 channel_ratio: float = 1.0, conv_mode: str = "standard", upsample_mode: str = "bilinear", **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.upsample_mode = upsample_mode
        c1_in_channels, c2_in_channels, c3_in_channels = 32, 64, 256
        inner_dim = max(64, int(round(embedding_dim * float(channel_ratio))))

        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=inner_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=inner_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=inner_dim)
        self.dropout = nn.Dropout2d(0.1)
        if conv_mode == "depthwise":
            self.linear_fuse = _DepthwiseSeparableBNReLU(inner_dim * 3, inner_dim)
        else:
            self.linear_fuse = _ConvBNReLU(inner_dim * 3, inner_dim)
        self.out_proj = nn.Conv2d(inner_dim, embedding_dim, kernel_size=1, bias=False) if inner_dim != embedding_dim else nn.Identity()
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3= x
        ############## MLP decoder on C1-C3 ###########
        n, _, h, w = c3.shape
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode=self.upsample_mode, align_corners=False if self.upsample_mode == "bilinear" else None)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode=self.upsample_mode, align_corners=False if self.upsample_mode == "bilinear" else None)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c3, _c2, _c1], dim=1))
        _c = self.out_proj(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class SegFormerHead_adaption(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=(32, 64, 256), embedding_dim=256, num_classes=8,
                 channel_ratio: float = 1.0, conv_mode: str = "standard", upsample_mode: str = "bilinear", **kwargs):
        super(SegFormerHead_adaption, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.upsample_mode = upsample_mode
        self.out_channels = embedding_dim
        c1_in_channels, c2_in_channels, c3_in_channels = 32, 64, 256
        inner_dim = max(64, int(round(embedding_dim * float(channel_ratio))))

        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=inner_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=inner_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=inner_dim)
        self.dropout = nn.Dropout2d(0.1)
        if conv_mode == "depthwise":
            self.linear_fuse = _DepthwiseSeparableBNReLU(inner_dim * 3, inner_dim)
        else:
            self.linear_fuse = _ConvBNReLU(inner_dim * 3, inner_dim)
        self.out_proj = nn.Conv2d(inner_dim, embedding_dim, kernel_size=1, bias=False) if inner_dim != embedding_dim else nn.Identity()
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3= x
        ############## MLP decoder on C1-C3 ###########
        n, _, h, w = c3.shape

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode=self.upsample_mode, align_corners=False if self.upsample_mode == "bilinear" else None)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode=self.upsample_mode, align_corners=False if self.upsample_mode == "bilinear" else None)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # feat = [_c1, _c2, _c3]
        
        _c = self.linear_fuse(torch.cat([_c3, _c2, _c1], dim=1))
        _c = self.out_proj(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x, _c
