import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DepthwiseConvblock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1):
        super(DepthwiseConvblock, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding='valid', 
            groups=in_channels, 
            bias=False
        )

        self.bn = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.silu(out)
        return out

class PointwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConvBlock, self).__init__()

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding='valid',
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pointwise(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        out = self.pool(x).view(batch_size, channels)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(batch_size, channels, 1, 1)
        out = x * out
        return out

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, reduction_ratio):
        super(MBConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.reduction_ratio = reduction_ratio

        self.expand_channels = int(round(in_channels * expand_ratio))
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        # Expansion phase
        if self.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=self.expand_channels, 
                kernel_size=1, 
                stride=1, 
                padding='valid',
                bias=False
            )
            self.bn = nn.BatchNorm2d(self.expand_channels)
            self.relu = nn.SiLU()

        # Depthwise convolution phase
        self.depthwise_conv = DepthwiseConvblock(
            in_channels=self.expand_channels, 
            kernel_size=kernel_size, 
            stride=stride
        )

        # SE phase
        num_reduced_channels = max(1, int(in_channels * reduction_ratio))
        self.se = SEBlock(self.expand_channels, num_reduced_channels)

        # Pointwise convolution phase
        self.pointwise_conv = PointwiseConvBlock(self.expand_channels, out_channels)

    def forward(self, x):
        if self.expand_ratio != 1:
            out = self.expand_conv(x)
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = x
        
        out = self.depthwise_conv(out)
        out = self.se(out)
        out = self.pointwise_conv(out)

        if self.use_residual:
            out += x
        
        return out

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()

        # stem phase
        self.stem_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)

        # Blocks
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 3, 1, 1, 4),

            MBConvBlock(16, 24, 3, 2, 6, 4),
            MBConvBlock(24, 24, 3, 1, 6, 4),
            
            MBConvBlock(24, 40, 5, 2, 6, 4),
            MBConvBlock(40, 40, 5, 1, 6, 4),

            MBConvBlock(40, 80, 3, 2, 6, 4),
            MBConvBlock(80, 80, 3, 1, 6, 4),
            MBConvBlock(80, 80, 3, 1, 6, 4),

            MBConvBlock(80, 112, 5, 1, 6, 4),
            MBConvBlock(112, 112, 5, 1, 6, 4),
            MBConvBlock(112, 112, 5, 1, 6, 4),

            MBConvBlock(112, 192, 5, 2, 6, 4),
            MBConvBlock(192, 192, 5, 1, 6, 4),
            MBConvBlock(192, 192, 5, 1, 6, 4),
            MBConvBlock(192, 192, 5, 1, 6, 4),

            MBConvBlock(192, 320, 3, 1, 6, 4)
        )

        # head phase
        self.head_conv = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding='valid', bias=False)

        # pooling phase
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # classifer
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x