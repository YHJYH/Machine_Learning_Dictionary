import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DepthwiseConvblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseConvblock, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=3, 
            stride=stride, 
            padding='same', 
            groups=in_channels, 
            bias=False
        )

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.relu(out)
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
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
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
    def __init__(self, in_channels, out_channels, expand_ratio, stride=1):
        super(MBConvBlock, self).__init__()

        self.expand_ratio = expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        # Expansion phase
        hidden_dim = round(in_channels * expand_ratio)
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding='valid', bias=False)
            self.bn = nn.BatchNorm2d(hidden_dim)
            self.relu = nn.ReLU(inplace=True)

        # Depthwise convolution phase
        self.depthwise_conv = DepthwiseConvblock(hidden_dim, hidden_dim, stride=stride)

        # SE phase
        self.se = SEBlock(hidden_dim)

        # Pointwise convolution phase
        self.pointwise_conv = PointwiseConvBlock(hidden_dim, out_channels)

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

