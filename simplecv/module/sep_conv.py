import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Sequential):
    """
    for Xception
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super(SeparableConv2d, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                      bias=bias, padding_mode=padding_mode),
            nn.Conv2d(in_channels, out_channels, 1)
        )


class SeparableConv2D(nn.Module):
    """
    for ASPP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True,
                 use_batchnorm=False,
                 norm_fn=nn.BatchNorm2d):
        super(SeparableConv2D, self).__init__()
        self.use_bn = use_batchnorm
        self.dilation = dilation
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        if use_batchnorm:
            self.bn = norm_fn(in_channels)
        nn.init.normal_(self.depthwise.weight, std=0.33)
        nn.init.normal_(self.pointwise.weight, std=0.06)

    def forward(self, x):
        x = self.depthwise(x)
        if self.use_bn:
            x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pointwise(x)
        return x
