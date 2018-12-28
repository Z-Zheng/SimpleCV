import torch
import torch.nn as nn
import torch.nn.functional as F
from simplecv import registry
from simplecv.interface import CVModule
from simplecv.module.gap import GlobalAvgPool2D


class InvResidulBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride=1, dilation=1,
                 bias=True):
        super(InvResidulBlock, self).__init__()
        t = expansion_factor
        self.s = stride
        padding = dilation
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1, bias=bias),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=self.s, padding=padding, dilation=dilation,
                      groups=in_channels * t, bias=bias),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=bias)
        )

    def forward(self, x):
        res = self.seq(x)
        if self.s == 1 and x.size(1) == res.size(1):
            out = res + x
        else:
            out = res
        return out


class MobileNetv2(CVModule):
    def __init__(self, config):
        super(MobileNetv2, self).__init__(config)
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)

        self.blocks = self.make_blocks(bias=True)

        self.conv1x1 = nn.Conv2d(320, 1280, 1)
        self.gap = GlobalAvgPool2D()
        if self.num_classes is not None:
            self.cls_conv = nn.Conv2d(1280, self.num_classes, 1)

    def make_blocks(self, bias=True):
        blocks = nn.ModuleList()
        in_c = self.conv1.out_channels
        for t, c, n, s in zip(self.expansion_factors, self.out_channels, self.repeats, self.strides):
            layers = [InvResidulBlock(in_c, c, t, s, bias=bias)]
            in_c = c
            if n > 1:
                s = 1
                layers += [InvResidulBlock(c, c, t, s, bias=bias) for _ in range(n - 1)]

            blocks.append(nn.Sequential(*layers))
        return blocks

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)

        x = self.conv1x1(x)
        x = self.gap(x)
        if self.num_classes is not None:
            x = self.cls_conv(x)
        return x

    def set_defalut_config(self):
        self.config.update(dict(
            num_classes=None,
            expansion_factors=(1, 6, 6, 6, 6, 6, 6),
            out_channels=(16, 24, 32, 64, 96, 160, 320),
            repeats=(1, 2, 3, 4, 3, 3, 1),
            strides=(1, 2, 2, 2, 1, 2, 1)
        ))


if __name__ == '__main__':
    m = MobileNetv2(dict(num_classes=1000))
    from simplecv.util import param_util

    conv = nn.Conv2d(32, 32, 3, 1, 1)
    irb = InvResidulBlock(32, 32, expansion_factor=3)
    param_util.count_model_parameters(conv)
    param_util.count_model_parameters(irb)

    im = torch.ones([1, 3, 224, 224])
