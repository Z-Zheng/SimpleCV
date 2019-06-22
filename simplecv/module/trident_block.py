import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilations=(1, 2, 3)):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilations = dilations

    def forward(self, x):
        out_list = []
        if isinstance(x, list):
            for d, x_i in zip(self.dilations, x):
                out_list.append(self._forward_with_dilation(x_i, dilation=d))
        else:
            for d in self.dilations:
                out_list.append(self._forward_with_dilation(x, dilation=d))

        return out_list

    def _forward_with_dilation(self, x, dilation):
        org_dilation = self.conv2.dilation
        org_padding = self.conv2.padding
        self.conv2.dilation = (dilation, dilation)
        self.conv2.padding = (dilation, dilation)

        out = self._forward(x)

        self.conv2.dilation = org_dilation
        self.conv2.padding = org_padding
        return out

    def _forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def plugin_to_resnet(module):
    pass
