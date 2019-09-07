from simplecv import registry
import torch.nn as nn

registry.OP.register('batchnorm', nn.BatchNorm2d)
registry.OP.register('groupnorm', nn.GroupNorm)
# basic component
from simplecv.module.aspp import AtrousSpatialPyramidPool
from simplecv.module.aspp import AtrousSpatialPyramidPoolv2
from simplecv.module.context_block import ContextBlock2d
from simplecv.module.sep_conv import SeparableConv2D
from simplecv.module.gap import GlobalAvgPool2D
from simplecv.module.se_block import SEBlock

# encoder
from simplecv.module.resnet import ResNetEncoder
from simplecv.module.fpn import FPN
from simplecv.module.fpn import LastLevelMaxPool
from simplecv.module.fpn import LastLevelP6P7

# loss
from simplecv.module import loss
