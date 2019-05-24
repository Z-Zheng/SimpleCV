import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from simplecv.util import param_util


class Conv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups,
                                     bias, padding_mode)

    def forward(self, input):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def convert_conv2d_with_ws(module):
    """

    Args:
        module: (nn.Module): containing module
    Returns:
        The original Conv2D with the converted `Conv2D with WS` layer

    Example::

            >>> # r16 ct c3-c5
            >>> from simplecv.module import ResNetEncoder
            >>> m = ResNetEncoder({})
            >>> m = convert_conv2d_with_ws(m)

    """
    classname = module.__class__.__name__
    module_output = module
    if classname.find('Conv') != -1:
        module_output = Conv2D(module.in_channels,
                               module.out_channels,
                               module.kernel_size,
                               module.stride,
                               module.padding, module.dilation, module.groups,
                               module.bias is not None,
                               module.padding_mode
                               )

        param_util.copy_conv_parameters(module, module_output)

    for name, sub_module in module.named_children():
        module_output.add_module(name, convert_conv2d_with_ws(sub_module))
    del module
    return module_output
