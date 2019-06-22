import torch
import torch.nn as nn
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

BaseLayer = namedtuple('BaseLayer', ['kernel_size', 'stride', 'dilation', 'padding',
                                     'input_size', 'output_size', 'receptive_field',
                                     'layer_name',
                                     'is_hide'])


def make_pair(x):
    return x, x


def from_xx(module, name=None):
    if isinstance(module, nn.Conv2d):
        return from_conv2d(module, name)
    elif isinstance(module, nn.MaxPool2d):
        return from_maxpool2d(module, name)
    else:
        raise TypeError('Unsupport {}'.format(type(module)))


def from_conv2d(module: nn.Conv2d, name=None):
    return BaseLayer(
        kernel_size=module.kernel_size,
        stride=module.stride,
        dilation=module.dilation,
        padding=module.padding,
        input_size=None,
        output_size=None,
        receptive_field=None,
        layer_name=module.__class__.__name__ if name is None else name,
        is_hide=False
    )


def from_maxpool2d(module: nn.MaxPool2d, name=None):
    return BaseLayer(
        kernel_size=make_pair(module.kernel_size),
        stride=make_pair(module.stride),
        dilation=make_pair(module.dilation),
        padding=make_pair(module.padding),
        input_size=None,
        output_size=None,
        receptive_field=None,
        layer_name=module.__class__.__name__ if name is None else name,
        is_hide=False
    )


def plot_receptive_field_growth_from_module(input_shape, module: nn.Module):
    layer_list = make_baselayers_from_tracking(module, input_shape)
    print_baselayers(layer_list)
    return plot_receptive_field_growth_from_baselayers(layer_list)


def plot_receptive_field_growth_from_baselayers(layer_list: list):
    """

    Args:
        layer_list:

    Returns:
    """
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # n = reduce(lambda layer1, layer2: int(not layer1.is_hide) + int(not layer2.is_hide), layer_list)
    assert all([layer.receptive_field[0] == layer.receptive_field[1] for layer in layer_list])
    rf_list = [layer.receptive_field[0] for layer in layer_list if not layer.is_hide]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    x = np.arange(len(rf_list))
    y = np.array(rf_list)
    plt.plot(x, y, color='b', marker='o', markersize=3,
             markeredgecolor='black', mfc='g')
    sparse_x = np.linspace(1, len(rf_list), len(rf_list) // 5)
    sparse_x = np.minimum(sparse_x, len(rf_list)) - 1
    smooth_y = interpolate.interp1d(x, y, kind='cubic')(sparse_x)
    plt.plot(sparse_x, smooth_y, 'r')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.fill_between(sparse_x, smooth_y, color="r", alpha=0.3)
    plt.grid()


def print_baselayers(layer_list: list):
    cnt = 1
    for layer in layer_list:
        print(cnt, layer)
        if not layer.is_hide:
            cnt += 1


def make_baselayers_from_tracking(model: nn.Module, input_shape):
    def _register_hook(module):
        def _hook(module, input, output):

            last_rf = getattr(input[0], 'receptive_field', last_mem_receptive_field[0])
            accumulated_stride = getattr(input[0], 'accumulated_stride', last_mem_accumulated_stride[0])
            if any([isinstance(module, nn.Conv2d),
                    isinstance(module, nn.MaxPool2d)]):
                baselayer = from_xx(module)
                baselayer = baselayer._replace(input_size=(input[0].size(2), input[0].size(3)))
                baselayer = baselayer._replace(output_size=(output.size(2), output.size(3)))

                baselayer.dilation[0] * (baselayer.kernel_size[0] - 1)

                cur_rf_H = last_rf[0] + (baselayer.kernel_size[0] - 1) * baselayer.dilation[0] * accumulated_stride[0]
                cur_rf_W = last_rf[1] + (baselayer.kernel_size[1] - 1) * baselayer.dilation[1] * accumulated_stride[1]

                baselayer = baselayer._replace(receptive_field=(cur_rf_H, cur_rf_W))

                baselayers.append(baselayer)
                output.receptive_field = baselayer.receptive_field
                output.accumulated_stride = (
                    accumulated_stride[0] * baselayer.stride[0], accumulated_stride[1] * baselayer.stride[1])

                last_mem_receptive_field[0] = baselayer.receptive_field
                last_mem_accumulated_stride[0] = output.accumulated_stride
            else:
                output.receptive_field = last_rf

        if not module._modules:
            hooks.append(module.register_forward_hook(_hook))

    hooks = []
    baselayers = []
    last_mem_receptive_field = [1]
    last_mem_accumulated_stride = [(1, 1)]
    model.apply(_register_hook)

    with torch.no_grad():
        input = torch.ones(input_shape)
        input.receptive_field = (1, 1)
        input.accumulated_stride = (1, 1)
        model(input)

    for hook in hooks:
        hook.remove()

    _rf = (1, 1)
    for idx, layer in enumerate(baselayers):
        if layer.receptive_field[0] < _rf[0] or layer.receptive_field[1] < _rf[1]:
            baselayers[idx] = layer._replace(is_hide=True)
        else:
            _rf = layer.receptive_field
    return baselayers


if __name__ == '__main__':
    class A(nn.Module):
        def __init__(self):
            super(A, self).__init__()
            self.a = nn.Sequential(
                nn.Conv2d(3, 3, 3, 2, 1)
            )
            self.b = nn.Sequential(
                nn.Conv2d(3, 3, 3, 2, 1)
            )
            self.c = nn.Sequential(
                nn.Conv2d(3, 3, 3, 2, 1)
            )

        def forward(self, x):
            return self.a(x) + self.b(x) + self.c(x)


    from torchvision.models.resnet import resnet50, resnet101

    resnet = resnet101()
    print(resnet)
    plt.subplot(121)
    plot_receptive_field_growth_from_module([1, 3, 256, 256], resnet)
    plt.subplot(122)
    plot_receptive_field_growth_from_module([1, 3, 256, 256], resnet50())
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # x = np.arange(50)
    # y = x
    #
    # # ax.set_xticks(x)
    # ax.set_xlabel('Layer Index')
    # ax.plot(x, y)
    # plt.ylim(bottom=0)
    # plt.xlim(0, 50)
    #
    # plt.show()
