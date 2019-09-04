import simplecv._impl.preprocess.function as pF
import torch.nn as nn
from simplecv.util.tensor_util import to_tensor


class Pipeline(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class FuncWrapper(nn.Module):
    def __init__(self, fn):
        super(FuncWrapper, self).__init__()
        self.fn = fn

    def forward(self, *input):
        return self.fn(*input)


class ToTensor(nn.Module):
    def forward(self, *input):
        return to_tensor(input)


class THChannelFirst(nn.Module):
    @staticmethod
    def _is_channel_first(tensor):
        return tensor.size(0) <= 8

    def forward(self, image):
        if THChannelFirst._is_channel_first(image):
            return image
        return image.permute(2, 0, 1)


class THChannelFirst2(THChannelFirst):
    def forward(self, image, other):
        return super(THChannelFirst2, self).forward(image), other


class THMeanStdNormalize(nn.Module):
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        super(THMeanStdNormalize, self).__init__()
        self._m = mean
        self._s = std

    def forward(self, image):
        image = image.float()
        nimage = pF.th_mean_std_normalize(image, self._m, self._s)

        return nimage


class THMeanStdNormalize2(THMeanStdNormalize):
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        super(THMeanStdNormalize2, self).__init__(mean, std)

    def forward(self, image, other):
        nimage = super(THMeanStdNormalize2, self).forward(image)
        return nimage, other


class THDivisiblePad(nn.Module):
    def __init__(self, size_divisor, mask_pad_value=255):
        super(THDivisiblePad, self).__init__()
        self.size_divisor = size_divisor
        self.mask_pad_value = mask_pad_value

    def forward(self, image, mask=None):
        pimage = pF.th_divisible_pad(image, self.size_divisor)
        if mask is not None:
            pmask = pF.th_divisible_pad(mask, self.size_divisor, value=self.mask_pad_value)
        else:
            pmask = mask

        return pimage, pmask
