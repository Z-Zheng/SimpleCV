import torch
import numpy as np
import torch.nn.functional as F
from simplecv.core.transform_base import TestTransform


class Rotate90k(TestTransform):
    def __init__(self, k=1):
        super(Rotate90k, self).__init__()
        assert k in [1, 2, 3]

        self.k = k

    def transform(self, inputs):
        transformed_inputs = torch.rot90(inputs, self.k, [2, 3])
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.rot90(transformed_inputs, 4 - self.k, [2, 3])
        return inputs


class HorizontalFlip(TestTransform):
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def transform(self, inputs):
        transformed_inputs = torch.flip(inputs, [3])
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.flip(transformed_inputs, [3])
        return inputs


class VerticalFlip(TestTransform):
    def __init__(self):
        super(VerticalFlip, self).__init__()

    def transform(self, inputs):
        transformed_inputs = torch.flip(inputs, [2])
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.flip(transformed_inputs, [2])
        return inputs


class Transpose(TestTransform):
    def __init__(self):
        super(Transpose, self).__init__()

    def transform(self, inputs):
        transformed_inputs = torch.transpose(inputs, 2, 3)
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.transpose(transformed_inputs, 2, 3)
        return inputs


class Scale(TestTransform):
    def __init__(self, size=None, scale_factor=None):
        super(Scale, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.input_shape = None

    def transform(self, inputs):
        self.input_shape = inputs.shape
        transformed_inputs = F.interpolate(inputs, size=self.size, scale_factor=self.scale_factor, mode='bilinear',
                                           align_corners=True)
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        size = (self.input_shape[2], self.input_shape[3])
        inputs = F.interpolate(transformed_inputs, size=size, mode='bilinear',
                               align_corners=True)
        return inputs


if __name__ == '__main__':
    # unit test
    TestTransform.unit_test(Rotate90k(k=1))
    TestTransform.unit_test(Rotate90k(k=2))
    TestTransform.unit_test(Rotate90k(k=3))

    TestTransform.unit_test(HorizontalFlip())
    TestTransform.unit_test(VerticalFlip())
    TestTransform.unit_test(Transpose())

    for scale_factor in np.linspace(0.25, 2.0, num=int((2.0 - 0.25) / 0.25 + 1)):
        TestTransform.unit_test(Scale(scale_factor=float(scale_factor)))

    TestTransform.unit_test(Scale(scale_factor=float(0.49)))

    TestTransform.unit_test(Scale(size=(894, 896)))
