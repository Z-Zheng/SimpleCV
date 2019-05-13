import torch
import numpy as np
import torch.nn.functional as F


class TestTransform(object):
    def __init__(self):
        pass

    def transform(self, inputs):
        """

        Args:
            inputs: 4-D tensor of shape [batch, channel, height, width]

        Returns:
            transformed_inputs
        """
        raise NotImplementedError

    def inv_transform(self, transformed_inputs):
        """ inverse transformation

        Args:
            transformed_inputs:

        Returns:
            inputs
        """
        raise NotImplementedError

    @staticmethod
    def unit_test(transform):
        inputs = torch.ones(2, 32, 128, 128)
        transformed_inputs = transform.transform(inputs)
        our_inputs = transform.inv_transform(transformed_inputs)
        np.testing.assert_almost_equal(our_inputs.numpy(), inputs.numpy())


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
        transformed_inputs = torch.flip(inputs, [1])
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.flip(transformed_inputs, [1])
        return inputs


class VerticalFlip(TestTransform):
    def __init__(self):
        super(VerticalFlip, self).__init__()

    def transform(self, inputs):
        transformed_inputs = torch.flip(inputs, [0])
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.flip(transformed_inputs, [0])
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
    def __init__(self, scale_factor):
        super(Scale, self).__init__()
        self.scale_factor = scale_factor

    def transform(self, inputs):
        transformed_inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = F.interpolate(transformed_inputs, scale_factor=1.0 / self.scale_factor, mode='bilinear',
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
