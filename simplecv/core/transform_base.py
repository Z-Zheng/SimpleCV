import torch
import numpy as np


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
