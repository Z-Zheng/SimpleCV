from albumentations import RandomScale, PadIfNeeded
from albumentations.pytorch import ToTensorV2
import random
import cv2
import numpy as np

__all__ = ['RandomDiscreteScale',
           'ToTensor',
           'ConstantPad']


class RandomDiscreteScale(RandomScale):
    def __init__(self, scales, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super(RandomDiscreteScale, self).__init__(0, interpolation, always_apply, p)
        self.scales = scales

    def get_params(self):
        return {"scale": random.choice(self.scales)}


class ToTensor(ToTensorV2):
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, 'masks': self.apply_to_masks}

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(m, **params) for m in masks]


class ConstantPad(PadIfNeeded):
    def __init__(self,
                 min_height=1024,
                 min_width=1024,
                 value=None,
                 mask_value=None,
                 always_apply=False,
                 p=1.0, ):
        super(ConstantPad, self).__init__(min_height, min_width, None, value, mask_value, always_apply, p)

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant',
                      constant_values=self.value)

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant',
                      constant_values=self.mask_value)
