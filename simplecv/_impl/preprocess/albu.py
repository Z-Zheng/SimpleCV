from albumentations import RandomScale
from albumentations.pytorch import ToTensorV2
import random
import cv2

__all__ = ['RandomDiscreteScale',
           'ToTensor']


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
