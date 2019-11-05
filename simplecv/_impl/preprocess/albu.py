from albumentations import RandomScale
import random
import cv2


class RandomDiscreteScale(RandomScale):
    def __init__(self, scales, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super(RandomDiscreteScale, self).__init__(0, interpolation, always_apply, p)
        self.scales = scales

    def get_params(self):
        return {"scale": random.choice(self.scales)}
