import random
import math
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image


def transpose(img):
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.transpose(Image.TRANSPOSE)


class ToTensor(object):
    def __call__(self, image, mask):
        return torch.from_numpy(np.array(image, copy=False)), torch.from_numpy(np.array(mask, copy=False))


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class RandomTranspose(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = transpose(image)
            mask = transpose(mask)
        return image, mask


class RandomRotate90K(object):
    def __init__(self, k=(0, 1, 2, 3)):
        self.k = k

    def __call__(self, image, mask):
        k = random.choice(self.k)
        if k == 0:
            return image, mask

        image = F.rotate(image, 90 * k, expand=True)
        mask = F.rotate(mask, 90 * k, expand=True)

        return image, mask


class RandomScale(object):
    def __init__(self, scale_factor, size_divisor=32):
        pass

    def __call__(self, image, mask):
        pass


class DivisiblePad(object):
    def __init__(self, size_divisor, mask_pad_value=255):
        self.size_divisor = size_divisor
        self.mask_pad_value = mask_pad_value

    def __call__(self, image, mask=None):
        ph = math.ceil(image.height / self.size_divisor) * self.size_divisor - image.height
        pw = math.ceil(image.width / self.size_divisor) * self.size_divisor - image.width

        if ph == 0 and pw == 0:
            if mask is None:
                return image
            return image, mask

        image = F.pad(image, (0, 0, pw, ph), 0)
        if mask is None:
            return image
        mask = F.pad(mask, (0, 0, pw, ph), self.mask_pad_value)
        return image, mask
