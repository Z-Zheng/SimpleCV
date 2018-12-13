import torch
import torch.nn.functional as F

def _th_resize_to_range(image, min_size, max_size):
    h = image.size(0)
    w = image.size(1)
    c = image.size(2)
    im_size_min = min(h, w)
    im_size_max = max(h, w)

    im_scale = min(min_size / im_size_min, max_size / im_size_max)

    image = F.interpolate(image.permute(2, 0, 1).view(1, c, h, w), scale_factor=im_scale, mode='bilinear')
    return image, im_scale