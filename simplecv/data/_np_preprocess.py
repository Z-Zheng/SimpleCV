import cv2
import numpy as np


def _np_resize_to_range(image, min_size, max_size):
    im_shape = image.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = min(min_size / im_size_min, max_size / im_size_max)

    image = cv2.resize(
        image,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    return image, im_scale
