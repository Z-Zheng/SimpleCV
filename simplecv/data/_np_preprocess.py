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


def _np_mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    """

    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray

    Returns:

    """
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean, np.float32)
    if not isinstance(std, np.ndarray):
        std = np.array(std, np.float32)
    shape = [1] * image.ndim
    shape[-1] = -1
    return (image - mean.reshape(shape)) / std.reshape(shape)

