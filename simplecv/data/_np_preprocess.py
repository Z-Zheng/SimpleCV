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


def _np_random_crop(image, crop_size):
    """

    Args:
        image: 3-D tensor of shape [h, w, c]
        crop_size: a tuple of (crop_h, crop_w)

    Returns:

    """
    im_h, im_w, _ = image.shape
    c_h, c_w = crop_size

    pad_h = c_h - im_h
    pad_w = c_w - im_w
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, [[0, pad_h], [0, pad_w], [0, 0]], mode='constant', constant_values=0)
    im_h, im_w, _ = image.shape

    y_lim = im_h - c_h + 1
    x_lim = im_w - c_w + 1
    ymin = np.random.randint(0, y_lim, 1)
    xmin = np.random.randint(0, x_lim, 1)

    xmax = xmin + c_w
    ymax = ymin + c_h

    crop_im = image[ymin:ymax, xmin:xmax, :]

    return crop_im
