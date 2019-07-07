import cv2
import numpy as np
import math

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
        image = np.pad(image, [[0, max(pad_h, 0)], [0, max(pad_w, 0)], [0, 0]], mode='constant', constant_values=0)
    im_h, im_w, _ = image.shape

    y_lim = im_h - c_h + 1
    x_lim = im_w - c_w + 1
    ymin = int(np.random.randint(0, y_lim, 1))
    xmin = int(np.random.randint(0, x_lim, 1))

    xmax = xmin + c_w
    ymax = ymin + c_h

    crop_im = image[ymin:ymax, xmin:xmax, :]

    return crop_im


def _np_im_scale(image, scale_factor, size_divisor=None):
    """

    Args:
        image: 3-D of shape [height, width, channel]
        scale_factor:
        size_divisor:

    Returns:

    """
    im_h, im_w = image.shape[0:2]
    if size_divisor is None:
        resized_im = cv2.resize(
            image,
            None,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR
        )
    else:
        dst_h = int(np.ceil(scale_factor * im_h / size_divisor) * size_divisor)
        dst_w = int(np.ceil(scale_factor * im_h / size_divisor) * size_divisor)
        resized_im = cv2.resize(
            image,
            (dst_h, dst_w),
            None,
            interpolation=cv2.INTER_LINEAR
        )
    return resized_im


def _np_im_random_scale(image, scale_factors, size_divisor=None, mask=None):
    """

    Args:
        image: 3-D of shape [height, width, channel]
        scale_factors:
        size_divisor:
        mask:

    Returns:

    """
    if not isinstance(scale_factors, list) and not isinstance(scale_factors, tuple):
        raise ValueError('param: scale_factors should be list or tuple.')

    im_h, im_w = image.shape[0:2]

    if size_divisor is None:
        dst_sizes = [(round(im_h * scale), round(im_w * scale)) for scale in scale_factors]
    else:
        dst_sizes = [(int(np.ceil(im_h * scale / size_divisor) * size_divisor),
                      int(np.ceil(im_w * scale / size_divisor) * size_divisor)) for scale in
                     scale_factors]

    inds = np.arange(len(dst_sizes))
    index = np.random.choice(inds)
    dst_size = dst_sizes[index]

    resized_im = cv2.resize(
        image,
        dst_size,
        None,
        interpolation=cv2.INTER_LINEAR
    )
    if mask is not None:
        resized_mask = cv2.resize(
            mask,
            dst_size,
            None,
            interpolation=cv2.INTER_LINEAR
        )
        return resized_im, resized_mask
    return resized_im


def sliding_window(input_size, kernel_size, stride: int):
    ih, iw = input_size
    kh, kw = kernel_size
    assert ih > 0 and iw > 0 and kh > 0 and kw > 0 and stride > 0

    kh = ih if kh > ih else kh
    kw = iw if kw > iw else kw

    num_rows = math.ceil((ih - kh) / stride) if math.ceil((ih - kh) / stride) * stride + kh >= ih else math.ceil(
        (ih - kh) / stride) + 1
    num_cols = math.ceil((iw - kw) / stride) if math.ceil((iw - kw) / stride) * stride + kw >= iw else math.ceil(
        (iw - kw) / stride) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * stride
    ymin = y * stride

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + kw > iw, iw - xmin - kw, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + kh > ih, ih - ymin - kh, np.zeros_like(ymin))
    boxes = np.stack([xmin + xmin_offset, ymin + ymin_offset,
                      np.minimum(xmin + kw, iw), np.minimum(ymin + kh, ih)], axis=1)

    return boxes
