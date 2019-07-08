import cv2
import numpy as np
import torch

from simplecv.data._th_preprocess import _th_resize_to_range
from simplecv.data._th_preprocess import _th_mean_std_normalize
from simplecv.data._np_preprocess import _np_resize_to_range
from simplecv.data._np_preprocess import _np_mean_std_normalize
from simplecv.data._np_preprocess import _np_random_crop
from simplecv.data._np_preprocess import _np_im_random_scale
from simplecv.data._np_preprocess import _np_im_scale
from simplecv.data._np_preprocess import sliding_window
from simplecv.data._th_preprocess import _th_divisible_pad as th_divisible_pad


def random_crop(image, crop_size):
    if isinstance(image, np.ndarray):
        return _np_random_crop(image, crop_size)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))


def divisible_pad(image_list, size_divisor=128, to_tensor=True):
    """

    Args:
        image_list: a list of images with shape [channel, height, width]
        size_divisor: int
        to_tensor: whether to convert to tensor
    Returns:
        blob: 4-D ndarray of shape [batch, channel, divisible_max_height, divisible_max_height]
    """
    max_shape = np.array([im.shape for im in image_list]).max(axis=0)

    max_shape[1] = int(np.ceil(max_shape[1] / size_divisor) * size_divisor)
    max_shape[2] = int(np.ceil(max_shape[2] / size_divisor) * size_divisor)

    if to_tensor:
        storage = torch.FloatStorage._new_shared(len(image_list) * np.prod(max_shape))
        out = torch.Tensor(storage).view([len(image_list), max_shape[0], max_shape[1], max_shape[2]])
        out = out.zero_()
    else:
        out = np.zeros([len(image_list), max_shape[0], max_shape[1], max_shape[2]], np.float32)

    for i, resized_im in enumerate(image_list):
        out[i, :, 0:resized_im.shape[1], 0:resized_im.shape[2]] = torch.from_numpy(resized_im)

    return out


def resize_to_range(image, min_size, max_size):
    """

    Args:
        image: [height, width, channel]
        min_size: int
        max_size: int

    Returns:

    """
    if isinstance(image, np.ndarray):
        return _np_resize_to_range(image, min_size, max_size)
    elif isinstance(image, torch.Tensor):
        return _th_resize_to_range(image, min_size, max_size)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))


def transpose(image, mask=None, boxes=None):
    ret = []
    new_image = np.transpose(image, axes=[1, 0, 2])
    ret.append(new_image)

    if mask is not None:
        new_mask = np.transpose(mask, axes=[0, 1])
        ret.append(new_mask)
    if boxes is not None:
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)

        new_boxes = np.concatenate([y1, x1, y2, x2], axis=1)
        ret.append(new_boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def rotate_90(image, mask=None, boxes=None):
    ret = []
    new_image = np.rot90(image, k=1)
    ret.append(new_image)

    if mask is not None:
        new_mask = np.rot90(mask, k=1)
        ret.append(new_mask)
    if boxes is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)

        new_boxes = np.concatenate([y1, w - x2, y2, w - x1], axis=1)
        ret.append(new_boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def rotate_180(image, mask=None, boxes=None):
    ret = []
    new_image = np.rot90(image, k=2)
    ret.append(new_image)

    if mask is not None:
        new_mask = np.rot90(mask, k=2)
        ret.append(new_mask)
    if boxes is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)

        new_boxes = np.concatenate([w - x2, h - y2, w - x1, h - y1], axis=1)
        ret.append(new_boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def rotate_270(image, mask=None, boxes=None):
    ret = []
    new_image = np.rot90(image, k=3)
    ret.append(new_image)

    if mask is not None:
        new_mask = np.rot90(mask, k=3)
        ret.append(new_mask)
    if boxes is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)

        new_boxes = np.concatenate([h - y2, x1, h - y1, x2], axis=1)
        ret.append(new_boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def flip_left_right(image, mask=None, boxes=None):
    """

    Args:
        image: 3-D of shape [height, width, channel]
        mask:
        boxes: 2-D of shape [N, 4] xmin, ymin, xmax, ymax
    Returns:

    """
    ret = []
    new_image = image[:, ::-1, :]
    ret.append(new_image)

    if mask is not None:
        new_mask = mask[:, ::-1]
        ret.append(new_mask)

    if boxes is not None:
        w = image.shape[1]
        new_xmin = w - boxes[:, 2] - 1
        new_xmax = w - boxes[:, 0] - 1
        new_boxes = boxes.copy()
        new_boxes[:, 0] = new_xmin
        new_boxes[:, 2] = new_xmax
        ret.append(new_boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def scale_image(image, scale_factor, size_divisor=None):
    # todo: support torch.Tensor
    if isinstance(image, np.ndarray):
        return _np_im_scale(image, scale_factor, size_divisor)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))


def random_scale_image(image, scale_factors, size_divisor=None, mask=None):
    # todo: support torch.Tensor
    if isinstance(image, np.ndarray):
        return _np_im_random_scale(image, scale_factors, size_divisor, mask)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))


def scale_image_and_label(image, scale_factor, max_stride=32, mask=None, fixed_size=None):
    """

    Args:
        image: 3-D of shape [height, width, channel]
        scale_factor: a float number
        max_stride:
        mask: 2-D of shape [height, width]
        fixed_size: a tuple of (fixed height, fixed width)
    Returns:

    """
    resized_im = cv2.resize(image, None, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    im_h, im_w = resized_im.shape[0:2]
    dst_h = int(np.ceil(im_h / max_stride) * max_stride)
    dst_w = int(np.ceil(im_w / max_stride) * max_stride)
    if fixed_size:
        padded_im = np.zeros([fixed_size[0], fixed_size[1], 3], dtype=image.dtype)
        padded_im[:im_h, :im_w, :] = resized_im
    else:
        padded_im = np.zeros([dst_h, dst_w, 3], dtype=image.dtype)
        padded_im[:im_h, :im_w, :] = resized_im

    if mask is not None:
        resized_mask = cv2.resize(mask, None, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        if fixed_size:
            padded_mask = np.zeros([fixed_size[0], fixed_size[1], 3], dtype=mask.dtype)
            padded_mask[:im_h, :im_w, :] = resized_mask
        else:
            padded_mask = np.zeros([dst_h, dst_w], dtype=image.dtype)
            padded_mask[:im_h, :im_w, :] = resized_mask
        return padded_im, padded_mask
    return padded_im


def mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    """

    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple
        std: a list or tuple

    Returns:

    """
    if isinstance(image, np.ndarray):
        return _np_mean_std_normalize(image, mean, std)
    elif isinstance(image, torch.Tensor):
        return _th_mean_std_normalize(image, mean, std)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))


def channel_last_to_first(image):
    """

    Args:
        image: 3-D numpy array of shape [height, width, channel]

    Returns:
        new_image: 3-D numpy array of shape [channel, height, width]
    """
    new_image = np.transpose(image, axes=[2, 0, 1])
    return new_image


def flip_up_down(image, mask=None, boxes=None):
    """

    Args:
        image: 3-D of shape [height, width, channel]
        mask:

    Returns:

    """
    ret = []
    new_image = image[::-1, :, :]
    ret.append(new_image)

    if mask is not None:
        new_mask = mask[::-1, :]
        ret.append(new_mask)

    if boxes is not None:
        h = image.shape[0]
        new_ymin = h - boxes[:, 3] - 1
        new_ymax = h - boxes[:, 1] - 1
        new_boxes = boxes.copy()
        new_boxes[:, 1] = new_ymin
        new_boxes[:, 3] = new_ymax
        ret.append(new_boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def random_flip_up_down(image, mask=None, boxes=None, prob=0.5):
    """

    Args:
        image: 3-D of shape [height, width, channel]
        mask:
        prob:

    Returns:

    """
    if np.random.random() < prob:
        return flip_up_down(image, mask, boxes)

    ret = [image]
    if mask is not None:
        ret.append(mask)

    if boxes is not None:
        ret.append(boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def random_flip_left_right(image, mask=None, boxes=None, prob=0.5):
    """

    Args:
        image: 3-D of shape [height, width, channel]
        mask:
        prob:

    Returns:

    """
    if np.random.random() < prob:
        return flip_left_right(image, mask, boxes)

    ret = [image]
    if mask is not None:
        ret.append(mask)

    if boxes is not None:
        ret.append(boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def random_rotate_90(image, mask=None, boxes=None, prob=0.5):
    if np.random.random() < prob:
        return rotate_90(image, mask, boxes)

    ret = [image]
    if mask is not None:
        ret.append(mask)

    if boxes is not None:
        ret.append(boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def random_rotate_180(image, mask=None, boxes=None, prob=0.5):
    if np.random.random() < prob:
        return rotate_180(image, mask, boxes)

    ret = [image]
    if mask is not None:
        ret.append(mask)

    if boxes is not None:
        ret.append(boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def random_rotate_270(image, mask=None, boxes=None, prob=0.5):
    if np.random.random() < prob:
        return rotate_270(image, mask, boxes)

    ret = [image]
    if mask is not None:
        ret.append(mask)

    if boxes is not None:
        ret.append(boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


def random_transpose(image, mask=None, boxes=None, prob=0.5):
    if np.random.random() < prob:
        return transpose(image, mask, boxes)

    ret = [image]
    if mask is not None:
        ret.append(mask)

    if boxes is not None:
        ret.append(boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]
