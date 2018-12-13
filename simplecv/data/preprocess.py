import cv2
import numpy as np
import torch
import torch.nn.functional as F


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


def _th_resize_to_range(image, min_size, max_size):
    h = image.size(0)
    w = image.size(1)
    c = image.size(2)
    im_size_min = min(h, w)
    im_size_max = max(h, w)

    im_scale = min(min_size / im_size_min, max_size / im_size_max)

    image = F.interpolate(image.permute(2, 0, 1).view(1, c, h, w), scale_factor=im_scale, mode='bilinear')
    return image, im_scale


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

        new_boxes = np.concatenate([y1, w - x2, h - y2, w - x1])
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
        new_xmin = w - boxes[:, 2]
        new_xmax = w - boxes[:, 0]
        new_boxes = boxes.copy()
        new_boxes[:, 0] = new_xmin
        new_boxes[:, 2] = new_xmax
        ret.append(new_boxes)

    return tuple(ret) if len(ret) != 1 else ret[0]


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
        new_ymin = h - boxes[:, 3]
        new_ymax = h - boxes[:, 1]
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
