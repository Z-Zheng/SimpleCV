"""
modified from https://github.com/dmlc/gluon-cv/blob/master/gluoncv/utils/viz
"""
from __future__ import division

import random

import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_image(img, ax=None, reverse_rgb=False):
    """Visualize image.
    Parameters
    ----------
    img : numpy.ndarray
        Image with shape `H, W, 3`.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    Returns
    -------
    matplotlib axes
        The ploted axes.
    Examples
    --------
    from matplotlib import pyplot as plt
    ax = plot_image(img)
    plt.show()
    """
    from matplotlib import pyplot as plt
    if ax is None:
        # create new axes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    img = img.copy()
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    ax.imshow(img.astype(np.uint8))
    return ax


def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True):
    """Visualize bounding boxes.
    Parameters
    ----------
    img : numpy.ndarray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).
    Returns
    -------
    matplotlib axes
        The ploted axes.
    """
    from matplotlib import pyplot as plt

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=1.5)
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            ax.text(xmin, ymin - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=12, color='white')
    return ax


def plot_mask(img, masks, alpha=0.5):
    """Visualize segmentation mask.
    Parameters
    ----------
    img : numpy.ndarray
        Image with shape `H, W, 3`.
    masks : numpy.ndarray
        Binary images with shape `N, H, W`.
    alpha : float, optional, default 0.5
        Transparency of plotted mask
    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks
    """
    rs = np.random.RandomState(567)
    for mask in masks:
        color = rs.random_sample(3) * 255
        mask = np.repeat((mask > 0)[:, :, np.newaxis], repeats=3, axis=2)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)
    return img.astype('uint8')


def plot_bbox_mask(img, bboxes, masks, scores=None, labels=None, thresh=0.5,
                   class_names=None, colors=None, ax=None,
                   reverse_rgb=False, absolute_coordinates=True):
    full_masks, _ = expand_mask(masks, bboxes, (img.shape[1], img.shape[0]), scores=scores)
    viz_im = plot_mask(img, full_masks)
    return plot_bbox(viz_im, bboxes, scores, labels, thresh, class_names, colors, ax, reverse_rgb, absolute_coordinates)


def fill(mask, bbox, size):
    """Fill mask to full image size
    Parameters
    ----------
    mask : numpy.ndarray with dtype=uint8
        Binary mask prediction of a box
    bbox : iterable of float
        They are :math:`(xmin, ymin, xmax, ymax)`.
    size : tuple
        Tuple of length 2: (width, height).
    Returns
    -------
    numpy.ndarray
        Full size binary mask of shape (height, width)
    """
    width, height = size
    # pad mask
    M = mask.shape[0]
    padded_mask = np.zeros((M + 2, M + 2))
    padded_mask[1:-1, 1:-1] = mask
    mask = padded_mask
    # expand boxes
    x1, y1, x2, y2 = bbox
    x, y, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    hw = hw * ((M + 2) * 1.0 / M)
    hh = hh * ((M + 2) * 1.0 / M)
    x1, y1, x2, y2 = x - hw, y - hh, x + hw, y + hh
    # quantize
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    w, h = (x2 - x1 + 1), (y2 - y1 + 1)
    # resize mask
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    # binarize and fill
    mask = (mask > 0.5).astype('uint8')
    ret = np.zeros((height, width), dtype='uint8')
    xx1, yy1 = max(0, x1), max(0, y1)
    xx2, yy2 = min(width, x2 + 1), min(height, y2 + 1)
    ret[yy1:yy2, xx1:xx2] = mask[yy1 - y1:yy2 - y1, xx1 - x1:xx2 - x1]
    return ret


def expand_mask(masks, bboxes, im_shape, scores=None, thresh=0.5, scale=1.0, sortby=None):
    """Expand instance segmentation mask to full image size.
    Parameters
    ----------
    masks : numpy.ndarray
        Binary images with shape `N, M, M`
    bboxes : numpy.ndarray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes
    im_shape : tuple
        Tuple of length 2: (width, height)
    scores : numpy.ndarray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    sortby : str, optional, default None
        If not None, sort the color palette for masks by the given attributes of each bounding box.
        Valid inputs are 'area', 'xmin', 'ymin', 'xmax', 'ymax'.
    scale : float
        The scale of output image, which may affect the positions of boxes
    Returns
    -------
    numpy.ndarray
        Binary images with shape `N, height, width`
    numpy.ndarray
        Index array of sorted masks
    """
    if len(masks) != len(bboxes):
        raise ValueError('The length of bboxes and masks mismatch, {} vs {}'
                         .format(len(bboxes), len(masks)))
    if scores is not None and len(masks) != len(scores):
        raise ValueError('The length of scores and masks mismatch, {} vs {}'
                         .format(len(scores), len(masks)))

    if sortby is not None:
        if sortby == 'area':
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            sorted_inds = np.argsort(-areas)
        elif sortby == 'xmin':
            sorted_inds = np.argsort(-bboxes[:, 0])
        elif sortby == 'ymin':
            sorted_inds = np.argsort(-bboxes[:, 1])
        elif sortby == 'xmax':
            sorted_inds = np.argsort(-bboxes[:, 2])
        elif sortby == 'ymax':
            sorted_inds = np.argsort(-bboxes[:, 3])
        else:
            raise ValueError('argument sortby cannot take value {}'
                             .format(sortby))
    else:
        sorted_inds = np.argsort(range(len(masks)))

    full_masks = []
    bboxes *= scale
    for i in sorted_inds:
        if scores is not None and scores[i] < thresh:
            continue
        mask = masks[i]
        bbox = bboxes[i]
        full_masks.append(fill(mask, bbox, im_shape))
    full_masks = np.array(full_masks)
    return full_masks, sorted_inds


def get_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


def plot_image_color_mask(image, mask, alpha=0.4):
    """

    Args:
        image: 3-D array of shape [H, W, 3] following RGB mode (0-255)
        mask: 3-D array of shape [H, W, 3] with color (0-255), background color needs to be (0, 0, 0).
        alpha: transparency of mask

    Returns:

    """
    image = image.astype(np.float32)
    fg_mask = (np.sum(mask, axis=2) > 0).astype(np.float32)
    im_factor = (fg_mask * (1 - alpha) + (1 - fg_mask))[:, :, None]
    mask_factor = (fg_mask * alpha)[:, :, None]

    render_image = image * im_factor + mask * mask_factor
    return render_image.astype(np.uint8)
