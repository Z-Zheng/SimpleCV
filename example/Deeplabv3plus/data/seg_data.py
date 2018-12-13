import torch
import torch.utils.data as data
import glob
import os
import logging
from skimage.io import imread
from simplecv.data.preprocess import random_flip_left_right, mean_std_normalize, channel_last_to_first
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SegData(data.Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 image_format='jpg',
                 mask_format='png',
                 training=True,
                 image_mask_mapping=None,
                 filenameList_path=None):
        super(SegData, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.im_format = image_format
        self.mask_format = mask_format
        self.training = training
        self.image_mask_mapping = image_mask_mapping
        self.filenameList_path = filenameList_path

        self.idx_to_im_mask_path = self._build_index()

    def _build_index(self):
        if self.filenameList_path is None:
            im_path_list = glob.glob(os.path.join(self.image_dir, '*.{}'.format(self.im_format)))
        else:
            with open(self.filenameList_path, 'r') as f:
                data = f.read()
                name_list = data.split('\n')[:-1]
            im_path_list = [os.path.join(self.image_dir, filename + '.jpg') for filename in name_list]

        if len(im_path_list) == 0:
            raise FileNotFoundError('The image is not found.')
        if self.image_mask_mapping is None:
            mask_path_list = [
                os.path.join(self.mask_dir, os.path.split(im_path)[-1].replace(self.im_format, self.mask_format)) for
                im_path in
                im_path_list]
        else:
            mask_path_list = [os.path.join(self.mask_dir, self.image_mask_mapping(os.path.split(im_path)[-1])) for
                              im_path in
                              im_path_list]
        idx_to_im_mask_path = list(map(lambda e: (e[0], e[1]), zip(im_path_list, mask_path_list)))
        return idx_to_im_mask_path

    def _preprocess(self, image, mask):
        # rgb order
        image = mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
        image = channel_last_to_first(image)
        image = image.astype(np.float32, copy=False)
        mask = mask.astype(np.float32, copy=False)
        return image, mask

    def __getitem__(self, idx):
        im_path, mask_path = self.idx_to_im_mask_path[idx]
        im = imread(im_path)
        mask = imread(mask_path)

        if self.training:
            im, mask = random_flip_left_right(im, mask, prob=0.5)

        im, mask = self._preprocess(im, mask)

        im_ts = torch.from_numpy(im)
        mask_ts = torch.from_numpy(mask)
        return im_ts, mask_ts

    def __len__(self):
        return len(self.idx_to_im_mask_path)


def demo_mapping(name):
    return name.replace('sat', 'mask')
