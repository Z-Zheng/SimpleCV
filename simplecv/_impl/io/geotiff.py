import gdal
import numpy as np
from PIL import Image


def open(filepath):
    data = gdal.Open(filepath)
    image = TiffImage()
    image._set_data(data)
    image._set_shape(height=data.RasterYSize, width=data.RasterXSize)
    return image


class TiffImage(object):
    """ A simple wrapper of gdal dataset for conveniently accessing image data

    """

    def __init__(self):
        self.data = None
        self.height = None
        self.width = None

    def __getitem__(self, item):
        if len(item) > 2:
            hw, c = item[:2], item[-1]
        else:
            hw = item
            c = slice(None, None, None)
        params = self._parse_slice(hw)
        nparray = self.data.ReadAsArray(**params)
        if nparray.ndim == 3:
            nparray = nparray.transpose([1, 2, 0])
            nparray = nparray[:, :, c]
        return nparray

    def __setitem__(self, key, value):
        pass

    @property
    def shape(self):
        c = self.data.RasterCount
        h = self.data.RasterYSize
        w = self.data.RasterXSize
        return h, w, c

    def size(self, index):
        return self.shape[index]

    @staticmethod
    def open(filepath):
        data = gdal.Open(filepath)
        image = TiffImage()
        image._set_data(data)
        image._set_shape(height=data.RasterYSize, width=data.RasterXSize)
        return image

    def crop(self, box, to_PIL_Image=False):
        """
        Args:
            box: The crop rectangle, as a (left, upper, right, lower)-tuple.

        Returns:
        """
        xmin, ymin, xmax, ymax = box
        region_image = self[int(ymin):int(ymax), int(xmin):int(xmax)]
        if to_PIL_Image:
            return Image.fromarray(region_image)
        return region_image

    def _set_data(self, data):
        self.data = data

    def _set_shape(self, height, width):
        self.height = height
        self.width = width

    def _parse_slice(self, item):
        y_slice, x_slice = item
        if isinstance(x_slice, int):
            x_off = x_slice
            x_size = 1
        elif isinstance(x_slice, slice):
            x_start = 0 if x_slice.start is None else x_slice.start
            x_end = self.width if x_slice.stop is None else x_slice.stop
            assert x_end <= self.width, 'out of bound, {} - {}'.format(x_end, self.width)
            x_off = x_start
            x_size = x_end - x_start
        else:
            raise ValueError()
        if isinstance(y_slice, int):
            y_off = y_slice
            y_size = 1
        elif isinstance(y_slice, slice):
            y_start = 0 if y_slice.start is None else y_slice.start
            y_end = self.height if y_slice.stop is None else y_slice.stop
            assert y_end <= self.height, 'out of bound, {} - {}'.format(y_end, self.height)
            y_off = y_start
            y_size = y_end - y_start
        else:
            raise ValueError()
        return {
            'xoff': x_off,
            'yoff': y_off,
            'xsize': x_size,
            'ysize': y_size
        }


if __name__ == '__main__':
    # im = imread(r'D:\2018 Open AI Tanzania Building Footprint Segmentation Challenge\5ae242fd0b093000130afd27.tif')
    import matplotlib.pyplot as plt

    im = TiffImage.open(
        r'C:\Users\zhengzhuo\Desktop\gf2\images\GF_2015_R_G_B_NIR.tif')
    print(im.shape)

    print()
