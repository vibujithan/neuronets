from math import ceil

import numpy as np
from scipy.ndimage import zoom


class Minmax:
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        try:
            image = image.astype('f8')
            maxv = np.max(image)
            minv = np.min(image)
            return ((image - minv) / maxv).astype('f4')
        except:
            return image


class CenterCrop3D(object):
    """Crop to size informed."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        original_size = image.shape
        if len(original_size) != 3:
            raise Exception(f'Crop3D only works with 3 dimensions. Input has {len(original_size)} dimensions')
        x_init = ceil((original_size[0] - self.size[0]) / 2)
        y_init = ceil((original_size[1] - self.size[1]) / 2)
        z_init = ceil((original_size[2] - self.size[2]) / 2)
        image = image[x_init:x_init + self.size[0], y_init:y_init + self.size[1], z_init:z_init + self.size[2]]

        return image


class Zoom(object):
    """resize to size informed."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return zoom(image, self.size)


class ToFloatUKBB:
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        try:
            image = image.astype('f8')
            maxv = np.max(image)
            minv = np.min(image)
            return ((image - minv) / maxv).astype('f4')
        except:
            return image


class MeanSub:
    def __init__(self, mean_img):
        self.mean_img = mean_img

    def __call__(self, img):
        try:
            img = img - self.mean_img
        except:
            print('Error Occured in subtracting mean image')

        return img
