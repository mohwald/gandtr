import random
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

cv2.setNumThreads(0)

from . import functional as tfunc
from .core_transforms import GenericTransform


#
# Crop & scale & flip
#

class RandomCrop(transforms.RandomCrop):

    def __call__(self, *pics): # pylint: disable=arguments-differ
        i, j, h, w = self.get_params(pics[0], self.size)
        return [x[i:i+h,j:j+w] for x in pics]


class RandomHorizontalFlip(GenericTransform):

    def __init__(self, p=0.5):
        super().__init__({"p": float(p)})

    def __call__(self, *pics):
        if random.random() < self.params["p"]:
            return [np.flip(x, axis=1).copy() for x in pics]
        return pics


class CenterCrop(GenericTransform):
    """Crop the center part of an image"""

    def __init__(self, size):
        super().__init__({"size": np.array(tfunc.parse_tuple(size, int))[::-1]})

    def __call__(self, *pics):
        if not isinstance(pics[0], np.ndarray):
            raise RuntimeError("{} is not np.ndarray".format(type(pics[0])))
        acc = []
        for pic in pics:
            pad = (pic.shape[:2] - self.params["size"]) / 2
            y0, y1, x0, x1 = int(np.floor(pad[0])), -int(np.ceil(pad[0])) or None, \
                             int(np.floor(pad[1])), -int(np.ceil(pad[1])) or None
            acc.append(pic[y0:y1,x0:x1])
        return acc


class SquareCrop(GenericTransform):
    """Crop the image to its center square"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _square_crop_boundaries(size):
        pad = (np.array(size) - min(size)) / 2
        return int(np.floor(pad[0])), size[0] - int(np.ceil(pad[0])), \
                int(np.floor(pad[1])), size[1] - int(np.ceil(pad[1]))

    def __call__(self, *pics):
        acc = []
        if isinstance(pics[0], Image.Image):
            for pic in pics:
                acc.append(pic.crop(self._square_crop_boundaries(pic.size)))
            return acc
        elif isinstance(pics[0], np.ndarray):
            for pic in pics:
                y0, y1, x0, x1 = self._square_crop_boundaries(pic.shape[:2])
                acc.append(pic[y0:y1,x0:x1])
            return acc
        raise RuntimeError("{} is not either PIL image, np.ndarray".format(type(pics[0])))


class Downscale(GenericTransform):
    """Downscale the image so that it fits in provided size while keeping its aspect ratio"""

    def __init__(self, size):
        super().__init__({"size": int(size)})

    def __call__(self, *pics):
        if not isinstance(pics[0], (Image.Image, np.ndarray)):
            raise RuntimeError("{} is not either PIL image or np.ndarray".format(type(pics[0])))
        acc = []
        for pic in pics:
            if isinstance(pic, Image.Image) and max(pic.size) > self.params["size"]:
                pic.thumbnail((self.params["size"], self.params["size"]), Image.LANCZOS)
            elif isinstance(pic, np.ndarray) and max(pic.shape) > self.params["size"]:
                img = Image.fromarray((pic*255).astype(np.uint8))
                img.thumbnail((self.params["size"], self.params["size"]), Image.LANCZOS)
                pic = np.array(img, dtype=np.float32) / 255.0
            acc.append(pic)
        return acc


class RandomScaleCrop(GenericTransform):
    """Randomly scale image (defined by lower and upper bound of scale) and random crop to required
        size. In implementation, these two steps are swapped for performance reasons. If the input
        images have the exact size as is the target, the scale-crop operation is skipped and images
        are returned unchanged"""

    def __init__(self, size, scale=(0.5, 0.8)):
        super().__init__({"size": np.array(tfunc.parse_tuple(size, int)),
                          "scale": tfunc.parse_tuple(scale, float)})

    def __call__(self, *pics):
        # Precompute random crop
        pic_min_size = self._pic_min_size(pics)
        if pic_min_size is None:
            return pics

        assert (self.params['size'] <= pic_min_size).all(), "{} <= {}".format(self.params['size'], pic_min_size)
        lowest_scale = max(float(np.max(self.params['size'] / pic_min_size)), self.params["scale"][0])
        scale = random.random() * (self.params["scale"][1] - lowest_scale) + lowest_scale
        cropped_size = np.ceil(self.params["size"][::-1] / scale).astype(int)
        assert (pic_min_size >= cropped_size).all()
        cropped_offset = [random.randint(0, x) for x in (pic_min_size - cropped_size)]

        return self._crop_downscale(pics, cropped_offset, cropped_size)

    def _pic_min_size(self, pics):
        if len(pics) == 1 or pics[0].shape[:2] == pics[1].shape[:2]:
            if (pics[0].shape[:2] == self.params["size"][::-1]).all():
                return None

        # Compute min image size inside batch
        pic_min_size = np.full(2, np.iinfo(np.int_).max)
        for pic in pics:
            pic_min_size = np.minimum(pic.shape[:2], pic_min_size)
        return pic_min_size

    def _crop_downscale(self, pics, cropped_offset, cropped_size):
        ystart, yend, xstart, xend = (cropped_offset[0], cropped_offset[0]+cropped_size[0],
                                      cropped_offset[1], cropped_offset[1]+cropped_size[1])
        acc = []
        for pic in pics:
            pic = cv2.resize(pic[ystart:yend,xstart:xend], tuple(self.params["size"]))
            assert pic.shape[0] == self.params["size"][1] and pic.shape[1] == self.params["size"][0], pic.shape
            acc.append(pic)
        return acc


class CenterScaleCrop(RandomScaleCrop):
    """Deterministic scale and crop the center of an image in the random scalecrop fashion."""

    def __init__(self, size, scale=0.6):
        super().__init__(size, (scale, scale))

    def __call__(self, *pics):
        # Precompute center crop
        pic_min_size = self._pic_min_size(pics)
        if pic_min_size is None:
            return pics

        scale = float(self.params["scale"][0])
        cropped_size = np.ceil(self.params["size"][::-1] / scale).astype(int)
        assert (pic_min_size >= cropped_size).all()
        cropped_offset = [x // 2 for x in (pic_min_size - cropped_size)]

        return self._crop_downscale(pics, cropped_offset, cropped_size)


#
# Noise
#

class AdditiveGaussianNoise(GenericTransform):
    """Adds Gaussian noise with zero mean and specified sigma to the input image only. Clips values
        to be between [0, 1]. Add noise only to the first image."""

    def __init__(self, sigma):
        super().__init__({"sigma": float(sigma)})

    def __call__(self, *pics):
        pics = list(pics)
        pics[0] = np.clip(pics[0] + np.random.normal(loc=0, scale=self.params["sigma"], size=pics[0].shape), 0, 1).astype(np.float32)
        return pics

