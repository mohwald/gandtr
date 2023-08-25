import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from mdir.tools import tensors


#
# Abstract & Lib
#

class GenericTransform(object):

    def __init__(self, params=None):
        self.params = params or {}

    def __repr__(self):
        return self.__class__.__name__ + '(%s)' % ", ".join("%s=%s" % (x, str(y)) for x, y in self.params.items())


#
# Core
#

class Compose(transforms.Compose):

    def __call__(self, *pics): # pylint: disable=arguments-differ
        for t in self.transforms:
            pics = t(*pics)
        if len(pics) == 1:
            return pics[0]
        return pics


class ToTensor(transforms.ToTensor):

    def __call__(self, *pics): # pylint: disable=arguments-differ
        acc = []
        for pic in pics:
            tensor = super(ToTensor, self).__call__(pic)
            if pic.dtype.metadata is not None:
                tensor = tensors.as_metadata_tensor(tensor, pic.dtype.metadata)
            acc.append(tensor)
        return acc


class Normalize(GenericTransform):

    def __init__(self, mean, std, strict_shape=True):
        strict_shape = bool(strict_shape) if not isinstance(strict_shape, str) or strict_shape.lower() != "false" else False
        super().__init__({"mean": mean, "std": std, "strict_shape": strict_shape})
        assert len(mean) == len(std)

    def __call__(self, *pics):
        acc = []
        for pic in pics:
            if self.params["strict_shape"]:
                assert pic.size(0) == len(self.params["mean"]), (pic.size(0), len(self.params["mean"]))
            else:
                assert pic.size(0) <= len(self.params["mean"]), (pic.size(0), len(self.params["mean"]))

            mean, std = self.params["mean"][:pic.size(0)], self.params["std"][:pic.size(0)]
            if isinstance(pic, tensors.MetadataTensor):
                tensor = transforms.functional.normalize(pic.tensor, mean, std)
                pic = tensors.as_metadata_tensor(tensor, pic.metadata)
            else:
                pic = transforms.functional.normalize(pic, mean, std)
            acc.append(pic)

        return acc


class Pil2Numpy(GenericTransform):
    """Convert pil image to numpy array with values between 0 and 1"""

    @staticmethod
    def _uint2float(pic, metadata):
        dtype = np.float32
        if metadata is not None:
            dtype = np.dtype(dtype, metadata=metadata)

        if pic.dtype == np.uint8:
            return pic.astype(dtype) / 255.0
        elif pic.dtype == np.uint16:
            return pic.astype(dtype) / 65535.0
        return pic.astype(dtype)

    def __call__(self, *pics):
        acc = []
        for pic in pics:
            metadata = None
            if isinstance(pic, Image.Image):
                if "_metadata" in pic.info:
                    metadata = pic.info["_metadata"]
                pic = np.asarray(pic.convert('RGB'))
            elif not isinstance(pic, np.ndarray):
                raise ValueError("Unsupported type '%s'" % type(pic))

            acc.append(self._uint2float(pic, metadata))
        return acc


class StackBatch(GenericTransform):
    """Convert a list of image tensors to a single tensor by concatenating them along the axis 0"""

    def __call__(self, *pics):
        return [torch.cat(pics, 0)]


class NanCheck(GenericTransform):
    """Check for nan in images, raise an exception when nan detected. Return input unchanged"""

    def __call__(self, *pics):
        for pic in pics:
            if np.isnan(pic).any():
                raise ValueError("Nan value occured in input")
        return pics
