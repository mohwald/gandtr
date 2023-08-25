import os
import warnings
import gzip
import numpy as np
import h5py
from PIL import Image
from PIL import ImageFile

import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", "Possibly corrupt EXIF data.", UserWarning)

def cid2filename(cid, prefix):
    """
    Creates a training image path out of its CID name

    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved

    Returns
    -------
    filename : full image filename
    """
    if cid[0] == "/":
        return cid

    cid = os.path.join(cid[-2:], cid[-4:-2], cid[-6:-4], cid)
    if "*" in prefix:
        return prefix.replace("*", cid)
    return os.path.join(prefix, cid)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.info['_metadata'] = {"path": path, "name": os.path.basename(path).rsplit(".", 1)[0]}
            return img.convert('RGB')
    except OSError as e:
        return e

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    if isinstance(path, np.ndarray):
        return path
    elif isinstance(path, h5py.Dataset):
        return path[:]
    elif path.endswith(".npy"):
        with open(path, 'rb') as handle:
            return np.load(handle)
    elif path.endswith(".npy.gz"):
        with gzip.open(path, 'rb') as handle:
            return np.load(handle)

    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def imresize(img, imsize):
    if isinstance(img, np.ndarray):
        # if imsize != max(img.shape):
        #     raise ValueError("Numpy array has incompatible size '%s'" % str(img.shape))
        return img

    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]
