import os.path
import warnings
import numpy as np
import cv2
import torch

from .functional_consts import HISTOGRAMS


def parse_tuple(tpl, dtype=int, fixed_size=0):
    if isinstance(tpl, str):
        tpl = tuple(dtype(x) for x in tpl.split("_"))
    if fixed_size:
        if len(tpl) == 1:
            tpl *= fixed_size
        elif len(tpl) != fixed_size:
            raise ValueError("Invalid tuple of size %s, required %s" % (len(tpl), fixed_size))
    return tpl


#
# Generic
#

def torchcat_npchan(img, add_chan):
    return torch.cat((img, torch.tensor(add_chan, dtype=torch.float32).unsqueeze(dim=0)), dim=0)

def rgb2normspace(img, colorspace):
    colorspace = colorspace.lower()
    if len(colorspace) == 4 and colorspace[0] == "s":
        img = np.power(img, 2.2)
        colorspace = colorspace[1:]

    if colorspace == "lab":
        return (cv2.cvtColor(img, cv2.COLOR_RGB2LAB) + np.array([0, 128, 128], dtype=np.float32)) / np.array([100.0, 255.0, 255.0], dtype=np.float32)
    elif colorspace == "luv":
        return (cv2.cvtColor(img, cv2.COLOR_RGB2LUV) + np.array([0, 134, 140], dtype=np.float32)) / np.array([100.0, 354.0, 262.0], dtype=np.float32)
    elif colorspace == "lsh":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) / np.array([360.0, 1.0, 1.0], dtype=np.float32)
        return np.stack((img[:,:,1], img[:,:,2], img[:,:,0]), axis=2)
    elif colorspace == "hsv":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV) / np.array([360.0, 1.0, 1.0], dtype=np.float32)
    elif colorspace == "yxz":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
        return np.stack((img[:,:,1], img[:,:,0], img[:,:,2]), axis=2)
    elif colorspace == "gray":
        return np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=2).astype(np.float32)
    elif colorspace == "bgr":
        return img[:, :, [2, 1, 0]]
    elif colorspace == "rgb":
        return img

    raise NotImplementedError("Colorspace %s is not supported" % colorspace)

def normspace2rgb(img, colorspace):
    colorspace = colorspace.lower()
    standard = False
    if len(colorspace) == 4 and colorspace[0] == "s":
        standard = True
        colorspace = colorspace[1:]

    if colorspace == "lab":
        img = cv2.cvtColor((img * np.array([100.0, 255.0, 255.0], dtype=np.float32)) - np.array([0, 128, 128], dtype=np.float32), cv2.COLOR_LAB2RGB)
    elif colorspace == "luv":
        img = cv2.cvtColor((img * np.array([100.0, 354.0, 262.0], dtype=np.float32)) - np.array([0, 134, 140], dtype=np.float32), cv2.COLOR_LUV2RGB)
    elif colorspace == "lsh":
        img = np.stack((img[:,:,2], img[:,:,0], img[:,:,1]), axis=2) * np.array([360.0, 1.0, 1.0], dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    elif colorspace == "hsv":
        img = cv2.cvtColor(img * np.array([360.0, 1.0, 1.0], dtype=np.float32), cv2.COLOR_HSV2RGB)
    elif colorspace == "yxz":
        img = np.stack((img[:,:,1], img[:,:,0], img[:,:,2]), axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_XYZ2RGB)
    elif colorspace != "rgb":
        raise NotImplementedError("Colorspace %s is not supported" % colorspace)

    if standard:
        return np.power(img, 1/2.2)
    return img

def apply_lightness_transform(img, colorspace, func):
    """Apply given function on a lightness channel in given colorspace"""
    spc = rgb2normspace(img, colorspace)
    spc[:,:,0] = func(spc[:,:,0])
    return normspace2rgb(spc, colorspace)


#
# Histogram
#

HISTOGRAM_BINS = np.linspace(-0.00196078431372549, 1.0019607843137255, 257)
HISTOGRAM_CENTERS = np.linspace(0, 1, 256)
HISTOGRAM_CDF = {x: np.cumsum(y) for x, y in HISTOGRAMS.items()}

def channel_histogram_matching(chan0, histogram):
    cdf0 = np.cumsum(np.histogram(chan0, HISTOGRAM_BINS)[0]) / chan0.size
    centers = HISTOGRAM_CENTERS
    if histogram == "eq":
        return np.interp(chan0, centers, cdf0*centers[-1]).astype(np.float32)
    return np.interp(chan0, centers, np.interp(cdf0, HISTOGRAM_CDF[histogram], centers)).astype(np.float32)

def image_histogram_matching(img, histogram, colorspace):
    return apply_lightness_transform(img, colorspace, lambda x: channel_histogram_matching(x, histogram))

def channel2channel_histogram_matching(chan0, chan1):
    cdf0 = np.cumsum(np.histogram(chan0, HISTOGRAM_BINS)[0]) / chan0.size
    cdf1 = np.cumsum(np.histogram(chan1, HISTOGRAM_BINS)[0]) / chan1.size
    return np.interp(chan0, HISTOGRAM_CENTERS, np.interp(cdf0, cdf1, HISTOGRAM_CENTERS)).astype(np.float32)


#
# Gamma
#

def channel_gamma_matching(channel, target):
    # Optimization definition
    func = lambda gamma: np.mean(np.power(channel, gamma)) - target
    # fprime = lambda gamma: [np.mean(np.power(channel, gamma) * np.ma.log(channel).filled(0))]
    x0 = np.log(target) / np.log(np.mean(channel))
    # Root finding
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            solution = scipy.optimize.newton(func, x0=x0, tol=1e-4, maxiter=50)
        except RuntimeError:
            solution = 0.1 if abs(func(0.1)) < abs(func(10)) else 10
    # criterion = abs(func(solution))
    solution = np.clip(solution, 0.1, 10)
    return np.power(channel, solution)

def image_gamma_matching(img, target, colorspace):
    return apply_lightness_transform(img, colorspace, lambda x: channel_gamma_matching(x, target))


#
# CLAHE
#

class ChannelClahe:

    def __init__(self, clip_limit, grid_size):
        if not isinstance(grid_size, tuple):
            grid_size = (int(grid_size), int(grid_size))
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def apply_clahe(self, chan):
        return self.clahe.apply((chan*255).astype(np.uint8)).astype(np.float32) / 255.0

    def apply(self, chan):
        return self.apply_clahe(chan)


class ImageClahe(ChannelClahe):

    def __init__(self, clip_limit, grid_size, colorspace):
        super().__init__(clip_limit, grid_size)
        self.colorspace = colorspace

    def apply(self, img):
        return apply_lightness_transform(img, self.colorspace, self.apply_clahe)


class ImageColorspaceClahe(ImageClahe):

    def apply(self, img):
        spc = rgb2normspace(img, self.colorspace)
        spc[:, :, 0] = self.apply_clahe(spc[:, :, 0])
        return spc
