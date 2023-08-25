import math
import torch
import torch.nn as nn


def init_weights_normal(m):
    """Initialize weights from normal distribution with mean=0 and std=1"""
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.normal_(m.bias.data)

def init_weights_uniform(m):
    """Initialize weights from a uniform distribution with bounds (0, 1)"""
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.uniform_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.uniform_(m.bias.data)

def _calculate_fan_in(tensor):
    """Calculate the number of input units of given tensor"""
    dimension = tensor.ndimension()
    if dimension < 2:
        raise ValueError("Fan in can not be computed for tensor with less than 2 dimensions")

    fan_in = tensor.size(1)
    if dimension > 2 and tensor.dim() > 2: # not Linear
        fan_in *= tensor[0][0].numel() # receptive field size
    return fan_in

def he_normal_(tensor):
    """HE normal initializer"""
    fan_in = _calculate_fan_in(tensor)
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)

def init_weights_he_normal(m):
    """Initialize weights using HE normal and biases using const, used in orig unet"""
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        he_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01) # Better than 0 if relu follows


#
# Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
# All rights reserved.
#

def init_weights_p2p(init_type, init_gain):
    """Initialize network weights for official pix2pix, cycleGAN, and CUT.
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | kaiming
        init_gain (float)    -- scaling factor for normal, and xavier.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    return init_func


def initialize_weights(weights, params):
    if "p2p" in weights:
        # Assuming weights as one of: normal_p2p | kaiming_p2p
        if params is None or "init_gain" not in params:
            params = {"init_gain": 0.2}
        return init_weights_p2p(weights.split("_")[0], **params)

    assert not params
    return WEIGHT_INITIALIZATIONS[weights]


WEIGHT_INITIALIZATIONS = {
    "uniform": init_weights_uniform,
    "normal": init_weights_normal,
    "he_normal": init_weights_he_normal,
}
