import torch.nn as nn

from . import cirnet, hed, p2p_networks, single_layer, unet, rcf


class Identity(nn.Module):
    """Returns its input unaltered"""

    def __init__(self):
        super().__init__()
        self.meta = {
            "out_channels": 3,
            "in_channels": 3,
        }

    def forward(self, x):
        return x


MODEL_LABELS = {
    "identity": Identity,
    "orig_unet": unet.OrigUNet,
    "p2p_unet": unet.P2pUNet,
    "outconv_unet": unet.OutconvP2pUNet,
    "outconv_dynint_unet": unet.OutconvP2pUNetDynamicInterpolate,

    "shallow_p2p_unet": unet.ShallowP2pUNet,
    "inconv_p2p_unet": unet.InconvP2pUNet,
    "aligned_p2p_unet": unet.AlignedP2pUNet,

    "official_p2p_unet_generator": p2p_networks.UnetGenerator,
    "official_p2p_discriminator": p2p_networks.NLayerDiscriminator,
    "official_resnet_generator": p2p_networks.ResnetGenerator,
    "official_p2p_mlp": p2p_networks.PatchSampleF,

    "cirnet": cirnet.init_cirnet,
    "cirnet_inchan": cirnet.init_cirnet_inchan,
    "cirnet_attention": cirnet.init_cirnet_attention,

    "hed_interpolation": hed.HedInterpolation,

    "normalization_l2": single_layer.NormalizationL2,

    "rcf": rcf.RCF,
}

def initialize_model(params):
    return MODEL_LABELS[params.pop("architecture")](**params)
