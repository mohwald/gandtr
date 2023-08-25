#
# Reimplemented architecture of HED [1] inspired by
#
# Sniklaus, Simon. "A Reimplementation of {HED} Using {PyTorch}"
# 2018. Available at: https://github.com/sniklaus/pytorch-hed
#
# [1]: Xie, Saining, and Zhuowen Tu. "Holistically-nested edge detection."
# Proceedings of the IEEE international conference on computer vision. 2015.
#

from collections import defaultdict
import torch
import torch.nn.functional as F

from mdir.tools.utils import load_pretrained
from ....external.daan.core.path_resolver import resolve_path


class HedInterpolation(torch.nn.Module):
    """
    HED pytorch implementation from https://github.com/sniklaus/pytorch-hed that uses bilinear
    interpolation to upsample score maps
    """

    meta = {"in_channels": 3, "out_channels": 1}

    def __init__(self, pretrained=None):
        super().__init__()

        self.vgg1 = torch.nn.Sequential(*self._create_vgg_block(3, [64, 64], first=True))
        self.vgg2 = torch.nn.Sequential(*self._create_vgg_block(64, [128, 128]))
        self.vgg3 = torch.nn.Sequential(*self._create_vgg_block(128, [256, 256, 256]))
        self.vgg4 = torch.nn.Sequential(*self._create_vgg_block(256, [512, 512, 512]))
        self.vgg5 = torch.nn.Sequential(*self._create_vgg_block(512, [512, 512, 512]))

        self.score1 = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.score2 = torch.nn.Conv2d(128, 1, kernel_size=1)
        self.score3 = torch.nn.Conv2d(256, 1, kernel_size=1)
        self.score4 = torch.nn.Conv2d(512, 1, kernel_size=1)
        self.score5 = torch.nn.Conv2d(512, 1, kernel_size=1)

        self.fusion = torch.nn.Sequential(
            torch.nn.Conv2d(5, 1, kernel_size=1)
        )

        if pretrained:
            self.load_state_dict(load_pretrained(resolve_path(pretrained)))

    @staticmethod
    def _create_vgg_block(in_channels, out_channels, first=False):
        acc = []
        if not first:
            acc += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]

        for out_chan in out_channels:
            acc += [torch.nn.Conv2d(in_channels, out_chan, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True)]
            in_channels = out_chan
        return acc

    def forward(self, x, no_sigmoid=False):
        vgg1 = self.vgg1(x)
        vgg2 = self.vgg2(vgg1)
        vgg3 = self.vgg3(vgg2)
        vgg4 = self.vgg4(vgg3)
        vgg5 = self.vgg5(vgg4)

        score1 = self.score1(vgg1)
        score2 = self.score2(vgg2)
        score3 = self.score3(vgg3)
        score4 = self.score4(vgg4)
        score5 = self.score5(vgg5)

        interp_kwargs = {"size": (x.size(2), x.size(3)), "mode": "bilinear", "align_corners": False}

        score1 = F.interpolate(score1, **interp_kwargs)
        score2 = F.interpolate(score2, **interp_kwargs)
        score3 = F.interpolate(score3, **interp_kwargs)
        score4 = F.interpolate(score4, **interp_kwargs)
        score5 = F.interpolate(score5, **interp_kwargs)

        if no_sigmoid:
            return self.fusion(torch.cat([score1, score2, score3, score4, score5], 1))
        return torch.sigmoid(self.fusion(torch.cat([score1, score2, score3, score4, score5], 1)))

    @staticmethod
    def _get_parameter_group(name):
        name = tuple(name.split("."))
        if name[0] in {"vgg1", "vgg2", "vgg3", "vgg4"}:
            return "conv.%s" % name[2]
        elif name[0] == "vgg5":
            return "conv5.%s" % name[2]
        elif name[0] in {"score1", "score2", "score3", "score4", "score5"}:
            return "score.%s" % name[1]
        elif name[0] == "fusion":
            return "fusion.%s" % name[2]
        raise KeyError("Parameter name not recognized '%s'" % ".".join(name))

    def parameter_groups(self, optimizer_opts):
        lr_mult = {"conv.weight": 1, "conv.bias": 2, "conv5.weight": 100, "conv5.bias": 200,
                   "score.weight": 0.01, "score.bias": 0.02, "fusion.weight": 0.001, "fusion.bias": 0.002}
        decay_mult = {"conv.weight": 1, "conv.bias": 0, "conv5.weight": 1, "conv5.bias": 0,
                      "score.weight": 1, "score.bias": 0, "fusion.weight": 1, "fusion.bias": 0}

        groups = defaultdict(list)
        for name, param in self.named_parameters():
            groups[self._get_parameter_group(name)].append(param)

        acc = []
        for key, params in groups.items():
            acc.append({"params": params, "lr": lr_mult[key] * optimizer_opts["lr"],
                        "weight_decay": decay_mult[key] * optimizer_opts["weight_decay"]})
        return acc
