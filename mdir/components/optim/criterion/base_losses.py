import torch
from torch import nn, Tensor


class L1Loss(nn.L1Loss):

    def __init__(self, **kwargs):
        super().__init__(**{"reduction": "mean", **kwargs})


class MSELoss(nn.MSELoss):

    def __init__(self, **kwargs):
        super().__init__(**{"reduction": "mean", **kwargs})


class BCELoss(nn.BCELoss):

    def __init__(self, **kwargs):
        super().__init__(**{"reduction": "mean", **kwargs})

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target.detach())


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def __init__(self, **kwargs):
        if "pos_weight" in kwargs and isinstance(kwargs["pos_weight"], float):
            kwargs["pos_weight"] = torch.Tensor([kwargs["pos_weight"]])
        super().__init__(**{"reduction": "mean", **kwargs})
