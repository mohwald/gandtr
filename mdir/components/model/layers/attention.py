import torch.nn as nn


class L2NormAttention(nn.Module):

    def __init__(self, normalize_max):
        super().__init__()
        self.normalize_max = normalize_max

    def forward(self, x):
        m = (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0).squeeze(0)

        if self.normalize_max:
            m = m / m.max()
        return m


ATTENTIONS = {
    "l2norm": L2NormAttention,
}
