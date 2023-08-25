import torch
import torch.nn as nn
import torch.nn.functional as F


class HordeCascadedKOrder(nn.Module):
    """
    Horde regularizing pooling implementation:

    - https://github.com/pierre-jacob/ICCV2019-Horde/blob/master/kerastools/models/horde_models.py
    - https://github.com/pierre-jacob/ICCV2019-Horde/blob/master/kerastools/layers/horde_layers.py
    """

    def __init__(self, dim, order, high_order_dims):
        # order: 5
        # high_order_dims: 10 000
        super().__init__()
        self.projections = []

        assert order > 0
        if order == 1:
            self.embeddings = []
            return

        for _ in range(order):
            conv = nn.Conv2d(in_channels=dim, out_channels=high_order_dims, kernel_size=1, bias=False)
            self.projections.append(conv)
        self.projections = nn.ModuleList(self.projections)

        self.pooling = nn.functional.avg_pool2d
        self.embeddings = nn.ModuleList([nn.Linear(high_order_dims, dim, bias=False) for _ in range(order-1)])

    def forward(self, x):
        if not self.projections:
            return []

        projected = [self.projections[0](x) * self.projections[1](x)]
        for projection in self.projections[2:]:
            projected.append(projected[-1] * projection(x))

        return [self.embeddings[i](self.pooling(x, x.shape[-2:]).squeeze(-1).squeeze(-1)) for i, x in enumerate(projected)]


class GeometricMedianWeiszfeld(nn.Module):
    """
    Implementation of Weiszfeld's iterative algorithm to approximate geometric median

    w_i = 1 / || y - x_i ||
    y = sum w_i x_i / sum w_i
    """

    def __init__(self, iterations, intermediate_gradients):
        super().__init__()
        self.iterations = iterations
        self.intermediate_gradients = intermediate_gradients

    def forward(self, x):
        kernel_size = (x.size(-2), x.size(-1))
        weights = torch.ones((1,) + kernel_size, device=x.device)

        effective_x = x if self.intermediate_gradients else x.detach()
        for iter in range(self.iterations):
            median = F.avg_pool2d(effective_x * weights, kernel_size, divisor_override=1) / weights.sum()
            weights = 1.0 / ((effective_x - median).pow(2.0).sum(1) + 1e-10).sqrt()

        median = F.avg_pool2d(x * weights, kernel_size, divisor_override=1) / weights.sum()
        return median


class WeightedGeometricMedianWeiszfeld(nn.Module):
    """
    Implementation of Weiszfeld's iterative algorithm to approximate weighted geometric median

    w_i = a_i / || y - x_i ||
    y = sum w_i x_i / sum w_i
    """

    def __init__(self, iterations, intermediate_gradients, attention):
        super().__init__()
        self.iterations = iterations
        self.intermediate_gradients = intermediate_gradients
        self.attention = attention

    def forward(self, x):
        kernel_size = (x.size(-2), x.size(-1))
        att = self.attention(x)
        weights = att

        effective_x = x if self.intermediate_gradients else x.detach()
        for iter in range(self.iterations):
            median = F.avg_pool2d(effective_x * weights, kernel_size, divisor_override=1) / weights.sum()
            weights = att / ((effective_x - median).pow(2.0).sum(1) + 1e-10).sqrt()

        median = F.avg_pool2d(x * weights, kernel_size, divisor_override=1) / weights.sum()
        return median


POOLINGS = {
    "HordeCascadedKOrder": HordeCascadedKOrder,
    "GeometricMedianWeiszfeld": GeometricMedianWeiszfeld,
}
