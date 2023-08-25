import warnings
import torch
from cirtorch.layers import loss as cirloss
from mdir.tools import loss_value


class ContrastiveLoss(cirloss.ContrastiveLoss):

    reduction = "sum"
    eps = 1e-6

    def __init__(self, margin, eps=None):
        if eps is not None:
            warnings.warn("Parameter 'eps' in ContrastiveLoss is deprecated and will be removed, remove from configuration", DeprecationWarning)
        super().__init__(margin=margin, eps=self.eps)

    def forward(self, x, label):
        if isinstance(label, (list, tuple)):
            label = torch.cat(label)
        return super().forward(x, label.to(x.device))

class ContrastiveLossMultipleDescriptors(ContrastiveLoss):
    """ContrastiveLoss when the network returns a tuple of descriptors instead of a single one"""

    def __init__(self, margin, weights):
        super().__init__(margin=margin)
        if isinstance(weights, str):
            weights = [float(x) for x in weights.split(",")]
        self.weights = weights

    def forward(self, x, label):
        # Fallback to non-multiple-descriptors
        if not isinstance(x, list):
            return super().forward(x, label)

        # Accumulate loss across descriptors
        weights = [1.0 / len(x)] * len(x) if self.weights is None else self.weights
        assert len(x) == len(weights), (len(x), len(weights))
        partial = {}
        total = loss_value.ZERO
        for i, xi in enumerate(x):
            loss = super().forward(xi, label)
            partial[str(i)] = loss
            total = total + weights[i] * loss
        return loss_value.TotalWithIntermediate(total, **partial)


class TripletLoss(cirloss.TripletLoss):

    reduction = "sum"

    def __init__(self, margin):
        super().__init__(margin=margin)

    def forward(self, x, label):
        if isinstance(label, (list, tuple)):
            label = torch.cat(label)
        return super().forward(x, label.to(x.device))
