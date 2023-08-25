import copy
import numpy as np
import torch
from torch import nn

from .. import criterion as crit
from mdir.tools import loss_value


class CycleLoss(torch.nn.Module):
    """One loss tailored to cycleGAN losses."""

    def __init__(self, loss_G_X, loss_G_Y, loss_D_X, loss_D_Y):
        super().__init__()
        self.loss_G_X = crit.CRITERIA[loss_G_X.pop("loss")](**loss_G_X)
        self.loss_G_Y = crit.CRITERIA[loss_G_Y.pop("loss")](**loss_G_Y)
        self.loss_D_X = crit.CRITERIA[loss_D_X.pop("loss")](**loss_D_X)
        self.loss_D_Y = crit.CRITERIA[loss_D_Y.pop("loss")](**loss_D_Y)
        self.reduction = "mixed"

    def forward(self, *inputs):
        raise NotImplementedError("Losses are handled manually through SupervisedCycleGanEpoch")


class DiscriminatorLoss(torch.nn.Module):
    """aka Adversarial Loss"""

    def __init__(self, criterion):
        super().__init__()
        self.criterion = crit.CRITERIA[criterion.pop("loss")](**criterion)
        self.reduction = "mixed"

    def forward(self, output, is_target_real, device):
        # Multiscale discriminator
        if isinstance(output, list):
            total = loss_value.ZERO
            partial = {}
            for i, y in enumerate(output):
                key = "layer" + str(len(output) - 1 - i)
                partial[key] = self.criterion(y, self.get_target_tensor(y, is_target_real, device))
                total = total + partial[key]  # Cannot use += as it changes the tensor on the left
            return loss_value.TotalWithIntermediate(total, **partial)
        # Single discriminator
        total = self.criterion(output, self.get_target_tensor(output, is_target_real, device))
        return loss_value.TotalWithIntermediate(total)

    @staticmethod
    def get_target_tensor(ref_tensor, is_target_real: bool, device):
        val = int(not is_target_real)
        return torch.full(ref_tensor.shape, val, dtype=torch.float32, device=device)


class LossSet(torch.nn.Module):
    def __init__(self, **losses):
        super().__init__()
        for k, v in losses.items():
            setattr(self, k, crit.CRITERIA[v.pop("loss")](**v))
        self.reduction = "mixed"
        self.loss_names = set(losses.keys())

    def forward(self, *inputs):
        raise NotImplementedError("Losses are handled manually through epoch iteration")


class MultiheadLoss(torch.nn.Module):
    """Combination loss for multi-headed networks where each loss is for one head of the network"""

    def __init__(self, weights, normalize_weights, **losses):
        super().__init__()
        self.losses = {}
        for loss in losses:
            self.losses[loss] = crit.CRITERIA[losses[loss].pop("loss")](**losses[loss])

        self.weights = weights
        if isinstance(self.weights, (int, float)):
            self.weights = {key: self.weights for key in self.losses}
        if normalize_weights:
            sum_weights = sum(self.weights.values())
            self.weights = {key: val/sum_weights for key, val in self.weights.items()}

        assert losses.keys() == self.weights.keys(), str(losses.keys()) + "!=" + str(self.weights.keys())

        reductions = [x.reduction for x in self.losses.values()]
        self.reduction = reductions[0] if len(set(reductions)) == 1 else "mixed"

    def forward(self, output, target):
        total = loss_value.ZERO
        partial = {}
        for loss in self.losses:
            partial[loss] = self.weights[loss] * self.losses[loss](output[loss], target[loss])
            total = total + partial[loss] # Cannot use += as it changes the tensor on the left
        return loss_value.TotalWithIntermediate(total, **partial)

    def to(self, device):
        for loss in self.losses:
            self.losses[loss] = self.losses[loss].to(device)


class CombinationLoss(MultiheadLoss):
    """Sum of multiple losses on the same data"""

    def forward(self, output, target):
        total = loss_value.ZERO
        partial = {}
        for loss in self.losses:
            partial[loss] = self.weights[loss] * self.losses[loss](output, target)
            total = total + partial[loss] # Cannot use += as it changes the tensor on the left
        return loss_value.TotalWithIntermediate(total, **partial)


# Patched from
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/patchnce.py
class PatchNCELoss(nn.Module):
    def __init__(self, batch_dim_for_bmm=1, temperature=0.07):
        """
        Args:
            batch_dim_for_bmm (int)     -- set 1 if mining negatives from the whole batch (single-image translation)
                                        -- set batch_size if mining negatives only from the input image patches
        """
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.batch_dim_for_bmm = batch_dim_for_bmm
        self.temperature = temperature
        self.reduction = "mixed"

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # reshape features to batch size
        feat_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(self.batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss


class MultilayerPatchNCELoss(nn.Module):
    def __init__(self, batch_dim_for_bmm, nce_layers, num_patches, temperature, weight):
        super().__init__()
        self.nce_layers = [int(i) for i in nce_layers.split(',')]
        self.losses = [PatchNCELoss(batch_dim_for_bmm, temperature) for _ in self.nce_layers]
        self.num_patches = num_patches
        self.weight = weight
        self.reduction = "mixed"

    def forward(self, feat_q_pool, feat_k_pool):
        total = loss_value.ZERO
        partial = {}
        for feat_q, feat_k, criterion, nce_layer in zip(feat_q_pool, feat_k_pool, self.losses, self.nce_layers):
            layer_key = "layer" + str(nce_layer)
            partial[layer_key] = torch.mean(criterion(feat_q, feat_k) * self.weight)
            total = total + partial[layer_key]
        return loss_value.TotalWithIntermediate(total / len(self.nce_layers), **partial)
