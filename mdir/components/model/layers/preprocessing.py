import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

#
# Edgemap pre-processing layers
#

class EdgeFilter(nn.Module):
    def __init__(self, w=10, p=0.5, beta=500, tau=0.1, eps=1e-6):
        super().__init__()
        self.w = float(w)
        self.p = Parameter(torch.FloatTensor([p]))
        self.beta = float(beta)
        self.tau = Parameter(torch.FloatTensor([tau]))
        self.eps = float(eps)

    def forward(self, x):
        # Restrict tau range
        if self.tau.data[0] < 0.01:
            self.tau.data[0] = 0.01
        if self.tau.data[0] > 0.9:
            self.tau.data[0] = 0.9
        # The input of exp is clipped, as exp output otherwise overflows to inf for t=0.19, x=0
        return (self.w * x.clamp(min=self.eps).pow(self.p)) / ((-self.beta * (x - self.tau)).clamp(max=50.0).exp() + 1)

    def __repr__(self):
        return "%s(w=%.4f, p=%.4f, beta=%.4f, tau=%.4f, eps=%f)" % \
            (self.__class__.__name__, self.w, self.p.item(), self.beta, self.tau.item(), self.eps)
