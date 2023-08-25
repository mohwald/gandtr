"""
Nets consisting of one layer only
"""

from cirtorch.layers import normalization

class NormalizationL2(normalization.L2N):
    """Return L2 normalization of the input tensor"""

    def __init__(self):
        super().__init__()
        self.meta = {}
