"""
lib_modules.py
--------------
Custom network modules, and their registration with the Ultralytics YAML parser.

Ultralytics resolves the module names in a model YAML from the globals of
``ultralytics.nn.tasks``. CBAM ships with the package but is not exported into
that namespace, and BiFPN is not shipped at all, so both are bound here rather
than by editing the installed package -- a site-packages edit would not survive
a reinstall and would not be reproducible for anyone rebuilding the paper.

Every script that builds, trains, loads or evaluates a model from models/ must
import this module first. That includes the evaluator: a checkpoint containing
BiFPNFuse cannot be unpickled unless the class is importable under the name it
was saved with.
"""

import torch
import torch.nn as nn

import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules.conv import CBAM, Conv


class BiFPNFuse(nn.Module):
    """
    One BiFPN fusion node: fast normalised weighted fusion followed by a
    depthwise-separable convolution (EfficientDet, Tan et al. 2020, Eq. 3).

    The learnable weights are what separates this from plain concatenation:
    each input level gets a scalar the network tunes, ReLU-clamped and
    normalised to sum to one, so the fusion can suppress a level that carries
    nothing useful at a given scale instead of being forced to average it in.

    Input is the channel-concatenated stack of ``n`` feature maps that each
    carry ``w`` channels; output is a one-element list. The list is not
    decoration. The YAML parser derives a layer's output width from the width
    of its input, which here is the concatenated ``n * w`` rather than the
    fused ``w``; returning a list lets the following Index layer declare the
    true width. This is the same idiom Ultralytics itself uses to take
    intermediate feature maps out of a TorchVision backbone.
    """

    def __init__(self, c1, w, n=2):
        super().__init__()
        self.w, self.n = w, n
        self.weight = nn.Parameter(torch.ones(n))
        self.dw = Conv(w, w, 3, 1, g=w)
        self.pw = Conv(w, w, 1, 1)

    def forward(self, x):
        parts = torch.split(x, self.w, dim=1)
        a = torch.relu(self.weight)
        a = a / (a.sum() + 1e-4)
        y = parts[0] * a[0]
        for i in range(1, self.n):
            y = y + parts[i] * a[i]
        return [self.pw(self.dw(y))]


tasks.CBAM = CBAM
tasks.BiFPNFuse = BiFPNFuse
