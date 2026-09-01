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



class CBAMOpenGate(CBAM):
    """
    CBAM whose attention gates start open, so the block begins as a uniform
    pass-through instead of a random mask.

    Stock CBAM initialises both gate convolutions the default way, so at step
    zero the channel gate and the spatial gate emit values scattered around 0.5
    that differ per channel and per pixel. Appended to a pretrained backbone
    and neck, that multiplies a converged representation by random noise on the
    first forward pass. Measured on this dataset, that cost YOLOv8s 5.8 points
    of detection rate and raised YOLOv11s false alarms by 4.3 points, and it
    showed up on the synthetic validation split too -- so it was a training
    problem, not overfitting.

    The fix is to zero both gate convolutions and open them with a positive
    bias. Zero weights make each gate constant across channels and pixels, so
    the block applies one uniform gain that the following layer simply absorbs;
    what it can no longer do is destroy the structure of the features it was
    handed. The gates still receive gradient and still learn to attend, they
    just start from "pass everything through" rather than from noise.

    GATE_BIAS is 2.0 rather than something larger. sigmoid(2) = 0.88 with a
    local gradient of 0.105, against 0.98 and 0.018 at a bias of 4: pushing the
    gate closer to a true identity saturates the sigmoid and starves the very
    parameters that are supposed to learn. Since the gain is uniform, the exact
    value does not matter; the gradient does.

    Ultralytics' SpatialAttention builds its convolution with bias=False, so
    the layer is replaced with an equivalent one that has a bias to open.
    """

    GATE_BIAS = 2.0

    def __init__(self, c1, kernel_size=7):
        super().__init__(c1, kernel_size)

        fc = self.channel_attention.fc
        nn.init.zeros_(fc.weight)
        nn.init.constant_(fc.bias, self.GATE_BIAS)

        old = self.spatial_attention.cv1
        opened = nn.Conv2d(old.in_channels, old.out_channels, old.kernel_size,
                           stride=old.stride, padding=old.padding, bias=True)
        nn.init.zeros_(opened.weight)
        nn.init.constant_(opened.bias, self.GATE_BIAS)
        self.spatial_attention.cv1 = opened


# The model YAMLs say CBAM; this is what they get. The open-gate variant is
# the only initialisation this project wants, so it is bound under that name
# rather than offered as an alternative someone has to remember to select.
tasks.CBAM = CBAMOpenGate
tasks.CBAMOpenGate = CBAMOpenGate
tasks.BiFPNFuse = BiFPNFuse
