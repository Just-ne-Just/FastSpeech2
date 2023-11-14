from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.metric.utils import calc_si_sdr
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

class SISDRMetric(BaseMetric):
    def __init__(self, device='cuda:0', *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, short, targets, **kwargs):
        short = short.squeeze(1) - torch.mean(short.squeeze(1), dim=-1, keepdim=True)
        targets = targets.squeeze(1) - torch.mean(targets.squeeze(1), dim=-1, keepdim=True)

        sisdr = calc_si_sdr(short, targets)
        return sisdr.mean().item()