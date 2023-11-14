from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric

class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, logits, speaker_ids, **kwargs):
        pred = logits.argmax(dim=-1)
        return (pred == speaker_ids).float().mean().item()