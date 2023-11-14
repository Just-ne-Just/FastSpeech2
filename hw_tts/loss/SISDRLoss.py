import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from hw_asr.metric.utils import calc_si_sdr


class SISDRLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.5, device='cuda:0'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sisdr = ScaleInvariantSignalDistortionRatio(zero_mean=True).to(device)

    def forward(self, short, middle, long, targets, speaker_ids, logits, train=True, **kwargs):
        short = short.squeeze(1) - torch.mean(short.squeeze(1), dim=-1, keepdim=True)
        middle = middle.squeeze(1) - torch.mean(middle.squeeze(1), dim=-1, keepdim=True)
        long = long.squeeze(1) - torch.mean(long.squeeze(1), dim=-1, keepdim=True)
        targets = targets.squeeze(1) - torch.mean(targets.squeeze(1), dim=-1, keepdim=True)

        sisdr_short = calc_si_sdr(short, targets)
        sisdr_middle = calc_si_sdr(middle, targets)
        sisdr_long = calc_si_sdr(long, targets)

        sisdr_loss = (1 - self.alpha - self.beta) * sisdr_short.sum() + self.alpha * sisdr_middle.sum() + self.beta * sisdr_long.sum()
        sisdr_loss = -sisdr_loss.mean()

        if not train:
            return sisdr_loss
        ce_loss = F.cross_entropy(logits, speaker_ids)
        return sisdr_loss + self.gamma * ce_loss

