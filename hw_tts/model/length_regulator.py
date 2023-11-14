import torch
import torch.nn as nn
import torch.nn.functional as F
from hw_tts.model.predictor import Predictor
from hw_tts.model.utils import create_alignment


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = Predictor(encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        if target is None:
            duration_prediction = self.duration_predictor(x)
            duration_prediction = ((torch.exp(duration_prediction) + 0.5) * alpha).int()
            output = self.LR(x, duration_prediction)
            mel_pos = torch.stack([torch.Tensor([i + 1 for i in range(output.size(1))])]).long().to(x.device)
            return output, mel_pos
        else:
            duration_prediction = self.duration_predictor(x)
            output = self.LR(x, target, mel_max_length)
            return output, duration_prediction