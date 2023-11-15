import torch
import torch.nn as nn

class TTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, 
                mel, 
                duration_predicted, 
                pitch_predicted, 
                energy_predicted, 
                mel_target, 
                duration_predictor_target, 
                pitch_predictor_target, 
                energy_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(duration_predicted,
                                               torch.log(duration_predictor_target.float() + 0.5))
        
        pitch_predictor_loss = self.mse_loss(pitch_predicted,
                                               torch.log(pitch_predictor_target.float() + 0.5))
        
        energy_predictor_loss = self.mse_loss(energy_predicted,
                                               torch.log(energy_predictor_target.float() + 0.5))

        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss