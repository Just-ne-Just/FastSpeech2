import torch
import torch.nn as nn
from hw_tts.model.predictor import Predictor

class PitchPredictor(Predictor):
    def __init__(self, 
                 encoder_dim, 
                 pitch_predictor_filter_size, 
                 pitch_predictor_kernel_size, 
                 dropout):
        super(PitchPredictor, self).__init__(encoder_dim, 
                                              pitch_predictor_filter_size, 
                                              pitch_predictor_kernel_size, 
                                              dropout)
    
    def get_pitch(self, x, pitch_bounds, pitch_embedding, target=None, alpha=1.0):
        if target is not None:
            pitch_prediction = self.forward(x)
            embedding = pitch_embedding(torch.bucketize(torch.log(target + 1), pitch_bounds))
        else:
            pitch_prediction = self.forward(x)
            energy_prediction_pred = pitch_prediction * alpha
            embedding = pitch_embedding(torch.bucketize(torch.log(energy_prediction_pred), pitch_bounds))
        
        return embedding, pitch_prediction
