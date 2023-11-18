import torch
import torch.nn as nn
from hw_tts.model.predictor import Predictor

class EnergyPredictor(Predictor):
    def __init__(self, 
                 encoder_dim, 
                 energy_predictor_filter_size, 
                 energy_predictor_kernel_size, 
                 dropout):
        super(EnergyPredictor, self).__init__(encoder_dim, 
                                              energy_predictor_filter_size, 
                                              energy_predictor_kernel_size, 
                                              dropout)
    
    def get_energy(self, x, energy_bounds, energy_embedding, target=None, alpha=1.0):
        if target is None:
            energy_prediction = self.forward(x)
            embedding = energy_embedding(torch.bucketize(torch.log(target + 1), energy_bounds))
        else:
            energy_prediction = self.forward(x)
            energy_prediction_pred = energy_prediction * alpha
            embedding = energy_embedding(torch.bucketize(torch.log(energy_prediction_pred), energy_bounds))
        
        return embedding, energy_prediction
