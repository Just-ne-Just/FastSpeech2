import torch
import torch.nn as nn
from hw_tts.model.layers import Encoder, Decoder
from hw_tts.model.length_regulator import LengthRegulator
from hw_tts.model.utils import get_mask_from_lengths
from hw_tts.model.pitch_predictor import PitchPredictor
from hw_tts.model.energy_predictor import EnergyPredictor
import numpy as np

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, 
                 max_seq_len, 
                 encoder_n_layer, 
                 vocab_size, 
                 encoder_dim, 
                 encoder_head, 
                 pad, 
                 encoder_conv1d_filter_size,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 decoder_n_layer,
                 decoder_dim,
                 decoder_head,
                 decoder_conv1d_filter_size,
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 pitch_predictor_filter_size,
                 pitch_predictor_kernel_size,
                 energy_predictor_filter_size,
                 energy_predictor_kernel_size,
                 num_mels,
                 num_bins,
                 min_pitch,
                 max_pitch,
                 min_energy,
                 max_energy,
                 dropout):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(max_seq_len, 
                               encoder_n_layer, 
                               vocab_size, 
                               encoder_dim, 
                               encoder_head, 
                               pad, 
                               encoder_conv1d_filter_size,
                               fft_conv1d_kernel,
                               fft_conv1d_padding,
                               dropout)
        
        self.length_regulator = LengthRegulator(encoder_dim,
                                                duration_predictor_filter_size,
                                                duration_predictor_kernel_size,
                                                dropout)
        self.decoder = Decoder(max_seq_len, 
                               decoder_n_layer, 
                               decoder_dim, 
                               decoder_head, 
                               pad, 
                               decoder_conv1d_filter_size,
                               fft_conv1d_kernel,
                               fft_conv1d_padding,
                               dropout)
        
        self.register_buffer('pitch_bounds', torch.linspace(np.log(min_pitch + 1), np.log(max_pitch + 2), num_bins))
        self.register_buffer('energy_bounds', torch.linspace(np.log(min_energy + 1), np.log(max_energy + 2), num_bins))

        self.pitch_embedding = nn.Embedding(num_bins, encoder_dim)
        self.energy_embedding = nn.Embedding(num_bins, encoder_dim)

        self.pitch_predictor = PitchPredictor(
            encoder_dim,
            pitch_predictor_filter_size,
            pitch_predictor_kernel_size,
            dropout
        )
        self.energy_predictor = EnergyPredictor(
            encoder_dim,
            energy_predictor_filter_size,
            energy_predictor_kernel_size,
            dropout
        )

        self.mel_linear = nn.Linear(decoder_dim, num_mels)
        self.pad = pad

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, 
                text, 
                src_pos, 
                mel_pos=None, 
                mel_max_len=None, 
                duration=None, 
                energy=None, 
                pitch=None, 
                alpha=1.0,
                beta=1.0,
                gamma=1.0,
                *args, 
                **kwargs):
        enc_output, _ = self.encoder(text, src_pos)

        output, duration_prediction = self.length_regulator(enc_output, 
                                                            alpha, 
                                                            target=duration if self.training else None,
                                                            mel_max_length=mel_max_len)
        
        pitch_embedding, pitch_prediction = self.pitch_predictor.get_pitch(output, 
                                                                           pitch_bounds=self.pitch_bounds,
                                                                           pitch_embedding=self.pitch_embedding,
                                                                           target=pitch if self.training else None, 
                                                                           alpha=beta)

        energy_embedding, energy_prediction = self.energy_predictor.get_energy(output, 
                                                                               energy_bounds=self.energy_bounds,
                                                                               energy_embedding=self.energy_embedding,
                                                                               target=energy if self.training else None, 
                                                                               alpha=gamma)

        # print(output.shape, pitch_embedding.shape, energy_embedding.shape)
        output = self.decoder(output + pitch_embedding + energy_embedding, mel_pos if mel_pos is not None else duration_prediction)

        if self.training:
            output = self.mask_tensor(output, mel_pos, mel_max_len)
            
        output = self.mel_linear(output)

        return {
            "mel_predicted": output, 
            "duration_predicted": duration_prediction if self.training else None,
            "pitch_predicted": pitch_prediction if self.training else None,
            "energy_predicted": energy_prediction if self.training else None
        }