import torch
import torch.nn as nn
from hw_tts.model.layers import Encoder, Decoder
from hw_tts.model.length_regulator import LengthRegulator
from hw_tts.model.utils import get_mask_from_lengths
from hw_tts.model.pitch_predictor import PitchPredictor
from hw_tts.model.energy_predictor import EnergyPredictor

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
        
        self.pitch_bounds = torch.linspace(torch.log(min_pitch + 1), torch.log(max_pitch + 2), num_bins)
        self.energy_bounds = torch.linspace(torch.log(min_energy + 1), torch.log(max_energy + 2), num_bins)

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

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, 
                src_seq, 
                src_pos, 
                mel_pos=None, 
                mel_max_length=None, 
                length_target=None, 
                energy_target=None, 
                pitch_target=None, 
                alpha=1.0):
        enc_output, _ = self.encoder(src_seq, src_pos)

        output, duration_prediction = self.length_regulator(enc_output, 
                                                            alpha, 
                                                            target=length_target if self.training else None,
                                                            mel_max_length=mel_max_length)
        
        pitch_embedding, pitch_prediction = self.pitch_predictor.get_pitch(output, 
                                                                           pitch_bounds=self.pitch_bounds,
                                                                           pitch_embedding=self.pitch_embedding,
                                                                           target=pitch_target if self.training else None, 
                                                                           alpha=alpha)

        energy_embedding, energy_prediction = self.pitch_predictor.get_pitch(output, 
                                                                             energy_bounds=self.energy_bounds,
                                                                             energy_embedding=self.energy_embedding,
                                                                             target=energy_target if self.training else None, 
                                                                             alpha=alpha)

        output = self.decoder(output + pitch_embedding + energy_embedding, mel_pos)

        output = self.mask_tensor(output, mel_pos, mel_max_length)
        output = self.mel_linear(output)

        return {
            "output": output, 
            "duration_prediction": duration_prediction if self.training else None,
            "pitch_prediction": pitch_prediction if self.training else None,
            "energy_prediction": energy_prediction if self.training else None
        }