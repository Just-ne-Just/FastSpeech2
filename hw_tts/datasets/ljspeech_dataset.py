import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path
import pyworld as pw
from scipy.interpolate import interp1d
import librosa

import torchaudio
from hw_tts.base.base_dataset import BaseDataset
from hw_tts.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
from torch.utils.data import Dataset
from hw_tts.utils.util import get_data_to_buffer
import numpy as np
from speechbrain.utils.data_utils import download_file

import gdown

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}


class LJspeechDataset(Dataset):
    def __init__(self,
                 data_path,
                 wav_path,
                 mel_ground_truth, 
                 alignment_path, 
                 text_cleaners, 
                 pitch_path, 
                 energy_path, 
                 batch_expand_size,
                 limit=None, 
                 *args, 
                 **kwargs):
        if not Path(data_path).exists():
            Path(data_path).parent.mkdir(exist_ok=True, parents=True)
            download_file("https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx", data_path)
        if not Path(wav_path).exists():
            Path(wav_path).mkdir(exist_ok=True, parents=True)
            if not (Path(wav_path).parent / "LJSpeech-1.1.tar.bz2").exists():
                download_file("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", str(Path(wav_path).parent / "LJSpeech-1.1.tar.bz2"))
            shutil.unpack_archive(str(Path(wav_path).parent / "LJSpeech-1.1.tar.bz2"), str(Path(wav_path).parent))
        if not Path(mel_ground_truth).exists():
            Path(mel_ground_truth).mkdir(exist_ok=True, parents=True)
            if not (Path(mel_ground_truth).parent / 'mel.tar.gz').exists():
                download_file("https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j", str(Path(mel_ground_truth).parent / 'mel.tar.gz'))
            shutil.unpack_archive(str(Path(mel_ground_truth).parent / 'mel.tar.gz'), str(Path(mel_ground_truth).parent))
        if not Path(alignment_path).exists():
            Path(alignment_path).mkdir(exist_ok=True, parents=True)
            if not (Path(mel_ground_truth).parent / 'alignments.zip').exists():
                download_file("https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip", str(Path(alignment_path).parent / 'alignments.zip'))
            shutil.unpack_archive(str(Path(alignment_path).parent / 'alignments.zip'), str(Path(alignment_path).parent))
        if not Path(energy_path).exists():
            self._generate_energy(energy_path, mel_ground_truth)
        else:
            self._get_energy_bounds(energy_path)
        
        if not Path(pitch_path).exists():
            self._generate_pitch(pitch_path, mel_ground_truth, wav_path)
        else:
            self._get_pitch_bounds(pitch_path)
        


        self.buffer = get_data_to_buffer(data_path,
                                         mel_ground_truth, 
                                         alignment_path, 
                                         text_cleaners, 
                                         pitch_path, 
                                         energy_path, 
                                         batch_expand_size)
        if limit is not None:
            self.buffer = self.buffer[:limit]
        
        self.length_dataset = len(self.buffer)

    def _get_energy_bounds(self, energy_path):
        energy_path = Path(energy_path)

        self.max_energy = None
        self.min_energy = None

        for energy in tqdm(energy_path.iterdir()):
            energy = np.load(energy)
            self.max_energy = max(energy.max(), self.max_energy) if self.max_energy is not None else energy.max()
            self.min_energy = max(energy.min(), self.min_energy) if self.min_energy is not None else energy.min()


    def _generate_energy(self, energy_path, mel_ground_truth):
        energy_path = Path(energy_path)
        mel_ground_truth = Path(mel_ground_truth)
        energy_path.mkdir(exist_ok=True, parents=True)

        self.max_energy = None
        self.min_energy = None

        for mel_path in tqdm(mel_ground_truth.iterdir()):
            mel = np.load(mel_path)
            energy = np.linalg.norm(mel, axis=-1)
            energy_name = mel_path.name.replace('mel', 'energy')
            np.save(energy_path / energy_name, energy)

            self.max_energy = max(energy.max(), self.max_energy) if self.max_energy is not None else energy.max()
            self.min_energy = max(energy.min(), self.min_energy) if self.min_energy is not None else energy.min()


    def _get_pitch_bounds(self, pitch_path):
        pitch_path = Path(pitch_path)

        self.max_pitch = None
        self.min_pitch = None

        for pitch in tqdm(pitch_path.iterdir()):
            pitch = np.load(pitch)
            self.max_pitch = max(pitch.max(), self.max_pitch) if self.max_pitch is not None else pitch.max()
            self.min_pitch = max(pitch.min(), self.min_pitch) if self.min_pitch is not None else pitch.min()


    def _generate_pitch(self, pitch_path, mel_ground_truth, wav_path):
        pitch_path = Path(pitch_path)
        mel_ground_truth = Path(mel_ground_truth)
        wav_path = Path(wav_path) / "wavs"

        pitch_path.mkdir(exist_ok=True, parents=True)
        self.max_pitch = None
        self.min_pitch = None
        for i, wav in tqdm(enumerate(sorted(list(wav_path.iterdir()), key=lambda x: x.name))):
            pitch_name = "ljspeech-pitch-%05d.npy" % (i + 1)
            mel = np.load(mel_ground_truth / ("ljspeech-mel-%05d.npy" % (i + 1)))
            wav, sr = librosa.load(wav)
            pitch, t = pw.dio(
                wav.astype(np.float64),
                sr,
                frame_period=len(wav) / sr * 1000 / mel.shape[0],
            )
            pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)[:mel.shape[0]]
            nonzero_idx = np.where(pitch != 0)[0]
            pitch = interp1d(
                nonzero_idx,
                pitch[nonzero_idx],
                fill_value=(pitch[nonzero_idx[0]], pitch[nonzero_idx[-1]]),
                bounds_error=False,
            )(np.arange(0, pitch.shape[0]))
            np.save(pitch_path / pitch_name, pitch)

            self.max_pitch = max(pitch.max(), self.max_pitch) if self.max_pitch is not None else pitch.max()
            self.min_pitch = max(pitch.min(), self.min_pitch) if self.min_pitch is not None else pitch.min()

        

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]