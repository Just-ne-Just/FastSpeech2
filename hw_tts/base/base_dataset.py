import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config_parser["preprocessing"]["log_spec"]

        # self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        mix_path = data_dict["mix_path"]
        ref_path = data_dict["ref_path"]
        target_path = data_dict["target_path"]


        mix_wave = self.load_audio(mix_path)
        ref_wave = self.load_audio(ref_path)
        target_wave = self.load_audio(target_path)
        
        return {
            "mix_wave": mix_wave,
            "ref_wave": ref_wave,
            "target_wave": target_wave,
            "speaker_id": data_dict["speaker_id"],
            "name": data_dict["name"]
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor
    
    def _count_speakers(self):
        return max(map(lambda x: x["speaker_id"], self._index)) + 1

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "reference" in entry
            assert "path" in entry
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )
    
    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index