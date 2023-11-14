import logging
from pathlib import Path

from hw_asr.datasets.custom_audio_dataset import CustomAudioDataset
import os

logger = logging.getLogger(__name__)


class CustomDirTripletDataset(CustomAudioDataset):
    def __init__(self, mix_path, ref_path, target_path, *args, **kwargs):
        data = []

        mix_paths = sorted(os.listdir(mix_path))
        ref_paths = sorted(os.listdir(ref_path))
        target_paths = sorted(os.listdir(target_path))

        for mix_path_, ref_path_, target_path_ in zip(mix_paths, ref_paths, target_paths):
            data.append({
                "mix_path": f"{mix_path}/{mix_path_}",
                "ref_path": f"{ref_path}/{ref_path_}",
                "target_path": f"{target_path}/{target_path_}",
                "speaker_id": -1,
                "name": "",
            })
        
        super().__init__(data, *args, **kwargs)
