import logging
from pathlib import Path

import torchaudio

from hw_asr.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            entry["mix_path"] = str(Path(entry["mix_path"]).absolute().resolve())
            entry["ref_path"] = str(Path(entry["ref_path"]).absolute().resolve())
            entry["target_path"] = str(Path(entry["target_path"]).absolute().resolve())

        super().__init__(index, *args, **kwargs)
