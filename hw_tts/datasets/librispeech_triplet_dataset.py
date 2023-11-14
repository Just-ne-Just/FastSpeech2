import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH
from hw_asr.mixer import MixtureGenerator, LibriSpeechSpeakerFiles

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechTripletDataset(BaseDataset):
    def __init__(self, part, mixer_config, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir)
        self.mixer_config = mixer_config
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        # index_path = self._data_dir / f"{part}_index.json"
        index_path = Path(f"{part}_index.json")
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index
    
    def _get_files(self, split_dir):
        result = []
        for speaker_id in os.listdir(split_dir):
            speaker_dir_path = os.path.join(split_dir, speaker_id)
            if os.path.isdir(speaker_dir_path):
                result.append(LibriSpeechSpeakerFiles(speaker_id, split_dir, "*.flac"))
        return result
    
    def _create_mix(self, split_dir, mix_dir):
        speakers_files = self._get_files(split_dir)
        
        nfiles = self.mixer_config["nfiles"]
        test = self.mixer_config["test"]
        mixer = MixtureGenerator(speakers_files, mix_dir, nfiles=nfiles, test=test)
        
        mixer.generate_mixes(
            snr_levels=self.mixer_config["snr_levels"],
            num_workers=self.mixer_config["num_workers"],
            update_steps=100,
            trim_db=self.mixer_config["trim_db"] if "trim_db" in self.mixer_config else None,
            vad_db=self.mixer_config["vad_db"],
            audioLen=self.mixer_config["audio_len"]
        )

    def _load_part(self, part):
        self._data_dir.mkdir(exist_ok=True, parents=True)
        arch_path = self._data_dir / f"{part}.tar.gz"
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        mix_dir = Path(self.mixer_config["path"]) / f"{part}"


        if not mix_dir.exists():
            if not split_dir.exists():
                self._load_part(part)
            self._create_mix(split_dir, mix_dir)
        
        id_map = {}
        
        mix_paths = []
        reference_paths = []
        target_paths = []
        for filename in os.listdir(str(mix_dir)):
            if filename.endswith("-target.wav"):
                target_paths.append(f"{str(mix_dir)}/{filename}")
            if filename.endswith("-ref.wav"):
                reference_paths.append(f"{str(mix_dir)}/{filename}")
            if filename.endswith("-mixed.wav"):
                mix_paths.append(f"{str(mix_dir)}/{filename}")
        mix_paths = sorted(mix_paths)
        reference_paths = sorted(reference_paths)
        target_paths = sorted(target_paths)

        for mix_path, reference_path, target_path in tqdm(
                zip(mix_paths, reference_paths, target_paths), desc=f"Preparing librispeech folders: {part}"
        ):
            id = reference_path.split('/')[-1].split('_')[0]
            if id not in id_map:
                id_map[id] = len(id_map)

            index.append({
                "mix_path": mix_path,
                "ref_path": reference_path,
                "target_path": target_path,
                "speaker_id": id_map[id],
                "name": mix_path.split('/')[-1]
            })
        return index
