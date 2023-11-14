import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    references = []
    references_length = []
    mixes = []
    mixes_length = []
    targets = []
    targets_length = []
    speaker_ids = []
    names = []

    for item in dataset_items:
        references.append(item['ref_wave'][0])
        references_length.append(item['ref_wave'].shape[-1])

        mixes.append(item['mix_wave'][0])
        mixes_length.append(item['mix_wave'].shape[-1])

        targets.append(item['target_wave'][0])
        targets_length.append(item['target_wave'].shape[-1])

        speaker_ids.append(item['speaker_id'])
        names.append(item['name'])
    
    references = pad_sequence(references, batch_first=True).unsqueeze(1)
    references_length = torch.tensor(references_length)

    mixes_targets = pad_sequence(mixes + targets, batch_first=True).unsqueeze(1)

    mixes = mixes_targets[:len(mixes_targets) // 2]
    mixes_length = torch.tensor(mixes_length)

    targets = mixes_targets[len(mixes_targets) // 2:]
    targets_length = torch.tensor(targets_length)


    return {
        "references": references,
        "references_length": references_length,
        "mixes": mixes,
        "mixes_length": mixes_length,
        "targets": targets,
        "targets_length": targets_length,
        "speaker_ids": torch.tensor(speaker_ids),
        "names": names
    }
