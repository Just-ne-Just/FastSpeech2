import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from hw_tts.utils import reprocess_tensor

logger = logging.getLogger(__name__)


def collate_fn(batch):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // batch[0]["batch_expand_size"]

    cut_list = list()
    for i in range(batch[0]["batch_expand_size"]):
        cut_list.append(index_arr[i*real_batchsize:(i + 1)*real_batchsize])

    output = list()
    for i in range(batch[0]["batch_expand_size"]):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output