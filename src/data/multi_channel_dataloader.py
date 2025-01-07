import json
import logging
import math
from pathlib import Path
import os
import re

import librosa
import numpy as np
import torch
import torch.utils.data as data

from .audio import Audioset

logger = logging.getLogger(__name__)


def sort(infos): return sorted(
    infos, key=lambda info: int(info[1]), reverse=True)


class Trainset:
    def __init__(self, json_dir, sample_rate=16000, segment=4.0, stride=1.0, pad=True):
        mix_json = os.path.join(json_dir, 'mix.json')
        s_jsons = list()
        s_infos = list()
        sets_re = re.compile(r's[0-9].json')
        for s in os.listdir(json_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(json_dir, s))

        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, 'r') as f:
                s_infos.append(json.load(f))

        length = int(sample_rate  * segment)
        stride = int(sample_rate * stride)

        kw = {'length': length, 'stride': stride, 'pad': pad}
        self.mix_set = Audioset(sort(mix_infos), **kw)

        self.sets = list()
        for s_info in s_infos:
            self.sets.append(Audioset(sort(s_info), **kw))

        # verify all sets has the same size
        for s in self.sets:
            assert len(s) == len(self.mix_set)

    def __getitem__(self, index):
        mix_sig = self.mix_set[index]
        tgt_sig = [self.sets[i][index] for i in range(len(self.sets))]
        return self.mix_set[index], torch.LongTensor([mix_sig.shape[0]]), torch.stack(tgt_sig)

    def __len__(self):
        return len(self.mix_set)


class Validset:
    """
    load entire wav.
    """

    def __init__(self, json_dir):
        mix_json = os.path.join(json_dir, 'mix.json')
        s_jsons = list()
        s_infos = list()
        sets_re = re.compile(r's[0-9].json')
        for s in os.listdir(json_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(json_dir, s))
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, 'r') as f:
                s_infos.append(json.load(f))
        self.mix_set = Audioset(sort(mix_infos))
        self.sets = list()
        for s_info in s_infos:
            self.sets.append(Audioset(sort(s_info)))
        for s in self.sets:
            assert len(s) == len(self.mix_set)

    def __getitem__(self, index):
        mix_sig = self.mix_set[index]
        tgt_sig = [self.sets[i][index] for i in range(len(self.sets))]
        return self.mix_set[index], torch.LongTensor([mix_sig.shape[0]]), torch.stack(tgt_sig)

    def __len__(self):
        return len(self.mix_set)