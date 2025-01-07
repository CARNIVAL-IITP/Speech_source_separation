import json
import logging
import math
from pathlib import Path
import os
import re

import torchaudio
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from glob import glob

from .audio import Audioset

logger = logging.getLogger(__name__)

def sort(infos): return sorted(
    infos, key=lambda info: int(info[1]), reverse=True)

class Train_dataset:
    def __init__(self, path_dir, sitec_data_path=None, segment=None,
                 pad=True, sample_rate=16000,spk_num = 2):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        path_dir = os.path.join(path_dir,'tr')
        mix_json = os.path.join(path_dir, 'mix.json')
        s_jsons = list()
        self.s_infos = list()
        sets_re = re.compile(r's[0-9].json')
        for s in os.listdir(path_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(path_dir, s))
        self.spk_num = len(s_jsons)
        with open(mix_json, 'r') as f:
            self.mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, 'r') as f:
                self.s_infos.append(json.load(f))

        self.length = segment
        self.pad = pad
        self.sample_rate = sample_rate
        self.spk_num = spk_num
        self.sitec_available = False
        if sitec_data_path:
            self.data_list = glob(sitec_data_path + "/**/*.wav", recursive=True)
            self.sitec_available = True
            from .real_time_datamixing import Data_Mixer
            self.sitec_data_mixer = Data_Mixer(self.data_list,segment=self.length,pad=self.pad)
    def __getitem__(self, index):
        import random
        if random.random() > 0.5:
            mixture ,sr = torchaudio.load(self.mix_infos[index][0])
            sources = []
            for i in range(self.spk_num):
                source,sr = torchaudio.load(self.s_infos[i][index][0])
                sources.append(source)

            if self.length:
                num_frames = self.length *self.sample_rate
                if mixture.shape[-1]>num_frames:
                    import random
                    offset = random.randint(0, mixture.shape[-1] - num_frames - 1)
                    mixture = mixture[:, offset:offset+num_frames]
                    for i in range(self.spk_num):
                        sources[i] = sources[i][:, offset:offset+num_frames]

                elif self.pad:
                    mixture = F.pad(mixture, (0, num_frames - mixture.shape[-1]))
                    for i in range(self.spk_num):
                        sources[i] = F.pad(sources[i], (0, num_frames - sources[i].shape[-1]))
                else:
                    mixture = mixture
                    sources = sources
                sources = torch.stack(sources,dim=0)

        else:
            mixture , sources = self.sitec_data_mixer(self.spk_num)

        return mixture.squeeze(dim=0), mixture.size(-1), sources.squeeze(dim=1)
    def __len__(self):
        return len(self.mix_infos)



class Valid_dataset:
    def __init__(self, path_dir, segment=None,
                 pad=True, sample_rate=16000,spk_num = 2):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        path_dir = os.path.join(path_dir,'tt')
        mix_json = os.path.join(path_dir, 'mix.json')
        s_jsons = list()
        self.s_infos = list()
        sets_re = re.compile(r's[0-9].json')
        for s in os.listdir(path_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(path_dir, s))
        self.spk_num = len(s_jsons)
        with open(mix_json, 'r') as f:
            self.mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, 'r') as f:
                self.s_infos.append(json.load(f))

        self.length = segment
        self.pad = pad
        self.sample_rate = sample_rate
        self.spk_num = spk_num
    def __getitem__(self, index):
        mixture ,sr = torchaudio.load(self.mix_infos[index][0])
        sources = []
        for i in range(self.spk_num):
            source,sr = torchaudio.load(self.s_infos[i][index][0])
            sources.append(source)

        if self.length:
            num_frames = self.length *self.sample_rate
            if mixture.shape[-1]>num_frames:
                import random
                offset = random.randint(0, mixture.shape[-1] - num_frames - 1)
                mixture = mixture[:, offset:offset+num_frames]
                for i in range(self.spk_num):
                    sources[i] = sources[i][:, offset:offset+num_frames]

            elif self.pad:
                mixture = F.pad(mixture, (0, num_frames - mixture.shape[-1]))
                for i in range(self.spk_num):
                    sources[i] = F.pad(sources[i], (0, num_frames - sources[i].shape[-1]))
            else:
                mixture = mixture
                sources = sources
            sources = torch.stack(sources,dim=0)
        return mixture.squeeze(dim=0),mixture.size(-1), sources.squeeze(dim=1)
    def __len__(self):
        return len(self.mix_infos)

class Test_dataset:
    def __init__(self, path_dir, segment=None,
                 pad=True, sample_rate=16000,spk_num = 2):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        path_dir = os.path.join(path_dir,'cv')
        mix_json = os.path.join(path_dir, 'mix.json')
        s_jsons = list()
        self.s_infos = list()
        sets_re = re.compile(r's[0-9].json')
        for s in os.listdir(path_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(path_dir, s))
        self.spk_num = len(s_jsons)
        with open(mix_json, 'r') as f:
            self.mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, 'r') as f:
                self.s_infos.append(json.load(f))

        self.length = segment
        self.pad = pad
        self.sample_rate = sample_rate
        self.spk_num = spk_num
    def __getitem__(self, index):
        mixture ,sr = torchaudio.load(self.mix_infos[index][0])
        sources = []
        for i in range(self.spk_num):
            source,sr = torchaudio.load(self.s_infos[i][index][0])
            sources.append(source)

        if self.length:
            num_frames = self.length *self.sample_rate
            if mixture.shape[-1]>num_frames:
                import random
                offset = random.randint(0, mixture.shape[-1] - num_frames - 1)
                mixture = mixture[:, offset:offset+num_frames]
                for i in range(self.spk_num):
                    sources[i] = sources[i][:, offset:offset+num_frames]

            elif self.pad:
                mixture = F.pad(mixture, (0, num_frames - mixture.shape[-1]))
                for i in range(self.spk_num):
                    sources[i] = F.pad(sources[i], (0, num_frames - sources[i].shape[-1]))
            else:
                mixture = mixture
                sources = sources
            sources = torch.stack(sources,dim=0)
        return mixture.squeeze(dim=0),mixture.size(-1), sources.squeeze(dim=1)
    def __len__(self):
        return len(self.mix_infos)
