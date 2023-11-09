import json
from pathlib import Path
import math
import os
import tqdm
import sys

import torchaudio
import soundfile as sf
import torch as th
from torch.nn import functional as F

class Audioset:
    def __init__(self, files, length=None, stride=None, pad=True, augment=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.augment = augment
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(
                    math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = -1
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            #  out = th.Tensor(sf.read(str(file), start=offset, frames=num_frames)[0]).unsqueeze(0)
            out = torchaudio.load(str(file), frame_offset=offset,
                                  num_frames=num_frames)[0]
            if self.augment:
                out = self.augment(out.squeeze(0).numpy()).unsqueeze(0)
            if num_frames != -1:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            return out
