import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from .conformer.model import ConformerEncoder

def rescale_conv(conv, reference):
    """
    Rescale a convolutional module with `reference`.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    """
    Rescale a module with `reference`.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def center_trim(tensor, reference):
    """
    Trim a tensor to match with the dimension of `reference`.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2:-(diff - diff // 2)]
    return tensor


def left_trim(tensor, reference):
    """
    Trim a tensor to match with the dimension of `reference`. Trims only the end.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., 0:-diff]
    return tensor

def normalize_input(data):
    """
    Normalizes the input to have mean 0 std 1 for each input
    Inputs:
        data - torch.tensor of size batch x n_mics x n_samples
    """
    data = (data * 2**15).round() / 2**15
    ref = data.mean(1)  # Average across the n microphones
    means = ref.mean(1).unsqueeze(1).unsqueeze(2)
    stds = ref.std(1).unsqueeze(1).unsqueeze(2)
    data = (data - means) / stds

    return data, means, stds

def unnormalize_input(data, means, stds):
    """
    Unnormalizes the step done in the previous function
    """
    data = (data * stds.unsqueeze(3) + means.unsqueeze(3))
    return data


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1)
        gamma = gamma.view(x.size(0), x.size(1), 1)

        x = gamma * x + beta

        return x

class CoSNetwork_spk(nn.Module):
    """
    Cone of Silence network based on the Demucs network for audio source separation.
    """
    def __init__(
            self,
            n_audio_channels: int = 4,  # pylint: disable=redefined-outer-name
            window_conditioning_size: int = 5,
            kernel_size: int = 7,
            stride: int = 4,
            context: int = 3,
            depth: int = 6, #6,
            channels: int = 64, #64,
            growth: float = 2.0,
            lstm_layers: int = 2,
            rescale: float = 0.1):  # pylint: disable=redefined-outer-name
        super().__init__()
        self.n_audio_channels = n_audio_channels
        self.window_conditioning_size = window_conditioning_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.context = context
        self.depth = depth
        self.channels = channels
        self.growth = growth
        self.lstm_layers = lstm_layers
        self.rescale = rescale

        self.encoder = nn.ModuleList()  # Source encoder
        self.decoder = nn.ModuleList()  # Audio output decoder

        activation = nn.GLU(dim=1)

        in_channels = n_audio_channels  # Number of input channels

        # Wave U-Net structure
        for index in range(depth):
            encode = nn.ModuleDict()
            encode["conv1"] = nn.Conv1d(in_channels, channels, kernel_size,
                                        stride)
            encode["relu"] = nn.ReLU()

            encode["conv2"] = nn.Conv1d(channels, 2 * channels, 1)
            encode["activation"] = activation

            encode["gc_embed1"] = nn.Conv1d(self.window_conditioning_size, channels, 1)
            encode["gc_embed2"] = nn.Conv1d(self.window_conditioning_size, 2 * channels, 1)

            self.encoder.append(encode)

            decode = nn.ModuleDict()
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 2 * n_audio_channels

            decode["conv1"] = nn.Conv1d(channels, 2 * channels, context)
            decode["activation"] = activation
            decode["conv2"] = nn.ConvTranspose1d(channels, out_channels,
                                                 kernel_size, stride)

            decode["gc_embed1"] = nn.Conv1d(self.window_conditioning_size, 2 * channels, 1)
            decode["gc_embed2"] = nn.Conv1d(self.window_conditioning_size, out_channels, 1)

            if index > 0:
                decode["relu"] = nn.ReLU()
            self.decoder.insert(0,
                                decode)  # Put it at the front, reverse order

            in_channels = channels
            channels = int(growth * channels)

        # Bi-directional LSTM for the bottleneck layer
        channels = in_channels
        self.conformer = ConformerEncoder(num_blocks=2, d_model=2048, num_heads=4, max_len=128)
        spk_embedding = 192
        self.film_generator = nn.Linear(spk_embedding,2*channels)
        self.film_block = FiLMBlock()
        rescale_module(self, reference=rescale)

    def forward(self, mix: torch.Tensor, angle_conditioning: torch.Tensor,spk_embedding):  # pylint: disable=arguments-differ
        """
        Forward pass. Note that in our current work the use of `locs` is disregarded.

        Args:
            mix (torch.Tensor) - An input recording of size `(batch_size, n_mics, time)`.

        Output:
            x - A source separation output at every microphone
        """
        x = mix
        saved = [x]

        # Encoder
        for encode in self.encoder:
            x = encode["conv1"](x)  # Conv 1d
            embedding = encode["gc_embed1"](angle_conditioning.unsqueeze(2))

            x = encode["relu"](x + embedding)
            x = encode["conv2"](x)

            embedding2 = encode["gc_embed2"](angle_conditioning.unsqueeze(2))
            x = encode["activation"](x + embedding2)
            saved.append(x)

        # Bi-directional LSTM at the bottleneck layer
        x = x.permute(2, 0, 1)  # prep input for LSTM
        x = self.conformer(x)
        x = x.permute(1, 2, 0)
        film_vector = self.film_generator(spk_embedding)
        film_vector = film_vector.view(x.size(0),x.size(1),2)
        beta = film_vector[:,:,0]
        gamma = film_vector[:,:,1]
        x = self.film_block(x,beta,gamma)

        # Source decoder
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip

            x = decode["conv1"](x)
            embedding = decode["gc_embed1"](angle_conditioning.unsqueeze(2))
            x = decode["activation"](x + embedding)
            x = decode["conv2"](x)
            embedding2 = decode["gc_embed2"](angle_conditioning.unsqueeze(2))
            if "relu" in decode:
                x = decode["relu"](x + embedding2)

        # Reformat the output
        x = x.view(x.size(0), 2, self.n_audio_channels, x.size(-1))

        return x

    def loss(self, voice_signals, gt_voice_signals):
        """Simple L1 loss between voice and gt"""
        return -si_snr(voice_signals,gt_voice_signals)
        # return F.l1_loss(voice_signals, gt_voice_signals)

    def valid_length(self, length: int) -> int:  # pylint: disable=redefined-outer-name
        """
        Find the length of the input to the network such that the output's length is
        equal to the given `length`.
        """
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1

        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        return int(length)

def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)

if __name__ == '__main__':
    model = CoSNetwork()
    wav= torch.randn([4,64000])
