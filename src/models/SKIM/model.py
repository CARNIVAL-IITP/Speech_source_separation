import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .skim_separator import SkiMSeparator
EPS = 1e-8


def custom_unfold(input_tensor, dimension, block_size, step):
    # Get the shape of the input tensor
    shape = input_tensor.shape
    # Calculate the number of blocks that can be extracted along the specified dimension
    num_blocks = (shape[dimension] - block_size) // step + 1
    # Calculate the shape of the output tensor
    output_shape = list(shape)
    output_shape[dimension] = num_blocks
    # Add the block size dimension
    output_shape.append(block_size)

    # Initialize an empty tensor with the output shape
    output_tensor = torch.empty(*output_shape, dtype=input_tensor.dtype)

    # Loop through the specified dimension and extract the blocks
    for i in range(num_blocks):
        # Calculate the starting index of the current block
        start_index = i * step
        # Calculate the ending index of the current block
        end_index = start_index + block_size
        # Slice the input tensor and place it in the output tensor
        slice_idx = [slice(None)] * len(shape)  # List of slices for each dimension
        slice_idx[dimension] = slice(start_index, end_index)
        output_tensor[..., i, :] = input_tensor[slice_idx]

    return output_tensor
def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    # print(subframe_length)
    # print(signal.shape)
    # print(outer_dimensions)
    # subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length) .to("cuda")
    # frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = torch.arange(0,output_subframes)
    frame = custom_unfold(frame,0,subframes_per_frame,subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1) .to("cuda")


    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length).to("cuda")
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

class ConvEncoder(nn.Module):
    """Convolutional encoder for speech enhancement and separation"""

    def __init__(
        self,
        channel: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            1, channel, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.stride = stride
        self.kernel_size = kernel_size

        self._output_dim = channel

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        assert input.dim() == 2, "Currently only support single channel input"

        input = torch.unsqueeze(input, 1)

        feature = self.conv1d(input)
        feature = torch.nn.functional.relu(feature)
        feature = feature.transpose(1, 2)


        return feature

    def forward_streaming(self, input: torch.Tensor):
        output = self.forward(input)
        return output

    def streaming_frame(self, audio: torch.Tensor):
        """streaming_frame. It splits the continuous audio into frame-level
        audio chunks in the streaming *simulation*. It is noted that this
        function takes the entire long audio as input for a streaming simulation.
        You may refer to this function to manage your streaming input
        buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
        """
        batch_size, audio_len = audio.shape

        hop_size = self.stride
        frame_size = self.kernel_size

        audio = [
            audio[:, i * hop_size : i * hop_size + frame_size]
            for i in range((audio_len - frame_size) // hop_size + 1)
        ]

        return audio
class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, W=2, N=64):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.W, self.N = W, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=W, stride=W // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [B, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        mixture = torch.unsqueeze(mixture, 1)  # [B, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [B, N, L]
        return mixture_w

    def streaming_frame(self, audio: torch.Tensor):
        """streaming_frame. It splits the continuous audio into frame-level
        audio chunks in the streaming *simulation*. It is noted that this
        function takes the entire long audio as input for a streaming simulation.
        You may refer to this function to manage your streaming input
        buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
        """
        batch_size, audio_len = audio.shape

        hop_size = self.stride
        frame_size = self.kernel_size

        audio = [
            audio[:, i * hop_size: i * hop_size + frame_size]
            for i in range((audio_len - frame_size) // hop_size + 1)
        ]

        return audio
class ConvDecoder(nn.Module):
    """Transposed Convolutional decoder for speech enhancement and separation"""

    def __init__(
        self,
        channel: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.convtrans1d = torch.nn.ConvTranspose1d(
            channel, 1, kernel_size, bias=False, stride=stride
        )
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: torch.Tensor):
        """Forward.

        Args:
        input (torch.Tensor): spectrum [Batch, F, T]
        """

        wav = self.convtrans1d(input.permute(0,2,1))
        wav = wav.squeeze(1)

        return wav

    def forward_streaming(self, input_frame: torch.Tensor):
        return self.forward(input_frame)

    def streaming_merge(self, chunks: torch.Tensor):
        """streaming_merge. It merges the frame-level processed audio chunks
        in the streaming *simulation*. It is noted that, in real applications,
        the processed audio should be sent to the output channel frame by frame.
        You may refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
        """
        hop_size = self.stride
        frame_size = self.kernel_size

        num_chunks = len(chunks)
        batch_size = chunks[0].shape[0]
        audio_len = (
            int(hop_size * num_chunks + frame_size - hop_size)
        )

        output = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )

        for i, chunk in enumerate(chunks):
            output[:, i * hop_size : i * hop_size + frame_size] += chunk

        return output

class Custom_Decoder(nn.Module):
    def __init__(self, E, W):
        super(Custom_Decoder, self).__init__()
        # Hyper-parameter
        self.E, self.W = E, W
        # Components
        self.basis_signals = nn.Linear(E, W, bias=False)

    def forward(self, dec_input):
        """
        Args:
            dec_input: [B, C, E, L]
        Returns:
            est_source: [B, C, T]
        """
        # D = W * M
        #print(mixture_w.shape)
        #print(est_mask.shape)
        # S = DV

        est_source = self.basis_signals(dec_input)  # [B, C, L, W]
        est_source = overlap_and_add(est_source, self.W//2) # B x C x T
        return est_source



class SKIM(nn.Module):
    def __init__(
            self,
            input_dim: int,
            kernel_size: int,
            causal: bool = True,
            num_spk: int = 2,
            predict_noise: bool = False,
            nonlinear: str = "relu",
            layer: int = 3,
            unit: int = 512,
            segment_size: int = 20,
            dropout: float = 0.0,
            mem_type: str = "hc",
            seg_overlap: bool = False,
    ):
        super().__init__()
        self.encoder = ConvEncoder(input_dim,kernel_size, kernel_size//2)  # [B T]-->[B N L]
        self.enc_LN = ChannelwiseLayerNorm(input_dim, shape="BTD")  # [B N L]-->[B N L]
        self.separator = SkiMSeparator( input_dim = input_dim,
          causal= causal,
          num_spk= num_spk,
          predict_noise= predict_noise,
          nonlinear =  nonlinear,
          layer = layer,
          unit = unit,
          segment_size = segment_size,
          dropout = dropout,
          mem_type = mem_type,
          seg_overlap = seg_overlap
        )
        self.num_spk = num_spk
        self.decoder = ConvDecoder(input_dim, kernel_size,kernel_size//2)

    def forward(self, input):
        """
        input: shape (batch, T)
        """

        # pass to a DPRNN
        # input = input.to(device)
        # mixture, rest = self.pad_input(input, self.window)
        # print('mixture.shape {}'.format(mixture.shape))
        mixture_w = self.encoder(input)  # B, E, L
        score_ = self.enc_LN(mixture_w)  # B, E, L
        # print('mixture_w.shape {}'.format(mixture_w.shape))
        masked, others = self.separator(score_)  # B, nspk, T, N
        est_source = [self.decoder(masked[i]) for i in range(self.num_spk)]
        # est_source = self.decoder(masked,ilens)  # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]

        # if rest > 0:
        #     est_source = est_source[:, :, :-rest]
        return torch.stack(est_source,dim=1)
class SKIM_Stream(nn.Module):
    def __init__(
            self,
            input_dim: int,
            kernel_size: int,
            causal: bool = True,
            num_spk: int = 2,
            predict_noise: bool = False,
            nonlinear: str = "relu",
            layer: int = 3,
            unit: int = 512,
            segment_size: int = 20,
            dropout: float = 0.0,
            mem_type: str = "hc",
            seg_overlap: bool = False,
    ):
        super().__init__()
        self.encoder = ConvEncoder(input_dim,kernel_size, kernel_size//2)  # [B T]-->[B N L]
        self.enc_LN = ChannelwiseLayerNorm(input_dim, shape="BTD")  # [B N L]-->[B N L]
        self.separator = SkiMSeparator( input_dim = input_dim,
          causal= causal,
          num_spk= num_spk,
          predict_noise= predict_noise,
          nonlinear =  nonlinear,
          layer = layer,
          unit = unit,
          segment_size = segment_size,
          dropout = dropout,
          mem_type = mem_type,
          seg_overlap = seg_overlap
        )
        self.num_spk = num_spk
        self.decoder = ConvDecoder(input_dim, kernel_size,kernel_size//2)
        self.streaming_states = None
    def forward(self, input):
        """
        input: shape (batch, T)
        """
        # pass to a DPRNN
        # input = input.to(device)
        # mixture, rest = self.pad_input(input, self.window)
        # print('mixture.shape {}'.format(mixture.shape))
        frame_feature = self.encoder.forward_streaming(input)

        # frame_separated: list of num_spk [(B, 1, F)]
        (
            frame_separated,
            self.streaming_states,

        ) = self.separator.forward_streaming(
            frame_feature, self.streaming_states
        )

        # frame_separated: list of num_spk [(B, frame_size)]
        waves = [self.decoder.forward_streaming(f) for f in frame_separated]

        # if rest > 0:
        #     est_source = est_source[:, :, :-rest]

        return waves
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            cLN_y: [M, N, K]
        """

        assert y.dim() == 3

        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            cLN_y = cLN_y.transpose(1, 2).contiguous()

        return cLN_y

if __name__ == '__main__':
    main()