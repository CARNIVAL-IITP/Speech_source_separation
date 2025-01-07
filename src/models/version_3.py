from typing import Any, Dict, Tuple, Optional
import time
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from .xLSTM.xlstm_vision import ViLBlock

def neg_si_sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    si_snr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_snr_val.view(batch_size, -1), dim=1)


class MCSS_V3(nn.Module):
    def __init__(
            self,
            n_channel: int = 8,
            n_speaker: int = 2,
            n_fft: int = 512,
            n_overlap: int = 256,
            ref_channel: int = 0,
            arch: str = "NB_BLSTM",  # could also be NBC, NBC2
            arch_kwargs: Dict[str, Any] = dict(),
    ):
        super().__init__()
        # self.separation_model: nn.Module = BLSTM2_FC1(input_size=n_channel * 2, output_size=n_speaker * 2, **arch_kwargs)
        self.separation_model: nn.Module = XLSTM2_FC1(input_size=n_channel * 2, output_size=n_speaker * 2, **arch_kwargs)


        self.register_buffer('window', torch.hann_window(n_fft), False)  # self.window, will be moved to self.device at training time
        self.n_fft = n_fft
        self.n_overlap = n_overlap
        self.ref_channel = ref_channel
        self.n_channel = n_channel
        self.n_speaker = n_speaker

    def forward(self, x: Tensor) -> Tensor:
        # STFT
        B, C, T = x.shape
        x = x.reshape((B * C, T))
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_overlap, window=self.window, win_length=self.n_fft, return_complex=True)
        X = X.reshape((B, C, X.shape[-2], X.shape[-1]))  # (batch, channel, freq, time frame)
        X = X.permute(0, 2, 3, 1)  # (batch, freq, time frame, channel)

        # normalization by using ref_channel
        F, TF = X.shape[1], X.shape[2]
        Xr = X[..., self.ref_channel].clone()  # copy
        XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of X
        X[:, :, :, :] /= (XrMM.reshape(B, F, 1, 1) + 1e-8)

        # to real
        X = torch.view_as_real(X)  # [B, F, T, C, 2]
        X = X.reshape(B * F, TF, C * 2)

        # network processing
        output = self.separation_model(X)
        # to complex
        output = output.reshape(B, F, TF, self.n_speaker, 2).contiguous()
        output = torch.view_as_complex(output)  # [B, F, TF, S]

        # inverse normalization
        Ys_hat = torch.empty(size=(B, self.n_speaker, F, TF), dtype=torch.complex64, device=output.device)
        XrMM = torch.unsqueeze(XrMM, dim=2).expand(-1, -1, TF)
        for spk in range(self.n_speaker):
            Ys_hat[:, spk, :, :] = output[:, :, :, spk] * XrMM[:, :, :]

        # iSTFT with frequency binding
        ys_hat = torch.istft(Ys_hat.reshape(B * self.n_speaker, F, TF), n_fft=self.n_fft, hop_length=self.n_overlap, window=self.window, win_length=self.n_fft, length=T)
        ys_hat = ys_hat.reshape(B, self.n_speaker, T)
        return ys_hat

    def forward_streaming(self,input_frames: Tensor,states=None):
        B, C, T = input_frames.shape
        x = input_frames.reshape((B * C, T))
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_overlap, window=self.window, win_length=self.n_fft,
                       return_complex=True)
        X = X.reshape((B, C, X.shape[-2], X.shape[-1]))  # (batch, channel, freq, time frame)
        X = X.permute(0, 2, 3, 1)  # (batch, freq, time frame, channel)

        # normalization by using ref_channel
        F, TF = X.shape[1], X.shape[2]
        Xr = X[..., self.ref_channel].clone()  # copy
        XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of X
        X[:, :, :, :] /= (XrMM.reshape(B, F, 1, 1) + 1e-8)

        # to real
        X = torch.view_as_real(X)  # [B, F, T, C, 2]
        X = X.reshape(B * F, TF, C * 2)

        output = self.arch.forward_stream(X,states=states)

        output = output.reshape(B, F, TF, self.n_speaker, 2).contiguous()
        output = torch.view_as_complex(output)  # [B, F, TF, S]

        # inverse normalization
        Ys_hat = torch.empty(size=(B, self.n_speaker, F, TF), dtype=torch.complex64, device=output.device)
        XrMM = torch.unsqueeze(XrMM, dim=2).expand(-1, -1, TF)
        for spk in range(self.n_speaker):
            Ys_hat[:, spk, :, :] = output[:, :, :, spk] * XrMM[:, :, :]

        # iSTFT with frequency binding
        ys_hat = torch.istft(Ys_hat.reshape(B * self.n_speaker, F, TF), n_fft=self.n_fft, hop_length=self.n_overlap,
                             window=self.window, win_length=self.n_fft, length=T)
        ys_hat = ys_hat.reshape(B, self.n_speaker, T)
        return ys_hat

class BLSTM2_FC1(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation: Optional[str] = "",
            hidden_size: Tuple[int, int] = (256, 128),
            n_repeat_last_lstm: int = 1,
            dropout: Optional[float] = None,
    ):
        """Two layers of BiLSTMs & one fully connected layer

        Args:
            input_size: the input size for the features of the first BiLSTM layer
            output_size: the output size for the features of the last BiLSTM layer
            hidden_size: the hidden size of each BiLSTM layer. Defaults to (256, 128).
        """

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout = dropout
        self.norm1 = GlobalLayerNorm(self.hidden_size[0], shape="BTD")
        self.norm2 = GlobalLayerNorm(self.hidden_size[1], shape="BTD")
        self.blstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0], batch_first=True, bidirectional=False)  # type:ignore
        self.blstm2 = nn.LSTM(input_size=self.hidden_size[0] , hidden_size=self.hidden_size[1], batch_first=True, bidirectional=False, num_layers=n_repeat_last_lstm)  # type:ignore
        if dropout is not None:
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

        self.linear = nn.Linear(self.hidden_size[1], self.output_size)  # type:ignore
        if self.activation is not None and len(self.activation) > 0:  # type:ignore
            self.activation_func = getattr(nn, self.activation)()  # type:ignore
        else:
            self.activation_func = None

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x: shape [batch, seq, input_size]

        Returns:
            Tensor: shape [batch, seq, output_size]
        """

        x, _ = self.blstm1(x)
        if self.dropout:
            x = self.dropout1(x)
        x = x + self.norm1(x)
        x, _ = self.blstm2(x)
        if self.dropout:
            x = self.dropout2(x)
        x = x + self.norm2(x)
        if self.activation_func is not None:
            y = self.activation_func(self.linear(x))
        else:
            y = self.linear(x)

        return y
class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

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

    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            gLN_y: [M, N, K]
        """
        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = y.mean(dim=(1, 2), keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 2), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta

        if self.shape == "BTD":
            gLN_y = gLN_y.transpose(1, 2).contiguous()
        return gLN_y


class XLSTM2_FC1(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation: Optional[str] = "",
            hidden_size: Tuple[int, int] = (256, 128),
            n_repeat_last_lstm: int = 1,
            dropout: Optional[float] = None,
    ):
        """Two layers of BiLSTMs & one fully connected layer

        Args:
            input_size: the input size for the features of the first BiLSTM layer
            output_size: the output size for the features of the last BiLSTM layer
            hidden_size: the hidden size of each BiLSTM layer. Defaults to (256, 128).
        """

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout = dropout
        self.norm1 = GlobalLayerNorm(self.hidden_size[0], shape="BTD")
        self.norm2 = GlobalLayerNorm(self.hidden_size[1], shape="BTD")

        # self.xlstm1 = CustomxLSTM(self.input_size, self.hidden_size[0], batch_first=True, num_layers=1)
        self.xlstm1 = ViLBlock(self.input_size)
        # self.xlstm2 = CustomxLSTM(self.hidden_size[0], self.hidden_size[1], batch_first=True, num_layers=1)
        self.xlstm2 = ViLBlock(self.input_size)
        if dropout is not None:
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

        # self.linear = nn.Linear(self.hidden_size[1], self.output_size)  # type:ignore
        self.linear = nn.Linear(self.input_size, self.output_size)  # type:ignore
        if self.activation is not None and len(self.activation) > 0:  # type:ignore
            self.activation_func = getattr(nn, self.activation)()  # type:ignore
        else:
            self.activation_func = None

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x: shape [batch, seq, input_size]

        Returns:
            Tensor: shape [batch, seq, output_size]
        """

        x = self.xlstm1(x)
        if self.dropout:
            x = self.dropout1(x)
        # x = x + self.norm1(x)
        x = self.xlstm2(x)
        if self.dropout:
            x = self.dropout2(x)
        # x = x + self.norm2(x)
        if self.activation_func is not None:
            y = self.activation_func(self.linear(x))
        else:
            y = self.linear(x)

        return y
