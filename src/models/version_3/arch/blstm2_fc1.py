import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


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
        self.norm1 = GlobalLayerNorm(self.hidden_size[0])
        self.norm2 = GlobalLayerNorm(self.hidden_size[1])
        self.blstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0], batch_first=True, bidirectional=False)  # type:ignore
        self.blstm2 = nn.LSTM(input_size=self.hidden_size[0] , hidden_size=self.hidden_size[1], batch_first=True, bidirectional=False, num_layers=n_repeat_last_lstm)  # type:ignore
        if dropout is not None:
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)
        
        self.linear = nn.Linear(self.hidden_size[1] , self.output_size)  # type:ignore
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
        x = self.norm1(x)
        x, _ = self.blstm2(x)
        if self.dropout:
            x = self.dropout2(x)
        x = self.norm2(x)
        if self.activation_func is not None:
            y = self.activation_func(self.linear(x))
        else:
            y = self.linear(x)

        return y

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, 1,channel_size))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-7, 0.5) + self.beta
        return gLN_y

if __name__ == '__main__':
    x = torch.randn(size=(2056, 251, 8))
    NBSS_with_NB_BLSTM = BLSTM2_FC1(8,4)
    ys_hat = NBSS_with_NB_BLSTM(x)
    # neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    # print(ys_hat.shape, neg_sisdr_loss.mean())
