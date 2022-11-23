from torch import nn

from .layers import Swish, Transpose


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        expand_dim = d_model * expansion_factor
        self.model = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expand_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(expand_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.model(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        assert (
            kernel_size % 2
        ), f"Expected `kernel_size` to be odd, but got {kernel_size}"
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(d_model),
            # pointwise conv is same as linear, but matmul is quicker
            # see https://stackoverflow.com/questions/55576314
            nn.Linear(d_model, d_model * 2),
            Transpose(1, 2),
            nn.GLU(dim=1),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model,
            ),
            nn.BatchNorm1d(d_model),
            Swish(),
            Transpose(1, 2),
            # same logic as above
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.model(x)

