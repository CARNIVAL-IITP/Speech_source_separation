from torch import nn

from .attention import RelativeMultiHeadSelfAttention
from .modules import ConvolutionModule, FeedForwardModule


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model=256,
        num_heads=4,
        max_len=512,
        expansion_factor=4,
        kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.attn = RelativeMultiHeadSelfAttention(d_model, num_heads, max_len, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        # half step residual connection
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.norm(x)
        return x


class ConformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        num_blocks=6,
        d_model=256,
        num_heads=4,
        max_len=512,
        expansion_factor=4,
        kernel_size=31,
        dropout=0.1,
    ):
        block = ConformerBlock(
            d_model, num_heads, max_len, expansion_factor, kernel_size, dropout
        )
        super().__init__(block, num_blocks)
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

