import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        # shape (1, seq_len, d_model)
        return self.pe[:, :seq_len]


class RelativeMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len, dropout):
        d_head, remainder = divmod(d_model, num_heads)
        assert remainder == 0, "`d_model` should be divisible by `num_heads`"
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(d_head)

        self.norm = nn.LayerNorm(d_model)
        #self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # process key, query, values in one matmul
        self.kqv_proj = nn.Linear(d_model, 3 * d_model)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.u_bias = nn.Parameter(torch.Tensor(num_heads, d_head))
        self.v_bias = nn.Parameter(torch.Tensor(num_heads, d_head))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # x.shape == (batch_size, seq_len, d_model)
        #pos_embedding = self.positional_encoding(seq_len)
        # pos_embedding.shape == (1, seq_len, d_model)
        #pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        # pos_embedding.shape == (batc_size, seq_len, d_model)

        kqv = self.kqv_proj(self.norm(x))
        # kqv.shape == (batch_size, seq_len, 3 * d_model)
        key, query, value = torch.chunk(kqv, 3, dim=-1)
        # shape == (batch_size, seq_len, d_model)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)
        # pos_embedding = (
        #     self.pos_proj(pos_embedding)
        #     .view(batch_size, seq_len, self.num_heads, -1)
        #     .permute(0, 2, 3, 1)
        # )
        # pos_embedding.shape == (batch_size, num_head, d_head, seq_len)

        content_score = torch.matmul((query + self.u_bias.unsqueeze(1)), key)
        #pos_score = torch.matmul((query + self.v_bias.unsqueeze(1)), pos_embedding)
        #pos_score = skew(pos_score)
        #score = self.scale * (content_score + pos_score)
        score = self.scale * content_score
        # score.shape == (batch_size, num_heads, seq_len, seq_len)

        attn = F.softmax(score, -1)
        attn = self.attn_dropout(attn)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        out = torch.matmul(attn, value).transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.out_proj(out)
        out = self.out_dropout(x)

        return out


def skew(QEr):
    # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
    padded = F.pad(QEr, (1, 0))
    # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
    batch_size, num_heads, num_rows, num_cols = padded.shape
    reshaped = padded.view(batch_size, num_heads, num_cols, num_rows)
    # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
    Srel = reshaped[:, :, 1:, :].view_as(QEr)
    # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
    return Srel
