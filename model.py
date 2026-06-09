import math
import torch
import torch.nn as nn
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):

        seq_len = x.size(1)

        mask = torch.triu(
            torch.ones(seq_len, seq_len),
            diagonal=1
        ).bool().to(x.device)

        out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=mask
        )

        return out
class GPTBlock(nn.Module):

    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()

        self.attn = MaskedMultiHeadAttention(
            d_model,
            num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        x = self.norm1(
            x + self.attn(x)
        )

        x = self.norm2(
            x + self.ff(x)
        )

        return x
class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        ff_dim=256,
        num_layers=4,
        max_len=100
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.pos_encoding = PositionalEncoding(
            d_model,
            max_len
        )

        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    d_model,
                    num_heads,
                    ff_dim
                )
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x)

        return self.fc(x)