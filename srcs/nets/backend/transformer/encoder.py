import torch.nn as nn

from srcs.nets.backend.transformer.embedding import PositionalEncoding
from srcs.nets.backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        normalized = self.norm1(inputs)
        attention, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=None if mask is None else ~mask,
            need_weights=False,
        )
        hidden = inputs + self.dropout1(attention)
        outputs = hidden + self.dropout2(self.feed_forward(self.norm2(hidden)))

        if mask is not None:
            outputs = outputs * mask.unsqueeze(-1)

        return outputs


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        ffn_dim=1024,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        max_length=5000,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        self.position = PositionalEncoding(
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_length=max_length,
        )
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, inputs, mask=None):
        if mask is not None:
            mask = mask.bool()

        hidden = self.position(inputs)

        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1)

        for layer in self.layers:
            hidden = layer(hidden, mask)

        outputs = self.norm(hidden)

        if mask is not None:
            outputs = outputs * mask.unsqueeze(-1)

        return outputs
