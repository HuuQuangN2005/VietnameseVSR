import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)

        frequencies = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )

        encoding = torch.zeros(max_length, hidden_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].size(1)]
        )

        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs):
        if inputs.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Input length {inputs.size(1)} exceeds the maximum positional "
                f"encoding length {self.encoding.size(1)}."
            )

        encoding = self.encoding[:, : inputs.size(1)].to(dtype=inputs.dtype)

        return self.dropout(inputs + encoding)
