import torch
import torch.nn as nn
import torch.nn.functional as F

from srcs.nets.backend.transformer.layer_norm import LayerNorm


class RefinerConv(nn.Module):
    def __init__(self, channels, kernel_size=7, dropout=0.1, act="silu", bias=False):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0

        self.pre_norm = LayerNorm(channels)

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, bias=bias)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )

        self.norm = LayerNorm(channels)
        if act == "gelu":
            self.activation = nn.GELU()
        elif act == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError("act must be 'gelu' or 'silu'.")

        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask

        h = self.pre_norm(x)
        if mask is not None:
            h = h * mask

        h = h.transpose(1, 2)  # [B, C, T]
        h = F.glu(self.pointwise_conv1(h), dim=1)
        h = self.depthwise_conv(h).transpose(1, 2)
        h = self.activation(self.norm(h))
        if mask is not None:
            h = h * mask

        h = self.pointwise_conv2(h.transpose(1, 2)).transpose(1, 2)

        out = x + self.dropout(h)
        if mask is not None:
            out = out * mask

        return out
