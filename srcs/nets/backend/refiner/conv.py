import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinerConv(nn.Module):
    def __init__(self, channels, kernel_size=31, dropout=0.1, act="gelu", bias=True):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0

        self.pre_norm = nn.LayerNorm(channels, elementwise_affine=bias)

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, bias=bias)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )

        self.norm = nn.BatchNorm1d(channels, affine=bias)
        self.activation = nn.GELU() if act == "gelu" else nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask

        h = self.pre_norm(x).transpose(1, 2)  # [B, C, T]
        h = F.glu(self.pointwise_conv1(h), dim=1)
        h = self.activation(self.norm(self.depthwise_conv(h)))
        h = self.pointwise_conv2(h).transpose(1, 2)  # [B, T, C]

        out = x + self.dropout(h)
        if mask is not None:
            out = out * mask

        return out
