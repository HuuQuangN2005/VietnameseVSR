import math

import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        dilation,
        dropout,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        conv1 = nn.Conv1d(
            input_size,
            output_size,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        conv2 = nn.Conv1d(
            output_size,
            output_size,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self._init_conv(conv1)
        self._init_conv(conv2)

        self.conv1 = weight_norm(conv1)
        self.conv2 = weight_norm(conv2)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.residual = None
        if input_size != output_size:
            self.residual = nn.Conv1d(input_size, output_size, kernel_size=1)
            self._init_conv(self.residual)

    @staticmethod
    def _init_conv(conv):
        nn.init.xavier_uniform_(conv.weight, gain=math.sqrt(2.0))
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def forward(self, inputs, mask=None):
        hidden = self.dropout1(self.act1(self.conv1(inputs)))

        if mask is not None:
            hidden = hidden * mask

        hidden = self.dropout2(self.act2(self.conv2(hidden)))

        residual = inputs if self.residual is None else self.residual(inputs)

        outputs = self.act3(hidden + residual)

        if mask is not None:
            outputs = outputs * mask

        return outputs


class TCN(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=3,
        dilations=None,
        dropout=0.1,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number.")

        if dilations is None:
            dilations = [2**index for index in range(len(num_channels))]

        if len(dilations) != len(num_channels):
            raise ValueError("dilations and num_channels must have equal lengths.")

        layers = []

        for index, output_size in enumerate(num_channels):
            input_size = num_inputs if index == 0 else num_channels[index - 1]

            layers.append(
                TemporalBlock(
                    input_size=input_size,
                    output_size=output_size,
                    kernel_size=kernel_size,
                    dilation=dilations[index],
                    dropout=dropout,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, mask=None):
        hidden = inputs.transpose(1, 2)
        temporal_mask = None if mask is None else mask.unsqueeze(1)

        if temporal_mask is not None:
            hidden = hidden * temporal_mask

        for layer in self.layers:
            hidden = layer(hidden, temporal_mask)

        return hidden.transpose(1, 2)
