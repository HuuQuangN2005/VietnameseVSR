# Source (modified):
# https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
# License: see THIRD_PARTY_LICENSES.md.

import torch.nn as nn


def conv3x3(input_size, output_size, stride=1):
    return nn.Conv2d(
        input_size, output_size, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, input_size, output_size, stride=1):
        super().__init__()
        self.conv1 = conv3x3(input_size, output_size, stride)
        self.norm1 = nn.BatchNorm2d(output_size)
        self.activation1 = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(output_size, output_size)
        self.norm2 = nn.BatchNorm2d(output_size)
        self.activation2 = nn.SiLU(inplace=True)

        self.residual = None
        if stride != 1 or input_size != output_size:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    input_size, output_size, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(output_size),
            )

    def forward(self, inputs):
        residual = inputs if self.residual is None else self.residual(inputs)
        hidden = self.activation1(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))

        return self.activation2(hidden + residual)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 64
        self.layers = nn.Sequential(
            self._make_layer(64, blocks=2),
            self._make_layer(128, blocks=2, stride=2),
            self._make_layer(256, blocks=2, stride=2),
            self._make_layer(512, blocks=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, output_size, blocks, stride=1):
        layers = [BasicBlock(self.input_size, output_size, stride)]
        self.input_size = output_size
        layers.extend(BasicBlock(output_size, output_size) for _ in range(1, blocks))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        hidden = self.layers(inputs)
        return self.pool(hidden).flatten(1)


class VideoResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNet18()

    def forward(self, videos):
        batch_size = videos.size(0)
        hidden = self.stem(videos.transpose(1, 2))
        time = hidden.size(2)
        hidden = hidden.transpose(1, 2).flatten(0, 1)
        hidden = self.resnet(hidden)

        return hidden.view(batch_size, time, -1)


def video_resnet():
    return VideoResNet()
