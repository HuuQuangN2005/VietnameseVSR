import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0


class VideoShuffleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 1024

        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                24,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

        network = shufflenet_v2_x1_0(weights=None)

        self.layers = nn.Sequential(
            network.stage2,
            network.stage3,
            network.stage4,
            network.conv5,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, videos):
        batch_size = videos.size(0)
        hidden = self.stem(videos.transpose(1, 2))
        time = hidden.size(2)
        hidden = hidden.transpose(1, 2).flatten(0, 1)
        hidden = self.layers(hidden)
        hidden = self.pool(hidden).flatten(1)

        return hidden.view(batch_size, time, -1)


def video_shufflenet():
    return VideoShuffleNet()
