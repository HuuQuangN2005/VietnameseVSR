# Source (modified):
# https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
# License: see THIRD_PARTY_LICENSES.md.
# Kept structurally identical to the upstream ShuffleNetV2 so that the released
# LRW checkpoints load without any key remapping.

import torch
import torch.nn as nn


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    return x.view(batchsize, -1, height, width)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super().__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            self.banch2 = nn.Sequential(
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.benchmodel == 1:
            x1 = x[:, : (x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2) :, :, :]
            out = self._concat(x1, self.banch2(x2))
        else:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=2.0):
        super().__init__()
        assert input_size % 32 == 0, "Input size needs to be divisible by 32"

        self.stage_repeats = [4, 8, 4]

        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise ValueError(
                "Width multiplier should be in [0.5, 1.0, 1.5, 2.0]. "
                f"Current value: {width_mult}"
            )

        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        self.features = nn.Sequential(*features)
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        return self.classifier(x)


class VideoShuffleNet(nn.Module):
    def __init__(self, width_mult=1.0, relu_type="prelu"):
        super().__init__()
        network = ShuffleNetV2(input_size=96, width_mult=width_mult)
        self.output_size = network.stage_out_channels[-1]

        if relu_type == "prelu":
            activation = nn.PReLU(num_parameters=24)
        elif relu_type == "relu":
            activation = nn.ReLU(inplace=True)
        else:
            raise ValueError("relu_type must be 'prelu' or 'relu'.")

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1, 24, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(24),
            activation,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = nn.Sequential(network.features, network.conv_last, network.globalpool)

    def forward(self, videos):
        batch_size = videos.size(0)
        hidden = self.frontend3D(videos.transpose(1, 2))
        time = hidden.size(2)
        hidden = hidden.transpose(1, 2).flatten(0, 1)
        hidden = self.trunk(hidden)

        return hidden.view(batch_size, time, -1)


def read_visual_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)

    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break

    return {key.removeprefix("module."): value for key, value in state.items()}


def video_shufflenet(pretrained=None, width_mult=1.0, relu_type="prelu"):
    if pretrained is None:
        return VideoShuffleNet(width_mult, relu_type)

    state = read_visual_state(pretrained)
    detected = "prelu" if "frontend3D.2.weight" in state else "relu"
    network = VideoShuffleNet(width_mult, detected)

    visual = {
        key: value
        for key, value in state.items()
        if key.startswith(("frontend3D.", "trunk."))
    }
    if not visual:
        raise ValueError(f"No frontend3D./trunk. weights found in {pretrained}")

    missing, unexpected = network.load_state_dict(visual, strict=False)
    missing = [key for key in missing if key.startswith(("frontend3D.", "trunk."))]

    if missing or unexpected:
        raise ValueError(
            f"Visual weights do not match: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    return network
