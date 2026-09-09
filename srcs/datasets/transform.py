# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/datamodule/transforms.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

import random

import torch
import torchvision


class ScaleVideo(torch.nn.Module):
    def forward(self, video):
        return video.float().div(255.0)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        widths = torch.randint(0, self.window, size=(n_mask,))
        for width in widths:
            width = int(width)
            if width <= 0 or length - width <= 0:
                continue
            t_start = random.randrange(0, length - width)
            cloned[t_start : t_start + width] = 0
        return cloned


class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                ScaleVideo(),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                ScaleVideo(),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        else:
            raise ValueError("subset must be train, val, or test.")

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)
