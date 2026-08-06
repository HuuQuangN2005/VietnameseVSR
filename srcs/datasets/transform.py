# Source (modified): https://github.com/mpc001/auto_avsr/blob/main/datamodule/transforms.py
# License: Apache-2.0 (https://github.com/mpc001/auto_avsr/blob/main/LICENSE)

import random

import torch
import torchvision
from torchcodec.decoders import VideoDecoder


def load_video(video_source, start_time=0.0, end_time=None):
    if isinstance(video_source, dict):
        video_source = video_source.get("bytes") or video_source.get("path")

    decoder = VideoDecoder(video_source, dimension_order="NCHW")
    end_time = decoder.metadata.duration_seconds if end_time is None else float(end_time)

    return decoder.get_frames_played_in_range(float(start_time), end_time).data


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
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
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
