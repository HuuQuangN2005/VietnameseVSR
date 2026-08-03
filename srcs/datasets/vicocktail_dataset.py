import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torchvision
from datasets import Dataset, DatasetDict
from torchcodec.decoders import VideoDecoder

from srcs.nlp import TextTransform


def source_id_from_sample_id(sample_id: object) -> str:
    if isinstance(sample_id, (bytes, bytearray, memoryview)):
        sample_id = bytes(sample_id).decode("utf-8")
    else:
        sample_id = str(sample_id)

    parts = sample_id.rsplit("_", 4)
    if len(parts) != 5 or not parts[0]:
        raise ValueError(f"Invalid ViCocktail sample ID: {sample_id!r}")

    return parts[0]


def split_train_validation(
    train_dataset: Dataset, validation_fraction: float = 0.03, seed: int = 42
) -> tuple[Dataset, Dataset]:
    if not isinstance(train_dataset, Dataset):
        raise TypeError("train_dataset must be a Hugging Face Dataset.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if "sample_id" not in train_dataset.column_names:
        raise KeyError("train_dataset must contain a sample_id column.")
    if len(train_dataset) == 0:
        raise ValueError("train_dataset must not be empty.")

    source_ids = [
        source_id_from_sample_id(sample_id) for sample_id in train_dataset["sample_id"]
    ]
    source_counts = Counter(source_ids)

    if len(source_counts) < 2:
        raise ValueError(
            "At least two source videos are required to create a validation split."
        )

    shuffled_sources = list(source_counts)
    random.Random(seed).shuffle(shuffled_sources)
    target_validation_size = max(1, round(len(train_dataset) * validation_fraction))

    validation_sources = set()
    validation_size = 0

    for source_id in shuffled_sources:
        if validation_size >= target_validation_size:
            break

        validation_sources.add(source_id)
        validation_size += source_counts[source_id]

    train_indices = []
    validation_indices = []

    for index, source_id in enumerate(source_ids):
        if source_id in validation_sources:
            validation_indices.append(index)
        else:
            train_indices.append(index)

    if not train_indices or not validation_indices:
        raise RuntimeError(
            "The grouped split produced an empty train or validation dataset."
        )

    return (train_dataset.select(train_indices), train_dataset.select(validation_indices))


def add_validation_split(
    dataset: DatasetDict, validation_fraction: float = 0.03, seed: int = 42
) -> DatasetDict:
    if not isinstance(dataset, DatasetDict):
        raise TypeError("dataset must be a Hugging Face DatasetDict.")
    if "train" not in dataset:
        raise KeyError("dataset must contain a train split.")
    if "validation" in dataset:
        raise ValueError("dataset already contains a validation split.")

    train_split, validation_split = split_train_validation(
        dataset["train"], validation_fraction=validation_fraction, seed=seed
    )
    output = DatasetDict({"train": train_split, "validation": validation_split})

    for split_name, split_dataset in dataset.items():
        if split_name != "train":
            output[split_name] = split_dataset

    return output


def load_video(path, start_time=0, end_time=None):
    """
    rtype: torch, T x C x H x W
    """
    video_decoder = VideoDecoder(path, dimension_order="NCHW")
    if end_time is None:
        end_time = video_decoder.metadata.duration_seconds
    vid = video_decoder.get_frames_played_in_range(start_time, end_time).data
    return vid


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


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
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)


# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "label" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


@dataclass
class DataCollator:
    text_transform: TextTransform
    video_transform: VideoTransform

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        samples = []
        for feature in features:
            if "start_time" in feature and "end_time" in feature:
                video = load_video(
                    feature["video"], feature["start_time"], feature["end_time"]
                )
            else:
                video = load_video(feature["video"])

            video = self.video_transform(video)

            if "label" in feature:
                label = feature["label"]
                if isinstance(label, bytes):
                    label = label.decode("utf-8")
                label = self.text_transform.encode(label)
                samples.append({"video": video, "label": label})
            else:
                samples.append({"video": video})

        batch = collate_pad(samples)

        return batch
