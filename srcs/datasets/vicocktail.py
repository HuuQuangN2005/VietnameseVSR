# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/dataset/av_dataset.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

import random
from collections import Counter
from dataclasses import dataclass

import torch
import torchvision
from datasets import DatasetDict, concatenate_datasets
from datasets import load_dataset as hf_load_dataset
from torchcodec.decoders import VideoDecoder

from srcs.datasets.utils import pad_seq, setup_hf, to_text
from srcs.spm.text_transofm import TextTransform

DATASETS = ("vicocktail",)
HF_NAMES = {"vicocktail": "nguyenvulebinh/ViCocktail"}


def _source(sample_id):
    value = to_text(sample_id)
    parts = value.rsplit("_", 4)

    return parts[0] if len(parts) == 5 else value


def _select_fraction(dataset, fraction, seed):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Dataset fractions must be in (0, 1].")

    if fraction == 1.0:
        return dataset

    sample_count = max(1, round(len(dataset) * fraction))
    return dataset.shuffle(seed=seed).select(range(sample_count))


def _split_validation(dataset, validation_size, seed):
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be in (0, 1).")

    if "sample_id" not in dataset.column_names:
        split_dataset = dataset.train_test_split(
            test_size=validation_size, seed=seed
        )
        return split_dataset["train"], split_dataset["test"]

    source_ids = [_source(item) for item in dataset["sample_id"]]
    source_counts = Counter(source_ids)
    if len(source_counts) < 2:
        split_dataset = dataset.train_test_split(
            test_size=validation_size, seed=seed
        )
        return split_dataset["train"], split_dataset["test"]

    source_keys = list(source_counts)
    random.Random(seed).shuffle(source_keys)
    target_size = max(1, round(len(dataset) * validation_size))
    validation_keys = set()
    total = 0

    for key in source_keys:
        if total >= target_size:
            break

        validation_keys.add(key)
        total += source_counts[key]

    train_indices = []
    validation_indices = []

    for index, key in enumerate(source_ids):
        target_indices = (
            validation_indices if key in validation_keys else train_indices
        )
        target_indices.append(index)

    if not train_indices or not validation_indices:
        split_dataset = dataset.train_test_split(
            test_size=validation_size, seed=seed
        )
        return split_dataset["train"], split_dataset["test"]

    return dataset.select(train_indices), dataset.select(validation_indices)


def _clean(dataset):
    columns = [
        name for name in ("__key__", "__url__") if name in dataset.column_names
    ]
    return dataset.remove_columns(columns) if columns else dataset


def load_dataset(
    name="all",
    train_fraction=1.0,
    validation_fraction=1.0,
    test_fraction=1.0,
    validation_size=0.03,
    seed=42,
    cache_dir=None,
    splits=("train", "val", "test"),
):
    if name not in (*DATASETS, "all"):
        raise ValueError(f"Unsupported dataset: {name}")

    config = setup_hf(cache_dir)
    splits = tuple(splits)

    if not set(splits).issubset({"train", "val", "test"}):
        raise ValueError("splits can only contain train, val, and test.")

    names = DATASETS if name == "all" else (name,)

    train_datasets = []
    validation_datasets = []
    test_datasets = []

    for dataset_name in names:
        dataset = hf_load_dataset(
            HF_NAMES[dataset_name],
            streaming=False,
            cache_dir=config["HF_DATASETS_CACHE"],
        )
        if "train" in splits or "val" in splits:
            train_dataset, validation_dataset = _split_validation(
                _clean(dataset["train"]), validation_size, seed
            )

            if "train" in splits:
                train_datasets.append(
                    _select_fraction(train_dataset, train_fraction, seed)
                )

            if "val" in splits:
                validation_datasets.append(
                    _select_fraction(validation_dataset, validation_fraction, seed)
                )

        if "test" in splits and "test" in dataset:
            test_datasets.append(
                _select_fraction(_clean(dataset["test"]), test_fraction, seed)
            )

    output = DatasetDict()

    if train_datasets:
        output["train"] = (
            train_datasets[0]
            if len(train_datasets) == 1
            else concatenate_datasets(train_datasets)
        )

    if validation_datasets:
        output["val"] = (
            validation_datasets[0]
            if len(validation_datasets) == 1
            else concatenate_datasets(validation_datasets)
        )

    if test_datasets:
        output["test"] = (
            test_datasets[0]
            if len(test_datasets) == 1
            else concatenate_datasets(test_datasets)
        )
    return output


def load_video(video_source, start_time=0.0, end_time=None):
    if isinstance(video_source, dict):
        video_source = video_source.get("bytes") or video_source.get("path")

    decoder = VideoDecoder(video_source, dimension_order="NCHW")
    end_time = (
        decoder.metadata.duration_seconds if end_time is None else float(end_time)
    )

    return decoder.get_frames_played_in_range(float(start_time), end_time).data


class ScaleVideo(torch.nn.Module):
    def forward(self, video):
        return video.float().div(255.0)


class TimeMask(torch.nn.Module):
    def __init__(self, window=10, stride=25):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        output = x.clone()
        size = output.size(0)
        count = int((size + self.stride - 0.1) // self.stride)
        for width in torch.randint(0, self.window, (count,)).tolist():
            if width <= 0 or width >= size:
                continue
            start = random.randrange(0, size - width)
            output[start : start + width] = 0
        return output


class VideoTransform:
    def __init__(self, split="train", crop=88):
        crop_transform = (
            torchvision.transforms.RandomCrop(crop)
            if split == "train"
            else torchvision.transforms.CenterCrop(crop)
        )

        transforms = [
            ScaleVideo(),
            crop_transform,
            torchvision.transforms.Grayscale(),
        ]

        if split == "train":
            transforms.append(TimeMask())

        transforms.append(torchvision.transforms.Normalize(0.421, 0.165))
        self.pipeline = torch.nn.Sequential(*transforms)

    def __call__(self, video):
        return self.pipeline(video)


@dataclass
class Collator:
    text_transform: TextTransform
    split: str = "train"
    crop: int = 88

    def __post_init__(self):
        self.video_transform = VideoTransform(self.split, self.crop)

    def __call__(self, items):
        videos = []
        labels = []

        for item in items:
            start = item.get("start_time", 0.0)
            end = item.get("end_time")
            video = self.video_transform(load_video(item["video"], start, end))
            videos.append(video)

            if "label" in item:
                labels.append(self.text_transform.encode(to_text(item["label"])))

        video_batch, video_lengths = pad_seq(videos)
        output = {"videos": video_batch, "video_lengths": video_lengths}

        if labels:
            label_batch, label_lengths = pad_seq(labels, -1)
            output["labels"] = label_batch
            output["label_lengths"] = label_lengths

        return output
