# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/dataset/av_dataset.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

import random
from collections import Counter
from dataclasses import dataclass

from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset

from srcs.datasets.transform import VideoTransform, load_video
from srcs.datasets.utils import pad_seq, setup_hf, to_text
from srcs.spm.text_transofm import TextTransform

HF_NAME = "nguyenvulebinh/ViCocktail"


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
        split_dataset = dataset.train_test_split(test_size=validation_size, seed=seed)
        return split_dataset["train"], split_dataset["test"]

    source_ids = [_source(item) for item in dataset["sample_id"]]
    source_counts = Counter(source_ids)
    if len(source_counts) < 2:
        split_dataset = dataset.train_test_split(test_size=validation_size, seed=seed)
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
        target_indices = validation_indices if key in validation_keys else train_indices
        target_indices.append(index)

    if not train_indices or not validation_indices:
        split_dataset = dataset.train_test_split(test_size=validation_size, seed=seed)
        return split_dataset["train"], split_dataset["test"]

    return dataset.select(train_indices), dataset.select(validation_indices)


def _clean(dataset):
    columns = [name for name in ("__key__", "__url__") if name in dataset.column_names]
    return dataset.remove_columns(columns) if columns else dataset


def _add_video_length(dataset):
    if "video_length" in dataset.column_names:
        return dataset
    if "length" not in dataset.column_names:
        raise ValueError("The dataset must contain a length column.")

    video_lengths = []
    for value in dataset["length"]:
        video_lengths.append(int(to_text(value)))
    return dataset.add_column("video_length", video_lengths)


def load_vicocktail(
    train_fraction=1.0,
    validation_fraction=1.0,
    test_fraction=1.0,
    validation_size=0.03,
    seed=42,
    splits=("train", "val", "test"),
):
    setup_hf()
    splits = tuple(splits)

    if not set(splits).issubset({"train", "val", "test"}):
        raise ValueError("splits can only contain train, val, and test.")

    output = DatasetDict()

    if "train" in splits or "val" in splits:
        train_source = hf_load_dataset(HF_NAME, split="train", streaming=False)
        clean_dataset = _add_video_length(_clean(train_source))
        train_dataset, validation_dataset = _split_validation(
            clean_dataset, validation_size, seed
        )

        if "train" in splits:
            output["train"] = _select_fraction(train_dataset, train_fraction, seed)

        if "val" in splits:
            output["val"] = _select_fraction(
                validation_dataset, validation_fraction, seed
            )

    if "test" in splits:
        test_dataset = hf_load_dataset(HF_NAME, split="test", streaming=False)
        output["test"] = _select_fraction(
            _add_video_length(_clean(test_dataset)), test_fraction, seed
        )
    return output


@dataclass
class Collator:
    text_transform: TextTransform
    split: str = "train"

    def __post_init__(self):
        self.video_transform = VideoTransform(self.split)

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
