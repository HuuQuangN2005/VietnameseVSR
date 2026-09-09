# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/dataset/av_dataset.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

from datasets import DatasetDict, load_dataset

from srcs.datasets.utils import (
    add_video_length,
    clean_dataset,
    filter_dataset,
    get_word_frequencies,
    load_valid_words,
    select_fraction,
)

DS_NAME = "nguyenvulebinh/ViCocktail"


def load_vicocktail(
    split="all",
    seed: int = 42,
    fraction: float = 1.0,
    val_size: float = 0.1,
    min_word_frequency: int = 5,
    apply_filter: bool = True,
):
    assert 0 < val_size < 1
    assert 0 < fraction <= 1
    assert min_word_frequency >= 1
    assert split in ("train", "test", "all")

    outputs = DatasetDict()

    if split in ("train", "all") or apply_filter:
        train_dataset = load_dataset(DS_NAME, split="train", streaming=False)
        train_dataset = add_video_length(clean_dataset(train_dataset))
        train_dataset = train_dataset.train_test_split(test_size=val_size, seed=seed)

        if apply_filter:
            word_frequencies = get_word_frequencies(train_dataset["train"])
            allowed_words = {
                word
                for word in load_valid_words()
                if word_frequencies.get(word, 0) >= min_word_frequency
            }

    if split in ("train", "all"):
        train_split = train_dataset["train"]
        validation_split = train_dataset["test"]

        if apply_filter:
            train_split = filter_dataset(train_split, allowed_words)
            validation_split = filter_dataset(validation_split, allowed_words)

        outputs["train"] = select_fraction(train_split, fraction, seed)
        outputs["val"] = select_fraction(validation_split, fraction, seed)

    if split in ("test", "all"):
        test_dataset = load_dataset(DS_NAME, split="test", streaming=False)
        test_dataset = add_video_length(clean_dataset(test_dataset))

        if apply_filter:
            test_dataset = filter_dataset(test_dataset, allowed_words)

        outputs["test"] = select_fraction(test_dataset, fraction, seed)

    return outputs
