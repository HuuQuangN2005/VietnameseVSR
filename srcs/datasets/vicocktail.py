from datasets import DatasetDict, load_dataset

from srcs.datasets.utils import (
    add_video_length,
    clean_dataset,
    filter_analysable,
    select_fraction,
)

DS_NAME = "nguyenvulebinh/ViCocktail"


def load_vicocktail(
    split="all",
    seed: int = 42,
    fraction: float = 1.0,
    val_size: float = 0.1,
    apply_filter: bool = True,
):
    assert 0 < val_size < 1
    assert 0 < fraction <= 1
    assert split in ("train", "test", "all")

    outputs = DatasetDict()

    if split in ("train", "all"):
        train_dataset = load_dataset(DS_NAME, split="train", streaming=False)
        train_dataset = add_video_length(clean_dataset(train_dataset))
        train_dataset = train_dataset.train_test_split(test_size=val_size, seed=seed)

        train_split = train_dataset["train"]
        val_split = train_dataset["test"]

        if apply_filter:
            train_split = filter_analysable(train_split)
            val_split = filter_analysable(val_split)

        outputs["train"] = select_fraction(train_split, fraction, seed)
        outputs["val"] = select_fraction(val_split, fraction, seed)

    if split in ("test", "all"):
        test_dataset = load_dataset(DS_NAME, split="test", streaming=False)
        test_dataset = add_video_length(clean_dataset(test_dataset))

        if apply_filter:
            test_dataset = filter_analysable(test_dataset)

        outputs["test"] = select_fraction(test_dataset, fraction, seed)

    return outputs
