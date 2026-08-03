import argparse
import os
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from srcs.datasets.utils import setup_hf

HF_CONFIG = setup_hf()

from datasets import load_dataset


def get_vicocktail():
    return load_dataset(
        "nguyenvulebinh/ViCocktail",
        streaming=False,
        cache_dir=HF_CONFIG["HF_DATASETS_CACHE"],
    )


DATASETS = {"vicocktail": get_vicocktail}


def download(dataset_names: list[str]) -> bool:
    selected_names = (
        list(DATASETS)
        if not dataset_names or "all" in dataset_names
        else list(dict.fromkeys(dataset_names))
    )

    for dataset_name in selected_names:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        print(f"Downloading {dataset_name}")
        DATASETS[dataset_name]()
        print(f"Downloaded {dataset_name}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets into the project Hugging Face cache."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=[*DATASETS, "all"],
        default=["all"],
        help="Dataset names to download, or 'all' to download every dataset.",
    )
    args = parser.parse_args()
    download(args.dataset)


if __name__ == "__main__":
    main()
