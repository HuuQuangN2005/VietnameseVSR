import os
from collections.abc import Sequence

import torch
from dotenv import find_dotenv, load_dotenv


def root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_hf(cache_dir=None):
    load_dotenv(find_dotenv())
    base = cache_dir or os.path.join(root_dir(), "data", ".hf")
    cache = os.path.join(base, ".cache")
    config = {
        "HF_HOME": base,
        "HF_HUB_CACHE": os.path.join(cache, "hub"),
        "HF_DATASETS_CACHE": os.path.join(cache, "datasets"),
        "HF_ASSETS_CACHE": os.path.join(cache, "assets"),
        "HF_XET_CACHE": os.path.join(cache, "xet"),
    }
    token = os.getenv("HF_TOKEN")
    for key, value in config.items():
        os.makedirs(value, exist_ok=True)
        os.environ[key] = value
    if token:
        os.environ["HF_TOKEN"] = token
        config["HF_TOKEN"] = token
    return config


def to_text(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    return str(value)


def pad_seq(sequences: Sequence[torch.Tensor], padding_value=0.0):
    if not sequences:
        raise ValueError("sequences must not be empty")
    lengths = torch.tensor([item.size(0) for item in sequences], dtype=torch.long)
    max_length = int(lengths.max())
    shape = (len(sequences), max_length, *sequences[0].shape[1:])
    output = sequences[0].new_full(shape, padding_value)
    for index, item in enumerate(sequences):
        output[index, : item.size(0)] = item
    return output, lengths
