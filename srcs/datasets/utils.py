from collections.abc import Sequence

import torch
from dotenv import find_dotenv, load_dotenv


def setup_hf():
    load_dotenv(find_dotenv())


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
