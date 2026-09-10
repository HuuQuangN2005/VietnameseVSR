import os
import torch

from collections import Counter
from collections.abc import Sequence
from torchcodec.decoders import VideoDecoder
from srcs.nlp.tokenizer import PhonemeTokenizer, WordTokenizer

VALID_WORD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "valid_word.txt"
)
word_tokenizer = WordTokenizer()
phoneme_tokenizer = PhonemeTokenizer()
DECODE_THREADS = 2


def get_word_frequencies(dataset):
    word_frequencies = Counter()

    for label in dataset["label"]:
        text = to_text(label)
        words = word_tokenizer.tokenize(text)

        for word in words:
            word_frequencies[word] += 1

    return word_frequencies


def load_valid_words(path=VALID_WORD_PATH):
    words = set()

    with open(path, encoding="utf-8") as file:
        for line in file:
            words.update(word_tokenizer.tokenize(line))

    if not words:
        raise ValueError("The valid word vocabulary must not be empty.")

    return words


def is_valid_label(label, allowed_words):
    text = to_text(label)
    words = word_tokenizer.tokenize(text)

    if not words:
        return False

    for word in words:
        if word not in allowed_words:
            return False

    return True


def filter_dataset(dataset, allowed_words):
    indices = []

    for index, label in enumerate(dataset["label"]):
        is_valid = is_valid_label(label, allowed_words)

        if is_valid:
            indices.append(index)

    return dataset.select(indices)


def is_analysable_label(label):
    words = word_tokenizer.tokenize(to_text(label))

    if not words:
        return False

    return all(phoneme_tokenizer.analyze(word)["is_valid"] for word in words)


def filter_analysable(dataset):
    indices = [
        index
        for index, label in enumerate(dataset["label"])
        if is_analysable_label(label)
    ]

    return dataset.select(indices)


def filter_by_length(dataset, max_frames):
    indices = [
        index
        for index, length in enumerate(dataset["video_length"])
        if int(length) <= max_frames
    ]

    return dataset.select(indices)


def load_video(video_source, start_time=0.0, end_time=None):
    if isinstance(video_source, dict):
        video_source = video_source.get("bytes") or video_source.get("path")

    decoder = VideoDecoder(
        video_source, dimension_order="NCHW", num_ffmpeg_threads=DECODE_THREADS
    )
    if end_time is None:
        end_time = decoder.metadata.duration_seconds
    else:
        end_time = float(end_time)

    return decoder.get_frames_played_in_range(float(start_time), end_time).data


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


def select_fraction(dataset, fraction, seed):
    if fraction == 1.0:
        return dataset

    count = max(1, round(len(dataset) * fraction))

    return dataset.shuffle(seed=seed).select(range(count))


def clean_dataset(dataset):
    columns = [name for name in ("__key__", "__url__") if name in dataset.column_names]
    return dataset.remove_columns(columns) if columns else dataset


def add_video_length(dataset):
    if "video_length" in dataset.column_names:
        return dataset

    if "length" not in dataset.column_names:
        raise ValueError("The dataset must contain a length column.")

    video_lengths = []

    for value in dataset["length"]:
        video_lengths.append(int(to_text(value)))

    return dataset.add_column("video_length", video_lengths)
