import torch

from collections.abc import Sequence
from torchcodec.decoders import VideoDecoder
from srcs.nlp.tokenizer import PhonemeTokenizer, WordTokenizer

word_tokenizer = WordTokenizer()
phoneme_tokenizer = PhonemeTokenizer()
DECODE_THREADS = 2


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


def pad_seq(seqs: Sequence[torch.Tensor], padding_value=0.0):
    if not seqs:
        raise ValueError("sequences must not be empty")

    lengths = torch.tensor([item.size(0) for item in seqs], dtype=torch.long)
    max_length = int(lengths.max())
    shape = (len(seqs), max_length, *seqs[0].shape[1:])

    output = seqs[0].new_full(shape, padding_value)

    for index, item in enumerate(seqs):
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
