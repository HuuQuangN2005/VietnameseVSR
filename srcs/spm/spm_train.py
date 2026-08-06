# Source (modified): https://github.com/facebookresearch/fairseq/blob/main/scripts/spm_train.py
# License: MIT (https://github.com/facebookresearch/fairseq/blob/main/LICENSE)

import argparse
import os

import sentencepiece as spm

from srcs.spm.norm import TextNormalizer

SPM_DIR = os.path.dirname(os.path.abspath(__file__))
UNIGRAM_DIR = os.path.join(SPM_DIR, "unigram")
VOCAB_SIZE = 5000


def get_paths(vocab_size=VOCAB_SIZE, out_dir=UNIGRAM_DIR):
    prefix = os.path.join(out_dir, f"unigram{vocab_size}_vi")
    return prefix + ".model", prefix + "_units.txt"


def build_units(model_path, units_path):
    processor = spm.SentencePieceProcessor(model_file=model_path)
    token_id = 1
    with open(units_path, "w", encoding="utf-8") as file:
        for piece_id in range(processor.vocab_size()):
            if processor.is_control(piece_id):
                continue
            file.write(f"{processor.id_to_piece(piece_id)} {token_id}\n")
            token_id += 1
    return units_path


def train(input_path, out_dir=UNIGRAM_DIR, vocab_size=VOCAB_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    model_path, units_path = get_paths(vocab_size, out_dir)
    prefix = os.path.splitext(model_path)[0]
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )
    build_units(model_path, units_path)
    return model_path, units_path


def train_dataset(dataset, out_dir=UNIGRAM_DIR, vocab_size=VOCAB_SIZE):
    if "label" not in dataset.column_names:
        raise ValueError("The training dataset must contain a label column.")

    normalizer = TextNormalizer()

    def sentences():
        for value in dataset["label"]:
            if isinstance(value, (bytes, bytearray, memoryview)):
                value = bytes(value).decode("utf-8")
            text = normalizer(str(value))
            if text:
                yield text

    os.makedirs(out_dir, exist_ok=True)
    model_path, units_path = get_paths(vocab_size, out_dir)
    prefix = os.path.splitext(model_path)[0]
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences(),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )
    build_units(model_path, units_path)
    return model_path, units_path


def ensure_unigram(dataset=None, out_dir=UNIGRAM_DIR, vocab_size=VOCAB_SIZE):
    model_path, units_path = get_paths(vocab_size, out_dir)
    if not os.path.isfile(model_path):
        if dataset is None:
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        print(f"Training SentencePiece model: {model_path}")
        return train_dataset(dataset, out_dir, vocab_size)
    if not os.path.isfile(units_path):
        print(f"Building SentencePiece units: {units_path}")
        build_units(model_path, units_path)
    return model_path, units_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", default=UNIGRAM_DIR)
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args()
    train(args.input, args.out_dir, args.vocab_size)


if __name__ == "__main__":
    main()
