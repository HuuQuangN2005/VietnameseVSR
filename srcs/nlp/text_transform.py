import json
import os
from collections import Counter

import torch

from srcs.nlp.tokenizer import (
    DATA_DIR,
    PHONEME_LOOKUP_PATH,
    PhonemeTokenizer,
    Tokenizer,
    WordTokenizer,
)

WORD_PATH = os.path.join(DATA_DIR, "word.txt")
INITIAL_PATH = os.path.join(DATA_DIR, "initial.txt")
RHYME_PATH = os.path.join(DATA_DIR, "rhyme.txt")
TONE_PATH = os.path.join(DATA_DIR, "tone.txt")

BLANK_TOKEN = "<blank>"
UNK_TOKEN = Tokenizer.unk_token


def vocabulary_paths(directory):
    return {
        "word_path": os.path.join(directory, "word.txt"),
        "initial_path": os.path.join(directory, "initial.txt"),
        "rhyme_path": os.path.join(directory, "rhyme.txt"),
        "tone_path": os.path.join(directory, "tone.txt"),
        "lookup_path": os.path.join(directory, "phoneme_lookup.json"),
    }


def load_vocabulary(path):
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def save_vocabulary(path, vocabulary):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(vocabulary) + "\n")


def build_word_vocabulary(train_dataset, path=WORD_PATH, min_frequency=1):
    tokenizer = WordTokenizer()
    frequencies = Counter()

    for label in train_dataset["label"]:
        frequencies.update(tokenizer.tokenize(label))

    words = {word for word, count in frequencies.items() if count >= min_frequency}
    vocabulary = [BLANK_TOKEN, UNK_TOKEN, *sorted(words)]
    save_vocabulary(path, vocabulary)

    return vocabulary


class TextTransform:
    ignore_id = -1
    unk_token = UNK_TOKEN

    @staticmethod
    def to_list(ids):
        if torch.is_tensor(ids):
            return ids.detach().cpu().tolist()

        return ids


class WordTransform(TextTransform):
    blank_token = BLANK_TOKEN
    metric_names = ("wer",)

    def __init__(self, train_dataset=None, word_path=WORD_PATH, min_frequency=5):
        self.tokenizer = WordTokenizer()

        if train_dataset is not None:
            build_word_vocabulary(train_dataset, word_path, min_frequency)

        vocabulary = load_vocabulary(word_path)

        self.token2id = {token: index for index, token in enumerate(vocabulary)}
        self.id2token = dict(enumerate(vocabulary))
        self.vocab_size = len(vocabulary)
        self.blank_id = self.token2id[self.blank_token]
        self.unk_id = self.token2id[self.unk_token]

    def encode(self, text):
        labels = [
            self.token2id.get(word, self.unk_id)
            for word in self.tokenizer.tokenize(text)
        ]

        return torch.tensor(labels, dtype=torch.long)

    def decode(self, ids):
        words = []

        for index in self.to_list(ids):
            index = int(index)

            if index in (self.ignore_id, self.blank_id):
                continue

            words.append(self.id2token.get(index, self.unk_token))

        return self.tokenizer.detokenize(words)

    def decode_for_metrics(self, ids):
        return {"wer": self.decode(ids)}


class PhonemeTransform(TextTransform):
    component_names = ("initial", "rhyme", "tone")
    metric_names = ("per_i", "per_r", "per_t")

    def __init__(
        self,
        train_dataset=None,
        word_path=WORD_PATH,
        initial_path=INITIAL_PATH,
        rhyme_path=RHYME_PATH,
        tone_path=TONE_PATH,
        lookup_path=PHONEME_LOOKUP_PATH,
    ):
        self.tokenizer = PhonemeTokenizer(lookup_path)
        self.paths = {
            "initial": initial_path,
            "rhyme": rhyme_path,
            "tone": tone_path,
        }

        if train_dataset is not None:
            word_vocabulary = build_word_vocabulary(train_dataset, word_path)
            self.__build_vocabularies(word_vocabulary, lookup_path)

        self.token2id = {}
        self.id2token = {}
        self.vocab_size = {}
        self.unk_id = {}

        for name, path in self.paths.items():
            vocabulary = load_vocabulary(path)
            self.token2id[name] = {
                token: index for index, token in enumerate(vocabulary)
            }
            self.id2token[name] = dict(enumerate(vocabulary))
            self.vocab_size[name] = len(vocabulary)
            self.unk_id[name] = self.token2id[name][self.unk_token]

    def __build_vocabularies(self, word_vocabulary, lookup_path):
        components = {name: set() for name in self.component_names}
        lookup = {}

        for word in word_vocabulary:
            if word in (BLANK_TOKEN, UNK_TOKEN):
                continue

            analysis = self.tokenizer.analyze(word)

            if not analysis["is_valid"]:
                continue

            phonemes = [
                analysis["initial"],
                self.tokenizer.merge_phoneme(
                    [analysis["glide"], analysis["vowel"], analysis["final"]]
                ),
                analysis["tone"],
            ]

            for name, phoneme in zip(self.component_names, phonemes):
                components[name].add(phoneme)

            key = self.tokenizer.merge_phoneme(phonemes)
            lookup.setdefault(key, []).append(word)

        for name, path in self.paths.items():
            save_vocabulary(path, [self.unk_token, *sorted(components[name])])

        with open(lookup_path, "w", encoding="utf-8") as file:
            json.dump(lookup, file, ensure_ascii=False, indent=2)

        self.tokenizer.lookup = lookup

    def encode(self, text):
        labels = []

        for phonemes in self.tokenizer.tokenize(text):
            if any(
                phoneme not in self.token2id[name]
                for name, phoneme in zip(self.component_names, phonemes)
            ):
                labels.append([self.unk_id[name] for name in self.component_names])
                continue

            labels.append(
                [
                    self.token2id[name][phoneme]
                    for name, phoneme in zip(self.component_names, phonemes)
                ]
            )

        if not labels:
            return torch.empty((0, len(self.component_names)), dtype=torch.long)

        return torch.tensor(labels, dtype=torch.long)

    def decode(self, ids):
        phonemes = []

        for syllable_ids in self.to_list(ids):
            if all(int(index) == self.ignore_id for index in syllable_ids):
                continue

            phonemes.append(
                [
                    self.id2token[name].get(int(index), self.unk_token)
                    for name, index in zip(self.component_names, syllable_ids)
                ]
            )

        return self.tokenizer.detokenize(phonemes)

    def decode_for_metrics(self, ids):
        sequences = {name: [] for name in self.metric_names}

        for syllable_ids in self.to_list(ids):
            for name, index in zip(self.metric_names, syllable_ids):
                sequences[name].append(str(int(index)))

        return {name: " ".join(tokens) for name, tokens in sequences.items()}
