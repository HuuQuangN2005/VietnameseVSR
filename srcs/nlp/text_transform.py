import json
import os
from collections import Counter

import torch

from srcs.nlp.tokenizer import PhonemeTokenizer, Tokenizer, WordTokenizer

BLANK_TOKEN = "<blank>"
UNK_TOKEN = Tokenizer.unk_token


def vocab_paths(dir_path):
    return {
        "word_path": os.path.join(dir_path, "word.txt"),
        "initial_path": os.path.join(dir_path, "initial.txt"),
        "rhyme_path": os.path.join(dir_path, "rhyme.txt"),
        "tone_path": os.path.join(dir_path, "tone.txt"),
        "lookup_path": os.path.join(dir_path, "phoneme_lookup.json"),
    }


def load_vocab(path):
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def save_vocab(path, vocab):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(vocab) + "\n")


def build_word_vocab(train_dataset, path, min_freq=1):
    tokenizer = WordTokenizer()
    freqs = Counter()

    for label in train_dataset["label"]:
        freqs.update(tokenizer.tokenize(label))

    words = {word for word, count in freqs.items() if count >= min_freq}
    vocab = [BLANK_TOKEN, UNK_TOKEN, *sorted(words)]
    save_vocab(path, vocab)

    return vocab, freqs


class TextTransform:
    ignore_id = -1
    unk_token = UNK_TOKEN

    @staticmethod
    def to_list(ids):
        if torch.is_tensor(ids):
            return ids.detach().cpu().tolist()

        return ids


class PhonemeTransform(TextTransform):
    comp_names = ("initial", "rhyme", "tone")
    metric_names = ("per_i", "per_r", "per_t")

    def __init__(
        self,
        word_path,
        initial_path,
        rhyme_path,
        tone_path,
        lookup_path,
        train_dataset=None,
    ):
        self.tokenizer = PhonemeTokenizer(lookup_path)
        self.paths = {
            "initial": initial_path,
            "rhyme": rhyme_path,
            "tone": tone_path,
        }

        if train_dataset is not None:
            word_vocab, freqs = build_word_vocab(train_dataset, word_path)
            self.__build_vocabs(word_vocab, freqs, lookup_path)

        self.token2id = {}
        self.id2token = {}
        self.vocab_size = {}
        self.unk_id = {}

        for name, path in self.paths.items():
            vocab = load_vocab(path)
            self.token2id[name] = {token: index for index, token in enumerate(vocab)}
            self.id2token[name] = dict(enumerate(vocab))
            self.vocab_size[name] = len(vocab)
            self.unk_id[name] = self.token2id[name][self.unk_token]

    def __build_vocabs(self, word_vocab, freqs, lookup_path):
        comps = {name: set() for name in self.comp_names}
        lookup = {}

        for word in word_vocab:
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

            for name, phoneme in zip(self.comp_names, phonemes):
                comps[name].add(phoneme)

            key = self.tokenizer.merge_phoneme(phonemes)
            lookup.setdefault(key, []).append(word)

        for candidates in lookup.values():
            candidates.sort(key=lambda word: -freqs[word])

        for name, path in self.paths.items():
            save_vocab(path, [self.unk_token, *sorted(comps[name])])

        with open(lookup_path, "w", encoding="utf-8") as file:
            json.dump(lookup, file, ensure_ascii=False, indent=2)

        self.tokenizer.lookup = lookup

    def encode(self, text):
        labels = []

        for phonemes in self.tokenizer.tokenize(text):
            if any(
                phoneme not in self.token2id[name]
                for name, phoneme in zip(self.comp_names, phonemes)
            ):
                labels.append([self.unk_id[name] for name in self.comp_names])
                continue

            labels.append(
                [
                    self.token2id[name][phoneme]
                    for name, phoneme in zip(self.comp_names, phonemes)
                ]
            )

        if not labels:
            return torch.empty((0, len(self.comp_names)), dtype=torch.long)

        return torch.tensor(labels, dtype=torch.long)

    def decode(self, ids):
        phonemes = []

        for syllable_ids in self.to_list(ids):
            if all(int(index) == self.ignore_id for index in syllable_ids):
                continue

            phonemes.append(
                [
                    self.id2token[name].get(int(index), self.unk_token)
                    for name, index in zip(self.comp_names, syllable_ids)
                ]
            )

        return self.tokenizer.detokenize(phonemes)

    def decode_for_metrics(self, ids):
        seqs = {name: [] for name in self.metric_names}

        for syllable_ids in self.to_list(ids):
            for name, index in zip(self.metric_names, syllable_ids):
                seqs[name].append(str(int(index)))

        return {name: " ".join(tokens) for name, tokens in seqs.items()}
