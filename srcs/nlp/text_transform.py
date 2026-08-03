import os
from collections import Counter
from typing import Iterable

import torch

from .tokenizer import Tokenizer

NLP_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOCAB_PATH = os.path.join(NLP_PATH, "data", "word_vocab.txt")


class TextTransform:
    blank_token = "<blank>"
    unknown_token = "<unk>"
    ignore_id = -1

    def __init__(self, tokenizer: Tokenizer, vocab_path: str = DEFAULT_VOCAB_PATH):
        self.tokenizer = tokenizer
        self.vocab_path = vocab_path
        self.token_list = []
        self.token_to_id = {}

        if os.path.isfile(self.vocab_path):
            self.load_vocab()

    def create_vocab(self, texts: Iterable[str], min_frequency: int = 1) -> str:
        token_counts = Counter()
        for text in texts:
            token_counts.update(self.tokenizer.tokenize(text))

        tokens = [
            token
            for token, frequency in token_counts.items()
            if frequency >= min_frequency
        ]
        tokens.sort(key=lambda token: (-token_counts[token], token))

        vocab_dir = os.path.dirname(os.path.abspath(self.vocab_path))
        os.makedirs(vocab_dir, exist_ok=True)
        with open(self.vocab_path, "w", encoding="utf-8") as vocab_file:
            for token in tokens:
                vocab_file.write(f"{token}\n")

        self.load_vocab()
        return self.vocab_path

    def load_vocab(self) -> list[str]:
        if not os.path.isfile(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        tokens = []
        seen_tokens = set()
        with open(self.vocab_path, "r", encoding="utf-8") as vocab_file:
            for line in vocab_file:
                values = line.strip().split()
                if not values:
                    continue

                token = values[0]
                if token in seen_tokens:
                    raise ValueError(f"Duplicate token in vocabulary: {token}")

                seen_tokens.add(token)
                tokens.append(token)

        special_tokens = {self.blank_token, self.unknown_token}
        tokens = [token for token in tokens if token not in special_tokens]
        self.token_list = [self.blank_token, self.unknown_token, *tokens]
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.token_list)
        }
        return self.token_list

    def encode(self, text: str) -> torch.Tensor:
        if not self.token_to_id:
            raise RuntimeError("Vocabulary has not been loaded")

        unknown_id = self.token_to_id[self.unknown_token]
        token_ids = [
            self.token_to_id.get(token, unknown_id)
            for token in self.tokenizer.tokenize(text)
        ]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        if not self.token_list:
            raise RuntimeError("Vocabulary has not been loaded")

        tokens = []
        for token_id in token_ids.tolist():
            if token_id == self.ignore_id:
                continue

            token = self.token_list[int(token_id)]
            if token not in {self.blank_token}:
                tokens.append(token)

        return self.tokenizer.detokenize(tokens)
