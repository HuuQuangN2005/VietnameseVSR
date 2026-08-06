# Source (modified): https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/src/tokenizer/spm_tokenizer.py
# License: CC BY-NC 4.0 (https://github.com/nguyenvulebinh/AVSRCocktail/blob/main/LICENSE)

import os

import sentencepiece as spm
import torch

from srcs.spm.norm import TextNormalizer
from srcs.spm.spm_train import get_paths


MODEL_PATH, UNITS_PATH = get_paths()


class TextTransform:
    def __init__(self, model_path=MODEL_PATH, units_path=UNITS_PATH, norm=None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        if not os.path.isfile(units_path):
            raise FileNotFoundError(f"SentencePiece units not found: {units_path}")
        self.spm = spm.SentencePieceProcessor(model_file=model_path)
        self.normalizer = norm or TextNormalizer()
        self.piece_to_id = {}
        max_id = 0
        with open(units_path, encoding="utf-8") as file:
            for line in file:
                parts = line.rstrip().rsplit(maxsplit=1)
                if len(parts) != 2:
                    continue
                piece, token_id = parts[0], int(parts[1])
                self.piece_to_id[piece] = token_id
                max_id = max(max_id, token_id)
        self.token_list = ["<blank>"] + ["<unk>"] * max_id + ["<eos>"]
        for piece, token_id in self.piece_to_id.items():
            self.token_list[token_id] = piece
        self.blank_id = 0
        self.eos_id = len(self.token_list) - 1
        self.unk_id = self.piece_to_id.get("<unk>", 1)
        self.ignore_id = -1

    @property
    def vocab_size(self):
        return len(self.token_list)

    def encode(self, text):
        pieces = self.spm.encode(self.normalizer(text), out_type=str)
        ids = [self.piece_to_id.get(piece, self.unk_id) for piece in pieces]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        pieces = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in (self.blank_id, self.eos_id, self.ignore_id):
                continue
            if 0 <= token_id < len(self.token_list):
                pieces.append(self.token_list[token_id])
        return "".join(pieces).replace("\u2581", " ").strip()

    def tokenize(self, text):
        return self.encode(text)

    def post_process(self, ids):
        return self.decode(ids)
