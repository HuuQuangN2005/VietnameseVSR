from abc import ABC, abstractmethod
from typing import List
from .norm import TextNormalizer


class Tokenizer(ABC):
    def __init__(self, normalizer: TextNormalizer = None):
        self.normalizer = normalizer

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError


class WordTokenizer(Tokenizer):
    def __init__(self, normalizer=None):
        super().__init__(normalizer)

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []

        if self.normalizer is not None:
            text = self.normalizer.normalize(text)

        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        if not isinstance(tokens, (list, tuple)):
            return ""

        return " ".join(tokens)
