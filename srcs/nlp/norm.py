import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Pattern, Union


class RegexPatternNormalizer(ABC):
    def __init__(self, pattern: Union[str, Pattern[str]]):
        if isinstance(pattern, str):
            self.pattern: Pattern[str] = re.compile(pattern, re.UNICODE)
        else:
            self.pattern = pattern

    @abstractmethod
    def apply(self, text: str) -> str:
        raise NotImplementedError

    def __call__(self, text: str) -> str:
        return self.apply(text)


class TextNormalizer:
    def __init__(
        self, lowercase: bool = False, rules: List[RegexPatternNormalizer] = None
    ):
        self.rules = rules or []
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFC", text).strip()

        if self.lowercase:
            text = text.lower()

        for rule in self.rules:
            text = rule(text)

        return text.strip()

    def __call__(self, text: str) -> str:
        return self.normalize(text)


class PunctNormalizer(RegexPatternNormalizer):
    def __init__(self):
        super().__init__(
            r"\s*([?.!,;:\u2026\"'\u201c\u201d\u2018\u2019()\[\]{}\u2014\u2013-])\s*"
        )

    def apply(self, text: str) -> str:
        return self.pattern.sub(r" \1 ", text)


class RemovePunctNormalizer(RegexPatternNormalizer):
    def __init__(self):
        super().__init__(
            r"[?.!,;:\u2026\"'\u201c\u201d\u2018\u2019()\[\]{}\u2014\u2013-]"
        )

    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)


class SpaceNormalizer(RegexPatternNormalizer):
    def __init__(self):
        super().__init__(r"\s+")

    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text).strip()


class FlatToneNormalizer(RegexPatternNormalizer):
    def __init__(self):
        super().__init__(r"[\u0300\u0301\u0309\u0303\u0323]")

    def apply(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        text = self.pattern.sub("", text)
        return unicodedata.normalize("NFC", text)


class RemoveNumericNormalizer(RegexPatternNormalizer):
    def __init__(self):
        super().__init__(r"\S*\d\S*")

    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)


class RemoveSpecialCharNormalizer(RegexPatternNormalizer):
    def __init__(self):
        super().__init__(r"[^\w\s]")

    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)
