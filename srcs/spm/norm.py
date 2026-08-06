import re
import unicodedata


class TextNormalizer:
    def __init__(self, lang="vi", lower=False, keep_punct=False):
        self.lang = lang
        self.lower = lower
        self.keep_punct = keep_punct

    def __call__(self, text):
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFC", text).strip()
        if self.lower:
            text = text.lower()
        if not self.keep_punct:
            text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return re.sub(r"\s+", " ", text).strip()


def normalize(text, lang="vi", lower=False, keep_punct=False):
    return TextNormalizer(lang, lower, keep_punct)(text)
