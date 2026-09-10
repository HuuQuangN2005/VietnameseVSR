from abc import ABC, abstractmethod
import json
import os
import re
import unicodedata

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PHONEME_LOOKUP_PATH = os.path.join(DATA_DIR, "phoneme_lookup.json")


class Tokenizer(ABC):
    unk_token = "<unk>"

    def clean_text(self, text: str):
        if isinstance(text, (bytes, bytearray, memoryview)):
            text = bytes(text).decode("utf-8")

        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFC", text).lower()
        text = re.sub(r"[\W\d_]+", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def to_word(self, text: str):
        return self.clean_text(text).split()

    @abstractmethod
    def tokenize(self, text: str):
        pass

    @abstractmethod
    def detokenize(self, tokens: list):
        pass


class WordTokenizer(Tokenizer):
    def tokenize(self, text: str):
        return self.to_word(text)

    def detokenize(self, tokens: list):
        return " ".join(tokens)


class PhonemeTokenizer(Tokenizer):
    def __init__(self, lookup_path=None):
        self.lookup_path = lookup_path or PHONEME_LOOKUP_PATH
        self.lookup = None

        self.initials = {
            "/b/": ["b"],
            "/m/": ["m"],
            "/f/": ["ph"],
            "/v/": ["v"],
            "/th/": ["th"],
            "/t/": ["t"],
            "/d/": ["đ"],
            "/n/": ["n"],
            "/sx/": ["x"],
            "/z/": ["gi", "d"],
            "/l/": ["l"],
            "/tr/": ["tr"],
            "/r/": ["r"],
            "/s/": ["s"],
            "/c/": ["ch"],
            "/nh/": ["nh"],
            "/k/": ["c", "k", "q"],
            "/ng/": ["ngh", "ng"],
            "/xh/": ["kh"],
            "/yg/": ["gh", "g"],
            "/h/": ["h"],
            "/null/": [""],
        }

        self.glides = {"/-u-/": ["o", "u"], "/zero/": [""]}

        self.vowels = {
            "/i/": ["i", "y"],
            "/ee/": ["ê"],
            "/e/": ["e"],
            "/ew/": ["a"],
            "/ie/": ["iê", "ia", "yê", "ya"],
            "/w/": ["ư"],
            "/ow/": ["ơ"],
            "/a/": ["a"],
            "/aa/": ["â"],
            "/aw/": ["a", "ă"],
            "/wow/": ["ươ", "ưa"],
            "/u/": ["u"],
            "/oo/": ["ôô", "ô"],
            "/o/": ["oo", "o"],
            "/o_short/": ["o"],
            "/uo/": ["uô", "ua"],
        }

        self.finals = {
            "/-p/": ["p"],
            "/-t/": ["t"],
            "/-k/": ["c", "ch"],
            "/-m/": ["m"],
            "/-n/": ["n"],
            "/-ng/": ["ng", "nh"],
            "/-u/": ["u", "o"],
            "/-i/": ["i", "y"],
            "/zero/": [""],
        }

        self.tones = {
            "": "/ngang/",
            "\u0300": "/huyen/",
            "\u0301": "/sac/",
            "\u0309": "/hoi/",
            "\u0303": "/nga/",
            "\u0323": "/nang/",
        }

    def merge_phoneme(self, phonemes: list[str]):
        return "_".join(phonemes)

    def __load_lookup(self):
        if self.lookup is None:
            with open(self.lookup_path, encoding="utf-8") as file:
                self.lookup = json.load(file)

        return self.lookup

    def analyze(self, word: str):
        word = self.clean_text(word)
        remainder, tone = self.__get_tone(word)
        remainder, initial, initial_grapheme = self.__get_initial(remainder)

        rhyme_grapheme = remainder

        remainder, glide, glide_grapheme = self.__get_glide(
            rhyme_grapheme, initial_grapheme
        )
        remainder, final, final_grapheme = self.__get_final(remainder)
        remainder, vowel, vowel_grapheme = self.__get_vowel(remainder, final_grapheme)

        analysis = {
            "is_valid": False,
            "initial": initial,
            "glide": glide,
            "vowel": vowel,
            "final": final,
            "tone": tone,
        }

        if remainder != "" or vowel == "":
            return analysis

        analysis["is_valid"] = self.__check_spelling(
            initial_grapheme,
            rhyme_grapheme,
            glide_grapheme,
            vowel_grapheme,
            final_grapheme,
        )
        return analysis

    def __get_tone(self, word: str):
        characters = []
        tone = self.tones[""]

        for character in unicodedata.normalize("NFD", word):
            if character in self.tones:
                tone = self.tones[character]
            else:
                characters.append(character)

        remainder = unicodedata.normalize("NFC", "".join(characters))

        return remainder, tone

    def __get_initial(self, word: str):
        initial = "/null/"
        matched = ""

        for symbol, graphemes in self.initials.items():
            for grapheme in graphemes:
                if word.startswith(grapheme) and len(grapheme) > len(matched):
                    initial = symbol
                    matched = grapheme

        remainder = word[len(matched) :]

        if matched == "gi":
            vowels = "aăâeêioôơuưy"

            if remainder.startswith("ê") or not any(
                character in vowels for character in remainder
            ):
                remainder = word[1:]

        return remainder, initial, matched

    def __get_glide(self, word: str, initial_grapheme: str):
        glide = "/zero/"
        remainder = word
        matched = ""

        if initial_grapheme == "q" and word.startswith("u"):
            remainder = word[1:]
            glide = "/-u-/"
            matched = "u"

            return remainder, glide, matched

        if word.startswith("u"):
            remainder = word[1:]

            if remainder.startswith(("y", "ê", "ơ", "â")):
                glide = "/-u-/"
                matched = "u"

                return remainder, glide, matched

        if word.startswith("o"):
            remainder = word[1:]

            if remainder.startswith(("a", "ă", "e")):
                glide = "/-u-/"
                matched = "o"

                return remainder, glide, matched

        return word, glide, matched

    def __get_vowel(self, word: str, final_grapheme: str):
        if word == "a":
            if final_grapheme in ("nh", "ch"):
                return "", "/ew/", "a"

            if final_grapheme in ("y", "u"):
                return "", "/aw/", "a"

            return "", "/a/", "a"

        if word == "o":
            if final_grapheme in ("ng", "c"):
                return "", "/o_short/", "o"

            return "", "/o/", "o"

        vowel = ""
        matched = ""

        for symbol, graphemes in self.vowels.items():
            for grapheme in graphemes:
                if grapheme in ("a", "o"):
                    continue

                if word == grapheme:
                    vowel = symbol
                    matched = grapheme

        remainder = word[len(matched) :]

        return remainder, vowel, matched

    def __get_final(self, rhyme: str):
        final = "/zero/"
        matched = ""

        for symbol, graphemes in self.finals.items():
            for grapheme in graphemes:
                if not grapheme or not rhyme.endswith(grapheme):
                    continue
                remainder = rhyme[: -len(grapheme)]

                if remainder == "":
                    continue

                if len(grapheme) > len(matched):
                    final = symbol
                    matched = grapheme

        if matched:
            remainder = rhyme[: -len(matched)]
        else:
            remainder = rhyme

        return remainder, final, matched

    def __check_spelling(self, initial, rhyme, glide, vowel, final):
        if initial == "q" and glide != "u":
            return False

        if initial in ("c", "k") and glide != "":
            return False

        if initial in ("gh", "ngh") and not rhyme.startswith(("e", "ê", "i")):
            return False

        if initial in ("g", "ng") and rhyme.startswith(("e", "ê", "i")):
            return False

        if initial == "k" and not rhyme.startswith(("e", "ê", "i", "y")):
            return False

        if initial == "c" and rhyme.startswith(("e", "ê", "i", "y")):
            return False

        if vowel in ("ia", "ya", "ua", "ưa") and final != "":
            return False

        if vowel in ("iê", "yê", "uô", "ươ") and final == "":
            return False

        if vowel == "ya" and glide != "u":
            return False

        if vowel == "ia" and glide != "":
            return False

        if vowel == "iê" and (initial == "" or glide != ""):
            return False

        if vowel == "yê" and initial != "" and glide == "":
            return False

        if final in ("nh", "ch") and vowel not in ("a", "i", "y", "ê"):
            return False

        if vowel in ("oo", "ôô") and final not in ("ng", "c"):
            return False

        if vowel == "ă" and final in ("u", "o", "i", "y"):
            return False

        if vowel == "â" and final == "i":
            return False

        if vowel == "e" and final == "u":
            return False

        if final == "y" and vowel not in ("a", "â"):
            return False

        if final == "o" and vowel not in ("a", "e"):
            return False

        if final in ("ng", "c") and vowel in ("i", "y", "ê"):
            return False

        return True

    def tokenize(self, text):
        tokens = []

        for word in self.to_word(text):
            analysis = self.analyze(word)

            if not analysis["is_valid"]:
                tokens.append([self.unk_token] * 3)
                continue

            rhyme = self.merge_phoneme(
                [analysis["glide"], analysis["vowel"], analysis["final"]]
            )
            tokens.append([analysis["initial"], rhyme, analysis["tone"]])

        return tokens

    def detokenize(self, tokens: list):
        if not tokens:
            return ""

        lookup = self.__load_lookup()
        words = []

        for token in tokens:
            if self.unk_token in token:
                words.append(self.unk_token)
                continue

            phoneme = token if isinstance(token, str) else self.merge_phoneme(token)
            candidates = lookup.get(phoneme)

            if not candidates:
                words.append(self.unk_token)
                continue

            words.append(candidates[0])

        return " ".join(words)
