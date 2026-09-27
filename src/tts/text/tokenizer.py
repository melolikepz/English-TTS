from typing import List

from phonemizer import phonemize
from phonemizer.separator import Separator

from src.tts.text.normalize import normalize_text
from src.tts.text.symbols import symbol_to_id, BOS, EOS


PUNCTUATION = {
    ",",
    ".",
    "!",
    "?",
    ";",
    ":",
    "'",
    '"',
    "-",
    "…",
}

STRESS_MARKS = {
    "ˈ",
    "ˌ",
}


# Multi-character phonemes must be checked first.
MULTI_CHAR_PHONEMES = [
    "tʃ",
    "dʒ",
    "iː",
    "uː",
    "ɜː",
    "ɑː",
    "ɔː",
    "eɪ",
    "aɪ",
    "ɔɪ",
    "aʊ",
    "oʊ",
]


def split_phoneme_string(text: str) -> List[str]:
    """
    Split a phoneme string into valid phoneme tokens.

    Multi-character phonemes such as:
        oʊ
        aɪ
        iː
        tʃ
        dʒ

    are kept as a single token.
    """

    tokens = []
    i = 0

    while i < len(text):

        # Ignore whitespace
        if text[i].isspace():
            i += 1
            continue

        # Word boundary
        if text[i] == "|":
            tokens.append("|")
            i += 1
            continue

        # Punctuation
        if text[i] in PUNCTUATION:
            tokens.append(text[i])
            i += 1
            continue

        # Stress markers
        if text[i] in STRESS_MARKS:
            tokens.append(text[i])
            i += 1
            continue

        # Try multi-character phonemes first
        matched = False

        for phoneme in MULTI_CHAR_PHONEMES:
            if text.startswith(phoneme, i):
                tokens.append(phoneme)
                i += len(phoneme)
                matched = True
                break

        if matched:
            continue

        # Single-character phoneme
        tokens.append(text[i])
        i += 1

    return tokens


def text_to_phonemes(text: str) -> List[str]:
    """
    Convert English text into phoneme tokens.
    """

    return texts_to_phonemes([text])[0]


def texts_to_phonemes(texts: List[str]) -> List[List[str]]:
    """Phonemize a batch with one backend initialization."""
    if not texts:
        return []
    normalized = [normalize_text(text) for text in texts]

    phonemes = phonemize(
        normalized,
        language="en-us",
        backend="espeak",
        separator=Separator(
            phone="",
            word=" | ",
            syllable="",
        ),
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )

    return [split_phoneme_string(phoneme) for phoneme in phonemes]


def text_to_ids(text: str) -> List[int]:
    """
    Convert text into integer token IDs.
    """

    phoneme_tokens = text_to_phonemes(text)

    ids = [symbol_to_id(BOS)]

    for token in phoneme_tokens:
        ids.append(symbol_to_id(token))

    ids.append(symbol_to_id(EOS))

    return ids
