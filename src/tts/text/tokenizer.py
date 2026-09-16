from typing import List

from phonemizer import phonemize
from phonemizer.separator import Separator

from src.tts.text.normalize import normalize_text
from src.tts.text.symbols import symbol_to_id, BOS, EOS


def text_to_phonemes(text: str) -> List[str]:
    text = normalize_text(text)

    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        separator=Separator(
            phone=" ",
            word=" | ",
            syllable="",
        ),
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )

    # Split into individual tokens
    tokens = phonemes.split()

    return tokens


def text_to_ids(text: str) -> List[int]:
    phoneme_tokens = text_to_phonemes(text)

    ids = [symbol_to_id(BOS)]

    for token in phoneme_tokens:
        ids.append(symbol_to_id(token))

    ids.append(symbol_to_id(EOS))

    return ids