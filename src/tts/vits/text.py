from .cleaners import english_cleaners2
from .commons import intersperse
from .symbols import symbols

SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(symbols)}


def encode_text(text, cleaned=False):
    phonemes = text if cleaned else english_cleaners2(text)
    if not phonemes.strip():
        raise ValueError("Text has no phonemes")
    unknown = sorted(set(phonemes) - SYMBOL_TO_ID.keys())
    if unknown:
        raise ValueError(f"Unsupported VITS phonemes: {unknown!r}")
    return intersperse([SYMBOL_TO_ID[phone] for phone in phonemes], 0)
