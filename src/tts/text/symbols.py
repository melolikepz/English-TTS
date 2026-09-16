PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


PHONEMES = [
    # Vowels
    "a", "ɑ", "ɐ", "ɒ", "ʌ", "æ", "e", "ɛ", "ɪ", "i",
    "ɨ", "o", "ɔ", "ʊ", "u", "ə", "ɜ", "ɝ", "ɚ",

    # Long vowels
    "iː", "uː", "ɜː", "ɑː", "ɔː",

    # Diphthongs
    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ",

    # Consonants
    "p", "b", "t", "d", "k", "ɡ",
    "f", "v", "θ", "ð", "s", "z",
    "ʃ", "ʒ", "h",
    "tʃ", "dʒ",
    "m", "n", "ŋ",
    "l", "ɹ", "r", "j", "w",

    # Stress
    "ˈ",
    "ˌ",

    # Punctuation
    ",", ".", "!", "?", ";", ":", "'", '"', "-", "…",
]


SYMBOLS = [PAD, BOS, EOS, UNK] + PHONEMES


# Symbol → ID
_symbol_to_id = {
    symbol: idx
    for idx, symbol in enumerate(SYMBOLS)
}


# ID → Symbol
_id_to_symbol = {
    idx: symbol
    for symbol, idx in _symbol_to_id.items()
}


def symbol_to_id(symbol: str) -> int:
    return _symbol_to_id.get(symbol, _symbol_to_id[UNK])


def id_to_symbol(idx: int) -> str:
    return _id_to_symbol.get(idx, UNK)


def num_symbols() -> int:
    return len(SYMBOLS)