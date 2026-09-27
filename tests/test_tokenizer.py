from src.tts.text.tokenizer import text_to_phonemes, text_to_ids, texts_to_phonemes, split_phoneme_string
from src.tts.text.symbols import SYMBOLS, UNK, symbol_to_id


def test_ljspeech_espeak_symbols():
    assert len(SYMBOLS) == len(set(SYMBOLS))
    for token in split_phoneme_string("ᵻɾn̩ʔ(x)[x]"):
        assert symbol_to_id(token) != symbol_to_id(UNK)


def test_batch_phonemization_matches_single():
    texts = ["Hello, world!", "A little better."]
    assert texts_to_phonemes(texts) == [text_to_phonemes(text) for text in texts]
    assert texts_to_phonemes([]) == []


def test_phonemization():
    text = "Hello, I love machine learning."

    phonemes = text_to_phonemes(text)

    print("\nPhonemes:")
    print(phonemes)

    assert len(phonemes) > 0


def test_tokenization():
    text = "Hello, I love machine learning."

    ids = text_to_ids(text)

    print("\nIDs:")
    print(ids)

    assert len(ids) > 0
    assert all(isinstance(x, int) for x in ids)
