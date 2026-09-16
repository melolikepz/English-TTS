from src.tts.text.tokenizer import text_to_phonemes, text_to_ids


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