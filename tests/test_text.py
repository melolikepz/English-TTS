from src.tts.text.normalize import normalize_text


def test_whitespace():
    text = "  Hello,    world!  "
    assert normalize_text(text) == "Hello, world!"


def test_quotes():
    text = "She said “Hello”."
    assert normalize_text(text) == 'She said "Hello".'


def test_punctuation_spacing():
    text = "Hello , world ! How are you ?"
    assert normalize_text(text) == "Hello, world! How are you?"