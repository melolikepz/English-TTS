from phonemizer import phonemize


text = "Hello, I love machine learning."

phonemes = phonemize(
    text,
    language="en-us",
    backend="espeak",
    strip=True,
    preserve_punctuation=True,
    with_stress=True,
)

print("Text:")
print(text)

print("\nPhonemes:")
print(phonemes)