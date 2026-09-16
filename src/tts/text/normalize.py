import re


def normalize_text(text: str) -> str:
    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    text = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text