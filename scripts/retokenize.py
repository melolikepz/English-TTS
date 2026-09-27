"""Rebuild cached text tokens while reusing mel features and dataset splits."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from src.tts.text.symbols import BOS, EOS, SYMBOLS, symbol_to_id
from src.tts.text.tokenizer import texts_to_phonemes


def retokenize(source, destination, batch_size=128):
    source, destination = Path(source), Path(destination)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    manifests = {
        split: [json.loads(line) for line in (source / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        for split in ("train", "val")
    }
    rows = manifests["train"] + manifests["val"]
    for row in rows:
        path = Path(row["file"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid feature path: {path}")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output must be empty; original cache is left unchanged")
    destination.mkdir(parents=True, exist_ok=True)
    unknown = Counter()
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        samples = [torch.load(source / row["file"], map_location="cpu", weights_only=True) for row in subset]
        phonemes = texts_to_phonemes([sample["text"] for sample in samples])
        if len(phonemes) != len(samples):
            raise ValueError("Phonemizer returned an unexpected number of utterances")
        for row, sample, phones in zip(subset, samples, phonemes):
            unknown.update(phone for phone in phones if phone not in SYMBOLS)
            ids = [symbol_to_id(BOS)] + [symbol_to_id(phone) for phone in phones] + [symbol_to_id(EOS)]
            sample["token_ids"] = torch.tensor(ids, dtype=torch.long)
            row["tokens"] = len(ids)
            path = destination / row["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample, path)
        print(f"Retokenized {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    for split, records in manifests.items():
        (destination / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8",
        )
    metadata["symbols"] = SYMBOLS
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (destination / "unknown_phonemes.json").write_text(
        json.dumps(dict(unknown.most_common()), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Unknown phoneme tokens: {sum(unknown.values())}")
    return unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/processed/ljspeech")
    parser.add_argument("--output", default="data/processed/ljspeech-v2")
    args = parser.parse_args()
    retokenize(args.source, args.output)


if __name__ == "__main__":
    main()
