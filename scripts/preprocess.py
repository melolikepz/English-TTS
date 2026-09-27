import argparse
import json
import random
from pathlib import Path

import torch

from src.tts.config import load_config
from src.tts.data.dataset import read_metadata
from src.tts.data.preprocessing import MelProcessor, load_wav
from src.tts.text.symbols import SYMBOLS, UNK, symbol_to_id
from src.tts.text.tokenizer import text_to_ids


def main():
    parser = argparse.ArgumentParser(description="Cache LJSpeech phonemes and log-mel features")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="data/processed/ljspeech")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--limit", type=int, help="Optional positive limit for a smoke run")
    args = parser.parse_args()
    config = load_config(args.config)
    records = read_metadata(args.data_root)
    if args.limit is not None:
        if args.limit < 2:
            parser.error("--limit must be at least 2")
        records = records[:args.limit]
    if len(records) < 2:
        parser.error("Need at least two utterances for training and validation")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        parser.error("Output directory must be empty; choose a new --output path")
    (destination / "features").mkdir()
    processor = MelProcessor(config["audio"])
    manifest = []
    unknown_count = 0
    for index, (name, audio_path, text) in enumerate(records, 1):
        waveform = load_wav(audio_path, config["audio"]["sample_rate"])
        tokens = torch.tensor(text_to_ids(text), dtype=torch.long)
        unknown_count += int((tokens == symbol_to_id(UNK)).sum())
        mel = processor.encode(waveform)
        if mel.shape[0] < 2:
            raise ValueError(f"Audio is too short: {audio_path}")
        relative_path = f"features/{name}.pt"
        torch.save({"token_ids": tokens, "mel": mel, "text": text}, destination / relative_path)
        manifest.append({"file": relative_path, "frames": len(mel), "tokens": len(tokens)})
        if index % 100 == 0 or index == len(records):
            print(f"Prepared {index}/{len(records)}", flush=True)
    random.Random(config["training"]["seed"]).shuffle(manifest)
    fraction = config["training"]["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    validation_size = min(len(manifest) - 1, max(1, round(len(manifest) * fraction)))
    for name, rows in [("val", manifest[:validation_size]), ("train", manifest[validation_size:])]:
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    (destination / "metadata.json").write_text(
        json.dumps({"audio": config["audio"], "symbols": SYMBOLS}, indent=2), encoding="utf-8",
    )
    print(f"Saved {len(manifest) - validation_size} training and {validation_size} validation samples.")
    print(f"Unknown phoneme tokens: {unknown_count}")


if __name__ == "__main__":
    main()
