"""Prepare the official LJSpeech VITS splits using the upstream cleaned text."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import wave

from src.tts.vits.runtime import load_config
from src.tts.vits.symbols import symbols
from src.tts.vits.text import encode_text


def prepare(data_root, output, config):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if not (data_root / "metadata.csv").is_file():
        raise FileNotFoundError(data_root / "metadata.csv")
    filelists = Path(__file__).resolve().parents[1] / "src/tts/vits/filelists"
    records, skipped, seen = {}, [], set()
    minimum_frames = config["train"]["segment_size"] // config["data"]["hop_length"]
    for split in ("train", "val", "test"):
        records[split] = []
        for line in (filelists / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            upstream_path, phonemes = line.split("|", 1)
            name = Path(upstream_path).name
            if name in seen:
                raise ValueError(f"Duplicate utterance across splits: {name}")
            seen.add(name)
            wav = data_root / "wavs" / name
            with wave.open(str(wav), "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, config["data"]["sampling_rate"]):
                    raise ValueError(f"Expected mono PCM16 at configured sample rate: {wav}")
                frames = audio.getnframes() // config["data"]["hop_length"]
            tokens = encode_text(phonemes, cleaned=True)
            if frames < max(minimum_frames, len(tokens)):
                skipped.append({"file": name, "split": split, "reason": "too few frames for alignment/segment"})
                continue
            records[split].append({"wav": os.path.relpath(wav, output), "tokens": tokens,
                                   "phonemes": phonemes, "frames": frames})
        if not records[split]:
            raise ValueError(f"No usable utterances in {split}")
    # Preparation is cheap and repeatable. Refuse to overwrite different manifests.
    contents = {f"{split}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                for split, rows in records.items()}
    fingerprint = hashlib.sha256("".join(contents.values()).encode()).hexdigest()
    metadata = {"architecture": "vits", "data": config["data"], "symbols": symbols,
                "fingerprint": fingerprint, "skipped": skipped}
    contents["metadata.json"] = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing preparation differs: choose a new output instead of {output}")
    for name, text in contents.items():
        (output / name).write_text(text, encoding="utf-8")
    print("VITS data: " + ", ".join(f"{split}={len(rows)}" for split, rows in records.items()))
    print(f"Unknown tokens: 0; excluded short samples: {len(skipped)}; output: {output}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/LJSpeech-1.1")
    parser.add_argument("--output", default="data/processed/vits")
    parser.add_argument("--config", default="configs/vits.json")
    args = parser.parse_args()
    prepare(args.data_root, args.output, load_config(args.config))


if __name__ == "__main__":
    main()
