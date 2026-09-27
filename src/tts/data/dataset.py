import json
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tts.text.tokenizer import text_to_ids


class RawLJSpeechDataset(Dataset):
    """Read raw WAV samples; training uses cached LJSpeechDataset below."""

    def __init__(self, root, sample_rate=22050):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.samples = []

        metadata_path = self.root / "metadata.csv"

        with metadata_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                parts = line.rstrip("\r\n").split("|")

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid metadata at line {line_number}"
                    )

                utterance_id, original_text, normalized_text = parts
                text = normalized_text.strip() or original_text.strip()

                if not text:
                    raise ValueError(
                        f"Empty text at line {line_number}"
                    )

                audio_path = self.root / "wavs" / f"{utterance_id}.wav"

                if not audio_path.is_file():
                    raise FileNotFoundError(audio_path)

                self.samples.append((audio_path, text))

        if not self.samples:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_path, text = self.samples[index]

        with wave.open(str(audio_path), "rb") as audio:
            if audio.getnchannels() != 1:
                raise ValueError(f"Expected mono audio: {audio_path}")

            if audio.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit PCM: {audio_path}")

            if audio.getframerate() != self.sample_rate:
                raise ValueError(
                    f"Expected {self.sample_rate} Hz, "
                    f"got {audio.getframerate()}: {audio_path}"
                )

            raw_audio = audio.readframes(audio.getnframes())

        waveform = np.frombuffer(raw_audio, dtype="<i2")
        waveform = waveform.astype(np.float32) / 32768.0

        if waveform.size == 0:
            raise ValueError(f"Empty audio: {audio_path}")

        token_ids = torch.tensor(text_to_ids(text), dtype=torch.long)

        return {
            "token_ids": token_ids,
            "waveform": torch.from_numpy(waveform),
            "text": text,
            "audio_path": str(audio_path),
        }


def read_metadata(root):
    root = Path(root)
    samples = []
    seen = set()
    with (root / "metadata.csv").open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("|")
            if len(fields) != 3:
                raise ValueError(f"Invalid metadata at line {number}")
            name, original, normalized = fields
            if not name or Path(name).name != name or name in seen:
                raise ValueError(f"Invalid or duplicate utterance ID: {name}")
            text = normalized.strip() or original.strip()
            if not text:
                raise ValueError(f"Empty text: {name}")
            path = root / "wavs" / f"{name}.wav"
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(name)
            samples.append((name, path, text))
    if not samples:
        raise ValueError("Dataset is empty")
    return samples


class LJSpeechDataset(Dataset):
    """Load cached token IDs and mel features made by scripts.preprocess."""

    def __init__(self, manifest):
        self.manifest = Path(manifest)
        with self.manifest.open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.load(
            self.manifest.parent / self.records[index]["file"],
            map_location="cpu", weights_only=True,
        )
