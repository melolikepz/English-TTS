import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.tts.data.preprocessing import load_wav
from .audio import VITSAudio


class VITSDataset(Dataset):
    def __init__(self, manifest, config):
        self.manifest = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty dataset: {manifest}")
        self.config = config
        self.audio = VITSAudio(config["data"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wave = load_wav(self.manifest.parent / row["wav"], self.config["data"]["sampling_rate"])
        spec = self.audio.spectrogram(wave[None])[0]
        tokens = torch.tensor(row["tokens"], dtype=torch.long)
        if spec.shape[1] < max(len(tokens), self.config["train"]["segment_size"] // self.config["data"]["hop_length"]):
            raise ValueError(f"Audio too short for VITS alignment/segment: {row['wav']}")
        return tokens, spec, wave


def collate_vits(samples):
    tokens, specs, waves = zip(*samples)
    return (
        pad_sequence(tokens, batch_first=True), torch.tensor([len(x) for x in tokens]),
        pad_sequence([x.T for x in specs], batch_first=True).transpose(1, 2),
        torch.tensor([x.shape[1] for x in specs]),
        pad_sequence(waves, batch_first=True).unsqueeze(1),
        torch.tensor([len(x) for x in waves]),
    )
