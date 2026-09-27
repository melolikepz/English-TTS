"""PCM WAV, log-magnitude mel features, and Griffin-Lim using torch only."""

import wave
from pathlib import Path

import numpy as np
import torch


def load_wav(path, sample_rate):
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        if audio.getframerate() != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz: {path}")
        frames = audio.readframes(audio.getnframes())
    values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if not values.size:
        raise ValueError(f"Empty WAV: {path}")
    return torch.from_numpy(values)


def save_wav(path, waveform, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().float().cpu()
    if not torch.isfinite(waveform).all():
        raise ValueError("Non-finite waveform")
    waveform = waveform / waveform.abs().max().clamp_min(1.0)
    samples = (waveform.clamp(-1, 1) * 32767).numpy().astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())


class MelProcessor:
    def __init__(self, config):
        self.config = config
        self.window = torch.hann_window(config["win_length"])
        hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
        edges = torch.linspace(
            float(hz_to_mel(config["f_min"])),
            float(hz_to_mel(config["f_max"])),
            config["n_mels"] + 2,
        )
        edges = 700.0 * (10.0 ** (edges / 2595.0) - 1.0)
        frequencies = torch.linspace(0, config["sample_rate"] / 2, config["n_fft"] // 2 + 1)
        lower = (frequencies[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
        upper = (edges[2:, None] - frequencies[None]) / (edges[2:] - edges[1:-1])[:, None]
        self.filters = torch.minimum(lower, upper).clamp_min(0)
        if (self.filters.sum(1) == 0).any():
            raise ValueError("Empty mel filter; reduce n_mels or increase n_fft")

    def stft(self, waveform):
        return torch.stft(
            waveform, n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"], win_length=self.config["win_length"],
            window=self.window, center=True, pad_mode="constant", return_complex=True,
        )

    def encode(self, waveform):
        magnitude = self.stft(waveform.cpu().float()).abs()
        return (self.filters @ magnitude).clamp_min(1e-5).log().transpose(0, 1).contiguous()

    def decode(self, log_mel, iterations=60):
        if iterations < 1 or log_mel.shape[0] < 2:
            raise ValueError("Need at least two frames and one Griffin-Lim iteration")
        mel = log_mel.detach().cpu().float().clamp(-11.512925, 8).exp().T
        magnitude = (torch.linalg.pinv(self.filters) @ mel).clamp_min(1e-7)
        phase = torch.ones_like(magnitude, dtype=torch.complex64)
        length = (mel.shape[1] - 1) * self.config["hop_length"]
        for _ in range(iterations):
            waveform = torch.istft(
                magnitude * phase, n_fft=self.config["n_fft"],
                hop_length=self.config["hop_length"], win_length=self.config["win_length"],
                window=self.window, center=True, length=length,
            )
            estimate = self.stft(waveform)
            phase = estimate / estimate.abs().clamp_min(1e-8)
        return waveform
