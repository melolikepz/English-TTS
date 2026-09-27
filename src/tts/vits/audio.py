"""Upstream VITS magnitude STFT and Slaney-normalized log-mel loss features."""
import torch
from torch import nn
from torch.nn import functional as F
from librosa.filters import mel


class VITSAudio(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("window", torch.hann_window(config["win_length"]))
        filters = mel(sr=config["sampling_rate"], n_fft=config["filter_length"],
                      n_mels=config["n_mel_channels"], fmin=config["mel_fmin"], fmax=config["mel_fmax"])
        self.register_buffer("filters", torch.from_numpy(filters))

    def spectrogram(self, waveform):
        c = self.config
        padding = (c["filter_length"] - c["hop_length"]) // 2
        waveform = F.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spec = torch.stft(waveform, c["filter_length"], hop_length=c["hop_length"],
                          win_length=c["win_length"], window=self.window,
                          center=False, return_complex=True)
        return (spec.real.square() + spec.imag.square() + 1e-6).sqrt()

    def spec_to_mel(self, spec):
        return (self.filters @ spec).clamp_min(1e-5).log()

    def forward(self, waveform):
        return self.spec_to_mel(self.spectrogram(waveform))
