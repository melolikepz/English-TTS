import json
from pathlib import Path


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    audio = config["audio"]
    if not 0 <= audio["f_min"] < audio["f_max"] <= audio["sample_rate"] / 2:
        raise ValueError("Invalid mel frequency range")
    if not 0 < audio["hop_length"] < audio["win_length"] <= audio["n_fft"]:
        raise ValueError("Invalid STFT settings")
    if config["model"]["encoder_dim"] % 2:
        raise ValueError("encoder_dim must be even")
    return config
