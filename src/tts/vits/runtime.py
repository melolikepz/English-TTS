import json
import math
from pathlib import Path

import torch

from .models import SynthesizerTrn, MultiPeriodDiscriminator
from .symbols import symbols


def load_config(path):
    config = json.loads(Path(path).read_text())
    data, model, train = config["data"], config["model"], config["train"]
    if math.prod(model["upsample_rates"]) != data["hop_length"]:
        raise ValueError("Vocoder upsampling must equal hop_length")
    if train["segment_size"] % data["hop_length"]:
        raise ValueError("segment_size must be divisible by hop_length")
    if train["segment_size"] < data["filter_length"]:
        raise ValueError("segment_size must be at least filter_length")
    if model.get("use_sdp") is not True:
        raise ValueError("This VITS configuration requires stochastic duration prediction")
    return config


def make_generator(config):
    return SynthesizerTrn(len(symbols), config["data"]["filter_length"] // 2 + 1,
                          config["train"]["segment_size"] // config["data"]["hop_length"],
                          **config["model"])


def make_discriminator(config):
    return MultiPeriodDiscriminator(config["model"].get("use_spectral_norm", False))


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda; MPS training has not been validated")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("This PyTorch installation has no available CUDA GPU")
    return torch.device(name)
