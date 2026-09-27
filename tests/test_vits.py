import itertools
import json

import numpy as np
import pytest
import torch

from src.tts.vits.audio import VITSAudio
from src.tts.vits.monotonic_align import maximum_path
from src.tts.vits.runtime import load_config, make_generator, make_discriminator
from src.tts.vits.text import encode_text
from src.tts.vits.training import train_step


@pytest.fixture
def smoke_config():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield load_config("configs/vits-smoke.json")
    torch.set_num_threads(previous_threads)


def test_mas_matches_exhaustive_optimum_and_masks():
    torch.manual_seed(5)
    scores = torch.randn(2, 6, 4)
    mask = torch.ones_like(scores)
    mask[1, 4:] = 0
    mask[1, :, 2:] = 0
    path = maximum_path(scores, mask)
    for b, (frames, tokens) in enumerate([(6, 4), (4, 2)]):
        candidates = []
        for transitions in itertools.combinations(range(1, frames), tokens - 1):
            alignment = np.zeros(frames, dtype=int)
            for transition in transitions:
                alignment[transition:] += 1
            candidates.append(sum(scores[b, t, x].item() for t, x in enumerate(alignment)))
        assert float((path[b] * scores[b]).sum()) == pytest.approx(max(candidates), abs=1e-5)
        assert torch.equal(path[b, :frames].sum(-1), torch.ones(frames))
        assert (path[b].sum(0)[:tokens] > 0).all()
    assert (path * (1 - mask)).sum() == 0
    with pytest.raises(ValueError, match="at least one"):
        maximum_path(torch.zeros(1, 2, 3), torch.ones(1, 2, 3))


def test_vits_spectrum_matches_reference_stft(smoke_config):
    processor = VITSAudio(smoke_config["data"])
    wave = torch.randn(1, 2048, requires_grad=True)
    result = processor.spectrogram(wave)
    padded = torch.nn.functional.pad(wave[:, None], (48, 48), mode="reflect")[:, 0]
    windows = padded.unfold(-1, 128, 32) * torch.hann_window(128)
    reference = (torch.fft.rfft(windows).abs().square() + 1e-6).sqrt().transpose(1, 2)
    assert torch.allclose(result, reference, atol=1e-5)
    processor(wave).mean().backward()
    assert torch.isfinite(wave.grad).all() and wave.grad.abs().sum() > 0


def test_real_vits_gan_step_and_inference(smoke_config):
    torch.manual_seed(1234)
    generator, discriminator = make_generator(smoke_config), make_discriminator(smoke_config)
    processor = VITSAudio(smoke_config["data"])
    tokens = torch.tensor([encode_text("hɛloʊ", cleaned=True)])
    wave = 0.1 * torch.sin(torch.arange(2048).float()[None] * 0.05)
    spec = processor.spectrogram(wave)
    batch = (tokens, torch.tensor([tokens.shape[1]]), spec, torch.tensor([spec.shape[-1]]),
             wave[:, None], torch.tensor([2048]))
    optim_g = torch.optim.AdamW(generator.parameters(), lr=0.0002)
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=0.0002)
    before = next(discriminator.parameters()).detach().clone()
    losses = train_step(generator, discriminator, optim_g, optim_d, processor, batch, smoke_config)
    assert all(np.isfinite(value) for value in losses.values())
    assert not torch.equal(before, next(discriminator.parameters()).detach())
    for component in [generator.enc_p, generator.enc_q, generator.flow, generator.dp, generator.dec]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    generator.eval()
    with torch.no_grad():
        audio, _, mask, _ = generator.infer(tokens, batch[1], max_len=12)
    assert audio.shape[0:2] == (1, 1)
    assert 0 < audio.shape[-1] <= 12 * smoke_config["data"]["hop_length"]
    assert mask.shape[-1] <= 12
    assert torch.isfinite(audio).all()


def test_vits_text_rejects_unknowns():
    tokens = encode_text("hɛloʊ", cleaned=True)
    assert tokens[::2] == [0] * 6
    with pytest.raises(ValueError, match="Unsupported"):
        encode_text("☃", cleaned=True)
