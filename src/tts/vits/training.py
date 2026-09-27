"""Single-device full-precision training with the original VITS objectives."""
import torch
from torch.nn import functional as F

from . import commons
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss


def train_step(generator, discriminator, optim_g, optim_d, audio, batch, config):
    x, x_lengths, spec, spec_lengths, wave, _ = batch
    output, duration, attention, starts, _, z_mask, latent = generator(x, x_lengths, spec, spec_lengths)
    _, z_p, m_p, logs_p, _, logs_q = latent
    segment = config["train"]["segment_size"]
    hop = config["data"]["hop_length"]
    target_mel = commons.slice_segments(audio.spec_to_mel(spec), starts, segment // hop)
    generated_mel = audio(output.squeeze(1))
    target_wave = commons.slice_segments(wave, starts * hop, segment)

    optim_d.zero_grad(set_to_none=True)
    real, fake, _, _ = discriminator(target_wave, output.detach())
    loss_d, _, _ = discriminator_loss(real, fake)
    if not torch.isfinite(loss_d):
        raise RuntimeError("Non-finite discriminator loss")
    loss_d.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float("inf"), error_if_nonfinite=True)
    optim_d.step()
    optim_d.zero_grad(set_to_none=True)

    # Discriminator weights are constants for the generator update, but gradients
    # still flow through its operations into the generated waveform.
    discriminator.requires_grad_(False)
    try:
        _, fake, real_maps, fake_maps = discriminator(target_wave, output)
        adversarial, _ = generator_loss(fake)
        matching = feature_loss(real_maps, fake_maps)
        mel = F.l1_loss(target_mel, generated_mel) * config["train"]["c_mel"]
        kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["c_kl"]
        duration_loss = duration.sum()
        loss_g = adversarial + matching + mel + kl + duration_loss
        if not torch.isfinite(loss_g):
            raise RuntimeError("Non-finite generator loss")
        optim_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), float("inf"), error_if_nonfinite=True)
        optim_g.step()
    finally:
        discriminator.requires_grad_(True)
    return {"g": loss_g.item(), "d": loss_d.item(), "mel": mel.item(),
            "kl": kl.item(), "duration": duration_loss.item(), "feature": matching.item(),
            "adversarial": adversarial.item()}


@torch.no_grad()
def validate(generator, audio, loader, device, config, max_batches):
    """Posterior-reconstruction mel metric, not a speech intelligibility score."""
    generator.eval()
    total, count = 0.0, 0
    # Fix stochastic posterior/segment draws without changing training's RNG.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["train"]["seed"])
        for index, batch in enumerate(loader):
            x, lengths, spec, spec_lengths, _, _ = [item.to(device) for item in batch]
            output, _, _, starts, _, _, _ = generator(x, lengths, spec, spec_lengths)
            frames = config["train"]["segment_size"] // config["data"]["hop_length"]
            target = commons.slice_segments(audio.spec_to_mel(spec), starts, frames)
            loss = F.l1_loss(audio(output.squeeze(1)), target)
            total += loss.item() * len(x)
            count += len(x)
            if index + 1 >= max_batches:
                break
    generator.train()
    return total / count
