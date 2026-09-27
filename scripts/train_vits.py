import argparse
import json
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader, Subset

from src.tts.vits.audio import VITSAudio
from src.tts.vits.data import VITSDataset, collate_vits
from src.tts.vits.runtime import load_config, make_generator, make_discriminator, select_device
from src.tts.vits.symbols import symbols
from src.tts.vits.training import train_step, validate


def save_checkpoint(path, generator, discriminator, optim_g, optim_d, config,
                    fingerprint, epoch, next_batch, step, best):
    checkpoint = {
        "architecture": "vits", "config": config, "symbols": symbols,
        "data_fingerprint": fingerprint, "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(), "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(), "epoch": epoch, "next_batch": next_batch,
        "step": step, "best": best, "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(generator.parameters()).is_cuda else None,
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the reference VITS architecture")
    parser.add_argument("--config", default="configs/vits.json")
    parser.add_argument("--data", default="data/processed/vits")
    parser.add_argument("--output", default="checkpoints/vits")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, help="Total target epochs")
    parser.add_argument("--max-steps", type=int, help="Total target optimizer steps; use for smoke tests")
    parser.add_argument("--resume", help="Path to a VITS last.pt checkpoint")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=8)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    settings = config["train"]
    epochs = args.epochs if args.epochs is not None else settings["epochs"]
    if min(epochs, args.threads, args.val_batches, settings["batch_size"], settings["save_interval"], settings["log_interval"]) < 1:
        parser.error("Epochs, threads, batch sizes and intervals must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(settings["seed"])
    device = select_device(args.device)
    root, destination = Path(args.data), Path(args.output)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("architecture") != "vits" or metadata["data"] != config["data"] or metadata["symbols"] != symbols:
        raise ValueError("Use scripts.prepare_vits with this config; baseline caches are incompatible")
    destination.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(destination.glob("*.pt")):
        parser.error("Use --resume or a fresh checkpoint directory")
    train_data = VITSDataset(root / "train.jsonl", config)
    validation = DataLoader(VITSDataset(root / "val.jsonl", config), batch_size=settings["batch_size"],
                            collate_fn=collate_vits, num_workers=settings["num_workers"])
    generator, discriminator = make_generator(config).to(device), make_discriminator(config).to(device)
    audio = VITSAudio(config["data"]).to(device)
    optim_args = dict(lr=settings["learning_rate"], betas=tuple(settings["betas"]), eps=settings["eps"])
    optim_g, optim_d = torch.optim.AdamW(generator.parameters(), **optim_args), torch.optim.AdamW(discriminator.parameters(), **optim_args)
    epoch, next_batch, step, best = 0, 0, 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if state.get("architecture") != "vits" or state["config"] != config or state["symbols"] != symbols or state["data_fingerprint"] != metadata["fingerprint"]:
            raise ValueError("Resume requires a VITS checkpoint with identical config, symbols and data")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optim_g.load_state_dict(state["optim_g"])
        optim_d.load_state_dict(state["optim_d"])
        epoch, next_batch, step, best = state["epoch"], state["next_batch"], state["step"], state["best"]
        torch.set_rng_state(state["rng"].cpu())
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        del state
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested: finishing this optimizer step, then saving last.pt.", flush=True)

    old_handler = signal.signal(signal.SIGINT, request_stop)
    def save(name="last.pt"):
        save_checkpoint(destination / name, generator, discriminator, optim_g, optim_d, config,
                        metadata["fingerprint"], epoch, next_batch, step, best)

    print(f"VITS | device={device} | batch_size={settings['batch_size']} | step={step}", flush=True)
    if device.type == "cpu":
        print("CPU is suitable for local checks; full VITS training is very slow.", flush=True)
    try:
        while epoch < epochs and (args.max_steps is None or step < args.max_steps):
            # Recreate the exact epoch ordering, then start at the next saved batch.
            ordering = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(settings["seed"] + epoch)).tolist()
            offset = next_batch * settings["batch_size"]
            loader = DataLoader(Subset(train_data, ordering[offset:]), batch_size=settings["batch_size"],
                                collate_fn=collate_vits, num_workers=settings["num_workers"],
                                generator=torch.Generator().manual_seed(settings["seed"] + epoch))
            for optimizer in (optim_g, optim_d):
                for group in optimizer.param_groups:
                    group["lr"] = settings["learning_rate"] * settings["lr_decay"] ** epoch
            generator.train()
            discriminator.train()
            for batch in loader:
                started = time.monotonic()
                losses = train_step(generator, discriminator, optim_g, optim_d, audio,
                                    [item.to(device) for item in batch], config)
                step += 1
                next_batch += 1
                if step == 1 or step % settings["log_interval"] == 0:
                    print(f"epoch={epoch + 1} batch={next_batch} step={step} " +
                          " ".join(f"{key}={value:.4f}" for key, value in losses.items()) +
                          f" seconds={time.monotonic() - started:.1f}", flush=True)
                with (destination / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch + 1, "step": step, **losses}) + "\n")
                if step % settings["save_interval"] == 0:
                    save()
                if stop_requested or (args.max_steps is not None and step >= args.max_steps):
                    save()
                    print(f"Saved {destination / 'last.pt'} at step {step}", flush=True)
                    return
            score = validate(generator, audio, validation, device, config, args.val_batches)
            if not torch.isfinite(torch.tensor(score)):
                raise RuntimeError("Non-finite validation mel loss")
            improved = score < best
            best = min(best, score)
            epoch += 1
            next_batch = 0
            save()
            if improved:
                save("best.pt")
            print(f"Epoch {epoch}: val_reconstruction_mel={score:.4f} (up to {args.val_batches} batches)", flush=True)
            if stop_requested:
                return
    finally:
        signal.signal(signal.SIGINT, old_handler)


if __name__ == "__main__":
    main()
