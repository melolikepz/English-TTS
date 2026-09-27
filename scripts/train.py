import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.tts.config import load_config
from src.tts.data.collate import TTSCollate
from src.tts.data.dataset import LJSpeechDataset
from src.tts.model import AttentionTTS, tts_loss
from src.tts.text.symbols import SYMBOLS


def run_epoch(model, loader, device, settings, optimizer=None, max_batches=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch["token_ids"], batch["token_lengths"], batch["mel"])
            loss, _ = tts_loss(output, batch, settings["guided_attention_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect audio and learning rate")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
            size = batch["token_ids"].shape[0]
            total += loss.item() * size
            count += size
            if (index + 1) % 20 == 0:
                print(f"  batch {index + 1}/{len(loader)} loss={loss.item():.4f}", flush=True)
            if max_batches is not None and index + 1 >= max_batches:
                break
    return total / count


def main():
    parser = argparse.ArgumentParser(description="Train an attention TTS baseline")
    parser.add_argument("--data", default="data/processed/ljspeech")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output", default="checkpoints/baseline")
    parser.add_argument("--device", default="cpu", help="cpu or cuda; CPU is the portable default")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--max-batches", type=int, help="Limit batches per epoch for smoke testing")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["training"]
    epochs = settings["epochs"] if args.epochs is None else args.epochs
    if epochs < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("epochs and max-batches must be positive")
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    device = torch.device(args.device)
    root = Path(args.data)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata["audio"] != config["audio"] or metadata["symbols"] != SYMBOLS:
        raise ValueError("Preprocessing settings or symbols changed; preprocess again")
    loaders = {
        split: DataLoader(LJSpeechDataset(root / f"{split}.jsonl"),
                          batch_size=settings["batch_size"], shuffle=split == "train",
                          num_workers=settings["num_workers"], collate_fn=TTSCollate())
        for split in ("train", "val")
    }
    model = AttentionTTS(len(SYMBOLS), config["audio"]["n_mels"], **config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    start_epoch, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["config"] != config or checkpoint["symbols"] != SYMBOLS:
            raise ValueError("Resume requires the same configuration and vocabulary")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best = checkpoint["epoch"], checkpoint["best_val"]
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output_dir.glob("*.pt")):
        parser.error("Checkpoint directory already contains models; use --resume or a new --output")
    for epoch in range(start_epoch, epochs):
        training = run_epoch(model, loaders["train"], device, settings, optimizer, args.max_batches)
        validation = run_epoch(model, loaders["val"], device, settings, max_batches=args.max_batches)
        improved = validation < best
        best = min(best, validation)
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": config, "symbols": SYMBOLS, "epoch": epoch + 1, "best_val": best,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        # Replace only after serialization completes, preserving the previous checkpoint on interruption.
        for name in (["last", "best"] if improved else ["last"]):
            temporary = output_dir / f"{name}.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output_dir / f"{name}.pt")
        print(f"Epoch {epoch + 1}: train={training:.4f} val={validation:.4f} best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
