import argparse

import torch

from src.tts.data.preprocessing import MelProcessor, save_wav
from src.tts.model import AttentionTTS
from src.tts.text.symbols import SYMBOLS
from src.tts.text.tokenizer import text_to_ids


def main():
    parser = argparse.ArgumentParser(description="Generate WAV from a trained baseline checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="outputs/speech.wav")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--griffin-lim-iters", type=int, default=60)
    args = parser.parse_args()
    if not args.text.strip() or args.max_frames < 10 or args.griffin_lim_iters < 1:
        parser.error("Need nonempty text, max-frames >= 10 and griffin-lim-iters >= 1")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["symbols"] != SYMBOLS:
        raise ValueError("Checkpoint vocabulary differs from the current tokenizer")
    config = checkpoint["config"]
    device = torch.device(args.device)
    model = AttentionTTS(len(SYMBOLS), config["audio"]["n_mels"], **config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens = torch.tensor([text_to_ids(args.text)], dtype=torch.long, device=device)
    lengths = torch.tensor([tokens.shape[1]], device=device)
    with torch.inference_mode():
        result = model(tokens, lengths, max_frames=args.max_frames)
    if not result["finished"].item():
        print("Warning: reached max-frames before stop prediction; speech may be truncated.")
    mel = result["mel"][0, :result["lengths"][0].item()]
    waveform = MelProcessor(config["audio"]).decode(mel, args.griffin_lim_iters)
    save_wav(args.output, waveform, config["audio"]["sample_rate"])
    print(f"Saved {args.output} ({waveform.numel() / config['audio']['sample_rate']:.2f} seconds)")


if __name__ == "__main__":
    main()
