import argparse
import torch

from src.tts.data.preprocessing import save_wav
from src.tts.vits.runtime import make_generator, select_device
from src.tts.vits.symbols import symbols
from src.tts.vits.text import encode_text


def synthesize(checkpoint, text, output, device="auto", max_frames=1500, seed=1234,
               noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8):
    if max_frames < 1 or length_scale <= 0 or min(noise_scale, noise_scale_w) < 0:
        raise ValueError("Invalid synthesis limits/scales")
    torch.manual_seed(seed)
    device = select_device(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("architecture") != "vits" or state["symbols"] != symbols:
        raise ValueError("Expected a VITS checkpoint with matching vocabulary")
    config = state["config"]
    generator = make_generator(config).to(device).eval()
    generator.load_state_dict(state["generator"])
    del state
    tokens = torch.tensor([encode_text(text)], device=device)
    with torch.no_grad():
        wave, _, mask, _ = generator.infer(tokens, torch.tensor([tokens.shape[1]], device=device),
                                           noise_scale=noise_scale, length_scale=length_scale,
                                           noise_scale_w=noise_scale_w, max_len=max_frames)
    if int(mask.sum()) >= max_frames:
        print("Warning: duration reached max-frames; output may be truncated.")
    wave = wave[0, 0].cpu()
    save_wav(output, wave, config["data"]["sampling_rate"])
    print(f"Saved VITS waveform: {output}")
    return wave, config["data"]["sampling_rate"]


def main():
    parser = argparse.ArgumentParser(description="Generate speech with the VITS neural vocoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="outputs/vits.wav")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=1500)
    args = parser.parse_args()
    torch.set_num_threads(4)
    synthesize(args.checkpoint, args.text, args.output, args.device, args.max_frames)


if __name__ == "__main__":
    main()
