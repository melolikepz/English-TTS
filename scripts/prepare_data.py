"""Validate an already downloaded, extracted LJSpeech dataset."""
import argparse

from src.tts.data.dataset import read_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    records = read_metadata(args.data_root)
    print(f"Validated {len(records)} metadata entries and WAV paths.")
    print("Next: python -m scripts.preprocess --data-root PATH")


if __name__ == "__main__":
    main()
