# English TTS from scratch

A single-speaker LJSpeech baseline: phoneme tokenization, cached log-magnitude mel
features, a bidirectional GRU encoder, an autoregressive decoder with
location-sensitive attention, masked mel/stop losses, guided attention, checkpoint
resume, and Griffin–Lim WAV synthesis. This is an educational baseline, not VITS.
Use `configs/baseline.json`; the empty `configs/vits.yaml` is an unused placeholder.

No pretrained weights or dataset are bundled. Meaningful speech requires training
and learned attention alignment. A smoke run does not establish speech quality.
Griffin–Lim is less natural than a trained neural vocoder.

## Setup

Run commands from the repository root with Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Phonemizer requires eSpeak/eSpeak NG. Install with `brew install espeak-ng` on
macOS or `sudo apt-get install espeak-ng` on Debian/Ubuntu. If library discovery
fails, set `PHONEMIZER_ESPEAK_LIBRARY` to the installed shared library's absolute path.

## Prepare LJSpeech

Extract the dataset into:

```text
data/raw/LJSpeech-1.1/
  metadata.csv
  wavs/
    LJ001-0001.wav
    ...
```

Metadata rows are `utterance_id|original_text|normalized_text`; normalized text is
preferred. Audio must be mono 16-bit PCM at the configured rate (22050 Hz by default).
Mismatches raise an error; no resampling happens silently.

```sh
python -m scripts.prepare_data --data-root data/raw/LJSpeech-1.1
python -m scripts.preprocess --data-root data/raw/LJSpeech-1.1
```

Preprocessing saves token IDs, `[frames, mel_bins]` features, reproducible
train/validation splits, and audio/vocabulary settings. Its output directory must
be empty. After a failed run, choose a fresh `--output` directory. Changes to audio
settings or symbols require preprocessing again. Unknown phonemes are counted.
The earlier raw-WAV helpers remain available as `RawLJSpeechDataset` and
`RawTTSCollate`. Training uses `LJSpeechDataset` and `TTSCollate` for cached mels.

If an older cache reports unknown tokens from missing vocabulary symbols, update
the vocabulary and rebuild only the tokens (reusing mels, keeping the original
cache and train/validation split):

```sh
python -m scripts.retokenize --source data/processed/ljspeech --output data/processed/ljspeech-v2
python -m scripts.train --data data/processed/ljspeech-v2 --device cpu
```

The destination must be empty. `unknown_phonemes.json` records any remaining
unknown symbols and their counts. Vocabulary changes require a fresh model;
older checkpoints cannot be resumed with the expanded vocabulary.

## Train and resume

```sh
python -m scripts.train --device cpu
python -m scripts.train --device cpu --resume checkpoints/baseline/last.pt --epochs 150
```

Use `--device cuda` with a supported NVIDIA setup. CPU works but full training is
slow; accelerator performance and model convergence have not been established.
Reduce `batch_size` in the configuration if memory is limited. Each epoch saves
`last.pt`; improved validation loss also saves `best.pt`.

Resume restores model, optimizer, epoch, best validation loss, and torch RNG states.
Keep the same preprocessed data and configuration. `--epochs` is the total target
epoch count, including completed epochs. Existing checkpoint directories require
`--resume` or a fresh `--output`.

## Generate WAV

```sh
python -m scripts.inference \
  --checkpoint checkpoints/baseline/best.pt \
  --text "Hello, this is my English text to speech model." \
  --output outputs/hello.wav
```

Inference reads settings from the checkpoint. `--max-frames` defaults to 1200
(about 14 seconds); a warning means decoding reached the cap before predicting
a stop. Split longer text into sentences. `--griffin-lim-iters` defaults to 60.

## Tests and smoke run

```sh
python -m pytest -q
python -m scripts.preprocess --data-root data/raw/LJSpeech-1.1 \
  --output data/processed/smoke --limit 8
python -m scripts.train --data data/processed/smoke \
  --output checkpoints/smoke --epochs 1 --max-batches 1
```

Smoke runs verify plumbing only. Tests cover audio conversion, cached samples,
padding masks, gradients, attention masking, and autoregressive stopping.
The existing phonemizer tests require the eSpeak backend.
