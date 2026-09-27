# English TTS — VITS

Local single-speaker training and synthesis with the **VITS architecture** from
[jaywalnut310/vits](https://github.com/jaywalnut310/vits), pinned to commit
`2e561ba58618d021b5b8323d3765880f7e0ecfdb` (MIT).
See [source provenance](src/tts/vits/NOTICE.md) and [license](src/tts/vits/LICENSE).

The model includes a Transformer text encoder, variational posterior encoder,
normalizing flows, stochastic duration predictor, neural waveform generator and
multi-period discriminator. The training losses are adversarial, feature matching,
mel reconstruction, KL and duration. It does **not** use Griffin–Lim.

## Start Jupyter locally

From this repository root, using Python 3.10 or newer:

```sh
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --sys-prefix --name english-tts --display-name "English TTS (.venv)"
python -m jupyterlab notebooks/VITS_Local.ipynb
```

Choose **English TTS (.venv)** as the notebook kernel. Run cells in order.
The notebook prepares the dataset, runs a small-model smoke test, resumes it,
and exposes separate full-model training, metrics and audio playback cells.
Long training and synthesis are explicitly toggled off by default.

Phonemizer requires eSpeak/eSpeak NG (`brew install espeak-ng` on macOS).
The training manifests use the official pre-cleaned LJSpeech phonemes; inference
uses the upstream English cleaner and vocabulary, with blank tokens inserted.
No pretrained weights are bundled or downloaded.

## Dataset

Keep the original extracted audio in:

```text
data/raw/LJSpeech-1.1/
  metadata.csv
  wavs/
```

```sh
python -m scripts.prepare_vits
```

This validates mono PCM16/22050 Hz audio headers and creates VITS manifests using
upstream splits: 12,500 training, 100 validation and 500 test utterances. Samples
that cannot fit the alignment or training segment are reported and excluded.
Preparation is repeatable with identical settings. WAV paths are relative to the
manifest directory; preserve the directory layout if moving the project.

VITS reads raw WAV and computes its linear-magnitude spectrogram on demand.
The previous baseline mel caches, tokenizer and checkpoints are incompatible.
They remain available for reference; nothing was deleted. VITS uses `configs/vits.json`,
not the empty legacy `configs/vits.yaml`.

## Train from the terminal

```sh
python -m scripts.train_vits --device cpu --epochs 1
```

CPU is appropriate for local functional checks; full VITS training is very slow.
With an available NVIDIA GPU, use `--device cuda`. `auto` selects CUDA if available,
otherwise CPU. MPS and mixed-precision training are not enabled or validated here.
The default batch size is 2. The full model architecture is retained; a smaller
batch does not make it the reduced smoke model.

```sh
python -m scripts.train_vits --device cpu --epochs 100 \
  --resume checkpoints/vits/last.pt
```

`--epochs` and `--max-steps` are total targets, including completed work.
A one-epoch run is only an initial check, not a speech-quality target.
Checkpoints include both networks, both AdamW optimizers, RNG states, epoch,
next batch, configuration and dataset fingerprint. Resume preserves epoch order
and learning-rate decay. Use identical settings/data, including batch size.
`last.pt` is saved every 100 steps and at epoch boundaries. Ctrl+C requests a stop
**after the current optimizer step** and saves `last.pt`; wait for confirmation.
An abrupt process kill/power loss can only recover the last completed save.
Checkpoint serialization uses a temporary file followed by replacement.

At each epoch boundary, up to 8 validation batches produce a deterministic
posterior-reconstruction mel metric, selecting `best.pt`. Change `--val-batches`
if needed. This metric is not intelligibility or MOS; listen to generated speech
and inspect alignment before judging quality. Component losses are recorded in
`metrics.jsonl` and cannot be compared to the old GRU model's loss.

## Synthesize

```sh
python -m scripts.infer_vits \
  --checkpoint checkpoints/vits/last.pt \
  --text "Hello, this is my English text to speech model." \
  --output outputs/vits.wav
```

This generates waveform directly through VITS. Duration allocation is capped by
`--max-frames` (default 1500); a warning indicates possible truncation. An untrained
checkpoint produces noise and is only useful for testing the code path.
Only this integration's VITS checkpoints are accepted by this command.

## Validation

```sh
python -m pytest -q
```

Tests include MAS against exhaustive optimal paths, differentiable reference STFT,
a real VITS generator/discriminator update, gradients through all main components,
and bounded neural-waveform inference. `configs/vits-smoke.json` reduces hidden
sizes for checks while retaining all architecture components and the discriminator.
The notebook additionally exercises save/resume and WAV generation on local data.
Local CPU tests do not establish GPU compatibility, convergence or voice quality.

The superseded GRU baseline documentation is in [docs/BASELINE.md](docs/BASELINE.md).
Its old commands remain available for reproducibility; use the VITS commands above
for this project going forward.
