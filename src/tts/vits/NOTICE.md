# VITS provenance

Upstream: https://github.com/jaywalnut310/vits
Commit: `2e561ba58618d021b5b8323d3765880f7e0ecfdb`
Copyright (c) 2021 Jaehyeon Kim. MIT license: see LICENSE in this directory.

models.py, modules.py, attentions.py, commons.py, transforms.py, losses.py,
symbols.py, cleaners.py and the cleaned LJSpeech filelists originate upstream.
Local changes: package-relative imports, removal of an unused scipy import,
whitespace cleanup, and bounded inference allocation in models.py.

The surrounding data/training/inference code is a local single-device integration.
Audio processing follows upstream's reflect-padded STFT and Slaney mel filters,
using modern torch/librosa APIs. Monotonic alignment is implemented in NumPy with
the same dynamic-programming recurrence (no compiled Cython dependency).
The full configured model retains stochastic duration prediction, posterior
encoder, normalizing flows, HiFi-GAN-style generator and multi-period discriminator.
The reduced test configuration is only for smoke tests, not quality evaluation.
