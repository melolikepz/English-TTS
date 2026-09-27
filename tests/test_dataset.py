import json

import pytest
import torch

from src.tts.data.collate import TTSCollate
from src.tts.data.dataset import LJSpeechDataset, read_metadata
from src.tts.data.preprocessing import MelProcessor, load_wav, save_wav
from src.tts.model import AttentionTTS, tts_loss
from src.tts.text.symbols import UNK, symbol_to_id


AUDIO = dict(sample_rate=22050, n_fft=128, win_length=128, hop_length=32,
             n_mels=8, f_min=0, f_max=8000)


def test_audio_features_and_reconstruction(tmp_path):
    signal = 0.2 * torch.sin(torch.arange(1000) * 0.1)
    path = tmp_path / "test.wav"
    save_wav(path, signal, AUDIO["sample_rate"])
    loaded = load_wav(path, AUDIO["sample_rate"])
    assert torch.allclose(signal, loaded, atol=1e-4)
    processor = MelProcessor(AUDIO)
    mel = processor.encode(loaded)
    assert mel.shape == (32, 8)
    reconstructed = processor.decode(mel, iterations=2)
    assert reconstructed.shape == (31 * 32,)
    assert torch.isfinite(reconstructed).all()
    assert reconstructed.abs().max() > 0
    with pytest.raises(ValueError, match="Expected 16000"):
        load_wav(path, 16000)


def test_metadata_and_cached_dataset(tmp_path):
    (tmp_path / "wavs").mkdir()
    save_wav(tmp_path / "wavs" / "LJ001.wav", torch.zeros(100), 22050)
    (tmp_path / "metadata.csv").write_text("LJ001|12|twelve\n", encoding="utf-8")
    assert read_metadata(tmp_path)[0][2] == "twelve"
    sample = {"token_ids": torch.tensor([1, 4, 2]), "mel": torch.zeros(4, 8), "text": "twelve"}
    torch.save(sample, tmp_path / "sample.pt")
    (tmp_path / "train.jsonl").write_text(json.dumps({"file": "sample.pt"}) + "\n")
    dataset = LJSpeechDataset(tmp_path / "train.jsonl")
    assert len(dataset) == 1
    assert torch.equal(dataset[0]["token_ids"], sample["token_ids"])
    (tmp_path / "metadata.csv").write_text("bad row\n")
    with pytest.raises(ValueError, match="Invalid metadata"):
        read_metadata(tmp_path)


def test_padding_loss_gradients_and_autoregression():
    torch.manual_seed(42)
    batch = TTSCollate()([
        {"token_ids": torch.tensor([1, 4, 2]), "mel": torch.randn(5, 8)},
        {"token_ids": torch.tensor([1, 2]), "mel": torch.randn(3, 8)},
    ])
    assert batch["stop_targets"].tolist() == [[0, 0, 0, 0, 1], [0, 0, 1, 1, 1]]
    model = AttentionTTS(10, 8, embedding_dim=8, encoder_dim=16,
                         decoder_dim=16, attention_dim=8, prenet_dim=8, dropout=0)
    output = model(batch["token_ids"], batch["token_lengths"], batch["mel"])
    assert output["attention"][1, :, 2].eq(0).all()
    loss, _ = tts_loss(output, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embedding.weight.grad.abs().sum() > 0
    changed = {**output, "mel": output["mel"].detach().clone(),
               "stop_logits": output["stop_logits"].detach().clone()}
    changed["mel"][1, 3:] = 1000
    changed["stop_logits"][1, 3:] = 1000
    other_loss, _ = tts_loss(changed, batch)
    assert torch.allclose(loss, other_loss)
    model.eval()
    with torch.no_grad():
        model.stop_projection.weight.zero_()
        model.stop_projection.bias.fill_(100)
        generated = model(batch["token_ids"], batch["token_lengths"], max_frames=5, min_frames=2)
    assert generated["mel"].shape == (2, 2, 8)
    assert generated["lengths"].tolist() == [2, 2]
    assert generated["finished"].all()


def test_word_boundary_has_its_own_symbol():
    assert symbol_to_id("|") != symbol_to_id(UNK)


def test_retokenize_preserves_mels_splits_and_source(tmp_path, monkeypatch):
    from scripts import retokenize as script
    from src.tts.text.symbols import SYMBOLS

    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    original = {"mel": torch.randn(4, 8), "token_ids": torch.tensor([1, 3, 2]), "text": "test"}
    (source / "metadata.json").write_text(json.dumps({"audio": AUDIO, "symbols": SYMBOLS[:-9]}))
    for split in ("train", "val"):
        torch.save(original, source / f"{split}.pt")
        (source / f"{split}.jsonl").write_text(json.dumps({"file": f"{split}.pt", "tokens": 3, "frames": 4}) + "\n")
    monkeypatch.setattr(script, "texts_to_phonemes", lambda texts: [["ɾ", "ᵻ"] for _ in texts])
    assert not script.retokenize(source, destination, batch_size=1)
    for split in ("train", "val"):
        old = LJSpeechDataset(source / f"{split}.jsonl")[0]
        new = LJSpeechDataset(destination / f"{split}.jsonl")[0]
        assert torch.equal(old["token_ids"], original["token_ids"])
        assert torch.equal(old["mel"], new["mel"])
        assert symbol_to_id(UNK) not in new["token_ids"].tolist()
        assert new["text"] == old["text"]
    with pytest.raises(ValueError, match="Output must be empty"):
        script.retokenize(source, destination)
