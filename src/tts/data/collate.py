import torch
from torch.nn.utils.rnn import pad_sequence

from src.tts.text.symbols import PAD, symbol_to_id


class RawTTSCollate:
    """Pad raw audio batches; cached mel training uses TTSCollate below."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        tokens = [sample["token_ids"] for sample in batch]
        waveforms = [sample["waveform"] for sample in batch]

        token_lengths = torch.tensor(
            [tokens_i.numel() for tokens_i in tokens],
            dtype=torch.long,
        )

        waveform_lengths = torch.tensor(
            [audio.numel() for audio in waveforms],
            dtype=torch.long,
        )

        padded_tokens = pad_sequence(
            tokens,
            batch_first=True,
            padding_value=self.pad_id,
        )

        padded_waveforms = pad_sequence(
            waveforms,
            batch_first=True,
            padding_value=0.0,
        )

        return {
            "token_ids": padded_tokens,
            "token_lengths": token_lengths,
            "waveform": padded_waveforms.unsqueeze(1),
            "waveform_lengths": waveform_lengths,
            "texts": [sample["text"] for sample in batch],
            "audio_paths": [sample["audio_path"] for sample in batch],
        }


class TTSCollate:
    def __call__(self, batch):
        if not batch:
            raise ValueError("Empty batch")
        tokens = [sample["token_ids"] for sample in batch]
        mels = [sample["mel"] for sample in batch]
        token_lengths = torch.tensor([len(item) for item in tokens])
        mel_lengths = torch.tensor([len(item) for item in mels])
        mel = pad_sequence(mels, batch_first=True, padding_value=0.0)
        steps = torch.arange(mel.shape[1])[None]
        return {
            "token_ids": pad_sequence(tokens, batch_first=True, padding_value=symbol_to_id(PAD)),
            "token_lengths": token_lengths,
            "mel": mel,
            "mel_lengths": mel_lengths,
            "stop_targets": (steps >= mel_lengths[:, None] - 1).float(),
        }
