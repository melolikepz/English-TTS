"""Small autoregressive attention TTS baseline (not VITS)."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionTTS(nn.Module):
    def __init__(self, vocab_size, n_mels, embedding_dim=192, encoder_dim=256,
                 decoder_dim=256, attention_dim=128, prenet_dim=128, dropout=0.1):
        super().__init__()
        self.n_mels = n_mels
        self.decoder_dim = decoder_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(embedding_dim, encoder_dim // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.prenet = nn.Sequential(nn.Linear(n_mels, prenet_dim), nn.ReLU(), nn.Dropout(dropout))
        self.decoder = nn.GRUCell(prenet_dim + encoder_dim, decoder_dim)
        self.key = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.query = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.location_conv = nn.Conv1d(2, 32, kernel_size=31, padding=15, bias=False)
        self.location = nn.Linear(32, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)
        self.mel_projection = nn.Linear(decoder_dim + encoder_dim, n_mels)
        self.stop_projection = nn.Linear(decoder_dim + encoder_dim, 1)

    def encode(self, tokens, lengths):
        embedded = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        memory, _ = pad_packed_sequence(encoded, batch_first=True, total_length=tokens.shape[1])
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None] >= lengths[:, None]
        return memory, self.key(memory), mask

    def step(self, previous, hidden, context, weights, cumulative, memory, keys, mask):
        hidden = self.decoder(torch.cat((self.prenet(previous), context), dim=-1), hidden)
        history = torch.stack((weights, cumulative), dim=1)
        location = self.location(self.location_conv(history).transpose(1, 2))
        energies = self.energy(torch.tanh(keys + self.query(hidden)[:, None] + location)).squeeze(-1)
        weights = energies.masked_fill(mask, float("-inf")).softmax(dim=-1)
        context = torch.bmm(weights[:, None], memory).squeeze(1)
        combined = torch.cat((hidden, context), dim=-1)
        return self.mel_projection(combined), self.stop_projection(combined).squeeze(-1), hidden, context, weights

    def forward(self, tokens, lengths, mel_targets=None, max_frames=1200,
                min_frames=10, stop_threshold=0.5):
        memory, keys, mask = self.encode(tokens, lengths)
        batch = tokens.shape[0]
        hidden = memory.new_zeros(batch, self.decoder_dim)
        context = memory.new_zeros(batch, memory.shape[-1])
        previous = memory.new_zeros(batch, self.n_mels)
        weights = memory.new_zeros(batch, tokens.shape[1])
        cumulative = torch.zeros_like(weights)
        mels, stops, alignments = [], [], []
        steps = mel_targets.shape[1] if mel_targets is not None else max_frames
        finished = torch.zeros(batch, dtype=torch.bool, device=tokens.device)
        output_lengths = torch.full((batch,), steps, dtype=torch.long, device=tokens.device)
        for index in range(steps):
            mel, stop, hidden, context, weights = self.step(
                previous, hidden, context, weights, cumulative, memory, keys, mask,
            )
            cumulative = cumulative + weights
            mels.append(mel)
            stops.append(stop)
            alignments.append(weights)
            previous = mel_targets[:, index] if mel_targets is not None else mel
            if mel_targets is None and index + 1 >= min_frames:
                done = stop.sigmoid() >= stop_threshold
                output_lengths[done & ~finished] = index + 1
                finished = finished | done
                if finished.all():
                    break
        return {
            "mel": torch.stack(mels, dim=1),
            "stop_logits": torch.stack(stops, dim=1),
            "attention": torch.stack(alignments, dim=1),
            "lengths": output_lengths,
            "finished": finished,
        }


def tts_loss(output, batch, guided_weight=1.0):
    prediction = output["mel"]
    frames = torch.arange(prediction.shape[1], device=prediction.device)[None]
    valid = frames < batch["mel_lengths"][:, None]
    mel_loss = ((prediction - batch["mel"]).abs() * valid[:, :, None]).sum()
    mel_loss = mel_loss / (valid.sum() * prediction.shape[-1])
    # Only real frames, including the final stop frame, contribute to stop loss.
    stop_loss = F.binary_cross_entropy_with_logits(
        output["stop_logits"], batch["stop_targets"], reduction="none",
        pos_weight=prediction.new_tensor(5.0),
    )
    stop_loss = (stop_loss * valid).sum() / valid.sum()
    attention = output["attention"]
    tokens = torch.arange(attention.shape[-1], device=prediction.device)[None]
    token_valid = tokens < batch["token_lengths"][:, None]
    frame_positions = frames / batch["mel_lengths"][:, None]
    token_positions = tokens / batch["token_lengths"][:, None]
    distance = frame_positions[:, :, None] - token_positions[:, None, :]
    penalty = 1.0 - torch.exp(-distance.square() / (2 * 0.2 ** 2))
    guided = (attention * penalty * valid[:, :, None] * token_valid[:, None]).sum() / valid.sum()
    loss = mel_loss + stop_loss + guided_weight * guided
    return loss, {"mel": mel_loss.detach(), "stop": stop_loss.detach(), "guided": guided.detach()}
