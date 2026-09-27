"""CPU NumPy MAS, equivalent to the upstream VITS dynamic program."""
import numpy as np
import torch


def maximum_path(scores, mask):
    values = scores.detach().float().cpu().numpy()
    valid = mask.detach().cpu().numpy().astype(bool)
    result = np.zeros_like(values)
    for b in range(values.shape[0]):
        frames = int(valid[b].any(axis=1).sum())
        tokens = int(valid[b].any(axis=0).sum())
        if tokens < 1 or frames < tokens:
            raise ValueError("MAS requires at least one audio frame per text token")
        previous = np.full(tokens, -np.inf, dtype=np.float32)
        previous[0] = 0.0
        advance = np.zeros((frames, tokens), dtype=bool)
        positions = np.arange(tokens)
        for t in range(frames):
            shifted = np.concatenate((np.array([-np.inf], dtype=np.float32), previous[:-1]))
            advance[t] = shifted > previous
            current = values[b, t, :tokens] + np.maximum(previous, shifted)
            reachable = (positions <= t) & (positions >= tokens + t - frames)
            previous = np.where(reachable, current, -np.inf)
        x = tokens - 1
        for t in range(frames - 1, -1, -1):
            result[b, t, x] = 1
            if advance[t, x]:
                x -= 1
    return torch.from_numpy(result).to(device=scores.device, dtype=scores.dtype)
