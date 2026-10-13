"""Stage 4 baselines. Both are causal by construction and both get the same care as M1.

**B0 -- per-frame classifier.** An MLP over the input row of frame ``t`` alone. The floor:
it has no memory at all, so anything M1 gains over it comes from using time.

**B1 -- sliding-window classifier.** The approach the author's supervisor proposed, built
honestly and well, not as a strawman: a window of the last ``window_s`` seconds ending at
frame ``t`` (never centred), summarised by its mean, its maximum and its last row, then an
MLP. Windows are evaluated every ``stride_s`` seconds on a grid anchored at frame 0, and
each prediction is held until the next window -- the same causal scheduling as the Stage 3
feature snippets, reusing the same tested helpers. The target of a window is the label of
its last frame, i.e. "what is happening now", which is what an online alarm needs; a
majority-label target would teach it to answer for the middle of the window, a second ago.

Prediction pads the final batch to a fixed size. Float32 matrix products are not invariant
to the number of rows in a batch, so without the padding a truncated stream would end in a
smaller batch and fail the truncation test for a reason that has nothing to do with the
future (the same effect found at Stage 3; see tests/test_causality.py).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn

from src.features import causal


class FrameMLP(nn.Module):
    """B0: one hidden layer over a single frame's input row."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, D] -> [N, C]
        return self.net(x)


class WindowMLP(nn.Module):
    """B1: mean, max and last row of a ``[N, L, D]`` window, then an MLP."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    @staticmethod
    def pool(windows: torch.Tensor) -> torch.Tensor:
        return torch.cat([windows.mean(dim=1), windows.amax(dim=1), windows[:, -1]], dim=1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:  # [N, L, D] -> [N, C]
        return self.net(self.pool(windows))


def window_index(ends: np.ndarray, length: int) -> np.ndarray:
    """``[S, L]`` frame indices of the windows ending at ``ends`` (backward-looking)."""
    return np.stack([causal.snippet_frame_indices(int(e), length) for e in ends])


def frames_to_steps(seconds: float, fps: float) -> int:
    """A duration in seconds as a whole number of frames, at least 1."""
    return max(1, int(round(seconds * fps)))


@torch.no_grad()
def _run_padded(
    fn: Callable[[torch.Tensor], torch.Tensor], items: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Apply ``fn`` in batches of exactly ``batch_size`` rows, padding the last one."""
    outs = []
    for start in range(0, items.shape[0], batch_size):
        chunk = items[start : start + batch_size]
        real = chunk.shape[0]
        if real < batch_size:
            pad = chunk[-1:].expand(batch_size - real, *chunk.shape[1:])
            chunk = torch.cat([chunk, pad], dim=0)
        outs.append(fn(chunk)[:real])
    return torch.cat(outs, dim=0)


@torch.no_grad()
def predict_frames(
    model: FrameMLP, x: np.ndarray, *, device: str, batch_size: int = 4096
) -> np.ndarray:
    """B0 probabilities ``[T, C]`` for one video."""
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    logits = _run_padded(model, xt, batch_size)
    return torch.softmax(logits, dim=1).cpu().numpy()


@torch.no_grad()
def predict_windows(
    model: WindowMLP,
    x: np.ndarray,
    *,
    length: int,
    stride: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """B1 probabilities ``[T, C]`` for one video: windows on a 0-anchored grid, held."""
    model.eval()
    num_frames = x.shape[0]
    ends = causal.snippet_ends(num_frames, stride)
    idx = torch.from_numpy(window_index(ends, length))
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)

    def run(index_batch: torch.Tensor) -> torch.Tensor:
        return model(xt[index_batch.to(device)])

    logits = _run_padded(run, idx, batch_size)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return causal.hold_last(probs, ends, num_frames)


__all__ = [
    "FrameMLP",
    "WindowMLP",
    "frames_to_steps",
    "predict_frames",
    "predict_windows",
    "window_index",
]
