"""One training loop for both baselines, so they differ only in the model.

* **Loss:** cross-entropy over all classes including ``none``, weighted by inverse class
  frequency on the training frames. Without weighting, three classes holding 82% of IPN
  Hand's frames would dominate and the 11 short gestures would barely be learnt.
* **Sampling:** every training frame is one sample per epoch, shuffled with a seeded
  generator. For B1 a sample is the window *ending* at that frame.
* **Model selection:** after each epoch the validation split is scored and the epoch with
  the best validation mcAP is kept. The test split is never touched here (Hard Rule 3).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

from src.data.datasets import SplitData
from src.eval.metrics import frame_metrics
from src.models.baselines import (
    FrameMLP,
    WindowMLP,
    frames_to_steps,
    predict_frames,
    predict_windows,
)

KINDS = ("b0", "b1")


@dataclass(frozen=True)
class TrainSettings:
    kind: str
    hidden: int = 512
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 1024
    window_s: float = 2.0
    stride_s: float = 0.5
    eval_batch_size: int = 512
    selection_metric: str = "mcAP"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"model kind must be one of {KINDS}, got {self.kind!r}")

    @classmethod
    def from_config(cls, cfg) -> "TrainSettings":
        keys = cls.__dataclass_fields__
        return cls(**{k: cfg[k] for k in keys if k in cfg})


def class_weights(labels: list[np.ndarray], num_classes: int) -> np.ndarray:
    """Inverse-frequency weights ``N / (C * n_c)``; absent classes get weight 0."""
    counts = np.bincount(np.concatenate(labels), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    with np.errstate(divide="ignore"):
        w = np.where(counts > 0, total / (num_classes * counts), 0.0)
    return w.astype(np.float32)


def training_window_index(
    frames: torch.Tensor, window: int, video_start: torch.Tensor
) -> torch.Tensor:
    """``[N, window]`` indices into the concatenated training frames.

    The window ends at each global frame index in ``frames`` and looks back ``window``
    frames, clamped at the start of that frame's own video -- the training-time twin of
    ``snippet_frame_indices`` at prediction time, so the model is trained on exactly the
    windows it will see live, and never on one that straddles two videos.
    """
    steps = torch.arange(window, device=frames.device)
    idx = (frames - window + 1)[:, None] + steps[None, :]
    return torch.maximum(idx, video_start[frames][:, None])


def build_model(s: TrainSettings, in_dim: int, num_classes: int) -> nn.Module:
    cls = FrameMLP if s.kind == "b0" else WindowMLP
    return cls(in_dim, num_classes, hidden=s.hidden, dropout=s.dropout)


def predict(
    s: TrainSettings, model: nn.Module, x: np.ndarray, *, fps: float, device: str
) -> np.ndarray:
    """``[T, C]`` probabilities for one video, causally."""
    if s.kind == "b0":
        return predict_frames(model, x, device=device)  # type: ignore[arg-type]
    return predict_windows(
        model,  # type: ignore[arg-type]
        x,
        length=frames_to_steps(s.window_s, fps),
        stride=frames_to_steps(s.stride_s, fps),
        device=device,
        batch_size=s.eval_batch_size,
    )


def predict_split(
    s: TrainSettings, model: nn.Module, data: SplitData, *, fps: float, device: str
) -> dict[str, np.ndarray]:
    return {v: predict(s, model, data.inputs[v], fps=fps, device=device) for v in data.videos}


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    best_score: float
    history: list[dict] = field(default_factory=list)


def train(
    s: TrainSettings,
    train_data: SplitData,
    val_data: SplitData,
    *,
    num_classes: int,
    class_names: list[str],
    fps: float,
    device: str,
    seed: int,
    on_epoch: Callable[[int, dict], None] | None = None,
) -> TrainResult:
    videos = train_data.videos
    x_all = torch.from_numpy(np.concatenate([train_data.inputs[v] for v in videos])).to(device)
    y_all = torch.from_numpy(np.concatenate([train_data.labels[v] for v in videos])).to(device)
    # Start offset of each frame's video in the concatenation, so a B1 window never
    # reaches back into the previous video: it is clamped at its own video's frame 0,
    # exactly as in prediction (warm-up repeats frame 0).
    lengths = [len(train_data.labels[v]) for v in videos]
    starts = np.repeat(np.cumsum([0, *lengths[:-1]]), lengths)
    video_start = torch.from_numpy(starts).to(device)
    window = frames_to_steps(s.window_s, fps)

    weights = torch.from_numpy(class_weights([train_data.labels[v] for v in videos], num_classes))
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    model = build_model(s, x_all.shape[1], num_classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
    generator = torch.Generator().manual_seed(seed)

    best_state, best_epoch, best_score = None, -1, -np.inf
    history: list[dict] = []
    for epoch in range(1, s.epochs + 1):
        model.train()
        order = torch.randperm(x_all.shape[0], generator=generator).to(device)
        total, batches = 0.0, 0
        for start in range(0, order.numel(), s.batch_size):
            g = order[start : start + s.batch_size]
            if s.kind == "b0":
                xb = x_all[g]
            else:
                xb = x_all[training_window_index(g, window, video_start)]
            loss = loss_fn(model(xb), y_all[g])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1

        scores = predict_split(s, model, val_data, fps=fps, device=device)
        frame = frame_metrics(
            np.concatenate([scores[v] for v in val_data.videos]),
            np.concatenate([val_data.labels[v] for v in val_data.videos]),
            class_names=class_names,
        )
        row = {
            "epoch": epoch,
            "train_loss": total / max(batches, 1),
            "val_mAP": frame["mAP"],
            "val_mcAP": frame["mcAP"],
            "val_balanced_accuracy": frame["balanced_accuracy"],
        }
        history.append(row)
        if on_epoch:
            on_epoch(epoch, row)
        score = frame[s.selection_metric]
        if score > best_score:
            best_score, best_epoch = float(score), epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, best_epoch=best_epoch, best_score=best_score, history=history)


__all__ = [
    "KINDS",
    "TrainResult",
    "TrainSettings",
    "build_model",
    "class_weights",
    "predict",
    "predict_split",
    "train",
    "training_window_index",
]
