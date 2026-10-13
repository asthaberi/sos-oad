"""The Stage 4 alarm rule: fire when the top class switches into a signal class.

Every method compared at Stage 4 -- B0, B1, and later M1 before the full Stage 6 layer --
turns its scores into alarms with this one rule, so differences in the alarm metrics come
from the scores and not from rules tuned per method.

At each frame the *active* class is the argmax, demoted to ``none`` when its probability is
below ``threshold``. An alarm fires on the first frame of every run of the same active
signal class. The rule reads only the current and previous frame, so it is causal; the
truncation test in ``tests/test_eval.py`` checks that.

This is deliberately not the decision layer the thesis studies. Smoothing, hysteresis,
k-of-n persistence, a refractory period and calibration arrive at Stage 6, where their
trade-off is the object of measurement.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.data.canonical import NONE_INDEX
from src.eval.metrics import Detection


def active_classes(scores: np.ndarray, threshold: float) -> np.ndarray:
    """``[T]`` active class per frame: the argmax, or ``none`` below ``threshold``."""
    scores = np.asarray(scores)
    top = scores.argmax(axis=1)
    confident = scores[np.arange(len(top)), top] >= threshold
    return np.where(confident, top, NONE_INDEX).astype(np.int64)


def onset_alarms(video: str, scores: np.ndarray, threshold: float) -> list[Detection]:
    """Alarms for one video: the first frame of every run of an active signal class."""
    scores = np.asarray(scores)
    active = active_classes(scores, threshold)
    out: list[Detection] = []
    previous = NONE_INDEX
    for t, c in enumerate(active.tolist()):
        if c != NONE_INDEX and c != previous:
            out.append(Detection(video, int(c), t, float(scores[t, c])))
        previous = c
    return out


def threshold_grid(values: Sequence[float] | None = None) -> list[float]:
    """Thresholds swept for the delay vs false-alarm curve."""
    if values is not None:
        return sorted(float(v) for v in values)
    return [round(v, 2) for v in np.arange(0.0, 1.0, 0.05).tolist()] + [0.99]


__all__ = ["active_classes", "onset_alarms", "threshold_grid"]
