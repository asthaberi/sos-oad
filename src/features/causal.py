"""Causal snippet scheduling and per-frame expansion.

Hard Rule 1 lives here. A snippet-based backbone (VideoMAEv2, X3D, TSM) consumes a clip of
``snippet_length`` frames and emits one vector, but the canonical format wants one row per
frame. How that gap is bridged is the single easiest place in this project to read a frame
from the future without noticing.

The rules, and what each one rules out
--------------------------------------
**A snippet ends at the frame it describes.** The clip for snippet end ``e`` covers
``[e - L + 1, e]`` -- it looks *backwards*. A snippet centred on ``e`` would cover
``[e - L/2, e + L/2]`` and read L/2 frames of future, which is the most natural way to
write this code and completely wrong. Centring is what every offline implementation does.

**The snippet grid is anchored at frame 0, not at the end of the video.** Ends are
``0, stride, 2*stride, ...``. Anchoring to the end -- "make sure a snippet lands on the
last frame" -- would mean the whole grid shifts when the video length changes, so the
feature for frame 10 would depend on how long the video turns out to be. That is a future
read dressed up as an implementation detail, and it is exactly what the truncation test in
``tests/test_causality.py`` catches.

**Between snippets, hold the last value. Never interpolate.** Frame ``t`` takes the feature
of the most recent snippet ending at or before ``t``. Interpolating between the previous
and the next snippet would blend in a value computed from frames after ``t``. CLAUDE.md
section 3 says this in one line: "hold last; never interpolate forward".

**Warm-up pads backwards, not forwards.** Before the first full snippet is available the
clip is left-padded by repeating frame 0, so a feature exists from frame 0 onward and it is
built only from frames that have actually arrived.

What is deliberately *not* here
-------------------------------
Feature normalisation. Rule 1 forbids statistics fitted over a whole sequence or a whole
dataset, so mean/std must be fitted on the training split only and applied as fixed
constants. That belongs to the stage that trains, not to the stage that caches raw
features. Writing normalised features here would bake a leak into the cache where nothing
downstream could see it.
"""

from __future__ import annotations

import numpy as np


def snippet_ends(num_frames: int, stride: int) -> np.ndarray:
    """Frame indices at which a snippet is computed, anchored at frame 0.

    Anchoring matters more than it looks. Because the grid starts at 0 and steps by a fixed
    stride, the grid for a stream of length ``t`` is exactly a prefix of the grid for the
    full stream -- which is what makes the truncation test pass bit-identically rather than
    approximately.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return np.arange(0, num_frames, stride, dtype=np.int64)


def snippet_frame_indices(end: int, length: int) -> np.ndarray:
    """Frame indices of the snippet ending at ``end``, inclusive.

    Covers ``[end - length + 1, end]``, clamped at 0 so the warm-up repeats frame 0 rather
    than reaching past the start. Every returned index is ``<= end``; that invariant is the
    whole of Rule 1 at snippet level and is asserted in the causality tests.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if end < 0:
        raise ValueError(f"end must be >= 0, got {end}")
    return np.clip(np.arange(end - length + 1, end + 1, dtype=np.int64), 0, None)


def hold_last(values: np.ndarray, ends: np.ndarray, num_frames: int) -> np.ndarray:
    """Expand ``[S, D]`` snippet features to ``[num_frames, D]`` by holding the last value.

    Frame ``t`` receives the feature of the most recent snippet ending at or before ``t``.
    Because ``ends[0] == 0``, every frame has one.

    This is a step function, not a ramp, and that is intentional: the smooth-looking
    alternative reads forward.
    """
    values = np.asarray(values)
    ends = np.asarray(ends, dtype=np.int64)
    if values.ndim < 1 or values.shape[0] != ends.shape[0]:
        raise ValueError(
            f"values has {values.shape[0] if values.ndim else 0} row(s) but there are "
            f"{ends.shape[0]} snippet end(s)"
        )
    if ends.size == 0:
        raise ValueError("no snippets to expand")
    if ends[0] != 0:
        raise ValueError(
            f"snippet grid must start at frame 0, starts at {int(ends[0])}; "
            "frames before the first snippet would have no causally available feature"
        )
    if np.any(np.diff(ends) <= 0):
        raise ValueError("snippet ends must be strictly increasing")

    # For frame t, the index of the last end <= t. side="right" then -1 gives exactly that,
    # and never selects an end greater than t -- which is the property being protected.
    frames = np.arange(num_frames, dtype=np.int64)
    picked = np.searchsorted(ends, frames, side="right") - 1
    return values[picked]


def assert_backward_looking(ends: np.ndarray, length: int) -> None:
    """Raise if any snippet would read past its own end frame.

    Cheap enough to call on every extraction. A backbone wrapper that quietly recentres its
    input is the failure this catches.
    """
    for end in np.asarray(ends).tolist():
        indices = snippet_frame_indices(int(end), length)
        if indices.size and int(indices.max()) > int(end):
            raise AssertionError(
                f"snippet ending at {end} reads frame {int(indices.max())}, which is in "
                "its future. See CLAUDE.md Rule 1."
            )


__all__ = [
    "snippet_ends",
    "snippet_frame_indices",
    "hold_last",
    "assert_backward_looking",
]
