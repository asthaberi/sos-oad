"""The feature extraction boundary: frame sources, backbones, and the causal extractor.

Stage 3 turns 800k JPEGs into ``features/<video_id>.npy`` -- one row per frame, float32,
causally derived. Three pieces, deliberately separated so that the part Rule 1 depends on
can be tested without downloading a pretrained network or reading a single JPEG.

``FrameSource``   where pixels come from. A directory of JPEGs in production; an in-memory
                  array in the tests. Truncating one is how the causality test simulates a
                  stream that has not finished arriving.
``SnippetBackbone`` clip -> vector. The only part that needs torch, weights, or a GPU.
``SnippetExtractor`` schedules snippets causally and expands them to per-frame rows. Knows
                  nothing about pixels or networks, so it is fully testable today.

The device is never hardcoded anywhere here; it arrives from config, because the training
hardware for this project is still undecided and the cache has to be extractable on
whatever turns up (CLAUDE.md section 4).
"""

from __future__ import annotations

import abc
from collections import deque
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from src.features import causal


# --------------------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------------------


class FrameSource(abc.ABC):
    """A sequence of RGB frames for one video."""

    @property
    @abc.abstractmethod
    def num_frames(self) -> int: ...

    @abc.abstractmethod
    def read(self, indices: Sequence[int]) -> np.ndarray:
        """Return frames at ``indices`` as ``[n, H, W, 3]`` uint8."""

    def __len__(self) -> int:
        return self.num_frames

    def iter_frames(self):
        """Yield frames in order, one at a time.

        The production extraction path consumes a source this way rather than by random
        access. With snippet length 16 and stride 6 the clips overlap heavily, so random
        access decodes most JPEGs ten or more times; iterating decodes each exactly once.
        Subclasses may override for a faster sequential read, but must yield the same
        frames in the same order.
        """
        for index in range(self.num_frames):
            yield self.read([index])[0]

    @abc.abstractmethod
    def truncated(self, num_frames: int) -> "FrameSource":
        """A view of the first ``num_frames`` frames.

        Required, not optional: this is how ``tests/test_causality.py`` feeds a partial
        stream. A source that cannot be truncated cannot be proven causal.
        """


class ArrayFrameSource(FrameSource):
    """Frames already in memory. The test fixture, and the reference implementation."""

    def __init__(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be [T, H, W, 3], got {frames.shape}")
        self.frames = frames

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    def read(self, indices: Sequence[int]) -> np.ndarray:
        return self.frames[np.asarray(indices, dtype=np.int64)]

    def truncated(self, num_frames: int) -> "ArrayFrameSource":
        return ArrayFrameSource(self.frames[:num_frames])


class JpegDirectorySource(FrameSource):
    """Frames as ``<directory>/<prefix>_%0<digits>d.jpg``, 1-indexed on disk.

    IPN Hand numbers its JPEGs from 1 while the canonical format is 0-indexed, so the
    offset is applied here, once, at the edge. Stage 1 established that the numbering is
    contiguous ``1..N`` for every video and that ``N`` equals the annotated length, so a
    missing file is a real error rather than something to skip over.

    Pillow is imported lazily: the causal machinery above must stay importable and testable
    on a machine with no image library installed.
    """

    def __init__(
        self,
        directory: Path | str,
        num_frames: int,
        *,
        prefix: str | None = None,
        digits: int = 6,
        index_offset: int = 1,
    ) -> None:
        self.directory = Path(directory)
        self._num_frames = int(num_frames)
        self.prefix = prefix if prefix is not None else self.directory.name
        self.digits = digits
        self.index_offset = index_offset

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def path_for(self, index: int) -> Path:
        return self.directory / (
            f"{self.prefix}_{index + self.index_offset:0{self.digits}d}.jpg"
        )

    def read(self, indices: Sequence[int]) -> np.ndarray:
        from PIL import Image  # lazy: see class docstring

        out = []
        for index in np.asarray(indices, dtype=np.int64).tolist():
            path = self.path_for(int(index))
            if not path.is_file():
                raise FileNotFoundError(
                    f"frame {index} of {self.directory.name} is missing at {path}. "
                    "Stage 1 verified the numbering is contiguous, so this is a corrupt "
                    "or incomplete extraction, not an expected gap."
                )
            with Image.open(path) as image:
                out.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        return np.stack(out)

    def truncated(self, num_frames: int) -> "JpegDirectorySource":
        return JpegDirectorySource(
            self.directory,
            min(num_frames, self._num_frames),
            prefix=self.prefix,
            digits=self.digits,
            index_offset=self.index_offset,
        )


# --------------------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------------------


class SnippetBackbone(abc.ABC):
    """Maps a clip of frames to one feature vector.

    Implementations must be **deterministic**: the same clip must produce bit-identical
    output every call. Rule 1's test asserts bit-identity, not closeness, so a backbone
    with dropout left on or non-deterministic kernels will fail it -- correctly, because a
    feature cache that changes between runs cannot back a reproducible number either.
    """

    #: Short identifier recorded in meta.yaml so a cache can be traced to what made it.
    name: str = "base"

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    @abc.abstractmethod
    def forward_batch(self, clips: np.ndarray) -> np.ndarray:
        """``[B, L, H, W, 3]`` uint8 -> ``[B, D]`` float32."""

    def describe(self) -> dict[str, Any]:
        """Everything needed to reproduce this backbone, for meta.yaml."""
        return {"backbone": self.name, "feature_dim": self.dim}


class ProjectionBackbone(SnippetBackbone):
    """A deterministic, dependency-free backbone: pooled pixels through a fixed projection.

    Not a toy left over from development, and not a stand-in for a real network. It exists
    so the extraction pipeline, the cache writer and the causality tests can run on a clean
    checkout with no torch weights, no downloads and no GPU -- the same reasoning as the
    synthetic adapter at Stage 0. It is registered as a real backbone so it exercises the
    same code path X3D will.

    It is deliberately *not* good. If a model ever scores well on these features, the
    harness is broken rather than the features being informative.
    """

    name = "projection"

    def __init__(
        self, dim: int = 32, seed: int = 1337, pool: int = 4, device: str = "cpu"
    ) -> None:
        # `device` is accepted and ignored. Every backbone takes it so that the caller can
        # pass the configured device uniformly instead of branching on the backbone name.
        del device
        self._dim = int(dim)
        self.seed = int(seed)
        self.pool = int(pool)
        # Weights derived from the seed alone, so two machines produce the same cache.
        rng = np.random.default_rng(self.seed)
        self._weights = rng.normal(0.0, 1.0, size=(self.pool * self.pool * 3, self._dim))
        self._weights = self._weights.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    def forward_batch(self, clips: np.ndarray) -> np.ndarray:
        clips = np.asarray(clips)
        if clips.ndim != 5:
            raise ValueError(f"clips must be [B, L, H, W, 3], got {clips.shape}")
        batch, length, height, width, _ = clips.shape

        # Mean over the clip's frames, then a coarse spatial pool. Both are averages over
        # frames already in the clip, and the clip only contains frames <= its end.
        pooled = clips.astype(np.float32).mean(axis=1) / 255.0
        rows = np.array_split(np.arange(height), self.pool)
        cols = np.array_split(np.arange(width), self.pool)
        cells = [
            pooled[:, r[0] : r[-1] + 1, c[0] : c[-1] + 1, :].mean(axis=(1, 2))
            for r in rows
            for c in cols
        ]
        flat = np.concatenate(cells, axis=1).astype(np.float32)
        return (flat @ self._weights).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "backbone": self.name,
            "feature_dim": self.dim,
            "seed": self.seed,
            "pool": self.pool,
            "notes": (
                "Dependency-free test backbone. Never report a metric computed on these "
                "features."
            ),
        }


# --------------------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------------------


class SnippetExtractor:
    """Runs a backbone over a stream on a causal snippet grid and expands to per-frame rows.

    Knows nothing about pixels or networks; all of Rule 1 for Stage 3 is in here and in
    :mod:`src.features.causal`.
    """

    def __init__(
        self,
        backbone: SnippetBackbone,
        *,
        snippet_length: int = 16,
        stride: int = 6,
        batch_size: int = 8,
    ) -> None:
        if snippet_length < 1:
            raise ValueError(f"snippet_length must be >= 1, got {snippet_length}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if stride > snippet_length:
            # Permitted, but it means frames fall into no snippet at all and are only ever
            # represented by a held value from before them. Worth refusing by default.
            raise ValueError(
                f"stride {stride} exceeds snippet_length {snippet_length}: some frames "
                "would contribute to no snippet and be described only by held-over values"
            )
        self.backbone = backbone
        self.snippet_length = snippet_length
        self.stride = stride
        self.batch_size = batch_size

    def snippet_batches(self, num_frames: int) -> Iterator[np.ndarray]:
        ends = causal.snippet_ends(num_frames, self.stride)
        for start in range(0, len(ends), self.batch_size):
            yield ends[start : start + self.batch_size]

    def extract_reference(self, source: FrameSource) -> np.ndarray:
        """Random-access implementation: re-reads each snippet's frames independently.

        Obviously correct and obviously slow. It exists as the reference half of a
        differential test: ``extract`` is the production path and must agree with this
        bit-for-bit. Optimising a causal scheduler is exactly the kind of change that can
        introduce a future read while still looking right, so the fast path is never
        trusted on its own.
        """
        num_frames = source.num_frames
        if num_frames <= 0:
            raise ValueError("cannot extract features from an empty stream")

        ends = causal.snippet_ends(num_frames, self.stride)
        causal.assert_backward_looking(ends, self.snippet_length)

        chunks: list[np.ndarray] = []
        for batch_ends in self.snippet_batches(num_frames):
            clips = np.stack(
                [
                    source.read(causal.snippet_frame_indices(int(end), self.snippet_length))
                    for end in batch_ends.tolist()
                ]
            )
            wanted = len(batch_ends)

            # Pad the final batch to the full batch size. This is not an optimisation and
            # must not be removed.
            #
            # Floating-point matrix multiply is not invariant to batch size: the same input
            # row produces different float32 output depending on how many rows are in the
            # batch, because the library blocks the reduction differently (measured here at
            # ~4e-6 for this backbone; a GPU kernel is typically worse). Position within the
            # batch and the contents of neighbouring rows do *not* matter -- only the row
            # count does.
            #
            # Without padding, the last batch of a truncated stream has a different size
            # from the batch the same snippet fell into on the full stream, so Rule 1's
            # bit-identity test fails for a reason that has nothing to do with reading the
            # future. That is the dangerous case: the obvious response is to loosen the test
            # to a tolerance, and a loosened causality test cannot catch a real violation.
            if wanted < self.batch_size:
                padding = np.repeat(clips[-1:], self.batch_size - wanted, axis=0)
                clips = np.concatenate([clips, padding], axis=0)

            features = np.asarray(self.backbone.forward_batch(clips), dtype=np.float32)
            if features.shape != (clips.shape[0], self.backbone.dim):
                raise ValueError(
                    f"backbone {self.backbone.name!r} returned {features.shape}, expected "
                    f"{(clips.shape[0], self.backbone.dim)}"
                )
            chunks.append(features[:wanted])

        values = np.concatenate(chunks, axis=0)
        expanded = causal.hold_last(values, ends, num_frames)
        return np.ascontiguousarray(expanded, dtype=np.float32)

    def extract(self, source: FrameSource) -> np.ndarray:
        """Return ``[T, D]`` float32 features, one row per frame. The production path.

        Consumes the stream once, keeping a rolling buffer of the last ``snippet_length``
        frames and firing the backbone whenever the frame index lands on the snippet grid.
        That is what a real streaming system does, and it decodes each frame exactly once.

        Causality is structural: the buffer physically cannot contain a frame that has not
        arrived. Warm-up left-pads by repeating the oldest frame held, which is frame 0.
        """
        num_frames = source.num_frames
        if num_frames <= 0:
            raise ValueError("cannot extract features from an empty stream")

        ends = causal.snippet_ends(num_frames, self.stride)
        causal.assert_backward_looking(ends, self.snippet_length)
        is_end = np.zeros(num_frames, dtype=bool)
        is_end[ends] = True

        buffer: deque = deque(maxlen=self.snippet_length)
        pending: list[np.ndarray] = []
        chunks: list[np.ndarray] = []

        for index, frame in enumerate(source.iter_frames()):
            if index >= num_frames:
                break
            buffer.append(frame)
            if not is_end[index]:
                continue

            held = list(buffer)
            if len(held) < self.snippet_length:
                # Before the first full snippet the buffer holds frames 0..index, so the
                # oldest frame it has *is* frame 0; repeating it reproduces the clamping in
                # causal.snippet_frame_indices without reaching forward.
                held = [held[0]] * (self.snippet_length - len(held)) + held
            pending.append(np.stack(held))

            if len(pending) == self.batch_size:
                chunks.append(self._run_batch(pending))
                pending = []

        if pending:
            chunks.append(self._run_batch(pending))

        if not chunks:
            raise RuntimeError("the frame source yielded no frames")

        values = np.concatenate(chunks, axis=0)
        if values.shape[0] != len(ends):
            raise ValueError(
                f"produced {values.shape[0]} snippet(s) for {len(ends)} grid position(s); "
                "the frame source yielded a different number of frames than it declared"
            )
        expanded = causal.hold_last(values, ends, num_frames)
        return np.ascontiguousarray(expanded, dtype=np.float32)

    def _run_batch(self, clips: list[np.ndarray]) -> np.ndarray:
        """Run one batch, padded to a fixed size. See the note in extract_reference."""
        wanted = len(clips)
        stacked = np.stack(clips)
        if wanted < self.batch_size:
            padding = np.repeat(stacked[-1:], self.batch_size - wanted, axis=0)
            stacked = np.concatenate([stacked, padding], axis=0)
        features = np.asarray(self.backbone.forward_batch(stacked), dtype=np.float32)
        if features.shape != (stacked.shape[0], self.backbone.dim):
            raise ValueError(
                f"backbone {self.backbone.name!r} returned {features.shape}, expected "
                f"{(stacked.shape[0], self.backbone.dim)}"
            )
        return features[:wanted]

    def describe(self) -> dict[str, Any]:
        return {
            **self.backbone.describe(),
            "snippet_length": self.snippet_length,
            "snippet_stride": self.stride,
            "per_frame_expansion": "causal hold-last (never interpolated forward)",
            "snippet_alignment": "clip covers [end-L+1, end]; grid anchored at frame 0",
            # Part of the cache's identity, not just a throughput knob: float32 matmul is
            # not batch-size invariant, so re-extracting with a different batch_size gives
            # numerically different bytes and a different checksum. Recorded so a cache can
            # be reproduced exactly rather than merely approximately.
            "batch_size": self.batch_size,
        }


__all__ = [
    "FrameSource",
    "ArrayFrameSource",
    "JpegDirectorySource",
    "SnippetBackbone",
    "ProjectionBackbone",
    "SnippetExtractor",
]
