"""Hard Rule 1: no module reads a frame from the future.

The canonical location, per CLAUDE.md. Every temporal module in the project gets its test
here, and the test is always the same one:

    feed a truncated stream of length t, assert the outputs for frames 0..t are
    **bit-identical** to the corresponding slice of the full-stream run.

Bit-identical, not close. A module that is *almost* causal is one that reads the future a
little, and "a little" is enough to inflate an onset-latency number, which is the headline
figure of this thesis.

A test that has only ever been run against causal code is not a test. So this file also
builds deliberately non-causal modules -- a centred snippet, an end-anchored grid, forward
interpolation -- and asserts the harness **rejects** each one. Those negative controls are
the reason to believe the positive results.

Currently covered: the Stage 3 snippet scheduler and per-frame expansion. Stage 5's causal
GRU and Stage 6's decision layer add their cases here as they land.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features import causal
from src.features.base import (
    ArrayFrameSource,
    FrameSource,
    ProjectionBackbone,
    SnippetBackbone,
    SnippetExtractor,
)

pytestmark = pytest.mark.causality


# ----------------------------------------------------------------------------------
# The harness: the truncation test itself
# ----------------------------------------------------------------------------------


def truncation_mismatch(
    extractor: SnippetExtractor, source: FrameSource, prefix: int
) -> int | None:
    """Run on the full stream and on its first ``prefix`` frames; return the first
    frame index whose output differs, or None if the prefix is bit-identical."""
    full = extractor.extract(source)
    partial = extractor.extract(source.truncated(prefix))
    assert partial.shape[0] == prefix
    same = np.equal(full[:prefix], partial).all(axis=tuple(range(1, full.ndim)))
    bad = np.flatnonzero(~same)
    return int(bad[0]) if bad.size else None


@pytest.fixture
def frames() -> np.ndarray:
    """A stream where every frame is visibly different from every other.

    Structured, not random: if frame content repeated, a module could read the future and
    coincidentally produce the right answer, and the test would pass for the wrong reason.
    """
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(97, 12, 16, 3), dtype=np.uint8)
    base[:, 0, 0, 0] = np.arange(97, dtype=np.uint8)  # a per-frame fingerprint
    return base


@pytest.fixture
def source(frames: np.ndarray) -> ArrayFrameSource:
    return ArrayFrameSource(frames)


@pytest.fixture
def extractor() -> SnippetExtractor:
    return SnippetExtractor(ProjectionBackbone(dim=8), snippet_length=16, stride=6)


# ----------------------------------------------------------------------------------
# The extractor is causal
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", [1, 2, 5, 6, 7, 17, 30, 48, 96])
def test_truncated_stream_is_bit_identical(extractor, source, prefix):
    """THE Rule 1 test, at every interesting prefix length.

    The prefixes straddle snippet boundaries deliberately: 5/6/7 sit either side of the
    stride, 17 just past the first full snippet.
    """
    assert truncation_mismatch(extractor, source, prefix) is None


@pytest.mark.parametrize("length,stride", [(16, 6), (8, 8), (32, 4), (4, 1), (16, 16)])
def test_causal_at_every_snippet_geometry(source, length, stride):
    backbone = ProjectionBackbone(dim=8)
    extractor = SnippetExtractor(backbone, snippet_length=length, stride=stride)
    for prefix in (1, 3, 9, 33, 64, source.num_frames):
        assert truncation_mismatch(extractor, source, prefix) is None, (
            f"length={length} stride={stride} prefix={prefix}"
        )


def test_extractor_emits_one_row_per_frame(extractor, source):
    features = extractor.extract(source)
    assert features.shape == (source.num_frames, extractor.backbone.dim)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_repeated_extraction_is_bit_identical(extractor, source):
    """Determinism is part of causality here: the truncation test asserts bit-identity,
    which is meaningless if a single run is not reproducible."""
    assert np.array_equal(extractor.extract(source), extractor.extract(source))


# ----------------------------------------------------------------------------------
# The fast path must equal the obvious path
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("length,stride,batch", [(16, 6, 8), (8, 8, 1), (32, 4, 3), (4, 1, 16)])
def test_streaming_equals_random_access_bit_for_bit(source, length, stride, batch):
    """Differential test between the production path and the reference implementation.

    ``extract`` keeps a rolling buffer and decodes each frame once; ``extract_reference``
    re-reads every snippet's frames independently. They must agree exactly. Optimising a
    causal scheduler is precisely the kind of change that can introduce a future read while
    still producing plausible output, so the fast path is never trusted alone.
    """
    extractor = SnippetExtractor(
        ProjectionBackbone(dim=8), snippet_length=length, stride=stride, batch_size=batch
    )
    assert np.array_equal(extractor.extract(source), extractor.extract_reference(source))


def test_streaming_decodes_each_frame_exactly_once(frames):
    """The reason the rolling buffer exists: overlapping clips would otherwise re-decode.

    With length 16 and stride 6 the random-access path reads each frame about 2.7x over;
    on the real dataset that is millions of redundant JPEG decodes, and decode -- not the
    GPU -- is the bottleneck on Kaggle.
    """
    reads: list[int] = []

    class Counting(ArrayFrameSource):
        def read(self, indices):
            reads.extend(np.asarray(indices).tolist())
            return super().read(indices)

    source = Counting(frames)
    extractor = SnippetExtractor(ProjectionBackbone(dim=8), snippet_length=16, stride=6)

    extractor.extract(source)
    streaming = len(reads)
    assert streaming == source.num_frames, f"expected one read per frame, got {streaming}"

    reads.clear()
    extractor.extract_reference(source)
    assert len(reads) > streaming, "the reference path should read more than once per frame"


def test_reference_path_is_also_causal(source):
    """Both implementations are certified, not just the one in production."""
    extractor = SnippetExtractor(ProjectionBackbone(dim=8), snippet_length=16, stride=6)
    full = extractor.extract_reference(source)
    for prefix in (1, 13, 47, 96):
        partial = extractor.extract_reference(source.truncated(prefix))
        assert np.array_equal(full[:prefix], partial), f"prefix={prefix}"


def test_a_source_whose_iteration_ends_early_is_caught(frames):
    """A stream that stops short must fail loudly, not yield a silently truncated cache.

    The realistic version of this is a frames directory that is missing its tail because
    an extraction or a transfer was interrupted. The features would still have the right
    dtype and a plausible length, and every frame past the cut would carry a held-over
    value from before it -- invisible downstream, and quietly wrong.
    """

    class StopsEarly(ArrayFrameSource):
        def iter_frames(self):
            for index, frame in enumerate(super().iter_frames()):
                if index >= 50:
                    return
                yield frame

    extractor = SnippetExtractor(ProjectionBackbone(dim=8), snippet_length=16, stride=6)
    with pytest.raises(ValueError, match="different number of frames"):
        extractor.extract(StopsEarly(frames))


# ----------------------------------------------------------------------------------
# The scheduling primitives
# ----------------------------------------------------------------------------------


def test_a_snippet_never_reads_past_its_own_end():
    for end in range(0, 50):
        indices = causal.snippet_frame_indices(end, 16)
        assert indices.max() <= end
        assert indices.min() >= 0


def test_warm_up_pads_backwards_not_forwards():
    """Before the first full snippet, the clip repeats frame 0 -- it does not borrow
    frames from ahead to fill itself out."""
    indices = causal.snippet_frame_indices(2, 8)
    assert indices.tolist() == [0, 0, 0, 0, 0, 0, 1, 2]


def test_snippet_grid_for_a_prefix_is_a_prefix_of_the_full_grid():
    """Anchoring at frame 0 is what makes truncation safe; this is that property alone."""
    full = causal.snippet_ends(100, 6)
    for prefix in (1, 7, 42, 99):
        partial = causal.snippet_ends(prefix, 6)
        assert np.array_equal(partial, full[: len(partial)])


def test_hold_last_never_uses_a_future_snippet():
    ends = causal.snippet_ends(20, 5)  # 0, 5, 10, 15
    values = np.arange(len(ends), dtype=np.float32)[:, None]
    held = causal.hold_last(values, ends, 20)
    # Frame 4 must still carry snippet 0's value; taking snippet 1 (computed at frame 5)
    # would be reading one frame ahead.
    assert held[4, 0] == 0.0
    assert held[5, 0] == 1.0
    assert held[9, 0] == 1.0
    assert held[19, 0] == 3.0


def test_hold_last_rejects_a_grid_that_does_not_start_at_zero():
    with pytest.raises(ValueError, match="start at frame 0"):
        causal.hold_last(np.zeros((2, 3)), np.array([4, 9]), 12)


def test_stride_wider_than_the_snippet_is_refused():
    with pytest.raises(ValueError, match="contribute to no snippet"):
        SnippetExtractor(ProjectionBackbone(), snippet_length=8, stride=9)


# ----------------------------------------------------------------------------------
# Batch size, which is part of the cache's identity
# ----------------------------------------------------------------------------------


def test_final_batch_is_padded_to_a_fixed_size(source):
    """Guard on the padding in SnippetExtractor.extract. Do not delete it as dead weight.

    float32 matmul is not invariant to batch size -- the same input row gives a different
    result depending on how many rows share the call. Without padding, a snippet near the
    end of a truncated stream lands in a smaller batch than it did on the full stream and
    the truncation test fails for a reason unrelated to reading the future. The tempting
    fix at that point is to relax Rule 1's bit-identity to a tolerance, which would leave
    the project with a causality test that cannot detect a causality violation.
    """
    sizes: list[int] = []

    class Recording(ProjectionBackbone):
        def forward_batch(self, clips):
            sizes.append(clips.shape[0])
            return super().forward_batch(clips)

    extractor = SnippetExtractor(Recording(dim=8), snippet_length=16, stride=6, batch_size=8)
    extractor.extract(source)
    assert set(sizes) == {8}, f"batches were not all full size: {sorted(set(sizes))}"


def test_batch_size_is_recorded_because_it_changes_the_cache_bytes(source):
    """Two batch sizes give numerically different features, so the cache is only
    reproducible if the batch size is recorded alongside it."""
    a = SnippetExtractor(ProjectionBackbone(dim=8), snippet_length=16, stride=6, batch_size=4)
    b = SnippetExtractor(ProjectionBackbone(dim=8), snippet_length=16, stride=6, batch_size=8)
    fa, fb = a.extract(source), b.extract(source)

    assert np.allclose(fa, fb, atol=1e-4), "different batch sizes should agree numerically"
    assert a.describe()["batch_size"] == 4
    assert b.describe()["batch_size"] == 8


@pytest.mark.parametrize("batch_size", [1, 3, 8, 64])
def test_causal_at_every_batch_size(source, batch_size):
    """Whatever batch size is configured, truncation must be exact at that batch size."""
    extractor = SnippetExtractor(
        ProjectionBackbone(dim=8), snippet_length=16, stride=6, batch_size=batch_size
    )
    for prefix in (1, 13, 47, 96):
        assert truncation_mismatch(extractor, source, prefix) is None, (
            f"batch_size={batch_size} prefix={prefix}"
        )


# ----------------------------------------------------------------------------------
# NEGATIVE CONTROLS -- the harness must reject known-bad modules
# ----------------------------------------------------------------------------------


class CentredExtractor(SnippetExtractor):
    """Wrong on purpose: the clip is centred on its index, so it reads L/2 frames ahead.

    This is the single most likely way for this code to go wrong, because centring is what
    every offline implementation does and it looks tidier.
    """

    def extract(self, source: FrameSource) -> np.ndarray:
        num_frames = source.num_frames
        ends = causal.snippet_ends(num_frames, self.stride)
        half = self.snippet_length // 2
        rows = []
        for end in ends.tolist():
            indices = np.clip(
                np.arange(end - half, end + self.snippet_length - half), 0, num_frames - 1
            )
            rows.append(self.backbone.forward_batch(source.read(indices)[None])[0])
        return causal.hold_last(np.stack(rows), ends, num_frames).astype(np.float32)


class EndAnchoredExtractor(SnippetExtractor):
    """Wrong on purpose: the grid is anchored so a snippet lands on the final frame.

    Subtle, and it validates perfectly. The feature for frame 10 silently depends on how
    long the video turned out to be, which is a future read wearing an implementation
    detail as a disguise.
    """

    def extract(self, source: FrameSource) -> np.ndarray:
        num_frames = source.num_frames
        offset = (num_frames - 1) % self.stride
        ends = np.concatenate(
            [np.array([0], dtype=np.int64), np.arange(offset, num_frames, self.stride)]
        )
        ends = np.unique(ends)
        rows = [
            self.backbone.forward_batch(
                source.read(causal.snippet_frame_indices(int(e), self.snippet_length))[None]
            )[0]
            for e in ends.tolist()
        ]
        return causal.hold_last(np.stack(rows), ends, num_frames).astype(np.float32)


class InterpolatingExtractor(SnippetExtractor):
    """Wrong on purpose: blends towards the *next* snippet instead of holding the last.

    The smooth-looking option. Every frame between two snippets carries a fraction of a
    value computed from frames that have not arrived.
    """

    def extract(self, source: FrameSource) -> np.ndarray:
        num_frames = source.num_frames
        ends = causal.snippet_ends(num_frames, self.stride)
        rows = np.stack(
            [
                self.backbone.forward_batch(
                    source.read(causal.snippet_frame_indices(int(e), self.snippet_length))[None]
                )[0]
                for e in ends.tolist()
            ]
        )
        frames = np.arange(num_frames)
        out = np.stack(
            [np.interp(frames, ends, rows[:, d]) for d in range(rows.shape[1])], axis=1
        )
        return out.astype(np.float32)


@pytest.mark.parametrize(
    "broken_cls, why",
    [
        (CentredExtractor, "clip centred on its index reads L/2 frames ahead"),
        (EndAnchoredExtractor, "grid anchored to the final frame shifts with video length"),
        (InterpolatingExtractor, "interpolation blends in the next snippet's value"),
    ],
)
def test_harness_rejects_a_non_causal_extractor(source, broken_cls, why):
    """If any of these ever passes, the truncation test has stopped working.

    That would be worse than a non-causal module, because everything downstream would then
    be certified causal by a test that cannot tell.
    """
    broken = broken_cls(ProjectionBackbone(dim=8), snippet_length=16, stride=6)
    mismatches = [
        truncation_mismatch(broken, source, prefix) for prefix in (20, 40, 60, 80)
    ]
    assert any(m is not None for m in mismatches), (
        f"{broken_cls.__name__} passed the truncation test but should not have: {why}"
    )


def test_negative_controls_would_pass_a_naive_shape_check(source):
    """The broken extractors are broken *causally*, not structurally.

    They produce the right shape, the right dtype and finite values, which is exactly why
    a shape assertion is not a causality check and Rule 1 needs its own test.
    """
    for broken_cls in (CentredExtractor, EndAnchoredExtractor, InterpolatingExtractor):
        broken = broken_cls(ProjectionBackbone(dim=8), snippet_length=16, stride=6)
        features = broken.extract(source)
        assert features.shape == (source.num_frames, 8)
        assert np.isfinite(features).all()


# ----------------------------------------------------------------------------------
# Backbone contract
# ----------------------------------------------------------------------------------


class NondeterministicBackbone(SnippetBackbone):
    """Wrong on purpose: dropout left on, or a non-deterministic kernel."""

    name = "nondeterministic"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    @property
    def dim(self) -> int:
        return 4

    def forward_batch(self, clips: np.ndarray) -> np.ndarray:
        return self._rng.normal(size=(clips.shape[0], 4)).astype(np.float32)


def test_a_nondeterministic_backbone_fails_the_truncation_test(source):
    """A feature cache that changes between runs cannot back a reproducible number.

    Rule 1's bit-identity requirement catches this for free, which is why it is stated as
    bit-identity rather than as closeness.
    """
    extractor = SnippetExtractor(NondeterministicBackbone(), snippet_length=16, stride=6)
    assert truncation_mismatch(extractor, source, 40) is not None


def test_backbone_shape_disagreement_is_caught(source):
    class WrongDim(ProjectionBackbone):
        @property
        def dim(self) -> int:
            return 999

    extractor = SnippetExtractor(WrongDim(dim=8), snippet_length=16, stride=6)
    with pytest.raises(ValueError, match="expected"):
        extractor.extract(source)


# ----------------------------------------------------------------------------------
# The pose stream (Stage 3b): per-frame RTMW behind the same extractor
# ----------------------------------------------------------------------------------
#
# The real backbone needs onnxruntime and weights; these fakes stand in for the detector
# and the pose model so the scheduling is tested anywhere. Their outputs depend on the
# frame's pixels, so reading the wrong frame changes the answer.


def fake_detector(bgr: np.ndarray) -> np.ndarray:
    """Two boxes whose geometry depends on the frame; none on every fifth frame."""
    tag = int(bgr[0, 0, 2])  # the RGB fingerprint pixel, now in the BGR red channel
    if tag % 5 == 4:
        return np.zeros((0, 4))
    w, h = bgr.shape[1], bgr.shape[0]
    return np.array([[0, 0, w // 2, h // 2], [1, 1, w - 1 - tag % 3, h - 1]], dtype=float)


def fake_estimator(bgr: np.ndarray, boxes: np.ndarray):
    """Five joints computed from the pixels inside the box."""
    x1, y1, x2, y2 = boxes[0].astype(int)
    crop = bgr[y1:y2, x1:x2].astype(np.float64)
    joints = np.stack([crop.mean(axis=(0, 1))[:2] + j for j in range(5)])
    scores = np.full(5, crop.std() / 255.0)
    return joints[None], scores[None]


def pose_extractor(length: int = 1, stride: int = 1) -> SnippetExtractor:
    from src.features.backbones.rtmw import RTMWPoseBackbone

    backbone = RTMWPoseBackbone(detector=fake_detector, estimator=fake_estimator)
    return SnippetExtractor(backbone, snippet_length=length, stride=stride, batch_size=1)


@pytest.mark.parametrize("prefix", [1, 2, 4, 5, 6, 30, 96])
def test_pose_stream_truncated_is_bit_identical(source, prefix):
    assert truncation_mismatch(pose_extractor(), source, prefix) is None


def test_pose_backbone_estimates_only_the_current_frame(source):
    """Given a longer clip, it must use the clip's last frame and nothing earlier."""
    per_frame = pose_extractor().extract(source)
    with_history = pose_extractor(length=4, stride=1).extract(source)
    assert np.array_equal(per_frame, with_history)
