"""A synthetic dataset in the canonical format.

This is not a toy left over from development -- it is load-bearing test infrastructure.
It gives us a dataset with known ground truth that exists before IPN Hand is downloaded,
which means:

* the canonical format and its validator can be tested on their own, with no dataset;
* the causality test (Hard Rule 1) has a stream to run against from Stage 0 onward;
* every downstream stage gets a fast fixture that does not touch 800k real frames;
* mutation tests can corrupt a *known-good* dataset one property at a time and assert
  the validator catches each corruption. A validator that has only ever seen valid data
  is not a tested validator.

It is registered as a real adapter so that it exercises the same code path a real dataset
takes, rather than a parallel shortcut that could drift from it.

The features are deliberately label-correlated but noisy: each class has a random mean
vector and features are drawn around the mean of the current frame's class. That makes a
trivially-learnable-but-not-perfect signal, which is what we want from a smoke fixture --
if a model scores 100% on this, the harness is broken, not brilliant.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.data import canonical as C
from src.data.adapters.base import AdapterConfig, DatasetAdapter

DEFAULT_OPTIONS: dict[str, Any] = {
    "name": "synthetic",
    "fps": 30.0,
    "num_videos": 8,
    "num_subjects": 4,
    "gesture_classes": ["wave", "point", "fist"],
    "feature_dim": 16,
    "num_joints": 5,
    "frames_per_video": (300, 600),
    "gestures_per_video": (3, 6),
    "gesture_duration": (30, 90),
    "min_gap": 20,
    "noise": 0.6,
    "seed": 1337,
}


class SyntheticAdapter(DatasetAdapter):
    """Generates a small, fully-specified canonical dataset."""

    name = "synthetic"

    def __init__(self, config: AdapterConfig) -> None:
        options = {**DEFAULT_OPTIONS, **(config.options or {})}
        super().__init__(AdapterConfig(config.source_root, config.output_root, options))
        self.rng = np.random.default_rng(int(options["seed"]))

    def gesture_classes(self) -> Sequence[str]:
        return list(self.config.options["gesture_classes"])

    def extra_meta(self) -> dict[str, Any]:
        opts = self.config.options
        return {
            "synthetic": True,
            "num_joints": int(opts["num_joints"]),
            "seed": int(opts["seed"]),
            "feature_backbone": "synthetic-gaussian",
            "pose_backbone": "synthetic-gaussian",
            "notes": "Test fixture. Not a real dataset; never report a metric from it.",
        }

    def build(self) -> None:
        opts = self.config.options
        classes = self.classes
        dim = int(opts["feature_dim"])
        num_joints = int(opts["num_joints"])
        noise = float(opts["noise"])

        # One mean vector per class, well separated so the task is learnable.
        class_means = self.rng.normal(0.0, 2.0, size=(len(classes), dim)).astype(np.float32)
        class_joint_means = self.rng.normal(0.0, 1.0, size=(len(classes), num_joints, 3))
        class_joint_means = class_joint_means.astype(np.float32)

        num_videos = int(opts["num_videos"])
        num_subjects = int(opts["num_subjects"])

        for index in range(num_videos):
            video_id = f"vid{index:03d}"
            # Subjects deliberately own more than one video, so the Stage 2 splitter has
            # a non-trivial grouping problem to solve.
            subject_id = f"subj{index % num_subjects:02d}"

            num_frames = int(self.rng.integers(*opts["frames_per_video"]))
            self._add_gestures(video_id, subject_id, num_frames, classes)
            self.fill_none_spans(video_id, num_frames, subject_id)

            labels = self._labels_for(video_id, num_frames, classes)
            features = class_means[labels] + self.rng.normal(
                0.0, noise, size=(num_frames, dim)
            ).astype(np.float32)
            poses = class_joint_means[labels] + self.rng.normal(
                0.0, noise * 0.5, size=(num_frames, num_joints, 3)
            ).astype(np.float32)

            self.emit_features(video_id, features)
            self.emit_poses(video_id, poses)

    # -- internals ----------------------------------------------------------------

    def _add_gestures(
        self, video_id: str, subject_id: str, num_frames: int, classes: list[str]
    ) -> None:
        """Lay down non-overlapping gesture spans, leaving room for `none` between them."""
        opts = self.config.options
        target = int(self.rng.integers(*opts["gestures_per_video"]))
        min_gap = int(opts["min_gap"])
        lo, hi = opts["gesture_duration"]

        cursor = int(self.rng.integers(min_gap, min_gap * 2))
        for _ in range(target):
            duration = int(self.rng.integers(lo, hi))
            start = cursor
            end = start + duration - 1
            if end >= num_frames - min_gap:
                break
            class_name = classes[int(self.rng.integers(1, len(classes)))]
            self.add_instance(video_id, class_name, start, end, subject_id)
            cursor = end + 1 + int(self.rng.integers(min_gap, min_gap * 3))

    def _labels_for(self, video_id: str, num_frames: int, classes: list[str]) -> np.ndarray:
        index = {name: i for i, name in enumerate(classes)}
        labels = np.full(num_frames, C.NONE_INDEX, dtype=np.int64)
        for row in self._rows:
            if row["video_id"] != video_id:
                continue
            labels[row["start_frame"] : row["end_frame"] + 1] = index[row["class"]]
        return labels


def make_synthetic_dataset(output_root, **options: Any) -> C.ValidationReport:
    """Convenience entry point used by tests and ``scripts/make_fixture.py``."""
    adapter = SyntheticAdapter(
        AdapterConfig(source_root=output_root, output_root=output_root, options=options)
    )
    return adapter.convert()


__all__ = ["SyntheticAdapter", "make_synthetic_dataset", "DEFAULT_OPTIONS"]
