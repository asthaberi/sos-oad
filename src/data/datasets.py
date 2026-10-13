"""Per-frame model inputs over the canonical format: features, normalised pose, labels.

What a model sees at frame ``t`` is built from frame ``t`` alone:

* the RGB feature row ``features[t]`` (already causal, from Stage 3);
* the pose ``poses[t]`` normalised *per frame*: translated so the root joints' midpoint is
  the origin and scaled by the distance between the scale joints, confidences kept. No
  statistic from any other frame enters, so this is causal by construction.

Then every input dimension is standardised with a mean and standard deviation fitted on
the **training videos only** and stored with the model. Fitting them on all videos would
leak the validation and test people's statistics into training -- one of the leak routes
Hard Rule 2 lists -- and refitting them on the evaluation videos would make the input
depend on the future of the stream.

Nothing here knows which dataset it reads. Joint indices for the pose normalisation come
from config, because they belong to the pose *format* (COCO-WholeBody for RTMW), not to a
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data import canonical as C


@dataclass(frozen=True)
class InputSpec:
    """Which streams feed the model, and how pose is normalised."""

    streams: tuple[str, ...] = ("features", "poses")
    pose_root_joints: tuple[int, ...] = (5, 6)
    pose_scale_joints: tuple[int, int] = (5, 6)
    #: Below this scale (pixels) a frame's pose is treated as absent: positions zeroed,
    #: confidences kept. Avoids dividing by a near-zero shoulder width.
    min_scale: float = 1.0

    @classmethod
    def from_config(cls, inputs_cfg, pose_cfg) -> "InputSpec":
        """``inputs_cfg`` is ``cfg.inputs``; ``pose_cfg`` is the dataset group's
        ``pose_normalisation`` block (joint indices belong to the pose format the dataset's
        cache was written in, so they live in the dataset YAML, not in code)."""
        streams = tuple(str(s) for s in inputs_cfg.streams)
        bad = [s for s in streams if s not in ("features", "poses")]
        if bad or not streams:
            raise ValueError(f"inputs.streams must be a subset of [features, poses], got {streams}")
        scale = tuple(int(j) for j in pose_cfg.scale_joints)
        if len(scale) != 2:
            raise ValueError("pose_normalisation.scale_joints must name exactly two joints")
        return cls(
            streams=streams,
            pose_root_joints=tuple(int(j) for j in pose_cfg.root_joints),
            pose_scale_joints=scale,  # type: ignore[arg-type]
            min_scale=float(pose_cfg.get("min_scale", 1.0)),
        )


def normalise_pose(pose: np.ndarray, spec: InputSpec) -> tuple[np.ndarray, int]:
    """``[T, J, 3]`` (x, y, conf) -> ``[T, J*3]`` normalised per frame.

    Returns the flattened array and the number of frames whose scale was too small to
    normalise (their positions are zeroed, so they read as "no pose" rather than as a pose
    blown up by a tiny divisor).
    """
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[2] != 3:
        raise ValueError(f"pose must be [T, J, 3], got {pose.shape}")
    joints = pose.shape[1]
    needed = [*spec.pose_root_joints, *spec.pose_scale_joints]
    if max(needed) >= joints:
        raise ValueError(f"pose has {joints} joints but config names joint {max(needed)}")
    xy = pose[:, :, :2]
    root = xy[:, list(spec.pose_root_joints), :].mean(axis=1, keepdims=True)
    a, b = spec.pose_scale_joints
    scale = np.linalg.norm(xy[:, a, :] - xy[:, b, :], axis=1)
    ok = scale >= spec.min_scale
    safe = np.where(ok, scale, 1.0)[:, None, None]
    out_xy = np.where(ok[:, None, None], (xy - root) / safe, 0.0)
    out = np.concatenate([out_xy, pose[:, :, 2:3]], axis=2).astype(np.float32)
    return out.reshape(len(pose), -1), int((~ok).sum())


def frame_inputs(dataset: C.CanonicalDataset, video_id: str, spec: InputSpec) -> np.ndarray:
    """``[T, D]`` raw (unstandardised) inputs for one video."""
    parts = []
    if "features" in spec.streams:
        parts.append(np.asarray(dataset.load_features(video_id), dtype=np.float32))
    if "poses" in spec.streams:
        pose = dataset.load_poses(video_id)
        if pose is None:
            raise FileNotFoundError(f"{video_id}: no pose array, but inputs.streams asks for poses")
        parts.append(normalise_pose(pose, spec)[0])
    lengths = {p.shape[0] for p in parts}
    if len(lengths) != 1:
        raise ValueError(f"{video_id}: streams disagree on frame count {sorted(lengths)}")
    out = np.concatenate(parts, axis=1)
    expected = dataset.num_frames(video_id)
    if out.shape[0] != expected:
        raise ValueError(f"{video_id}: {out.shape[0]} input rows for {expected} annotated frames")
    return out


@dataclass
class Standardiser:
    """Per-dimension z-scoring with statistics fitted on training data only."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "Standardiser":
        return cls(np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32))


def fit_standardiser(train_arrays: Sequence[np.ndarray], eps: float = 1e-6) -> Standardiser:
    """Mean and std over every training frame, two-pass in float64.

    The training split is ~480k frames x ~1.2k dims; a single-pass float32 variance loses
    precision at that size.
    """
    count = sum(a.shape[0] for a in train_arrays)
    if count == 0:
        raise ValueError("cannot fit a standardiser on no data")
    mean = sum(a.astype(np.float64).sum(axis=0) for a in train_arrays) / count
    var = sum(((a.astype(np.float64) - mean) ** 2).sum(axis=0) for a in train_arrays) / count
    std = np.sqrt(var)
    std = np.where(std < eps, 1.0, std)  # constant dimensions pass through unscaled
    return Standardiser(mean=mean.astype(np.float32), std=std.astype(np.float32))


@dataclass
class SplitData:
    """Standardised inputs and labels for every video of one split."""

    inputs: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]

    @property
    def videos(self) -> list[str]:
        return sorted(self.inputs)

    @property
    def num_frames(self) -> int:
        return int(sum(len(v) for v in self.labels.values()))


def load_split(
    dataset: C.CanonicalDataset,
    video_ids: Sequence[str],
    spec: InputSpec,
    standardiser: Standardiser | None,
) -> tuple[SplitData, list[np.ndarray]]:
    """Inputs (standardised if a standardiser is given) and labels for ``video_ids``.

    Also returns the raw arrays, so the training split can fit the standardiser and reuse
    them without reading every file twice.
    """
    raw = [frame_inputs(dataset, v, spec) for v in video_ids]
    inputs = {
        v: (standardiser.apply(a) if standardiser is not None else a) for v, a in zip(video_ids, raw)
    }
    labels = {v: dataset.frame_labels(v) for v in video_ids}
    return SplitData(inputs=inputs, labels=labels), raw


__all__ = [
    "InputSpec",
    "SplitData",
    "Standardiser",
    "fit_standardiser",
    "frame_inputs",
    "load_split",
    "normalise_pose",
]
