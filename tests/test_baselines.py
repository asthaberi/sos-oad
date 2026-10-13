"""Stage 4 inputs and baselines: pose normalisation, train-only standardisation, class
weights, B1's training windows, and an end-to-end run on the synthetic fixture.

Causality of B0 and B1 is tested in tests/test_causality.py, with the other temporal
modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import canonical as C
from src.data.datasets import InputSpec, fit_standardiser, load_split, normalise_pose
from src.eval.evaluate import evaluate_split
from src.features import causal
from src.models.train import (
    TrainSettings,
    class_weights,
    predict_split,
    train,
    training_window_index,
)

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic"
SPEC = InputSpec(streams=("features", "poses"), pose_root_joints=(0, 1), pose_scale_joints=(0, 1), min_scale=1e-6)


# ----------------------------------------------------------------------------------
# Pose normalisation
# ----------------------------------------------------------------------------------


def test_pose_is_centred_on_the_root_and_scaled_by_the_pair():
    # joints 0 and 1 at (10,0) and (20,0): root (15,0), scale 10. Joint 2 at (25,10) -> (1,1).
    pose = np.array([[[10, 0, 0.9], [20, 0, 0.8], [25, 10, 0.7]]], dtype=np.float32)
    spec = InputSpec(pose_root_joints=(0, 1), pose_scale_joints=(0, 1))
    out, absent = normalise_pose(pose, spec)
    assert absent == 0
    expected = [[-0.5, 0.0, 0.9], [0.5, 0.0, 0.8], [1.0, 1.0, 0.7]]
    assert np.allclose(out.reshape(3, 3), expected)


def test_a_degenerate_scale_zeroes_positions_but_keeps_confidence():
    pose = np.array([[[10, 0, 0.9], [10, 0, 0.8], [25, 10, 0.7]]], dtype=np.float32)
    spec = InputSpec(pose_root_joints=(0, 1), pose_scale_joints=(0, 1), min_scale=1.0)
    out, absent = normalise_pose(pose, spec)
    assert absent == 1
    assert out.reshape(3, 3)[:, :2].tolist() == [[0, 0], [0, 0], [0, 0]]
    assert out.reshape(3, 3)[:, 2].tolist() == pytest.approx([0.9, 0.8, 0.7])


def test_pose_normalisation_is_per_frame():
    """Each frame's output depends on that frame only: shuffling frames shuffles outputs."""
    rng = np.random.default_rng(0)
    pose = rng.uniform(0, 100, size=(50, 6, 3)).astype(np.float32)
    spec = InputSpec(pose_root_joints=(0, 1), pose_scale_joints=(0, 1), min_scale=1e-6)
    order = rng.permutation(50)
    assert np.array_equal(normalise_pose(pose, spec)[0][order], normalise_pose(pose[order], spec)[0])


def test_joint_indices_beyond_the_pose_are_refused():
    with pytest.raises(ValueError):
        normalise_pose(np.zeros((2, 5, 3), np.float32), InputSpec(pose_root_joints=(5, 6), pose_scale_joints=(5, 6)))


# ----------------------------------------------------------------------------------
# Standardisation and class weights
# ----------------------------------------------------------------------------------


def test_standardiser_uses_training_statistics_only():
    train = [np.array([[0.0, 10.0], [2.0, 10.0]], np.float32)]
    std = fit_standardiser(train)
    assert std.mean.tolist() == [1.0, 10.0]
    assert std.std.tolist() == [1.0, 1.0]  # constant dimension passes through unscaled
    # A validation array with a different mean is NOT re-centred: it keeps its offset.
    val = np.array([[5.0, 10.0]], np.float32)
    assert std.apply(val).tolist() == [[4.0, 0.0]]


def test_class_weights_by_hand():
    # 6 frames: four of class 0, two of class 1, none of class 2. N/(C*n_c) with C=3.
    w = class_weights([np.array([0, 0, 0, 0, 1, 1])], 3)
    assert w.tolist() == pytest.approx([0.5, 1.0, 0.0])


# ----------------------------------------------------------------------------------
# B1 training windows
# ----------------------------------------------------------------------------------


def test_training_windows_never_cross_into_the_previous_video():
    # Two videos of 5 and 4 frames concatenated: video 2 starts at global index 5.
    starts = torch.tensor([0, 0, 0, 0, 0, 5, 5, 5, 5])
    idx = training_window_index(torch.tensor([5, 6, 8]), window=3, video_start=starts)
    assert idx.tolist() == [[5, 5, 5], [5, 5, 6], [6, 7, 8]]


def test_training_windows_match_prediction_windows():
    """The window B1 learns from for frame t is the window it is shown live at frame t."""
    starts = torch.zeros(100, dtype=torch.long)
    for t in (0, 3, 59, 60, 99):
        train_idx = training_window_index(torch.tensor([t]), 60, starts)[0].numpy()
        assert np.array_equal(train_idx, causal.snippet_frame_indices(t, 60))


# ----------------------------------------------------------------------------------
# End to end on the synthetic fixture
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["b0", "b1"])
def test_baseline_trains_and_scores_on_the_fixture(kind):
    dataset = C.CanonicalDataset(FIXTURE)
    train_ids = dataset.video_ids_for_split("train")
    val_ids = dataset.video_ids_for_split("val")
    train_data, raw = load_split(dataset, train_ids, SPEC, None)
    std = fit_standardiser(raw)
    train_data.inputs = {v: std.apply(a) for v, a in train_data.inputs.items()}
    val_data, _ = load_split(dataset, val_ids, SPEC, std)

    s = TrainSettings(kind=kind, epochs=2, hidden=32, batch_size=256)
    result = train(
        s, train_data, val_data,
        num_classes=dataset.num_classes, class_names=dataset.classes,
        fps=dataset.fps, device="cpu", seed=0,
    )
    assert len(result.history) == 2 and result.best_epoch in (1, 2)
    scores = predict_split(s, result.model, val_data, fps=dataset.fps, device="cpu")
    out = evaluate_split(
        scores, val_data.labels, fps=dataset.fps, class_names=dataset.classes, signals=None,
        threshold=0.5, match_pre_s=0.5, match_post_s=1.0, point_tolerances_s=[1, 2],
    )
    assert out["videos"] == len(val_ids)
    for v in val_ids:
        assert scores[v].shape == (dataset.num_frames(v), dataset.num_classes)
        assert np.allclose(scores[v].sum(axis=1), 1.0, atol=1e-5)


def test_training_is_reproducible_with_a_seed():
    dataset = C.CanonicalDataset(FIXTURE)
    ids = dataset.video_ids_for_split("train")[:3]
    data, raw = load_split(dataset, ids, SPEC, None)
    std = fit_standardiser(raw)
    data.inputs = {v: std.apply(a) for v, a in data.inputs.items()}
    s = TrainSettings(kind="b0", epochs=1, hidden=16, batch_size=128)
    kw = dict(num_classes=dataset.num_classes, class_names=dataset.classes, fps=dataset.fps, device="cpu", seed=5)
    torch.manual_seed(5)
    a = train(s, data, data, **kw).history
    torch.manual_seed(5)
    b = train(s, data, data, **kw).history
    assert a == b
