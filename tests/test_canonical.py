"""Tests for the canonical data format and its validator.

Stage 0 gate: the validator passes on a synthetic fixture *and* rejects each way the
format can be violated. The second half is the part that matters -- these corruptions are
exactly the mistakes that produce a meaningless 98% at Stage 7, so each one gets a test
that names it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import canonical as C
from src.data.adapters.base import AdapterConfig
from src.data.adapters.synthetic import SyntheticAdapter

from .conftest import first_video

# ======================================================================================
# The gate: a well-formed dataset validates
# ======================================================================================


def test_good_dataset_validates(good_dataset: Path) -> None:
    report = C.validate_dataset(good_dataset)
    assert report.ok, report.render()
    assert not report.errors


def test_good_dataset_statistics(good_dataset: Path) -> None:
    report = C.validate_dataset(good_dataset)
    assert report.stats["videos"] == 6
    assert report.stats["subjects"] == 3
    assert report.stats["classes"] == 4          # none + 3 gestures
    assert report.stats["gesture_classes"] == 3
    assert report.stats["gesture_instances"] > 0
    assert report.stats["none_instances"] > 0
    assert report.stats["feature_dim"] == 16
    assert report.stats["num_joints"] == 5


def test_none_class_dominates_frames(good_dataset: Path) -> None:
    """Sanity-check the fixture has the shape of a real continuous-video dataset.

    If `none` is not the majority of frames, the fixture is not exercising the class
    imbalance the real problem has, and Stage 4's baselines would be tuned against a
    task that does not resemble IPN Hand.
    """
    report = C.validate_dataset(good_dataset)
    assert float(report.stats["none_frame_fraction"]) > 0.5


# ======================================================================================
# classes.txt
# ======================================================================================


def test_rejects_none_missing_from_index_0(dataset: Path) -> None:
    """The single most important check in the format.

    Dropping `none` as "background" turns online detection into trimmed-clip
    classification and every metric downstream becomes meaningless.
    """
    classes = C.read_classes(dataset)
    (dataset / C.CLASSES_FILE).write_text("\n".join(classes[1:]) + "\n", encoding="utf-8")

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "classes.none_not_index_0" in report.codes()


def test_rejects_none_demoted_from_index_0(dataset: Path) -> None:
    """`none` present but not at index 0 is just as broken -- label indices shift."""
    classes = C.read_classes(dataset)
    reordered = [classes[1], C.NONE_CLASS, *classes[2:]]
    (dataset / C.CLASSES_FILE).write_text("\n".join(reordered) + "\n", encoding="utf-8")

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "classes.none_not_index_0" in report.codes()


def test_rejects_duplicate_classes(dataset: Path) -> None:
    classes = C.read_classes(dataset)
    (dataset / C.CLASSES_FILE).write_text(
        "\n".join([*classes, classes[1]]) + "\n", encoding="utf-8"
    )

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "classes.duplicate" in report.codes()


def test_rejects_empty_classes(dataset: Path) -> None:
    (dataset / C.CLASSES_FILE).write_text("\n", encoding="utf-8")
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "classes.empty" in report.codes()


def test_write_classes_refuses_bad_ordering(tmp_path: Path) -> None:
    """The writer refuses too, so an adapter cannot emit it in the first place."""
    with pytest.raises(ValueError, match="must be 'none'"):
        C.write_classes(tmp_path, ["wave", "none", "point"])


def test_write_classes_refuses_duplicates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        C.write_classes(tmp_path, ["none", "wave", "wave"])


# ======================================================================================
# annotations.csv structure
# ======================================================================================


def test_rejects_wrong_columns(dataset: Path, annotations_of) -> None:
    annotations_of(dataset, lambda df: df.rename(columns={"subject_id": "subject"}))
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.bad_columns" in report.codes()


def test_rejects_unknown_class(dataset: Path, annotations_of) -> None:
    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df.index[-1], "class"] = "not_a_real_class"
        return df

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.unknown_class" in report.codes()


def test_rejects_missing_subject_id(dataset: Path, annotations_of) -> None:
    """subject_id is the unit of splitting; a row without one cannot be split safely."""

    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df.index[0], "subject_id"] = ""
        return df

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.missing_id" in report.codes()


def test_validator_reports_rather_than_raises_on_garbage(dataset: Path) -> None:
    """A validator that crashes on malformed input fails exactly when it is needed.

    Regression test: a blank subject_id used to read back as a float NaN and blow up a
    later comparison with a TypeError, so the run died instead of producing findings.
    """
    (dataset / C.ANNOTATIONS_FILE).write_text(
        "video_id,class,start_frame,end_frame,subject_id,split\n"
        "vid000,none,0,10,,\n"
        ",wave,5,3,,\n"
        "vid000,,0,10,s0,\n",
        encoding="utf-8",
    )
    report = C.validate_dataset(dataset)          # must not raise
    assert not report.ok
    assert "annotations.missing_id" in report.codes()


def test_rejects_video_with_two_subjects(dataset: Path, annotations_of) -> None:
    video = first_video(dataset)

    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        rows = df.index[df["video_id"] == video]
        df.loc[rows[0], "subject_id"] = "impostor"
        return df

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.multi_subject_video" in report.codes()


# ======================================================================================
# annotations.csv semantics -- the tiling invariant
# ======================================================================================


def test_rejects_gap_in_coverage(dataset: Path, annotations_of) -> None:
    """A gap means an adapter inferred `none` from absence instead of writing it down.

    That is precisely how the non-gesture class evaporates.
    """
    video = first_video(dataset)

    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        none_rows = df.index[(df["video_id"] == video) & (df["class"] == C.NONE_CLASS)]
        # Drop an interior `none` row, leaving a hole between two gestures.
        return df.drop(index=none_rows[len(none_rows) // 2])

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.gap" in report.codes()


def test_rejects_gap_at_start(dataset: Path, annotations_of) -> None:
    video = first_video(dataset)

    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        rows = df[df["video_id"] == video].sort_values("start_frame")
        return df.drop(index=rows.index[0])

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.gap" in report.codes()


def test_rejects_overlapping_intervals(dataset: Path, annotations_of) -> None:
    """A frame with two labels is ambiguous and silently corrupts the target vector."""
    video = first_video(dataset)

    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        rows = df[df["video_id"] == video].sort_values("start_frame")
        first_idx = rows.index[0]
        df.loc[first_idx, "end_frame"] = int(rows["end_frame"].iloc[0]) + 10
        return df

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.overlap" in report.codes()


def test_rejects_inverted_interval(dataset: Path, annotations_of) -> None:
    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index[0]
        start = int(df.loc[idx, "start_frame"])
        end = int(df.loc[idx, "end_frame"])
        df.loc[idx, "start_frame"] = end + 5
        df.loc[idx, "end_frame"] = start
        return df

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.inverted_interval" in report.codes()


def test_rejects_negative_frame(dataset: Path, annotations_of) -> None:
    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df.index[0], "start_frame"] = -5
        return df

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "annotations.negative_frame" in report.codes()


# ======================================================================================
# Arrays
# ======================================================================================


def test_rejects_feature_length_mismatch(dataset: Path) -> None:
    """One row per frame. A truncated feature array silently misaligns every label."""
    video = first_video(dataset)
    path = C.feature_path(dataset, video)
    features = np.load(path)
    np.save(path, features[:-5])

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "features.length_mismatch" in report.codes()


def test_rejects_feature_wrong_dtype(dataset: Path) -> None:
    video = first_video(dataset)
    path = C.feature_path(dataset, video)
    np.save(path, np.load(path).astype(np.float64))

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "features.bad_dtype" in report.codes()


def test_rejects_feature_wrong_rank(dataset: Path) -> None:
    video = first_video(dataset)
    path = C.feature_path(dataset, video)
    np.save(path, np.load(path).ravel())

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "features.bad_shape" in report.codes()


def test_rejects_non_finite_features(dataset: Path) -> None:
    video = first_video(dataset)
    path = C.feature_path(dataset, video)
    features = np.load(path)
    features[3, 0] = np.nan
    np.save(path, features)

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "features.non_finite" in report.codes()


def test_rejects_inconsistent_feature_dim(dataset: Path) -> None:
    video = first_video(dataset)
    path = C.feature_path(dataset, video)
    features = np.load(path)
    np.save(path, np.concatenate([features, features], axis=1))

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "features.inconsistent_dim" in report.codes()


def test_rejects_missing_feature_file(dataset: Path) -> None:
    video = first_video(dataset)
    C.feature_path(dataset, video).unlink()

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "files.missing_features" in report.codes()


def test_rejects_pose_length_mismatch(dataset: Path) -> None:
    """Features and poses share the frame index; a mismatch desynchronises the streams."""
    video = first_video(dataset)
    path = C.pose_path(dataset, video)
    np.save(path, np.load(path)[:-3])

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "poses.length_mismatch" in report.codes()


def test_rejects_pose_wrong_shape(dataset: Path) -> None:
    video = first_video(dataset)
    path = C.pose_path(dataset, video)
    poses = np.load(path)
    np.save(path, poses[:, :, :2])          # [T, J, 2] instead of [T, J, 3]

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "poses.bad_shape" in report.codes()


def test_rejects_partial_pose_stream(dataset: Path) -> None:
    """The pose stream is optional, but it is all-or-nothing -- a half-present stream
    would silently train one branch on a subset of the data."""
    video = first_video(dataset)
    C.pose_path(dataset, video).unlink()

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "files.missing_poses" in report.codes()


def test_require_poses_flag(good_dataset: Path, tmp_path: Path) -> None:
    import shutil

    target = tmp_path / "no_poses"
    shutil.copytree(good_dataset, target)
    shutil.rmtree(target / C.POSES_DIR)

    assert C.validate_dataset(target, require_poses=False).ok
    strict = C.validate_dataset(target, require_poses=True)
    assert not strict.ok
    assert "files.missing_poses" in strict.codes()


# ======================================================================================
# Splits -- Hard Rule 3
# ======================================================================================


def test_unassigned_splits_warn_but_pass(good_dataset: Path) -> None:
    """Adapters leave splits empty; Stage 2 assigns them. That must not be an error yet."""
    report = C.validate_dataset(good_dataset)
    assert report.ok
    assert "splits.unassigned" in {i.code for i in report.warnings}


def test_require_splits_flag(good_dataset: Path) -> None:
    report = C.validate_dataset(good_dataset, require_splits=True)
    assert not report.ok
    assert "splits.unassigned" in report.codes()


def test_valid_subject_disjoint_splits_pass(dataset: Path, assign_splits) -> None:
    subjects = sorted(C.read_annotations(dataset)["subject_id"].unique())
    mapping = {subjects[0]: "train", subjects[1]: "val", subjects[2]: "test"}
    assign_splits(dataset, mapping)

    report = C.validate_dataset(dataset, require_splits=True)
    assert report.ok, report.render()


def test_rejects_subject_in_two_splits(dataset: Path) -> None:
    """Hard Rule 3. This is the classic route to a meaningless 98%."""
    df = C.read_annotations(dataset)
    subjects = sorted(df["subject_id"].unique())
    df["split"] = df["subject_id"].map(
        {subjects[0]: "train", subjects[1]: "val", subjects[2]: "test"}
    )
    leaked = df.index[df["subject_id"] == subjects[0]]
    df.loc[leaked[: len(leaked) // 2], "split"] = "test"
    df.to_csv(dataset / C.ANNOTATIONS_FILE, index=False)

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "splits.subject_leak" in report.codes()


def test_rejects_video_split_across_splits(dataset: Path) -> None:
    """Never split by frame or window -- half a video in train and half in test puts
    frames from the same gesture instance on both sides."""
    df = C.read_annotations(dataset)
    subjects = sorted(df["subject_id"].unique())
    df["split"] = df["subject_id"].map(
        {subjects[0]: "train", subjects[1]: "val", subjects[2]: "test"}
    )
    video = str(df["video_id"].iloc[0])
    rows = df.index[df["video_id"] == video]
    df.loc[rows[: len(rows) // 2], "split"] = "train"
    df.loc[rows[len(rows) // 2 :], "split"] = "test"
    df.to_csv(dataset / C.ANNOTATIONS_FILE, index=False)

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert {"splits.video_leak", "splits.subject_leak"} & report.codes()


def test_rejects_invalid_split_name(dataset: Path, annotations_of) -> None:
    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df["split"] = "validation"          # not one of train/val/test
        return df

    annotations_of(dataset, mutate)
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "splits.invalid_name" in report.codes()


# ======================================================================================
# meta.yaml
# ======================================================================================


def test_rejects_meta_class_count_mismatch(dataset: Path) -> None:
    meta = C.read_meta(dataset)
    meta["num_classes"] = 99
    C.write_meta(dataset, meta)

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "meta.num_classes_mismatch" in report.codes()


def test_rejects_missing_meta_key(dataset: Path) -> None:
    import yaml

    meta = C.read_meta(dataset)
    del meta["fps"]
    (dataset / C.META_FILE).write_text(yaml.safe_dump(meta), encoding="utf-8")

    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "meta.missing_key" in report.codes()


def test_write_meta_refuses_missing_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        C.write_meta(tmp_path, {"dataset_name": "x"})


# ======================================================================================
# Layout
# ======================================================================================


def test_rejects_missing_root(tmp_path: Path) -> None:
    report = C.validate_dataset(tmp_path / "nope")
    assert not report.ok
    assert "layout.missing_root" in report.codes()


def test_rejects_missing_required_file(dataset: Path) -> None:
    (dataset / C.CLASSES_FILE).unlink()
    report = C.validate_dataset(dataset)
    assert not report.ok
    assert "layout.missing_file" in report.codes()


# ======================================================================================
# Reader
# ======================================================================================


def test_reader_exposes_dataset(good_dataset: Path) -> None:
    ds = C.CanonicalDataset(good_dataset)
    assert ds.classes[C.NONE_INDEX] == C.NONE_CLASS
    assert ds.num_classes == 4
    assert len(ds.video_ids) == 6
    assert len(ds.subjects()) == 3
    assert ds.fps == 30.0
    assert ds.has_poses()


def test_frame_labels_tile_the_video(good_dataset: Path) -> None:
    ds = C.CanonicalDataset(good_dataset)
    for video_id in ds.video_ids:
        labels = ds.frame_labels(video_id)
        features = ds.load_features(video_id)
        assert labels.shape[0] == features.shape[0] == ds.num_frames(video_id)
        assert labels.dtype == np.int64
        assert labels.min() >= 0
        assert labels.max() < ds.num_classes


def test_frame_labels_match_annotation_rows(good_dataset: Path) -> None:
    """Every annotated span must be exactly its label in the dense vector."""
    ds = C.CanonicalDataset(good_dataset)
    annotations = ds.annotations
    for video_id in ds.video_ids:
        labels = ds.frame_labels(video_id)
        rows = annotations[annotations["video_id"] == video_id]
        # Column access by name: `class` is a keyword and itertuples renames it.
        for start, end, name in zip(
            rows["start_frame"].astype(int),
            rows["end_frame"].astype(int),
            rows["class"],
        ):
            expected = ds.class_to_index[name]
            assert (labels[start : end + 1] == expected).all(), (
                f"{video_id}: frames {start}..{end} do not all carry the annotated label"
            )


def test_frame_labels_include_none(good_dataset: Path) -> None:
    ds = C.CanonicalDataset(good_dataset)
    labels = np.concatenate([ds.frame_labels(v) for v in ds.video_ids])
    assert (labels == C.NONE_INDEX).any(), "no `none` frames -- the fixture is degenerate"
    assert (labels != C.NONE_INDEX).any(), "no gesture frames -- the fixture is degenerate"


def test_unknown_video_raises(good_dataset: Path) -> None:
    ds = C.CanonicalDataset(good_dataset)
    with pytest.raises(KeyError):
        ds.frame_labels("does_not_exist")


# ======================================================================================
# Adapter boundary
# ======================================================================================


def test_fill_none_spans_tiles_exactly(tmp_path: Path) -> None:
    """The helper adapters use to make non-gesture spans explicit."""
    adapter = SyntheticAdapter(
        AdapterConfig(source_root=tmp_path, output_root=tmp_path, options={})
    )
    adapter.add_instance("v0", "wave", 100, 149, "s0")
    adapter.add_instance("v0", "point", 200, 249, "s0")
    added = adapter.fill_none_spans("v0", num_frames=400, subject_id="s0")

    assert added == 3                       # before, between, after
    spans = sorted(
        (r["start_frame"], r["end_frame"]) for r in adapter._rows if r["video_id"] == "v0"
    )
    assert spans[0][0] == 0
    assert spans[-1][1] == 399
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start == prev_end + 1, "spans must be contiguous with no gap or overlap"


def test_adapter_refuses_none_in_gesture_classes(tmp_path: Path) -> None:
    """`none` is added at index 0 automatically; an adapter listing it is a bug."""
    adapter = SyntheticAdapter(
        AdapterConfig(
            source_root=tmp_path,
            output_root=tmp_path,
            options={"gesture_classes": ["none", "wave"]},
        )
    )
    with pytest.raises(ValueError, match="must not include"):
        _ = adapter.classes


def test_adapter_does_not_assign_splits(good_dataset: Path) -> None:
    """Stage 2 owns splitting. An adapter that invents splits oversteps its boundary."""
    df = C.read_annotations(good_dataset)
    assert (df["split"] == C.UNASSIGNED_SPLIT).all()


def test_convert_validates_its_own_output(tmp_path: Path) -> None:
    """An adapter's output is trusted because it passed the validator, not because an
    adapter produced it."""
    from src.data.adapters.synthetic import make_synthetic_dataset

    report = make_synthetic_dataset(tmp_path / "ds", num_videos=3, num_subjects=2, seed=11)
    assert report.ok, report.render()


# ======================================================================================
# Report rendering
# ======================================================================================


def test_report_render_is_readable(dataset: Path) -> None:
    (dataset / C.CLASSES_FILE).unlink()
    text = C.validate_dataset(dataset).render()
    assert "FAIL" in text
    assert "layout.missing_file" in text


def test_report_collapses_repeated_codes(dataset: Path) -> None:
    """One systemic fault must not bury every other finding under identical lines."""
    for path in (dataset / C.FEATURES_DIR).glob("*.npy"):
        np.save(path, np.load(path).astype(np.float64))

    text = C.validate_dataset(dataset).render(max_per_code=2)
    assert "suppressed" in text
