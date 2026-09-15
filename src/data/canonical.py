"""The canonical data format: the contract between datasets and everything else.

Every dataset -- IPN Hand now, the author's own SOS recordings later -- is converted
into this layout by a small adapter. Training, evaluation and model code read *only*
this. Nothing downstream may branch on dataset name. See CLAUDE.md sections 3 and 5.

    data/<dataset>/
      features/<video_id>.npy      # [T, D]    float32
      poses/<video_id>.npy         # [T, J, 3] float32, optional
      annotations.csv              # video_id,class,start_frame,end_frame,subject_id,split
      classes.txt                  # one class per line; index 0 is ALWAYS "none"
      meta.yaml                    # fps, feature dim, backbone id, settings, checksums

This module is the format's single source of truth: the constants, the reader, the
writer used by adapters, and the validator that decides whether an adapter's output
is admissible. If the format changes, it changes here and everywhere else follows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------------------
# Format constants. Import these; never hardcode the strings.
# --------------------------------------------------------------------------------------

#: The non-gesture class. It is the largest class in a continuous-video dataset and the
#: entire detection problem depends on it existing as a real, predicted class. It is
#: never dropped, never merged, never treated as "background to be ignored". Dropping it
#: silently converts online detection into trimmed-clip classification.
NONE_CLASS: str = "none"

#: `none` is always class index 0. Not a convention -- a guarantee downstream code relies
#: on (loss weighting, event extraction, the false-alarm metrics).
NONE_INDEX: int = 0

FEATURES_DIR: str = "features"
POSES_DIR: str = "poses"
ANNOTATIONS_FILE: str = "annotations.csv"
CLASSES_FILE: str = "classes.txt"
META_FILE: str = "meta.yaml"

#: Exact column names, in order. annotations.csv must match this header exactly.
ANNOTATION_COLUMNS: tuple[str, ...] = (
    "video_id",
    "class",
    "start_frame",
    "end_frame",
    "subject_id",
    "split",
)

VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")

#: Adapters leave `split` empty. Stage 2 assigns splits and freezes them to disk; an
#: adapter that invents its own splits is overstepping its boundary.
UNASSIGNED_SPLIT: str = ""

#: Required keys in meta.yaml.
REQUIRED_META_KEYS: tuple[str, ...] = ("dataset_name", "fps", "num_classes", "feature_dim")


# --------------------------------------------------------------------------------------
# Validation report
# --------------------------------------------------------------------------------------


class Level(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class Issue:
    """One validation finding, tied to a machine-readable code so tests can assert on it."""

    level: Level
    code: str
    message: str
    where: str = ""

    def __str__(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        return f"{self.level.value:<7} {self.code}{loc}: {self.message}"


@dataclass
class ValidationReport:
    root: Path
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def error(self, code: str, message: str, where: str = "") -> None:
        self.issues.append(Issue(Level.ERROR, code, message, where))

    def warn(self, code: str, message: str, where: str = "") -> None:
        self.issues.append(Issue(Level.WARNING, code, message, where))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level is Level.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level is Level.WARNING]

    @property
    def ok(self) -> bool:
        """True when the dataset is admissible. Warnings do not block; errors do."""
        return not self.errors

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def render(self, max_per_code: int = 5) -> str:
        """Human-readable report. Repeated codes are collapsed so one systemic fault
        does not bury the others under a thousand identical lines."""
        lines: list[str] = [f"Canonical format validation: {self.root}", "=" * 78]

        if self.stats:
            lines.append("Statistics")
            width = max(len(k) for k in self.stats)
            for key, value in self.stats.items():
                lines.append(f"  {key:<{width}}  {value}")
            lines.append("")

        for level in (Level.ERROR, Level.WARNING):
            group = [i for i in self.issues if i.level is level]
            if not group:
                continue
            lines.append(f"{level.value}S ({len(group)})")
            seen: dict[str, int] = {}
            for issue in group:
                seen[issue.code] = seen.get(issue.code, 0) + 1
                if seen[issue.code] <= max_per_code:
                    lines.append(f"  {issue}")
                elif seen[issue.code] == max_per_code + 1:
                    lines.append(f"  ... more {issue.code} suppressed")
            lines.append("")

        verdict = "PASS" if self.ok else "FAIL"
        lines.append(
            f"{verdict}  ({len(self.errors)} errors, {len(self.warnings)} warnings)"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------


def read_classes(root: Path | str) -> list[str]:
    """Read classes.txt. Blank lines are dropped; order is the label index order."""
    path = Path(root) / CLASSES_FILE
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_annotations(root: Path | str) -> pd.DataFrame:
    """Read annotations.csv with the dtypes the format promises.

    Every string column is NaN-filled to "". Without that, a blank cell reads back as a
    float nan: equality checks silently fail, and any comparison that mixes it with a
    real string raises. The validator must be able to *report* a blank subject_id rather
    than crash on it -- a validator that dies on malformed input fails exactly when it
    is needed.
    """
    path = Path(root) / ANNOTATIONS_FILE
    df = pd.read_csv(
        path,
        dtype={
            "video_id": str,
            "class": str,
            "start_frame": "Int64",
            "end_frame": "Int64",
            "subject_id": str,
            "split": str,
        },
        keep_default_na=False,
        na_values=[""],
    )
    for column in ("video_id", "class", "subject_id", "split"):
        if column in df.columns:
            df[column] = df[column].fillna(UNASSIGNED_SPLIT).astype(str)
    return df


def read_meta(root: Path | str) -> dict[str, Any]:
    path = Path(root) / META_FILE
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def feature_path(root: Path | str, video_id: str) -> Path:
    return Path(root) / FEATURES_DIR / f"{video_id}.npy"


def pose_path(root: Path | str, video_id: str) -> Path:
    return Path(root) / POSES_DIR / f"{video_id}.npy"


class CanonicalDataset:
    """Read-only accessor for a canonical-format dataset.

    Arrays are memory-mapped, so opening a dataset costs nothing and a training run
    touches only the frames it samples.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.classes: list[str] = read_classes(self.root)
        self.annotations: pd.DataFrame = read_annotations(self.root)
        self.meta: dict[str, Any] = read_meta(self.root)
        self.class_to_index: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    # -- identity ----------------------------------------------------------------

    @property
    def name(self) -> str:
        return str(self.meta.get("dataset_name", self.root.name))

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def video_ids(self) -> list[str]:
        return sorted(self.annotations["video_id"].unique().tolist())

    def subjects(self, split: str | None = None) -> list[str]:
        df = self.annotations
        if split is not None:
            df = df[df["split"] == split]
        return sorted(df["subject_id"].unique().tolist())

    def video_ids_for_split(self, split: str) -> list[str]:
        df = self.annotations
        return sorted(df[df["split"] == split]["video_id"].unique().tolist())

    def has_poses(self) -> bool:
        return (self.root / POSES_DIR).is_dir()

    # -- arrays ------------------------------------------------------------------

    def load_features(self, video_id: str, mmap: bool = True) -> np.ndarray:
        return np.load(feature_path(self.root, video_id), mmap_mode="r" if mmap else None)

    def load_poses(self, video_id: str, mmap: bool = True) -> np.ndarray | None:
        path = pose_path(self.root, video_id)
        if not path.exists():
            return None
        return np.load(path, mmap_mode="r" if mmap else None)

    def num_frames(self, video_id: str) -> int:
        """Frame count from the annotation tiling.

        Annotations tile the video exactly (the validator enforces it), so the last
        end_frame is the video length minus one. This deliberately avoids touching the
        feature array, so label logic stays usable before features are extracted.
        """
        rows = self.annotations[self.annotations["video_id"] == video_id]
        if rows.empty:
            raise KeyError(f"unknown video_id: {video_id}")
        return int(rows["end_frame"].max()) + 1

    def frame_labels(self, video_id: str) -> np.ndarray:
        """Dense per-frame label vector, shape [T], dtype int64.

        Initialised to NONE_INDEX so that any frame an annotation fails to cover reads
        as `none` rather than as a wrong gesture. The validator separately refuses
        uncovered frames -- this is belt and braces, not a licence to leave gaps.
        """
        rows = self.annotations[self.annotations["video_id"] == video_id]
        if rows.empty:
            raise KeyError(f"unknown video_id: {video_id}")
        labels = np.full(self.num_frames(video_id), NONE_INDEX, dtype=np.int64)
        # Column access by name rather than itertuples: `class` is a Python keyword and
        # itertuples silently renames it to a positional alias.
        starts = rows["start_frame"].astype(int).to_numpy()
        ends = rows["end_frame"].astype(int).to_numpy()
        names = rows["class"].to_numpy()
        for start, end, name in zip(starts, ends, names):
            labels[start : end + 1] = self.class_to_index[name]
        return labels

    def iter_videos(self, split: str | None = None) -> Iterator[str]:
        ids = self.video_ids_for_split(split) if split else self.video_ids
        yield from ids

    def __repr__(self) -> str:
        return (
            f"CanonicalDataset(name={self.name!r}, videos={len(self.video_ids)}, "
            f"classes={self.num_classes}, subjects={len(self.subjects())})"
        )


# --------------------------------------------------------------------------------------
# Writers -- the only sanctioned way for an adapter to emit canonical data
# --------------------------------------------------------------------------------------


def write_classes(root: Path | str, classes: Sequence[str]) -> None:
    """Write classes.txt, refusing any ordering that violates the `none`-is-0 guarantee."""
    classes = list(classes)
    if not classes:
        raise ValueError("classes must be non-empty")
    if classes[NONE_INDEX] != NONE_CLASS:
        raise ValueError(
            f"class index {NONE_INDEX} must be {NONE_CLASS!r}, got {classes[NONE_INDEX]!r}. "
            "The non-gesture class is not optional -- see CLAUDE.md section 3."
        )
    if len(set(classes)) != len(classes):
        raise ValueError("duplicate class names")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / CLASSES_FILE).write_text("\n".join(classes) + "\n", encoding="utf-8")


def write_annotations(root: Path | str, rows: pd.DataFrame | Iterable[dict]) -> None:
    """Write annotations.csv with the exact column set, sorted for reproducibility."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    missing = [c for c in ANNOTATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"annotations missing required columns: {missing}")
    df = df[list(ANNOTATION_COLUMNS)].copy()
    df["split"] = df["split"].fillna(UNASSIGNED_SPLIT)
    df = df.sort_values(["video_id", "start_frame"]).reset_index(drop=True)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / ANNOTATIONS_FILE, index=False)


def write_features(root: Path | str, video_id: str, array: np.ndarray) -> None:
    """Write one [T, D] float32 feature array."""
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"features for {video_id} must be [T, D], got shape {array.shape}")
    path = feature_path(root, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def write_poses(root: Path | str, video_id: str, array: np.ndarray) -> None:
    """Write one [T, J, 3] float32 pose array."""
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"poses for {video_id} must be [T, J, 3], got shape {array.shape}")
    path = pose_path(root, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def write_meta(root: Path | str, meta: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"meta.yaml missing required keys: {missing}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / META_FILE).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, default_flow_style=False)


def checksum_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, so a shipped feature cache can be verified on arrival.

    Feature extraction happens on whatever GPU box is available and the cache is copied
    to wherever training runs; these digests are how we know the copy is the cache the
    metrics were computed against.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------------------


def validate_dataset(
    root: Path | str,
    *,
    check_arrays: bool = True,
    require_poses: bool = False,
    require_splits: bool = False,
) -> ValidationReport:
    """Validate a canonical-format dataset.

    Args:
        root: dataset directory, e.g. ``data/ipn_hand``.
        check_arrays: load every .npy to verify shape/dtype/finiteness. Disable for a
            fast structural check on a large dataset.
        require_poses: treat a missing pose stream as an error rather than a note.
            The pose stream is optional in the format; Stage 3 turns this on.
        require_splits: treat unassigned splits as an error. Adapters leave splits
            empty (Stage 1); Stage 2 turns this on once splits are frozen.

    Returns:
        A report. ``report.ok`` is False iff there is at least one error.
    """
    root = Path(root)
    report = ValidationReport(root=root)

    if not _check_layout(root, report):
        return report

    classes = _check_classes(root, report)
    if classes is None:
        return report

    annotations = _check_annotations_structure(root, classes, report)
    if annotations is None:
        return report

    _check_meta(root, classes, report)
    lengths = _check_annotation_semantics(annotations, report)
    _check_splits(annotations, report, require_splits=require_splits)
    _check_files(root, annotations, lengths, report, check_arrays, require_poses)
    _collect_stats(annotations, classes, lengths, report)

    return report


def _check_layout(root: Path, report: ValidationReport) -> bool:
    if not root.is_dir():
        report.error("layout.missing_root", f"dataset root does not exist: {root}")
        return False
    ok = True
    for name in (CLASSES_FILE, ANNOTATIONS_FILE, META_FILE):
        if not (root / name).is_file():
            report.error("layout.missing_file", f"required file missing: {name}")
            ok = False
    if not (root / FEATURES_DIR).is_dir():
        report.error("layout.missing_file", f"required directory missing: {FEATURES_DIR}/")
        ok = False
    return ok


def _check_classes(root: Path, report: ValidationReport) -> list[str] | None:
    try:
        classes = read_classes(root)
    except Exception as exc:  # noqa: BLE001 - surfaced as a validation error
        report.error("classes.unreadable", f"could not read {CLASSES_FILE}: {exc}")
        return None

    if not classes:
        report.error("classes.empty", f"{CLASSES_FILE} is empty")
        return None

    # The single most important check in this file.
    if classes[NONE_INDEX] != NONE_CLASS:
        report.error(
            "classes.none_not_index_0",
            f"class index {NONE_INDEX} must be {NONE_CLASS!r}, found {classes[NONE_INDEX]!r}. "
            "The non-gesture class is the largest class and the detection task is "
            "meaningless without it. Do not drop it as background.",
            where=CLASSES_FILE,
        )

    duplicates = sorted({c for c in classes if classes.count(c) > 1})
    if duplicates:
        report.error(
            "classes.duplicate", f"duplicate class names: {duplicates}", where=CLASSES_FILE
        )

    if len(classes) < 2:
        report.error(
            "classes.too_few",
            f"only {len(classes)} class(es); need `none` plus at least one gesture",
            where=CLASSES_FILE,
        )

    return classes


def _check_meta(root: Path, classes: list[str], report: ValidationReport) -> None:
    try:
        meta = read_meta(root)
    except Exception as exc:  # noqa: BLE001
        report.error("meta.unreadable", f"could not read {META_FILE}: {exc}")
        return

    for key in REQUIRED_META_KEYS:
        if key not in meta:
            report.error("meta.missing_key", f"missing required key {key!r}", where=META_FILE)

    fps = meta.get("fps")
    if fps is not None and (not isinstance(fps, (int, float)) or fps <= 0):
        report.error("meta.bad_fps", f"fps must be a positive number, got {fps!r}", where=META_FILE)

    declared = meta.get("num_classes")
    if declared is not None and declared != len(classes):
        report.error(
            "meta.num_classes_mismatch",
            f"meta says num_classes={declared} but {CLASSES_FILE} has {len(classes)}",
            where=META_FILE,
        )


def _check_annotations_structure(
    root: Path, classes: list[str], report: ValidationReport
) -> pd.DataFrame | None:
    try:
        df = read_annotations(root)
    except Exception as exc:  # noqa: BLE001
        report.error("annotations.unreadable", f"could not read {ANNOTATIONS_FILE}: {exc}")
        return None

    if tuple(df.columns) != ANNOTATION_COLUMNS:
        report.error(
            "annotations.bad_columns",
            f"columns must be exactly {list(ANNOTATION_COLUMNS)}, got {list(df.columns)}",
            where=ANNOTATIONS_FILE,
        )
        return None

    if df.empty:
        report.error("annotations.empty", f"{ANNOTATIONS_FILE} has no rows")
        return None

    for column in ("video_id", "subject_id"):
        blank = df[df[column].isna() | (df[column].astype(str).str.strip() == "")]
        if not blank.empty:
            report.error(
                "annotations.missing_id",
                f"{len(blank)} row(s) with empty {column}; "
                f"{column} is required for every row",
                where=ANNOTATIONS_FILE,
            )

    unknown = sorted(set(df["class"]) - set(classes))
    if unknown:
        report.error(
            "annotations.unknown_class",
            f"classes not present in {CLASSES_FILE}: {unknown}",
            where=ANNOTATIONS_FILE,
        )

    for column in ("start_frame", "end_frame"):
        if df[column].isna().any():
            report.error(
                "annotations.non_integer_frame",
                f"{column} contains non-integer or missing values",
                where=ANNOTATIONS_FILE,
            )
            return None

    return df


def _check_annotation_semantics(
    df: pd.DataFrame, report: ValidationReport
) -> dict[str, int]:
    """Per-video interval checks. Returns video_id -> frame count."""
    lengths: dict[str, int] = {}

    for video_id, rows in df.groupby("video_id", sort=True):
        rows = rows.sort_values("start_frame")
        starts = rows["start_frame"].astype(int).to_numpy()
        ends = rows["end_frame"].astype(int).to_numpy()

        if (starts < 0).any():
            report.error(
                "annotations.negative_frame", "start_frame must be >= 0", where=video_id
            )
        bad = starts > ends
        if bad.any():
            report.error(
                "annotations.inverted_interval",
                f"{int(bad.sum())} row(s) with start_frame > end_frame "
                "(intervals are closed and inclusive)",
                where=video_id,
            )
            continue

        # str() defensively: this runs even when earlier checks have already flagged a
        # blank or malformed id, and the report is more useful than the traceback.
        subjects = {str(s) for s in rows["subject_id"]}
        if len(subjects) > 1:
            report.error(
                "annotations.multi_subject_video",
                f"video maps to multiple subject_ids {sorted(subjects)}; "
                "subject-disjoint splitting cannot be guaranteed",
                where=video_id,
            )

        # The tiling invariant. Non-gesture spans are stored as explicit `none` rows, so
        # the intervals must cover [0, T-1] exactly: no gaps, no overlaps. A gap means an
        # adapter inferred `none` from absence instead of writing it down -- the exact
        # mistake that makes the non-gesture class evaporate.
        if starts[0] != 0:
            report.error(
                "annotations.gap",
                f"first annotation starts at frame {starts[0]}, expected 0; "
                f"frames 0..{starts[0] - 1} are uncovered",
                where=video_id,
            )
        gaps = starts[1:] - ends[:-1] - 1
        if (gaps > 0).any():
            first = int(np.argmax(gaps > 0))
            report.error(
                "annotations.gap",
                f"{int((gaps > 0).sum())} uncovered frame range(s); first is "
                f"{ends[first] + 1}..{starts[first + 1] - 1}. Non-gesture spans must be "
                f"explicit {NONE_CLASS!r} rows, not gaps.",
                where=video_id,
            )
        if (gaps < 0).any():
            first = int(np.argmax(gaps < 0))
            report.error(
                "annotations.overlap",
                f"{int((gaps < 0).sum())} overlapping interval(s); first is "
                f"{starts[first]}..{ends[first]} vs {starts[first + 1]}..{ends[first + 1]}. "
                "A frame with two labels is ambiguous.",
                where=video_id,
            )

        lengths[str(video_id)] = int(ends.max()) + 1

    return lengths


def _check_splits(
    df: pd.DataFrame, report: ValidationReport, *, require_splits: bool
) -> None:
    splits = set(df["split"]) - {UNASSIGNED_SPLIT}
    invalid = sorted(splits - set(VALID_SPLITS))
    if invalid:
        report.error(
            "splits.invalid_name",
            f"split must be one of {list(VALID_SPLITS)} or empty, got {invalid}",
            where=ANNOTATIONS_FILE,
        )

    unassigned = int((df["split"] == UNASSIGNED_SPLIT).sum())
    if unassigned:
        if require_splits:
            report.error(
                "splits.unassigned",
                f"{unassigned} row(s) have no split; splits must be frozen before training",
                where=ANNOTATIONS_FILE,
            )
        else:
            report.warn(
                "splits.unassigned",
                f"{unassigned} row(s) have no split assigned yet (expected before Stage 2)",
                where=ANNOTATIONS_FILE,
            )

    # Hard Rule 3. Checked here as well as in the Stage 2 test, because this is the
    # failure that silently produces a meaningless 98%.
    assigned = df[df["split"] != UNASSIGNED_SPLIT]
    if not assigned.empty:
        per_subject = assigned.groupby("subject_id")["split"].nunique()
        leaked = sorted(per_subject[per_subject > 1].index.tolist())
        if leaked:
            report.error(
                "splits.subject_leak",
                f"subject(s) appear in more than one split: {leaked}. "
                "Every reported number must come from a subject-disjoint split.",
                where=ANNOTATIONS_FILE,
            )

        per_video = assigned.groupby("video_id")["split"].nunique()
        split_videos = sorted(per_video[per_video > 1].index.tolist())
        if split_videos:
            report.error(
                "splits.video_leak",
                f"video(s) split across multiple splits: {split_videos}. "
                "Never split by frame or window.",
                where=ANNOTATIONS_FILE,
            )


def _check_files(
    root: Path,
    df: pd.DataFrame,
    lengths: dict[str, int],
    report: ValidationReport,
    check_arrays: bool,
    require_poses: bool,
) -> None:
    annotated = set(lengths)
    on_disk = {p.stem for p in (root / FEATURES_DIR).glob("*.npy")}

    for video_id in sorted(annotated - on_disk):
        report.error(
            "files.missing_features",
            f"annotated video has no {FEATURES_DIR}/{video_id}.npy",
            where=video_id,
        )
    for video_id in sorted(on_disk - annotated):
        report.warn(
            "files.orphan_features",
            f"{FEATURES_DIR}/{video_id}.npy has no annotation rows",
            where=video_id,
        )

    poses_dir = root / POSES_DIR
    has_pose_dir = poses_dir.is_dir()
    if not has_pose_dir:
        message = f"no {POSES_DIR}/ directory; the pose stream is absent"
        if require_poses:
            report.error("files.missing_poses", message)
        else:
            report.warn("files.no_pose_stream", message + " (optional in the format)")

    if not check_arrays:
        return

    feature_dims: set[int] = set()
    joint_counts: set[int] = set()

    for video_id in sorted(annotated & on_disk):
        expected_t = lengths[video_id]
        try:
            features = np.load(feature_path(root, video_id), mmap_mode="r")
        except Exception as exc:  # noqa: BLE001
            report.error("files.unreadable", f"could not load features: {exc}", where=video_id)
            continue

        if features.ndim != 2:
            report.error(
                "features.bad_shape",
                f"features must be [T, D], got shape {features.shape}",
                where=video_id,
            )
            continue
        if features.dtype != np.float32:
            report.error(
                "features.bad_dtype",
                f"features must be float32, got {features.dtype}",
                where=video_id,
            )
        if features.shape[0] != expected_t:
            report.error(
                "features.length_mismatch",
                f"features have {features.shape[0]} frames but annotations tile "
                f"{expected_t} frames. One row per frame is required.",
                where=video_id,
            )
        if not np.isfinite(np.asarray(features)).all():
            report.error(
                "features.non_finite", "features contain NaN or Inf", where=video_id
            )
        feature_dims.add(int(features.shape[1]))

        if not has_pose_dir:
            continue
        pose_file = pose_path(root, video_id)
        if not pose_file.exists():
            report.error(
                "files.missing_poses",
                f"{POSES_DIR}/ exists but {video_id}.npy is missing; the pose stream "
                "must be complete or absent, not partial",
                where=video_id,
            )
            continue
        try:
            poses = np.load(pose_file, mmap_mode="r")
        except Exception as exc:  # noqa: BLE001
            report.error("files.unreadable", f"could not load poses: {exc}", where=video_id)
            continue
        if poses.ndim != 3 or poses.shape[2] != 3:
            report.error(
                "poses.bad_shape",
                f"poses must be [T, J, 3], got shape {poses.shape}",
                where=video_id,
            )
            continue
        if poses.dtype != np.float32:
            report.error(
                "poses.bad_dtype", f"poses must be float32, got {poses.dtype}", where=video_id
            )
        if poses.shape[0] != expected_t:
            report.error(
                "poses.length_mismatch",
                f"poses have {poses.shape[0]} frames but annotations tile {expected_t}; "
                "features and poses share the frame index",
                where=video_id,
            )
        joint_counts.add(int(poses.shape[1]))

    if len(feature_dims) > 1:
        report.error(
            "features.inconsistent_dim",
            f"feature dimension differs across videos: {sorted(feature_dims)}",
        )
    if len(joint_counts) > 1:
        report.error(
            "poses.inconsistent_joints",
            f"joint count differs across videos: {sorted(joint_counts)}",
        )

    if feature_dims:
        report.stats["feature_dim"] = sorted(feature_dims)[0]
    if joint_counts:
        report.stats["num_joints"] = sorted(joint_counts)[0]


def _collect_stats(
    df: pd.DataFrame,
    classes: list[str],
    lengths: dict[str, int],
    report: ValidationReport,
) -> None:
    gestures = df[df["class"] != NONE_CLASS]
    none_rows = df[df["class"] == NONE_CLASS]
    total_frames = sum(lengths.values())

    report.stats.update(
        {
            "videos": df["video_id"].nunique(),
            "subjects": df["subject_id"].nunique(),
            "classes": len(classes),
            "gesture_classes": len(classes) - 1,
            "gesture_instances": len(gestures),
            "none_instances": len(none_rows),
            "total_frames": total_frames,
        }
    )

    if total_frames:
        none_frames = int(
            (none_rows["end_frame"].astype(int) - none_rows["start_frame"].astype(int) + 1).sum()
        )
        report.stats["none_frame_fraction"] = f"{none_frames / total_frames:.3f}"

    for split in VALID_SPLITS:
        rows = df[df["split"] == split]
        if not rows.empty:
            report.stats[f"split.{split}"] = (
                f"{rows['video_id'].nunique()} videos, "
                f"{rows['subject_id'].nunique()} subjects"
            )


__all__ = [
    "NONE_CLASS",
    "NONE_INDEX",
    "ANNOTATION_COLUMNS",
    "VALID_SPLITS",
    "UNASSIGNED_SPLIT",
    "FEATURES_DIR",
    "POSES_DIR",
    "ANNOTATIONS_FILE",
    "CLASSES_FILE",
    "META_FILE",
    "Level",
    "Issue",
    "ValidationReport",
    "CanonicalDataset",
    "read_classes",
    "read_annotations",
    "read_meta",
    "feature_path",
    "pose_path",
    "write_classes",
    "write_annotations",
    "write_features",
    "write_poses",
    "write_meta",
    "checksum_file",
    "validate_dataset",
]
