"""The dataset adapter boundary.

An adapter is the *only* place in this repository that is allowed to know anything about
a specific dataset: its directory layout, its annotation quirks, its subject naming, its
frame rate. Everything downstream -- splits, sampling, models, decision layer, metrics --
reads the canonical format and nothing else.

That boundary is what makes Phase 2 cheap. When the author's own SOS recordings arrive,
the work is one new subclass here plus one YAML file under ``configs/dataset/``. No
training, model or evaluation code should need to change. See CLAUDE.md Rule 5.

To add a dataset:

1. Subclass :class:`DatasetAdapter`.
2. Implement :meth:`build`, using the ``emit_*`` helpers to write canonical data.
3. Register it in ``ADAPTERS`` in ``src/data/adapters/__init__.py``.
4. Add ``configs/dataset/<name>.yaml``.

:meth:`convert` then runs the build and validates the result, so an adapter cannot
quietly emit something malformed.
"""

from __future__ import annotations

import abc
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data import canonical as C


@dataclass
class AdapterConfig:
    """Inputs an adapter needs. Dataset-specific extras go in ``options``."""

    #: Where the raw, unconverted dataset lives (downloads, frames, annotation files).
    source_root: Path
    #: Where the canonical dataset is written, e.g. ``data/ipn_hand``.
    output_root: Path
    #: Dataset-specific knobs, loaded from configs/dataset/<name>.yaml.
    options: dict[str, Any]

    def __post_init__(self) -> None:
        self.source_root = Path(self.source_root)
        self.output_root = Path(self.output_root)


class DatasetAdapter(abc.ABC):
    """Base class for dataset adapters.

    Subclasses accumulate annotation rows with :meth:`add_instance` and write arrays with
    :meth:`emit_features` / :meth:`emit_poses`, then :meth:`convert` finalises and
    validates. Subclasses never write ``annotations.csv`` or ``classes.txt`` by hand.
    """

    #: Short identifier, matching the key in ADAPTERS and the config file name.
    name: str = "base"

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self._rows: list[dict[str, Any]] = []

    # -- interface subclasses implement -------------------------------------------

    @abc.abstractmethod
    def gesture_classes(self) -> Sequence[str]:
        """Gesture class names, **excluding** ``none``.

        ``none`` is prepended automatically at index 0 so no adapter can forget it or
        order it wrongly.
        """

    @abc.abstractmethod
    def build(self) -> None:
        """Read the raw dataset and emit canonical content.

        Implementations call :meth:`add_instance` for every labelled span -- including
        the non-gesture spans -- and :meth:`emit_features` / :meth:`emit_poses` per video.
        """

    def extra_meta(self) -> dict[str, Any]:
        """Dataset-specific fields to record in meta.yaml. Override as needed."""
        return {}

    # -- helpers subclasses use ---------------------------------------------------

    @property
    def classes(self) -> list[str]:
        """Full class list with ``none`` guaranteed at index 0."""
        gestures = [c for c in self.gesture_classes()]
        if C.NONE_CLASS in gestures:
            raise ValueError(
                f"gesture_classes() must not include {C.NONE_CLASS!r}; it is added "
                "automatically at index 0"
            )
        return [C.NONE_CLASS, *gestures]

    def add_instance(
        self,
        video_id: str,
        class_name: str,
        start_frame: int,
        end_frame: int,
        subject_id: str,
    ) -> None:
        """Record one labelled span. Interval is closed and inclusive: ``[start, end]``.

        Non-gesture spans are recorded here too, with ``class_name=NONE_CLASS``. They are
        never left as gaps for downstream code to infer -- see :meth:`fill_none_spans`.
        """
        if end_frame < start_frame:
            raise ValueError(
                f"{video_id}: end_frame {end_frame} < start_frame {start_frame}"
            )
        self._rows.append(
            {
                "video_id": str(video_id),
                "class": str(class_name),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "subject_id": str(subject_id),
                # Adapters never assign splits. Stage 2 owns that and freezes it to disk.
                "split": C.UNASSIGNED_SPLIT,
            }
        )

    def fill_none_spans(self, video_id: str, num_frames: int, subject_id: str) -> int:
        """Insert explicit ``none`` rows into every gap for one video.

        Call this after adding a video's gesture instances. Many datasets, IPN Hand
        included, annotate only the gestures and leave the rest implicit. The canonical
        format requires the annotations to tile the video exactly, so the non-gesture
        class is a first-class, countable, trainable label rather than "whatever is left
        over". Returns the number of ``none`` rows added.
        """
        spans = sorted(
            ((r["start_frame"], r["end_frame"]) for r in self._rows if r["video_id"] == video_id),
        )
        added = 0
        cursor = 0
        for start, end in spans:
            if start > cursor:
                self.add_instance(video_id, C.NONE_CLASS, cursor, start - 1, subject_id)
                added += 1
            cursor = max(cursor, end + 1)
        if cursor < num_frames:
            self.add_instance(video_id, C.NONE_CLASS, cursor, num_frames - 1, subject_id)
            added += 1
        return added

    def emit_features(self, video_id: str, array: np.ndarray) -> None:
        C.write_features(self.config.output_root, video_id, array)

    def emit_poses(self, video_id: str, array: np.ndarray) -> None:
        C.write_poses(self.config.output_root, video_id, array)

    # -- driver -------------------------------------------------------------------

    def convert(
        self,
        *,
        validate: bool = True,
        check_arrays: bool = True,
        require_features: bool = True,
    ) -> C.ValidationReport:
        """Run the adapter and validate its output.

        An adapter's output is not trusted because it was produced by an adapter; it is
        trusted because it passed the validator.

        ``require_features`` is a caller's decision rather than an adapter's, so that an
        annotation-only conversion is visible at the call site instead of hidden in a
        class attribute. See :func:`src.data.canonical.validate_dataset`.
        """
        root = self.config.output_root
        root.mkdir(parents=True, exist_ok=True)

        self._rows.clear()
        self.build()

        if not self._rows:
            raise RuntimeError(f"adapter {self.name!r} produced no annotation rows")

        C.write_classes(root, self.classes)
        C.write_annotations(root, pd.DataFrame(self._rows))
        # Later stages record into meta.yaml too -- Stage 3's feature-cache checksums above
        # all. Re-running an adapter rewrites the keys it owns and keeps the rest; replacing
        # the file wholesale would leave hours of extracted features unverifiable.
        meta = self._meta()
        if (root / C.META_FILE).is_file():
            recorded = C.read_meta(root)
            # An adapter value of None means "could not tell" (e.g. feature_dim before any
            # features exist); it must not erase a value a later stage measured.
            meta = {**recorded, **{k: v for k, v in meta.items()
                                   if v is not None or recorded.get(k) is None}}
        C.write_meta(root, meta)

        if not validate:
            return C.ValidationReport(root=root)
        return C.validate_dataset(
            root, check_arrays=check_arrays, require_features=require_features
        )

    def _meta(self) -> dict[str, Any]:
        options = self.config.options
        feature_dim = options.get("feature_dim")
        if feature_dim is None:
            # Infer from what was actually written, so meta describes reality rather
            # than intent.
            written = sorted((self.config.output_root / C.FEATURES_DIR).glob("*.npy"))
            if written:
                feature_dim = int(np.load(written[0], mmap_mode="r").shape[1])

        meta: dict[str, Any] = {
            "dataset_name": options.get("name", self.name),
            "fps": options.get("fps"),
            "num_classes": len(self.classes),
            "feature_dim": feature_dim,
            "adapter": self.name,
            "source_root": str(self.config.source_root),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
        meta.update(self.extra_meta())
        return meta


__all__ = ["AdapterConfig", "DatasetAdapter"]
