"""IPN Hand -> canonical format.

Benitez-Garcia et al., "IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous
Hand Gesture Recognition", ICPR 2020. CC BY 4.0.
https://gibranbenitez.github.io/IPN_Hand/

Phase 1's engineering vehicle. The 13 classes are touchless-interface commands, not
distress signals; what makes IPN Hand the right stand-in is its *shape* -- continuous
video, a dominant non-gesture class, per-subject metadata, fuzzy onsets. Phase 2 swaps in
the author's own SOS recordings behind this same boundary and nothing downstream moves.

This module is the only place in the repository allowed to know any of the following.

Source layout (``source_root`` points at ``.../IPN_Hand`` or at its ``annotations/``)::

    IPN_Hand/
      annotations/
        Annot_List.txt        video,label,id,t_start,t_end,frames   -- every span
        metadata.csv          Video Name,Frames,...,Set             -- video lengths
        classIdx.txt          id,label                              -- the 14 codes
        Annot_{Train,Test}List.txt, Video_{Train,Test}List.txt      -- official split
      frames/<video_name>/<video_name>_%06d.jpg                     -- Stage 3 reads this

Decisions taken here, with reasons, because each is a place a thesis number could go
quietly wrong.

*Frame indexing.* IPN's ``t_start``/``t_end`` are 1-indexed and inclusive. The canonical
format is 0-indexed and inclusive, so both endpoints lose one. An off-by-one here shifts
every onset by 33 ms and quietly biases the Stage 7 latency metric.

*Video length.* Taken from the annotation spans -- the last ``t_end`` for each video --
and **not** from ``metadata.csv``'s ``Frames`` column. That is the opposite of the obvious
choice, so here is the evidence.

Counting the JPEGs in ``frames/`` gives 800,491 across the 200 videos, and for every
single video the JPEG count equals both the highest frame number present and the last
``t_end``. ``metadata.csv`` (and ``Video_{Train,Test}List.txt``, which carry the same
numbers) instead totals 800,505: it is larger by exactly one for exactly 14 videos, and
those 14 are precisely the 14 whose frame directory contains a stray Windows
``desktop.ini``. The ``Frames`` column was evidently produced by counting directory
entries, so it counts that artefact as a frame.

Trusting ``Frames`` would therefore append one non-existent frame to each of those 14
videos. It would tile cleanly, validate cleanly, and give Stage 3 fourteen feature rows
with no image behind them. The annotation spans are the authority; ``metadata.csv`` is
read only to cross-check, and any disagreement beyond that known ``+1`` is an error.

A consequence worth stating: because the spans already tile ``[1, last_t_end]`` exactly,
``fill_none_spans`` adds nothing on this dataset, and the ``none`` instance count matches
the published 1,431 with no caveat.

*Subject identity.* ``metadata.csv`` has no subject column. The video name
``1CM1_4_R_#229`` decomposes as ``<camera>_<subject>_<hand>_#<clip>``, and the subject
token alone is **not** an identity: it runs 1..32 within each camera and repeats across
cameras. ``(camera, subject)`` yields exactly 50 groups of exactly 4 videos, matching the
published "50 subjects, 200 videos", and the authors' own train/test split is disjoint
under ``(camera, subject)`` while eight bare subject tokens straddle it. So ``subject_id``
is ``"<camera>_<subject>"``. Getting this wrong is a Hard Rule 3 violation with a
plausible-looking accuracy attached to it.

*Splits.* The ``Set`` column is ignored on purpose. Adapters do not assign splits; Stage 2
generates subject-disjoint splits and freezes them to disk.

*Features.* Not written here. Stage 3 extracts them into ``features/``; until then the
dataset is annotation-only and ``validation.require_features`` stays off.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data import canonical as C
from src.data.adapters.base import AdapterConfig, DatasetAdapter

#: IPN's own code for the non-gesture class. It is class id 1 in classIdx.txt and the
#: largest class in the dataset; it becomes canonical ``none`` at index 0.
NONE_CODE: str = "D0X"

#: The 13 gesture codes in classIdx.txt order, paired positionally with the readable names
#: in configs/dataset/ipn_hand.yaml. That keeps the names in config (Rule 5) and the
#: dataset-specific codes here. Both order and membership are checked against the
#: classIdx.txt actually on disk, so a re-released annotation set fails loudly instead of
#: silently remapping every label in the dataset.
GESTURE_CODES: tuple[str, ...] = (
    "B0A",
    "B0B",
    "G01",
    "G02",
    "G03",
    "G04",
    "G05",
    "G06",
    "G07",
    "G08",
    "G09",
    "G10",
    "G11",
)

ANNOT_FILE = "Annot_List.txt"
METADATA_FILE = "metadata.csv"
CLASS_INDEX_FILE = "classIdx.txt"
ANNOTATIONS_DIR = "annotations"
FRAMES_DIR = "frames"

#: <camera>_<subject>_<hand>_#<clip>, e.g. 1CM1_4_R_#229.
VIDEO_NAME_RE = re.compile(
    r"^(?P<camera>[^_]+)_(?P<subject>[^_]+)_(?P<hand>[^_]+)_#(?P<clip>.+)$"
)


class IPNHandAdapter(DatasetAdapter):
    """Converts IPN Hand's annotation files into the canonical format."""

    name = "ipn_hand"

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._stats: dict[str, Any] = {}

    # -- interface ----------------------------------------------------------------

    def gesture_classes(self) -> Sequence[str]:
        names = list(self.config.options.get("gesture_classes") or [])
        if len(names) != len(GESTURE_CODES):
            raise ValueError(
                f"configs/dataset/ipn_hand.yaml lists {len(names)} gesture_classes but "
                f"IPN Hand has {len(GESTURE_CODES)} gesture codes {list(GESTURE_CODES)}. "
                "The two are paired positionally; a mismatch mislabels every frame."
            )
        return names

    def build(self) -> None:
        annot, metadata = self._read_source()
        code_to_class = self._code_map()

        # Video length comes from the annotations, which the JPEGs on disk confirm and
        # metadata.csv does not. See the module docstring.
        lengths = {
            str(video): int(end)
            for video, end in annot.groupby("video")["t_end"].max().items()
        }
        declared = {
            str(name): int(frames)
            for name, frames in zip(metadata["Video Name"], metadata["Frames"])
        }

        unannotated = sorted(set(declared) - set(lengths))
        if unannotated:
            raise ValueError(
                f"{len(unannotated)} video(s) in {METADATA_FILE} have no rows in "
                f"{ANNOT_FILE}: {unannotated[:5]}. A video with no annotations would be "
                "tiled as one long `none` span, which is a fabricated label."
            )
        undeclared = sorted(set(lengths) - set(declared))
        if undeclared:
            raise ValueError(
                f"{len(undeclared)} video(s) in {ANNOT_FILE} are absent from "
                f"{METADATA_FILE}: {undeclared[:5]}"
            )

        overcounts = self._reconcile_lengths(lengths, declared)

        none_rows_added = 0
        for video_id, rows in annot.groupby("video", sort=True):
            video_id = str(video_id)
            subject_id = subject_of(video_id)
            num_frames = lengths[video_id]

            rows = rows.sort_values("t_start")
            for start, end, code in zip(rows["t_start"], rows["t_end"], rows["label"]):
                # 1-indexed inclusive -> 0-indexed inclusive.
                start0, end0 = int(start) - 1, int(end) - 1
                if start0 < 0:
                    raise ValueError(
                        f"{video_id}: t_start={start} is below IPN's 1-indexed origin"
                    )
                if end0 >= num_frames:
                    raise ValueError(
                        f"{video_id}: a span ends at frame {end} but {METADATA_FILE} says "
                        f"the video has {num_frames} frames"
                    )
                self.add_instance(
                    video_id, code_to_class[str(code)], start0, end0, subject_id
                )

            none_rows_added += self.fill_none_spans(video_id, num_frames, subject_id)

        source_none = int((annot["label"] == NONE_CODE).sum())
        self._stats = {
            "num_videos": len(lengths),
            "num_subjects": len({subject_of(v) for v in lengths}),
            "num_gesture_classes": len(GESTURE_CODES),
            "num_gesture_instances": int((annot["label"] != NONE_CODE).sum()),
            # Counted from the source file, which is what the paper counts.
            "num_source_none_instances": source_none,
            # Counted from the emitted rows. Equal to the source count unless
            # fill_none_spans had to cover a gap, which on IPN Hand it does not.
            "num_none_instances": sum(
                1 for r in self._rows if r["class"] == C.NONE_CLASS
            ),
            "num_frames": int(sum(lengths.values())),
            "num_filled_none_rows": none_rows_added,
            "num_declared_frames": int(sum(declared.values())),
            "videos_with_inflated_frame_count": overcounts,
        }

    def extra_meta(self) -> dict[str, Any]:
        options = self.config.options
        source = dict(options.get("source") or {})
        return {
            "frame_size": list(options.get("frame_size") or []),
            "source_url": source.get("url"),
            "license": source.get("license"),
            # Stage 3 fills these in when it writes the feature cache.
            "feature_backbone": None,
            "pose_backbone": None,
            "class_code_map": {NONE_CODE: C.NONE_CLASS, **self._code_map_names()},
            "frame_index_origin": "0-based inclusive, converted from IPN's 1-based",
            "video_length_source": (
                f"{ANNOT_FILE}:max(t_end), confirmed against the JPEG count in frames/. "
                f"{METADATA_FILE}:Frames is NOT used -- it counts a stray desktop.ini as "
                "a frame in 14 videos."
            ),
            "official_split_ignored": (
                "metadata.csv's Set column is not used. Stage 2 generates and freezes "
                "subject-disjoint splits (CLAUDE.md Rule 3)."
            ),
            "computed": dict(self._stats),
        }

    # -- internals ----------------------------------------------------------------

    def _reconcile_lengths(
        self, lengths: dict[str, int], declared: dict[str, int]
    ) -> list[str]:
        """Cross-check the annotation-derived length against ``metadata.csv``.

        Returns the videos where ``metadata.csv`` claims exactly one frame more than the
        annotations cover -- the known ``desktop.ini`` artefact documented in the module
        docstring. Any other disagreement is an error: a metadata count *below* the
        annotations would mean a span runs past the end of the video, and a gap of more
        than one frame is not something this adapter has evidence to explain away.
        """
        overcounts: list[str] = []
        unexplained: list[str] = []
        for video_id, length in sorted(lengths.items()):
            difference = declared[video_id] - length
            if difference == 0:
                continue
            if difference == 1:
                overcounts.append(video_id)
            else:
                unexplained.append(f"{video_id}: {METADATA_FILE}={declared[video_id]} "
                                   f"vs annotations={length}")
        if unexplained:
            raise ValueError(
                f"{len(unexplained)} video(s) where {METADATA_FILE} and {ANNOT_FILE} "
                f"disagree by something other than the known +1: {unexplained[:5]}. "
                "Establish which is right before converting -- do not assume."
            )
        return overcounts


    @property
    def annotations_dir(self) -> Path:
        """Where the annotation text files live.

        Accepts either the dataset root (``.../IPN_Hand``) or the ``annotations/``
        directory itself, because both are things a person plausibly types.
        """
        root = self.config.source_root
        if (root / ANNOTATIONS_DIR / ANNOT_FILE).is_file():
            return root / ANNOTATIONS_DIR
        return root

    def _read_source(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        directory = self.annotations_dir
        for filename in (ANNOT_FILE, METADATA_FILE, CLASS_INDEX_FILE):
            if not (directory / filename).is_file():
                raise FileNotFoundError(
                    f"{filename} not found under {directory}. Expected IPN Hand's "
                    "annotations; run scripts/prepare_ipn_hand.py to unpack them."
                )

        annot = pd.read_csv(directory / ANNOT_FILE)
        required = {"video", "label", "t_start", "t_end"}
        if not required.issubset(annot.columns):
            raise ValueError(
                f"{ANNOT_FILE} columns {list(annot.columns)} lack {sorted(required)}"
            )

        metadata = pd.read_csv(directory / METADATA_FILE)
        if not {"Video Name", "Frames"}.issubset(metadata.columns):
            raise ValueError(
                f"{METADATA_FILE} columns {list(metadata.columns)} lack "
                "'Video Name' / 'Frames'"
            )
        return annot, metadata

    def _code_map(self) -> dict[str, str]:
        """IPN code -> canonical class name, validated against classIdx.txt on disk."""
        index = pd.read_csv(self.annotations_dir / CLASS_INDEX_FILE)
        on_disk = [str(c) for c in index["label"]]
        expected = [NONE_CODE, *GESTURE_CODES]
        if on_disk != expected:
            raise ValueError(
                f"{CLASS_INDEX_FILE} lists {on_disk} but this adapter maps {expected}. "
                "The class codes or their order changed; fix the mapping rather than the "
                "check -- every label in the dataset hangs off it."
            )
        return {NONE_CODE: C.NONE_CLASS, **self._code_map_names()}

    def _code_map_names(self) -> dict[str, str]:
        return dict(zip(GESTURE_CODES, self.gesture_classes()))


def subject_of(video_id: str) -> str:
    """``1CM1_4_R_#229`` -> ``1CM1_4``.

    The subject token is unique only within a camera, so the camera is part of the
    identity. See the module docstring; this is the unit Stage 2 splits on.
    """
    match = VIDEO_NAME_RE.match(video_id)
    if match is None:
        raise ValueError(
            f"video id {video_id!r} does not match IPN Hand's "
            "<camera>_<subject>_<hand>_#<clip> naming, so its subject is undeterminable"
        )
    return f"{match['camera']}_{match['subject']}"


def verification_table(
    computed: dict[str, Any], expected: dict[str, Any]
) -> tuple[list[tuple[str, Any, Any, bool, str]], bool]:
    """Compare computed dataset statistics against the published figures.

    Returns ``(rows, all_ok)``, each row ``(key, expected, computed, ok, note)``.
    ``approx_num_frames`` is compared with the fractional tolerance from the config
    because the paper quotes "~800,000"; everything else must match exactly.

    The published non-gesture count is compared against the count in ``Annot_List.txt``
    rather than against the emitted rows, since the tail fills described in the module
    docstring add a handful of ``none`` instances the paper never counted.

    This returns the numbers rather than a bare pass/fail: CLAUDE.md is explicit that
    mismatches get investigated, which needs the values in front of you.
    """
    tolerance = float(expected.get("frames_tolerance") or 0.0)
    checks: list[tuple[str, Any, Any, float]] = [
        ("num_videos", expected.get("num_videos"), computed.get("num_videos"), 0.0),
        ("num_subjects", expected.get("num_subjects"), computed.get("num_subjects"), 0.0),
        (
            "num_gesture_classes",
            expected.get("num_gesture_classes"),
            computed.get("num_gesture_classes"),
            0.0,
        ),
        (
            "num_gesture_instances",
            expected.get("num_gesture_instances"),
            computed.get("num_gesture_instances"),
            0.0,
        ),
        (
            "num_none_instances",
            expected.get("num_none_instances"),
            computed.get("num_source_none_instances"),
            0.0,
        ),
        (
            "approx_num_frames",
            expected.get("approx_num_frames"),
            computed.get("num_frames"),
            tolerance,
        ),
    ]

    rows: list[tuple[str, Any, Any, bool, str]] = []
    all_ok = True
    for key, want, got, tol in checks:
        if want is None or got is None:
            ok, note = False, "missing"
        elif tol:
            ok, note = abs(got - want) <= tol * want, f"within +/-{tol:.0%}"
        else:
            ok, note = got == want, "exact"
        all_ok = all_ok and ok
        rows.append((key, want, got, ok, note))
    return rows, all_ok


def per_class_table(
    annotations: "pd.DataFrame",
    classes: Sequence[str],
    expected: Sequence[dict[str, Any]],
    *,
    duration_tolerance: float = 1.0,
) -> tuple[list[tuple[str, Any, ...]], bool]:
    """Compare per-class counts and span durations against Table II of the paper.

    This is the check that pins down the class *mapping*. ``classIdx.txt`` ships only
    codes -- ``B0A``, ``G01``, ... -- and nothing in the download says which code is
    which gesture, so the names in ``configs/dataset/ipn_hand.yaml`` are matched to the
    codes positionally from the paper's class table. If that pairing were wrong, every
    total in :func:`verification_table` would still reconcile exactly and the error would
    surface only as a nonsensical confusion matrix in Stage 8, or not at all.

    Table II gives instances, mean duration and standard deviation per class. Those
    numbers differ enough between classes that 90 of the 91 possible pairwise label swaps
    change at least one of them. The lone exception is ``throw_right`` vs ``zoom_out``,
    which the paper rounds to the same 64 (28); those two are separated only by the
    ``classIdx.txt`` ordering, which the other twelve rows corroborate.

    Returns ``(rows, all_ok)``, each row
    ``(class_name, index_ok, instances, want_instances, mean, want_mean, std, want_std,
    ok)``. Durations are in frames, and the paper prints means rounded to whole frames,
    hence ``duration_tolerance``.
    """
    durations = annotations["end_frame"].astype(int) - annotations["start_frame"].astype(int) + 1
    grouped = annotations.assign(_duration=durations).groupby("class")["_duration"]
    stats = {
        str(name): (int(group.size), float(group.mean()), float(group.std()))
        for name, group in grouped
    }
    index_of = {name: i for i, name in enumerate(classes)}

    rows: list[tuple[str, Any, ...]] = []
    all_ok = True
    for paper_index, entry in enumerate(expected):
        name = str(entry["name"])
        want = (
            int(entry["instances"]),
            float(entry["mean_duration"]),
            float(entry["std_duration"]),
        )
        got = stats.get(name)
        # classes.txt order is the label index order, so the paper's id must be our index.
        index_ok = index_of.get(name) == paper_index
        if got is None:
            rows.append((name, index_ok, None, want[0], None, want[1], None, want[2], False))
            all_ok = False
            continue
        ok = (
            index_ok
            and got[0] == want[0]
            and abs(got[1] - want[1]) <= duration_tolerance
            and abs(got[2] - want[2]) <= duration_tolerance
        )
        all_ok = all_ok and ok
        rows.append((name, index_ok, got[0], want[0], got[1], want[1], got[2], want[2], ok))
    return rows, all_ok


def render_per_class_table(rows: Sequence[tuple[str, Any, ...]]) -> str:
    """Format :func:`per_class_table` output for the Stage 1 gate."""
    header = f"{'id':>2}  {'class':<30} {'inst':>5}{'/pub':>5}  {'mean':>6}{'/pub':>5}  {'std':>6}{'/pub':>5}  verdict"
    lines = [header, "-" * len(header)]
    for index, (name, index_ok, inst, w_inst, mean, w_mean, std, w_std, ok) in enumerate(rows):
        shown = lambda v: "  --  " if v is None else f"{v:6.1f}"  # noqa: E731
        verdict = "OK" if ok else ("MISMATCH" + ("" if index_ok else " (index)"))
        lines.append(
            f"{index:>2}  {name:<30} {inst if inst is not None else '--':>5}"
            f"{w_inst:>5}  {shown(mean)}{w_mean:>5.0f}  {shown(std)}{w_std:>5.0f}  {verdict}"
        )
    return "\n".join(lines)


def render_verification_table(rows: Sequence[tuple[str, Any, Any, bool, str]]) -> str:
    """Format :func:`verification_table` output as the side-by-side Stage 1 gate table."""
    header = ("metric", "published", "computed", "verdict")
    body = [
        (key, str(want), str(got), f"{'OK' if ok else 'MISMATCH'} ({note})")
        for key, want, got, ok, note in rows
    ]
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(4)]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), line]
    out += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in body]
    return "\n".join(out)


__all__ = [
    "IPNHandAdapter",
    "GESTURE_CODES",
    "NONE_CODE",
    "subject_of",
    "verification_table",
    "render_verification_table",
    "per_class_table",
    "render_per_class_table",
]
