"""Subject-disjoint splits, generated once and frozen to disk.

Hard Rule 3: every reported number comes from a subject-disjoint split. Never split by
frame, by window, or by clip. Splits are frozen, and the test set is never tuned against.

This module is the only thing that assigns a split. Adapters write ``split=""``; this
writes the real value. It is also dataset-agnostic: it reads ``source_split`` and
``subject_attributes`` from ``meta.yaml`` under those generic names and never asks which
dataset it is looking at. See CLAUDE.md Rule 5.

Why frozen files rather than recomputing on the fly
---------------------------------------------------
A split recomputed at each run is a split that can drift -- a new seed, a reordered
subject list, a pandas version that groups differently -- and then two numbers in the
thesis are not comparable and nothing says so. The frozen JSON records the assignment, the
spec that produced it, the git commit, and a digest of the subject list it was built from.
:func:`load` refuses a file whose digest no longer matches the dataset, which is what
catches "the adapter was re-run and gained a subject" before it silently becomes a leak.

The relationship between the primary split and the LOSGO folds
--------------------------------------------------------------
The non-test subjects are partitioned into ``num_folds`` folds. Fold *k* serves as the
validation set for cross-validation fold *k*; the rest are train. The **primary** split is
simply one of those folds (``primary_val_fold``) promoted to be *the* validation set.

That is deliberate. If the primary validation subjects were drawn separately, they would
appear as training subjects in most CV folds, and the cross-validation would no longer be
a clean superset of the primary experiment -- the seed-variance estimate in Stage 8 would
be measuring something subtly different from the headline number it is meant to bound.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.data import canonical as C
from src.utils.run import git_provenance

#: Where frozen splits live inside a canonical dataset.
SPLITS_DIR: str = "splits"

#: ``test_source`` values.
PUBLISHER = "publisher"
GENERATED = "generated"


@dataclass(frozen=True)
class SplitSpec:
    """Everything that determines a split. Loaded from ``configs/split/<name>.yaml``."""

    #: File name the frozen split is written under.
    name: str
    #: Seed for every random choice here. Same seed + same subjects -> same split.
    seed: int = 1337
    #: ``publisher`` honours meta.yaml's ``source_split``; ``generated`` draws our own.
    test_source: str = GENERATED
    #: Which ``source_split`` value designates the test set (``publisher`` only).
    publisher_test_value: str = "test"
    #: Share of subjects held out as test (``generated`` only).
    test_fraction: float = 0.2
    #: Number of LOSGO folds over the non-test subjects.
    num_folds: int = 5
    #: Which fold is promoted to be the primary validation set.
    primary_val_fold: int = 0
    #: ``subject_attributes`` keys to balance across folds, e.g. ``["camera"]``.
    stratify_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.test_source not in (PUBLISHER, GENERATED):
            raise ValueError(
                f"test_source must be {PUBLISHER!r} or {GENERATED!r}, got {self.test_source!r}"
            )
        if self.num_folds < 2:
            raise ValueError(f"num_folds must be >= 2, got {self.num_folds}")
        if not 0 <= self.primary_val_fold < self.num_folds:
            raise ValueError(
                f"primary_val_fold {self.primary_val_fold} is outside "
                f"[0, {self.num_folds})"
            )
        if self.test_source == GENERATED and not 0 < self.test_fraction < 1:
            raise ValueError(f"test_fraction must be in (0, 1), got {self.test_fraction}")

    @classmethod
    def from_config(cls, cfg: Any) -> "SplitSpec":
        """Build from an OmegaConf node or a plain mapping."""
        data = dict(cfg) if not hasattr(cfg, "keys") or isinstance(cfg, dict) else dict(cfg)
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown split config keys {unknown}; expected a subset of {sorted(known)}"
            )
        if "stratify_by" in data and data["stratify_by"] is not None:
            data["stratify_by"] = tuple(str(k) for k in data["stratify_by"])
        return cls(**data)


@dataclass
class SplitAssignment:
    """A frozen split: which subject is in which split, plus the LOSGO folds."""

    spec: SplitSpec
    #: subject_id -> "train" | "val" | "test".
    subject_split: dict[str, str]
    #: LOSGO folds over the non-test subjects; fold k is that fold's validation set.
    folds: list[list[str]]
    #: SHA-256 of the sorted subject list this was built from. Guards against reuse
    #: against a dataset that has since changed.
    subjects_digest: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- queries ------------------------------------------------------------------

    def subjects(self, split: str) -> list[str]:
        return sorted(s for s, v in self.subject_split.items() if v == split)

    @property
    def test_subjects(self) -> list[str]:
        return self.subjects("test")

    def fold(self, index: int) -> tuple[list[str], list[str]]:
        """``(train_subjects, val_subjects)`` for LOSGO fold ``index``.

        The test subjects are excluded from both. Cross-validation never touches them --
        that is the whole point of freezing a test set.
        """
        if not 0 <= index < len(self.folds):
            raise IndexError(f"fold {index} out of range [0, {len(self.folds)})")
        val = sorted(self.folds[index])
        train = sorted(s for i, f in enumerate(self.folds) if i != index for s in f)
        return train, val

    # -- persistence --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "subject_split": dict(sorted(self.subject_split.items())),
            "folds": [sorted(f) for f in self.folds],
            "subjects_digest": self.subjects_digest,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitAssignment":
        spec_data = dict(data["spec"])
        if "stratify_by" in spec_data and spec_data["stratify_by"] is not None:
            spec_data["stratify_by"] = tuple(spec_data["stratify_by"])
        return cls(
            spec=SplitSpec(**spec_data),
            subject_split=dict(data["subject_split"]),
            folds=[list(f) for f in data["folds"]],
            subjects_digest=str(data.get("subjects_digest", "")),
            provenance=dict(data.get("provenance", {})),
        )


def split_path(root: Path | str, name: str) -> Path:
    return Path(root) / SPLITS_DIR / f"{name}.json"


def subjects_digest(subjects: Iterable[str]) -> str:
    """Stable digest of a subject list, order-independent."""
    joined = "\n".join(sorted(str(s) for s in subjects))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Stratified partitioning
# --------------------------------------------------------------------------------------


def _stratum_key(
    subject: str, attributes: dict[str, dict[str, Any]], keys: Sequence[str]
) -> str:
    """A hashable, printable label for the stratum a subject belongs to."""
    entry = attributes.get(subject, {})
    parts = []
    for key in keys:
        value = entry.get(key, "?")
        # An attribute that varies across a subject's videos arrives as a list; keep it
        # as its own stratum rather than picking one value arbitrarily.
        parts.append("+".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value))
    return "|".join(parts)


def partition_stratified(
    subjects: Sequence[str],
    num_parts: int,
    *,
    rng: np.random.Generator,
    attributes: dict[str, dict[str, Any]] | None = None,
    stratify_by: Sequence[str] = (),
) -> list[list[str]]:
    """Split ``subjects`` into ``num_parts`` near-equal groups, balancing strata.

    Each stratum is shuffled and dealt round-robin across the parts, with the starting
    part rotated per stratum. Dealing rather than slicing is what keeps a stratum smaller
    than ``num_parts`` from landing entirely in one part: three "Dark" subjects across
    five folds end up in three different folds rather than all in fold 0.

    Subjects are sorted before shuffling so the result depends on the seed and the *set*
    of subjects, never on the order they happened to arrive in.
    """
    ordered = sorted(str(s) for s in subjects)
    if num_parts > len(ordered):
        raise ValueError(
            f"cannot split {len(ordered)} subject(s) into {num_parts} parts; "
            "a fold with no subjects in it is not a fold"
        )

    attributes = attributes or {}
    keys = tuple(stratify_by or ())
    if keys:
        strata: dict[str, list[str]] = {}
        for subject in ordered:
            strata.setdefault(_stratum_key(subject, attributes, keys), []).append(subject)
    else:
        strata = {"": ordered}

    parts: list[list[str]] = [[] for _ in range(num_parts)]
    # Largest strata first: the big groups establish balanced part sizes and the small
    # ones then fill the emptiest parts, instead of the reverse.
    offset = 0
    for _, members in sorted(strata.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        shuffled = list(members)
        rng.shuffle(shuffled)
        for i, subject in enumerate(shuffled):
            # Rotate the starting part per stratum, then deal into whichever part is
            # currently smallest among the rotation candidates.
            order = sorted(
                range(num_parts), key=lambda p: (len(parts[(p + offset) % num_parts]), p)
            )
            parts[(order[0] + offset) % num_parts].append(subject)
        offset = (offset + len(shuffled)) % num_parts

    return [sorted(p) for p in parts]


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def generate(dataset: C.CanonicalDataset, spec: SplitSpec) -> SplitAssignment:
    """Build a subject-disjoint split and its LOSGO folds.

    Subject-disjointness is structural here, not checked after the fact: every subject is
    assigned exactly once, from a dict keyed by subject. :func:`verify` re-checks it
    anyway, because the property is load-bearing enough to be worth asserting twice.
    """
    all_subjects = sorted(dataset.subjects())
    if not all_subjects:
        raise ValueError(f"dataset {dataset.root} has no subjects to split")

    attributes = dict(dataset.meta.get("subject_attributes") or {})
    rng = np.random.default_rng(spec.seed)

    # -- test set ------------------------------------------------------------------
    if spec.test_source == PUBLISHER:
        source_split = dict(dataset.meta.get("source_split") or {})
        if not source_split:
            raise ValueError(
                f"split spec {spec.name!r} asks for the publisher's partition, but "
                f"{dataset.root}/meta.yaml has no 'source_split'. Either the dataset has "
                "no recommended partition, or the adapter does not record it."
            )
        missing = sorted(set(all_subjects) - set(source_split))
        if missing:
            raise ValueError(
                f"{len(missing)} subject(s) absent from meta.yaml's source_split: "
                f"{missing[:5]}. A partial publisher partition cannot be honoured."
            )
        test = sorted(
            s for s in all_subjects if source_split[s] == spec.publisher_test_value
        )
        if not test:
            raise ValueError(
                f"no subject has source_split == {spec.publisher_test_value!r}; "
                f"values present: {sorted(set(source_split.values()))}"
            )
    else:
        # Partition into 1/test_fraction parts and take one whole part, rather than
        # slicing a fixed count off a part -- slicing would discard the stratification
        # that partition_stratified just established. Clamped to the number of subjects,
        # because Phase 2's first SOS recordings will be a small dataset and a request for
        # more parts than subjects should degrade, not crash.
        num_parts = max(2, min(round(1 / spec.test_fraction), len(all_subjects)))
        parts = partition_stratified(
            all_subjects,
            num_parts,
            rng=rng,
            attributes=attributes,
            stratify_by=spec.stratify_by,
        )
        test = sorted(parts[0])

    remaining = [s for s in all_subjects if s not in set(test)]
    if len(remaining) < spec.num_folds:
        raise ValueError(
            f"{len(remaining)} non-test subject(s) cannot form {spec.num_folds} folds "
            f"({len(all_subjects)} subjects, {len(test)} of them test). Either the dataset "
            "is too small for this protocol or num_folds is too high -- do not silently "
            "reduce the fold count, because the number of folds is part of what a reported "
            "variance estimate means."
        )

    # -- folds over the non-test subjects -------------------------------------------
    folds = partition_stratified(
        remaining,
        spec.num_folds,
        rng=rng,
        attributes=attributes,
        stratify_by=spec.stratify_by,
    )

    # The primary split is fold `primary_val_fold` promoted to validation; see module
    # docstring for why it is one of the folds rather than a separate draw.
    val = set(folds[spec.primary_val_fold])
    subject_split = {
        s: "test" if s in set(test) else ("val" if s in val else "train")
        for s in all_subjects
    }

    return SplitAssignment(
        spec=spec,
        subject_split=subject_split,
        folds=folds,
        subjects_digest=subjects_digest(all_subjects),
        provenance={
            "dataset": dataset.name,
            "dataset_root": str(dataset.root),
            "num_subjects": len(all_subjects),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "git": git_provenance(),
        },
    )


# --------------------------------------------------------------------------------------
# Verification -- Hard Rule 3, asserted rather than assumed
# --------------------------------------------------------------------------------------


def verify(assignment: SplitAssignment, dataset: C.CanonicalDataset) -> list[str]:
    """Return a list of problems; empty means the split is admissible.

    Checks the properties a thesis number depends on: every subject assigned exactly once,
    no video straddling splits, folds partitioning the non-test subjects exactly, and the
    primary validation set being exactly the promoted fold.
    """
    problems: list[str] = []
    expected = set(dataset.subjects())
    assigned = set(assignment.subject_split)

    for subject in sorted(expected - assigned):
        problems.append(f"subject {subject} has no split assigned")
    for subject in sorted(assigned - expected):
        problems.append(f"split assigns unknown subject {subject}")

    invalid = sorted({v for v in assignment.subject_split.values()} - set(C.VALID_SPLITS))
    if invalid:
        problems.append(f"invalid split names: {invalid}")

    for split in C.VALID_SPLITS:
        if not assignment.subjects(split):
            problems.append(f"split {split!r} is empty")

    # A video maps to one subject (the validator enforces that), so subject-disjointness
    # implies video-disjointness -- but a video appearing under two splits is the exact
    # failure Rule 3 exists to prevent, so it is checked directly rather than inferred.
    annotations = dataset.annotations.copy()
    annotations["_split"] = annotations["subject_id"].map(assignment.subject_split)
    per_video = annotations.groupby("video_id")["_split"].nunique(dropna=False)
    straddling = sorted(per_video[per_video > 1].index.tolist())
    if straddling:
        problems.append(f"video(s) straddling splits: {straddling[:5]}")

    # Folds must partition the non-test subjects: no overlap, no omission.
    non_test = set(assignment.subjects("train")) | set(assignment.subjects("val"))
    flat = [s for fold in assignment.folds for s in fold]
    if len(flat) != len(set(flat)):
        duplicates = sorted({s for s in flat if flat.count(s) > 1})
        problems.append(f"subject(s) in more than one fold: {duplicates[:5]}")
    if set(flat) != non_test:
        if missing := sorted(non_test - set(flat)):
            problems.append(f"non-test subject(s) in no fold: {missing[:5]}")
        if extra := sorted(set(flat) - non_test):
            problems.append(f"fold(s) contain test subjects: {extra[:5]}")

    promoted = set(assignment.folds[assignment.spec.primary_val_fold])
    if promoted != set(assignment.subjects("val")):
        problems.append(
            f"primary val set is not fold {assignment.spec.primary_val_fold}; "
            "the cross-validation would not be a superset of the primary experiment"
        )

    for index in range(len(assignment.folds)):
        train, val = assignment.fold(index)
        if overlap := sorted(set(train) & set(val)):
            problems.append(f"fold {index}: subject(s) in both train and val: {overlap[:5]}")
        if leaked := sorted(set(train + val) & set(assignment.test_subjects)):
            problems.append(f"fold {index}: test subject(s) used in CV: {leaked[:5]}")

    return problems


# --------------------------------------------------------------------------------------
# Persistence and application
# --------------------------------------------------------------------------------------


def freeze(assignment: SplitAssignment, root: Path | str) -> Path:
    """Write the split to ``<root>/splits/<name>.json``."""
    path = split_path(root, assignment.spec.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(assignment.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return path


def load(
    root: Path | str, name: str, *, dataset: C.CanonicalDataset | None = None
) -> SplitAssignment:
    """Read a frozen split, refusing one that no longer matches the dataset.

    The digest check is the point. If the adapter is re-run and the subject set changes,
    a stale split file would still load and still look subject-disjoint while silently
    describing a different dataset.
    """
    path = split_path(root, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"no frozen split at {path}. Generate it with scripts/make_splits.py"
        )
    assignment = SplitAssignment.from_dict(json.loads(path.read_text(encoding="utf-8")))

    if dataset is not None:
        current = subjects_digest(dataset.subjects())
        if assignment.subjects_digest and assignment.subjects_digest != current:
            raise ValueError(
                f"frozen split {path} was built from a different subject set "
                f"(digest {assignment.subjects_digest[:12]}, dataset now {current[:12]}). "
                "Re-freezing changes what every past number means -- work out what changed "
                "in the dataset first."
            )
    return assignment


def apply(root: Path | str, assignment: SplitAssignment) -> int:
    """Write the split into ``annotations.csv``'s ``split`` column. Returns rows written."""
    root = Path(root)
    annotations = C.read_annotations(root)
    mapped = annotations["subject_id"].map(assignment.subject_split)
    if mapped.isna().any():
        unknown = sorted(annotations.loc[mapped.isna(), "subject_id"].unique())
        raise ValueError(f"no split assigned for subject(s): {unknown[:5]}")
    annotations["split"] = mapped
    C.write_annotations(root, annotations)
    return int(len(annotations))


def summarise(assignment: SplitAssignment, dataset: C.CanonicalDataset) -> str:
    """Human-readable summary, printed at the gate and written into the run directory."""
    annotations = dataset.annotations
    videos = annotations.groupby("subject_id")["video_id"].nunique().to_dict()
    lines = [f"Split {assignment.spec.name!r}  (seed {assignment.spec.seed})", "=" * 78]

    for split in C.VALID_SPLITS:
        subjects = assignment.subjects(split)
        lines.append(
            f"{split:<6} {len(subjects):>3} subjects  "
            f"{sum(videos.get(s, 0) for s in subjects):>4} videos"
        )
    lines.append("")

    keys = assignment.spec.stratify_by or ("camera",)
    attributes = dict(dataset.meta.get("subject_attributes") or {})
    if attributes:
        for key in keys:
            values = sorted(
                {_stratum_key(s, attributes, (key,)) for s in assignment.subject_split}
            )
            lines.append(f"subjects by {key}:")
            header = "  " + " " * 10 + "".join(f"{v:>14}" for v in values)
            lines.append(header)
            for split in C.VALID_SPLITS:
                counts = [
                    sum(
                        1
                        for s in assignment.subjects(split)
                        if _stratum_key(s, attributes, (key,)) == v
                    )
                    for v in values
                ]
                lines.append(f"  {split:<10}" + "".join(f"{c:>14}" for c in counts))
            lines.append("")

    lines.append(f"LOSGO folds ({len(assignment.folds)}), over the non-test subjects:")
    for index, fold in enumerate(assignment.folds):
        marker = "  <- primary val" if index == assignment.spec.primary_val_fold else ""
        lines.append(f"  fold {index}: {len(fold):>2} subjects{marker}")
    return "\n".join(lines)


__all__ = [
    "SPLITS_DIR",
    "PUBLISHER",
    "GENERATED",
    "SplitSpec",
    "SplitAssignment",
    "generate",
    "verify",
    "freeze",
    "load",
    "apply",
    "summarise",
    "partition_stratified",
    "split_path",
    "subjects_digest",
]
