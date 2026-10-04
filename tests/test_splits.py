"""Stage 2: subject-disjoint splits.

Hard Rule 3 is the reason this file exists, and the reason CLAUDE.md calls it
load-bearing. The failure it guards against does not look like a failure: a split with a
subject on both sides still trains, still validates, still produces a confusion matrix,
and produces a number that is twenty points too high. Nothing downstream can detect it.
So the properties are asserted directly, and several of them are asserted twice -- once
structurally on a generated split, once by corrupting a good split and checking the
validator refuses it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data import canonical as C
from src.data import splits as S
from src.data.adapters.synthetic import make_synthetic_dataset

pytestmark = pytest.mark.splits


# ----------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def many_subjects(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic dataset with enough subjects to form five folds."""
    root = tmp_path_factory.mktemp("splits") / "synthetic"
    report = make_synthetic_dataset(root, num_videos=40, num_subjects=20, seed=11)
    assert report.ok, report.render()
    return root


@pytest.fixture
def dataset(many_subjects: Path) -> C.CanonicalDataset:
    return C.CanonicalDataset(many_subjects)


@pytest.fixture
def spec() -> S.SplitSpec:
    return S.SplitSpec(name="test", seed=1337, test_source=S.GENERATED, test_fraction=0.2)


@pytest.fixture
def assignment(dataset: C.CanonicalDataset, spec: S.SplitSpec) -> S.SplitAssignment:
    return S.generate(dataset, spec)


# ----------------------------------------------------------------------------------
# THE GATE -- no subject_id in more than one split
# ----------------------------------------------------------------------------------


def test_no_subject_appears_in_more_than_one_split(assignment, dataset):
    """The Stage 2 gate."""
    seen: dict[str, str] = {}
    for split in C.VALID_SPLITS:
        for subject in assignment.subjects(split):
            assert subject not in seen, (
                f"subject {subject} is in both {seen[subject]!r} and {split!r}"
            )
            seen[subject] = split
    assert set(seen) == set(dataset.subjects())


def test_no_video_straddles_a_split(assignment, dataset):
    """Never split by frame, by window, or by clip.

    Subject-disjointness implies this given one subject per video, but it is checked
    directly because it is the property that actually matters.
    """
    annotations = dataset.annotations.copy()
    annotations["_split"] = annotations["subject_id"].map(assignment.subject_split)
    per_video = annotations.groupby("video_id")["_split"].nunique()
    assert (per_video == 1).all(), sorted(per_video[per_video > 1].index)


def test_generated_split_verifies_clean(assignment, dataset):
    assert S.verify(assignment, dataset) == []


def test_every_split_is_non_empty(assignment):
    for split in C.VALID_SPLITS:
        assert assignment.subjects(split), f"{split} is empty"


# ----------------------------------------------------------------------------------
# LOSGO folds
# ----------------------------------------------------------------------------------


def test_folds_partition_the_non_test_subjects_exactly(assignment):
    flat = [s for fold in assignment.folds for s in fold]
    assert len(flat) == len(set(flat)), "a subject is in two folds"
    non_test = set(assignment.subjects("train")) | set(assignment.subjects("val"))
    assert set(flat) == non_test


def test_test_subjects_never_enter_cross_validation(assignment):
    """The frozen test set is not a validation set of convenience."""
    held_out = set(assignment.test_subjects)
    for index in range(len(assignment.folds)):
        train, val = assignment.fold(index)
        assert not (set(train) & held_out)
        assert not (set(val) & held_out)


def test_each_fold_has_disjoint_train_and_val(assignment):
    for index in range(len(assignment.folds)):
        train, val = assignment.fold(index)
        assert not (set(train) & set(val))


def test_primary_val_is_the_promoted_fold(assignment):
    """Cross-validation must be a strict superset of the primary experiment.

    If the primary validation subjects were drawn separately they would appear as
    *training* subjects in most folds, and Stage 8's variance estimate would be bounding
    a different experiment from the headline number.
    """
    promoted = set(assignment.folds[assignment.spec.primary_val_fold])
    assert promoted == set(assignment.subjects("val"))


def test_folds_are_near_equal_in_size(assignment):
    sizes = [len(f) for f in assignment.folds]
    assert max(sizes) - min(sizes) <= 1, sizes


# ----------------------------------------------------------------------------------
# Determinism -- a split that drifts is not frozen
# ----------------------------------------------------------------------------------


def test_same_seed_gives_an_identical_split(dataset, spec):
    first, second = S.generate(dataset, spec), S.generate(dataset, spec)
    assert first.subject_split == second.subject_split
    assert first.folds == second.folds


def test_a_different_seed_gives_a_different_split(dataset, spec):
    import dataclasses

    other = S.generate(dataset, dataclasses.replace(spec, seed=spec.seed + 1))
    assert S.generate(dataset, spec).folds != other.folds


def test_split_does_not_depend_on_subject_ordering(dataset, spec):
    """Sorted before shuffling, so a reordered annotations.csv cannot move a subject."""
    shuffled = dataset.annotations.sample(frac=1.0, random_state=0).reset_index(drop=True)
    reordered = C.CanonicalDataset(dataset.root)
    reordered.annotations = shuffled
    assert S.generate(reordered, spec).subject_split == S.generate(dataset, spec).subject_split


# ----------------------------------------------------------------------------------
# Stratification
# ----------------------------------------------------------------------------------


def test_a_rare_stratum_is_spread_across_folds_not_dumped_in_one():
    """Three subjects in a stratum, five folds: they must land in three different folds.

    Slicing a shuffled list would put them all in whichever fold the slice covers, and the
    Stage 8 per-attribute breakdown would then have empty cells that look like a finding.
    """
    subjects = [f"s{i:02d}" for i in range(25)]
    attributes = {s: {"lighting": "bright"} for s in subjects}
    for s in subjects[:3]:
        attributes[s]["lighting"] = "dark"

    folds = S.partition_stratified(
        subjects,
        5,
        rng=np.random.default_rng(0),
        attributes=attributes,
        stratify_by=["lighting"],
    )
    holding_dark = [i for i, f in enumerate(folds) if any(s in subjects[:3] for s in f)]
    assert len(holding_dark) == 3


def test_stratified_parts_stay_near_equal():
    subjects = [f"s{i:02d}" for i in range(23)]
    attributes = {s: {"g": "a" if i % 3 else "b"} for i, s in enumerate(subjects)}
    parts = S.partition_stratified(
        subjects, 5, rng=np.random.default_rng(3), attributes=attributes, stratify_by=["g"]
    )
    sizes = [len(p) for p in parts]
    assert sum(sizes) == len(subjects)
    assert max(sizes) - min(sizes) <= 1, sizes
    assert sorted(s for p in parts for s in p) == subjects


def test_partition_refuses_more_parts_than_subjects():
    with pytest.raises(ValueError, match="fold with no subjects"):
        S.partition_stratified(["a", "b"], 5, rng=np.random.default_rng(0))


def test_a_multi_valued_attribute_becomes_its_own_stratum():
    """A subject recorded under both plain and cluttered backgrounds genuinely has both.

    Collapsing that to one value would quietly mislead the stratification.
    """
    attributes = {"a": {"bg": ["Clutter", "Plain"]}, "b": {"bg": "Plain"}}
    assert S._stratum_key("a", attributes, ("bg",)) != S._stratum_key("b", attributes, ("bg",))


def test_a_small_dataset_degrades_instead_of_crashing(tmp_path):
    """Phase 2 starts with few SOS subjects, and 1/test_fraction can exceed them.

    Asking for five parts from four subjects used to raise. It must clamp instead, so the
    first small recording session is still splittable.
    """
    root = tmp_path / "tiny"
    report = make_synthetic_dataset(root, num_videos=8, num_subjects=4, seed=5)
    assert report.ok
    assignment = S.generate(
        C.CanonicalDataset(root),
        S.SplitSpec(name="tiny", test_fraction=0.2, num_folds=2),
    )
    assert S.verify(assignment, C.CanonicalDataset(root)) == []
    assert len(assignment.test_subjects) >= 1


def test_too_many_folds_for_the_dataset_is_refused_clearly(tmp_path):
    """Silently reducing the fold count would change what a variance estimate means."""
    root = tmp_path / "small"
    assert make_synthetic_dataset(root, num_videos=8, num_subjects=4, seed=5).ok
    with pytest.raises(ValueError, match="cannot form 5 folds"):
        S.generate(C.CanonicalDataset(root), S.SplitSpec(name="x", num_folds=5))


# ----------------------------------------------------------------------------------
# Honouring a publisher's partition
# ----------------------------------------------------------------------------------


def test_publisher_mode_reproduces_the_source_split_exactly(many_subjects, tmp_path):
    import shutil

    root = tmp_path / "with_source"
    shutil.copytree(many_subjects, root)
    dataset = C.CanonicalDataset(root)
    designated = sorted(dataset.subjects())[:5]

    meta = dict(dataset.meta)
    meta["source_split"] = {
        s: ("test" if s in designated else "train") for s in dataset.subjects()
    }
    C.write_meta(root, meta)

    assignment = S.generate(
        C.CanonicalDataset(root),
        S.SplitSpec(name="pub", test_source=S.PUBLISHER, num_folds=3),
    )
    assert assignment.test_subjects == designated


def test_publisher_mode_refuses_when_the_dataset_has_no_partition(dataset):
    with pytest.raises(ValueError, match="no 'source_split'"):
        S.generate(dataset, S.SplitSpec(name="pub", test_source=S.PUBLISHER))


def test_publisher_mode_refuses_a_partial_partition(many_subjects, tmp_path):
    import shutil

    root = tmp_path / "partial"
    shutil.copytree(many_subjects, root)
    dataset = C.CanonicalDataset(root)
    meta = dict(dataset.meta)
    # One subject left out: honouring this would silently drop them.
    meta["source_split"] = {s: "train" for s in sorted(dataset.subjects())[:-1]}
    meta["source_split"][sorted(dataset.subjects())[0]] = "test"
    C.write_meta(root, meta)

    with pytest.raises(ValueError, match="absent from"):
        S.generate(
            C.CanonicalDataset(root), S.SplitSpec(name="pub", test_source=S.PUBLISHER)
        )


# ----------------------------------------------------------------------------------
# Freezing, reloading, and the staleness guard
# ----------------------------------------------------------------------------------


def test_freeze_then_load_round_trips(assignment, dataset, tmp_path):
    root = tmp_path / "frozen"
    root.mkdir()
    S.freeze(assignment, root)
    loaded = S.load(root, assignment.spec.name)
    assert loaded.subject_split == assignment.subject_split
    assert loaded.folds == assignment.folds
    assert loaded.spec == assignment.spec


def test_load_refuses_a_split_built_from_a_different_subject_set(
    assignment, dataset, tmp_path
):
    """The check that catches a re-run adapter before it becomes a silent leak.

    A stale split file still loads, still looks subject-disjoint, and still describes a
    dataset that no longer exists.
    """
    root = tmp_path / "stale"
    root.mkdir()
    assignment.subjects_digest = S.subjects_digest(["someone", "else"])
    S.freeze(assignment, root)
    with pytest.raises(ValueError, match="different subject set"):
        S.load(root, assignment.spec.name, dataset=dataset)


def test_frozen_file_records_what_is_needed_to_defend_it(assignment, tmp_path):
    root = tmp_path / "prov"
    root.mkdir()
    payload = json.loads(S.freeze(assignment, root).read_text(encoding="utf-8"))
    assert payload["spec"]["seed"] == assignment.spec.seed
    assert payload["subjects_digest"]
    assert "git" in payload["provenance"]
    assert payload["provenance"]["num_subjects"] == len(assignment.subject_split)


# ----------------------------------------------------------------------------------
# Application, and the validator's own leak check
# ----------------------------------------------------------------------------------


def test_apply_writes_the_split_column_and_passes_strict_validation(
    many_subjects, tmp_path, assignment
):
    import shutil

    root = tmp_path / "applied"
    shutil.copytree(many_subjects, root)
    S.apply(root, assignment)

    annotations = C.read_annotations(root)
    assert set(annotations["split"]) <= set(C.VALID_SPLITS)
    assert C.UNASSIGNED_SPLIT not in set(annotations["split"])

    report = C.validate_dataset(root, require_splits=True)
    assert report.ok, report.render()


def test_validator_catches_a_leaked_subject(many_subjects, tmp_path, assignment):
    """Corrupt a good split one property at a time and check the validator refuses it.

    Belt and braces for the gate above: even if something bypassed this module and wrote
    the split column by hand, the validator still has to reject a leak.
    """
    import shutil

    root = tmp_path / "leaked"
    shutil.copytree(many_subjects, root)
    S.apply(root, assignment)

    annotations = C.read_annotations(root)
    victim = assignment.subjects("train")[0]
    rows = annotations.index[annotations["subject_id"] == victim]
    annotations.loc[rows[: max(1, len(rows) // 2)], "split"] = "test"
    annotations.to_csv(root / C.ANNOTATIONS_FILE, index=False)

    report = C.validate_dataset(root, require_splits=True)
    assert not report.ok
    assert "splits.subject_leak" in report.codes()


def test_apply_refuses_a_subject_it_has_no_split_for(many_subjects, tmp_path, assignment):
    import shutil

    root = tmp_path / "unknown"
    shutil.copytree(many_subjects, root)
    assignment.subject_split.pop(assignment.subjects("train")[0])
    with pytest.raises(ValueError, match="no split assigned"):
        S.apply(root, assignment)


# ----------------------------------------------------------------------------------
# Spec validation
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"test_source": "whatever"}, "test_source"),
        ({"num_folds": 1}, "num_folds"),
        ({"num_folds": 5, "primary_val_fold": 5}, "primary_val_fold"),
        ({"test_fraction": 0.0}, "test_fraction"),
        ({"test_fraction": 1.5}, "test_fraction"),
    ],
)
def test_bad_spec_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        S.SplitSpec(name="bad", **kwargs)


def test_unknown_config_key_is_rejected():
    """A typo in configs/split/*.yaml must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="unknown split config keys"):
        S.SplitSpec.from_config({"name": "x", "stratify_bY": ["camera"]})


# ----------------------------------------------------------------------------------
# The real dataset -- the Stage 2 gate on real data
# ----------------------------------------------------------------------------------

REAL_ROOT = Path("data/ipn_hand")

#: Guard on the *dataset*, not on the split file.
#:
#: The frozen split is the one artefact under data/ that is committed to git, which makes
#: it the worst possible sentinel for "has the dataset been built here?" -- on a fresh
#: clone it is always present while annotations.csv, classes.txt and meta.yaml are not, so
#: a guard on it lets these tests run against a dataset that does not exist. Found by
#: cloning onto a GPU server: these failed where the Stage 1 tests correctly skipped.
#:
#: Both conditions are required: the dataset has to be converted *and* the split frozen.
needs_real = pytest.mark.skipif(
    not (REAL_ROOT / C.ANNOTATIONS_FILE).is_file()
    or not S.split_path(REAL_ROOT, "ipn_official").is_file(),
    reason="run scripts/prepare_ipn_hand.py then scripts/make_splits.py first",
)


@pytest.mark.slow
@needs_real
def test_real_split_has_no_subject_in_two_splits():
    dataset = C.CanonicalDataset(REAL_ROOT)
    assignment = S.load(REAL_ROOT, "ipn_official", dataset=dataset)
    assert S.verify(assignment, dataset) == []
    assert len(assignment.subject_split) == 50
    assert len(assignment.test_subjects) == 13


@pytest.mark.slow
@needs_real
def test_real_test_set_is_exactly_the_publishers():
    """We did not choose the test set. That is the strongest form of 'never tuned against'."""
    dataset = C.CanonicalDataset(REAL_ROOT)
    assignment = S.load(REAL_ROOT, "ipn_official", dataset=dataset)
    source_split = dataset.meta["source_split"]
    assert assignment.test_subjects == sorted(
        s for s, v in source_split.items() if v == "test"
    )


@pytest.mark.slow
@needs_real
def test_real_dataset_passes_strict_split_validation():
    report = C.validate_dataset(REAL_ROOT, require_splits=True, require_features=False)
    assert report.ok, report.render()
