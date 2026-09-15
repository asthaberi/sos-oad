"""Shared test fixtures.

The strategy throughout: build one known-good canonical dataset, then corrupt a *copy*
of it one property at a time and assert the validator catches exactly that corruption.
A validator that has only ever been run on valid data is not a tested validator, and the
whole point of it is to catch the leakage that would otherwise surface as a meaningless
98% in Stage 7.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.data import canonical as C
from src.data.adapters.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def good_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A valid canonical dataset, generated once per session."""
    root = tmp_path_factory.mktemp("canonical") / "synthetic"
    report = make_synthetic_dataset(root, num_videos=6, num_subjects=3, seed=7)
    assert report.ok, "the synthetic generator itself produced an invalid dataset:\n" + report.render()
    return root


@pytest.fixture
def dataset(good_dataset: Path, tmp_path: Path) -> Path:
    """A fresh, writable copy of the good dataset for a single test to corrupt."""
    target = tmp_path / "synthetic"
    shutil.copytree(good_dataset, target)
    return target


@pytest.fixture
def annotations_of():
    """Read/modify/write helper for annotations.csv."""

    def _round_trip(root: Path, mutate) -> None:
        df = C.read_annotations(root)
        mutated = mutate(df)
        if mutated is None:
            mutated = df
        # Written directly rather than via write_annotations, which would reject some of
        # the corruptions we are deliberately introducing.
        mutated.to_csv(root / C.ANNOTATIONS_FILE, index=False)

    return _round_trip


@pytest.fixture
def assign_splits():
    """Assign splits by subject, so split-related tests start from a valid state."""

    def _assign(root: Path, mapping: dict[str, str]) -> None:
        df = C.read_annotations(root)
        df["split"] = df["subject_id"].map(mapping).fillna(C.UNASSIGNED_SPLIT)
        df.to_csv(root / C.ANNOTATIONS_FILE, index=False)

    return _assign


def codes_of(report: C.ValidationReport) -> set[str]:
    return report.codes()


def first_video(root: Path) -> str:
    df: pd.DataFrame = C.read_annotations(root)
    return str(df["video_id"].iloc[0])
