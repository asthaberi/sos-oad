"""Stage 1: the IPN Hand adapter.

Two layers of test here.

The first builds a miniature IPN-shaped annotation tree in a tmp directory and asserts
the adapter's conversion rules on data whose ground truth is known by construction: the
1-indexed-to-0-indexed shift, the tiling invariant, the video-length rule, and the subject
derivation. These run on a clean checkout with nothing downloaded.

The second is marked ``slow`` and runs only when ``data/ipn_hand`` exists. It asserts the
published figures, which is the Stage 1 gate itself. Keeping the gate in the test suite
rather than only in a script means a later change that quietly breaks the conversion --
an off-by-one, a dropped class, a subject regex that stops matching -- fails CI rather
than surfacing as a suspiciously good number in Stage 7.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import canonical as C
from src.data.adapters import AdapterConfig, get_adapter
from src.data.adapters.ipn_hand import (
    GESTURE_CODES,
    NONE_CODE,
    IPNHandAdapter,
    per_class_table,
    subject_of,
    verification_table,
)

#: Readable names for the 13 codes. Only the pairing order matters to the adapter, so the
#: test uses obvious stand-ins rather than duplicating the real config.
GESTURE_NAMES = [f"gesture_{code.lower()}" for code in GESTURE_CODES]

OPTIONS = {
    "name": "ipn_hand",
    "fps": 30.0,
    "frame_size": [640, 480],
    "gesture_classes": GESTURE_NAMES,
    "source": {"url": "https://example.invalid", "license": "CC BY 4.0"},
}


def write_source(
    root: Path,
    spans: dict[str, list[tuple[str, int, int]]],
    lengths: dict[str, int],
    *,
    codes: list[str] | None = None,
) -> Path:
    """Write a miniature ``annotations/`` tree in IPN Hand's own file formats.

    ``spans`` maps video name to ``(code, t_start, t_end)`` triples, 1-indexed and
    inclusive exactly as IPN stores them.
    """
    directory = root / "annotations"
    directory.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "video": video,
            "label": code,
            "id": 1,
            "t_start": start,
            "t_end": end,
            "frames": end - start + 1,
        }
        for video, triples in spans.items()
        for code, start, end in triples
    ]
    pd.DataFrame(rows).to_csv(directory / "Annot_List.txt", index=False)

    pd.DataFrame(
        [
            {"Video Name": video, "Frames": length, "Set": "train"}
            for video, length in lengths.items()
        ]
    ).to_csv(directory / "metadata.csv", index=False)

    all_codes = codes if codes is not None else [NONE_CODE, *GESTURE_CODES]
    pd.DataFrame(
        [{"id": i + 1, "label": code} for i, code in enumerate(all_codes)]
    ).to_csv(directory / "classIdx.txt", index=False)
    return root


def convert(source: Path, out: Path, **overrides) -> C.ValidationReport:
    adapter = IPNHandAdapter(
        AdapterConfig(
            source_root=source, output_root=out, options={**OPTIONS, **overrides}
        )
    )
    # Features are Stage 3's output; Stage 1 produces the annotation layer only.
    return adapter.convert(require_features=False)


@pytest.fixture
def converted(tmp_path: Path) -> tuple[Path, C.ValidationReport]:
    """Three videos, one of them reproducing the inflated-metadata case."""
    source = write_source(
        tmp_path / "raw",
        spans={
            # Annotations and metadata agree: 100 frames.
            "CAM1_1_R_#1": [(NONE_CODE, 1, 10), ("B0A", 11, 40), (NONE_CODE, 41, 100)],
            # Annotations cover 79; metadata claims 80. One of the real dataset's 14
            # videos whose directory holds a stray desktop.ini.
            "CAM1_2_R_#2": [("G01", 1, 20), (NONE_CODE, 21, 79)],
            # Same subject *token* as CAM1_1, different camera: a different person.
            "CAM2_1_R_#3": [(NONE_CODE, 1, 30), ("G11", 31, 50)],
        },
        lengths={"CAM1_1_R_#1": 100, "CAM1_2_R_#2": 80, "CAM2_1_R_#3": 50},
    )
    out = tmp_path / "canonical"
    return out, convert(source, out)


# ----------------------------------------------------------------------------------
# Subject identity -- Hard Rule 3 depends on this being right
# ----------------------------------------------------------------------------------


def test_subject_is_camera_and_subject_token():
    assert subject_of("1CM1_4_R_#229") == "1CM1_4"
    assert subject_of("4CM11_7_R_#34") == "4CM11_7"


def test_same_subject_token_under_different_cameras_is_a_different_subject():
    """IPN's subject tokens restart per camera, so the token alone is not an identity.

    If these collapsed to one subject, Stage 2 would put four videos of one person and
    four of another in the same group -- and, worse, a real person could end up on both
    sides of the split via the other camera.
    """
    assert subject_of("1CM1_1_R_#217") != subject_of("4CM11_1_R_#1")


def test_unparseable_video_name_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="subject"):
        subject_of("not-an-ipn-name")


def test_every_video_maps_to_exactly_one_subject(converted):
    out, _ = converted
    annotations = C.read_annotations(out)
    per_video = annotations.groupby("video_id")["subject_id"].nunique()
    assert (per_video == 1).all()
    assert set(annotations["subject_id"]) == {"CAM1_1", "CAM1_2", "CAM2_1"}


# ----------------------------------------------------------------------------------
# Conversion rules
# ----------------------------------------------------------------------------------


def test_conversion_validates(converted):
    out, report = converted
    assert report.ok, report.render()


def test_frame_indices_shift_from_one_based_to_zero_based(converted):
    """An off-by-one here moves every onset by one frame (33 ms at 30 fps).

    It would survive validation untouched and show up only as a biased latency number in
    Stage 7, so it is asserted directly against a hand-written span.
    """
    out, _ = converted
    annotations = C.read_annotations(out)
    rows = annotations[annotations["video_id"] == "CAM1_1_R_#1"].sort_values("start_frame")
    assert list(zip(rows["start_frame"], rows["end_frame"])) == [(0, 9), (10, 39), (40, 99)]


def test_none_is_class_zero_and_gestures_follow_classidx_order(converted):
    out, _ = converted
    classes = C.read_classes(out)
    assert classes[C.NONE_INDEX] == C.NONE_CLASS
    assert classes == [C.NONE_CLASS, *GESTURE_NAMES]


def test_codes_map_to_the_positionally_paired_names(converted):
    out, _ = converted
    annotations = C.read_annotations(out)
    labelled = dict(
        zip(annotations["video_id"] + "@" + annotations["start_frame"].astype(str),
            annotations["class"])
    )
    # B0A is the first gesture code, G01 the third, G11 the last.
    assert labelled["CAM1_1_R_#1@10"] == "gesture_b0a"
    assert labelled["CAM1_2_R_#2@0"] == "gesture_g01"
    assert labelled["CAM2_1_R_#3@30"] == "gesture_g11"


def test_video_length_comes_from_annotations_not_metadata(converted):
    """``metadata.csv`` overcounts 14 videos by one; the annotations are the authority.

    Believing ``Frames`` would append a frame that has no JPEG behind it. It would tile
    cleanly and validate cleanly, and Stage 3 would then be asked to extract a feature
    vector from an image that does not exist.
    """
    out, _ = converted
    dataset = C.CanonicalDataset(out)
    assert dataset.num_frames("CAM1_2_R_#2") == 79
    assert dataset.num_frames("CAM1_1_R_#1") == 100
    meta = C.read_meta(out)
    assert meta["computed"]["videos_with_inflated_frame_count"] == ["CAM1_2_R_#2"]
    assert meta["computed"]["num_frames"] == 79 + 100 + 50
    assert meta["computed"]["num_declared_frames"] == 80 + 100 + 50


def test_no_none_rows_are_invented(converted):
    """IPN's spans already tile every video, so fill_none_spans must add nothing.

    If this starts adding rows, either the conversion has developed a gap or the length
    rule has drifted back to metadata.csv -- both of which fabricate `none` labels.
    """
    out, _ = converted
    meta = C.read_meta(out)
    assert meta["computed"]["num_filled_none_rows"] == 0
    assert (
        meta["computed"]["num_none_instances"]
        == meta["computed"]["num_source_none_instances"]
    )


def test_annotations_tile_every_video_exactly(converted):
    out, _ = converted
    dataset = C.CanonicalDataset(out)
    for video_id in dataset.video_ids:
        rows = dataset.annotations[dataset.annotations["video_id"] == video_id]
        rows = rows.sort_values("start_frame")
        assert rows["start_frame"].iloc[0] == 0
        gaps = rows["start_frame"].to_numpy()[1:] - rows["end_frame"].to_numpy()[:-1] - 1
        assert not gaps.any(), f"{video_id} is not tiled: {gaps}"


def test_adapter_leaves_splits_unassigned(converted):
    """Stage 2 owns splits. An adapter that assigned them would be overstepping."""
    out, _ = converted
    assert set(C.read_annotations(out)["split"]) == {C.UNASSIGNED_SPLIT}


def test_meta_records_the_decisions_a_viva_would_ask_about(converted):
    out, _ = converted
    meta = C.read_meta(out)
    assert meta["fps"] == 30.0
    assert meta["num_classes"] == 1 + len(GESTURE_CODES)
    assert meta["class_code_map"][NONE_CODE] == C.NONE_CLASS
    assert "0-based" in meta["frame_index_origin"]
    assert "Set column is not used" in meta["official_split_ignored"]


# ----------------------------------------------------------------------------------
# The adapter refuses to guess
# ----------------------------------------------------------------------------------


def test_reordered_classidx_is_rejected(tmp_path):
    """A re-released annotation set with different codes must fail, not remap silently."""
    source = write_source(
        tmp_path / "raw",
        spans={"CAM1_1_R_#1": [(NONE_CODE, 1, 10)]},
        lengths={"CAM1_1_R_#1": 10},
        codes=[NONE_CODE, *reversed(GESTURE_CODES)],
    )
    with pytest.raises(ValueError, match="classIdx"):
        convert(source, tmp_path / "canonical")


def test_wrong_number_of_gesture_names_is_rejected(tmp_path):
    source = write_source(
        tmp_path / "raw",
        spans={"CAM1_1_R_#1": [(NONE_CODE, 1, 10)]},
        lengths={"CAM1_1_R_#1": 10},
    )
    with pytest.raises(ValueError, match="paired positionally"):
        convert(source, tmp_path / "canonical", gesture_classes=GESTURE_NAMES[:5])


def test_unexplained_length_disagreement_is_rejected(tmp_path):
    """Only the known +1 is tolerated. Anything else means one source is wrong.

    A metadata count *below* the annotations would mean a span runs past the end of the
    video; a gap wider than one frame is not something the desktop.ini story explains.
    Either way the adapter stops rather than picking a side.
    """
    source = write_source(
        tmp_path / "raw",
        spans={"CAM1_1_R_#1": [(NONE_CODE, 1, 10), ("B0A", 11, 40)]},
        lengths={"CAM1_1_R_#1": 20},
    )
    with pytest.raises(ValueError, match="disagree"):
        convert(source, tmp_path / "canonical")


def test_video_with_no_annotations_is_rejected(tmp_path):
    """Tiling an unannotated video with one long `none` span would fabricate labels."""
    source = write_source(
        tmp_path / "raw",
        spans={"CAM1_1_R_#1": [(NONE_CODE, 1, 10)]},
        lengths={"CAM1_1_R_#1": 10, "CAM1_2_R_#2": 50},
    )
    with pytest.raises(ValueError, match="no rows"):
        convert(source, tmp_path / "canonical")


def test_adapter_is_registered():
    assert get_adapter("ipn_hand") is IPNHandAdapter


# ----------------------------------------------------------------------------------
# The verification table itself
# ----------------------------------------------------------------------------------


def test_verification_table_flags_a_mismatch():
    rows, ok = verification_table(
        {
            "num_videos": 199,
            "num_subjects": 50,
            "num_gesture_classes": 13,
            "num_gesture_instances": 4218,
            "num_source_none_instances": 1431,
            "num_frames": 800505,
        },
        {
            "num_videos": 200,
            "num_subjects": 50,
            "num_gesture_classes": 13,
            "num_gesture_instances": 4218,
            "num_none_instances": 1431,
            "approx_num_frames": 800000,
            "frames_tolerance": 0.02,
        },
    )
    assert not ok
    assert [r[0] for r in rows if not r[3]] == ["num_videos"]


def test_verification_table_tolerance_applies_only_to_the_frame_count():
    computed = {
        "num_videos": 200,
        "num_subjects": 50,
        "num_gesture_classes": 13,
        "num_gesture_instances": 4218,
        "num_source_none_instances": 1431,
        "num_frames": 800505,
    }
    expected = {
        "num_videos": 200,
        "num_subjects": 50,
        "num_gesture_classes": 13,
        "num_gesture_instances": 4218,
        "num_none_instances": 1431,
        "approx_num_frames": 800000,
        "frames_tolerance": 0.02,
    }
    _, ok = verification_table(computed, expected)
    assert ok
    # A tenth of a percent off on an exact field still fails.
    _, ok = verification_table({**computed, "num_gesture_instances": 4222}, expected)
    assert not ok


# ----------------------------------------------------------------------------------
# The real dataset -- the Stage 1 gate
# ----------------------------------------------------------------------------------

REAL_ROOT = Path("data/ipn_hand")
needs_real = pytest.mark.skipif(
    not (REAL_ROOT / C.ANNOTATIONS_FILE).is_file(),
    reason="run scripts/prepare_ipn_hand.py first",
)


@pytest.mark.slow
@needs_real
def test_published_figures_reproduce():
    from src.utils import config as config_utils

    cfg = config_utils.load_config("configs/base.yaml", ["dataset=ipn_hand"])
    meta = C.read_meta(REAL_ROOT)
    rows, ok = verification_table(meta["computed"], config_utils.to_dict(cfg.dataset)["expected"])
    assert ok, "\n".join(f"{r[0]}: published={r[1]} computed={r[2]}" for r in rows if not r[3])


@pytest.mark.slow
@needs_real
def test_real_dataset_has_fifty_subjects_of_four_videos_each():
    dataset = C.CanonicalDataset(REAL_ROOT)
    per_subject = dataset.annotations.groupby("subject_id")["video_id"].nunique()
    assert len(per_subject) == 50
    assert set(per_subject) == {4}


@pytest.mark.slow
@needs_real
def test_real_dataset_uses_annotation_lengths_for_the_fourteen_inflated_videos():
    """Regression guard for the desktop.ini quirk, on the real files.

    Fourteen frames out of 800k is a small error, but it is the kind that survives every
    check downstream and only shows up as a missing image at Stage 3.
    """
    computed = C.read_meta(REAL_ROOT)["computed"]
    assert len(computed["videos_with_inflated_frame_count"]) == 14
    assert computed["num_frames"] == 800491
    assert computed["num_declared_frames"] == 800505
    assert computed["num_filled_none_rows"] == 0
    assert computed["num_none_instances"] == 1431


@pytest.mark.slow
@needs_real
def test_per_class_statistics_match_table_ii():
    """The check that pins down the class *mapping*, not just the class *count*.

    classIdx.txt ships only codes (B0A, G01, ...); nothing in the download says which
    code is which gesture. The names are matched positionally from the paper's class
    table, and if that pairing were wrong every total would still reconcile exactly.
    Table II's per-class instance counts and durations are what make it checkable.
    """
    from src.utils import config as config_utils

    cfg = config_utils.load_config("configs/base.yaml", ["dataset=ipn_hand"])
    expected = config_utils.to_dict(cfg.dataset)["expected"]
    rows, ok = per_class_table(
        C.read_annotations(REAL_ROOT),
        C.read_classes(REAL_ROOT),
        expected["per_class"],
        duration_tolerance=float(expected["duration_tolerance"]),
    )
    assert len(rows) == 14
    assert ok, "\n".join(
        f"{r[0]}: n={r[2]}/{r[3]} mean={r[4]}/{r[5]} std={r[6]}/{r[7]} index_ok={r[1]}"
        for r in rows
        if not r[8]
    )


@pytest.mark.slow
@needs_real
def test_table_ii_would_catch_a_swapped_class_mapping():
    """Matching Table II is only evidence if a wrong mapping would have failed it.

    Every pairwise label swap is simulated and the check re-run. All but one must fail.
    The survivor is throw_right vs zoom_out, which the paper rounds to the same 64 (28);
    those two rest on classIdx.txt ordering alone, corroborated by the other twelve rows.
    If this ever reports a second undetectable pair, the per-class check has lost teeth
    and the mapping needs evidence from somewhere else.
    """
    import itertools

    from src.utils import config as config_utils

    cfg = config_utils.load_config("configs/base.yaml", ["dataset=ipn_hand"])
    expected = config_utils.to_dict(cfg.dataset)["expected"]
    annotations = C.read_annotations(REAL_ROOT)
    classes = C.read_classes(REAL_ROOT)
    tolerance = float(expected["duration_tolerance"])

    survived = []
    for left, right in itertools.combinations(classes, 2):
        swapped = annotations.copy()
        swapped["class"] = annotations["class"].map(
            lambda c, a=left, b=right: b if c == a else (a if c == b else c)
        )
        _, ok = per_class_table(
            swapped, classes, expected["per_class"], duration_tolerance=tolerance
        )
        if ok:
            survived.append((left, right))

    assert survived == [("throw_right", "zoom_out")], survived


@pytest.mark.slow
@needs_real
def test_real_dataset_validates():
    report = C.validate_dataset(REAL_ROOT, require_features=False)
    assert report.ok, report.render()
