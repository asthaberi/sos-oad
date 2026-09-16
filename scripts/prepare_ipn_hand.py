"""Stage 1: unpack IPN Hand, convert it to the canonical format, and check the
published figures reproduce.

    # from the Google Drive download (16 zip parts, ~28 GB, in one directory)
    python scripts/prepare_ipn_hand.py --zips ~/Downloads

    # from an already-extracted IPN_Hand/ tree
    python scripts/prepare_ipn_hand.py --raw raw/IPN_Hand

    # also count the JPEGs in frames/, streamed from the archives (slow, ~10 min)
    python scripts/prepare_ipn_hand.py --zips ~/Downloads --verify-frames

Only the annotation text files are unpacked by default -- a few hundred kilobytes. The
~800k RGB frames are Stage 3's problem; pass ``--unpack frames`` when you get there.

The gate for this stage is the table this prints: the figures published in Benitez-Garcia
et al. (2020) beside the ones computed from the files on disk. A mismatch is something to
investigate, not to round away, so the script exits non-zero and prints both numbers.
"""

from __future__ import annotations

import argparse
import collections
import glob
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import canonical as C  # noqa: E402
from src.data.adapters import AdapterConfig  # noqa: E402
from src.data.adapters.ipn_hand import (  # noqa: E402
    ANNOTATIONS_DIR,
    FRAMES_DIR,
    IPNHandAdapter,
    per_class_table,
    render_per_class_table,
    render_verification_table,
    subject_of,
    verification_table,
)
from src.utils import config as config_utils  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.run import create_run  # noqa: E402

#: Google Drive splits a large folder into several independent zips, each holding a
#: subset of the files -- they are not the parts of one split archive, so each is opened
#: on its own and the members are unioned.
ZIP_GLOB = "IPN_Hand-*.zip"

log = get_logger("ipn-hand")


# ----------------------------------------------------------------------------------
# Unpacking
# ----------------------------------------------------------------------------------


def _zip_members(zips_dir: Path) -> Iterator[tuple[Path, zipfile.ZipInfo]]:
    paths = sorted(Path(p) for p in glob.glob(str(zips_dir / ZIP_GLOB)))
    if not paths:
        raise FileNotFoundError(f"no {ZIP_GLOB} found in {zips_dir}")
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    yield path, info


def unpack(zips_dir: Path, raw_root: Path, groups: list[str]) -> None:
    """Extract the requested groups out of the zip parts into ``raw_root``.

    ``raw_root`` is the directory that will contain ``annotations/``, ``frames/`` and so
    on -- i.e. the ``IPN_Hand`` directory itself. Anything already present is left alone,
    so re-running is cheap and safe.
    """
    wanted = {g.strip("/") for g in groups}
    raw_root.mkdir(parents=True, exist_ok=True)

    for zip_path, info in _zip_members(zips_dir):
        # Members are named IPN_Hand/<group>/<file>; strip the leading component so the
        # caller's --raw choice decides where the tree lands.
        parts = Path(info.filename).parts
        if len(parts) < 2:
            continue
        relative = Path(*parts[1:])
        if relative.parts[0] not in wanted:
            continue

        target = raw_root / relative
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("extracting %s -> %s", info.filename, target)
        with zipfile.ZipFile(zip_path) as archive, archive.open(info) as src:
            with target.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)

    # The frame and flow groups arrive as tarballs inside the zips; unpack those too, or
    # Stage 3 would find nothing but .tgz files where it expects directories of JPEGs.
    for tgz in sorted(raw_root.glob("*/*.tgz")) + sorted(raw_root.glob("*.tgz")):
        if tgz.parent.name not in wanted and tgz.stem not in wanted:
            continue
        marker = tgz.with_suffix(".extracted")
        if marker.exists():
            continue
        log.info("untarring %s", tgz)
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(raw_root)
        marker.write_text("ok\n", encoding="utf-8")


# ----------------------------------------------------------------------------------
# Frame cross-check
# ----------------------------------------------------------------------------------


def count_frames_on_disk(raw_root: Path) -> dict[str, int] | None:
    """Count JPEGs per video under ``frames/``, or None if it has not been unpacked."""
    frames_dir = raw_root / FRAMES_DIR
    if not frames_dir.is_dir():
        return None
    counts = {
        d.name: sum(1 for _ in d.glob("*.jpg"))
        for d in sorted(frames_dir.iterdir())
        if d.is_dir()
    }
    return counts or None


def count_frames_streaming(zips_dir: Path) -> dict[str, int]:
    """Count JPEGs per video by streaming frames0N.tgz straight out of the zip parts.

    Nothing is written to disk. This exists so the frame counts can be confirmed at Stage
    1 without committing to the ~30 GB unpack that Stage 3 needs.
    """
    counts: collections.Counter[str] = collections.Counter()
    for zip_path, info in _zip_members(zips_dir):
        if f"/{FRAMES_DIR}/" not in info.filename or not info.filename.endswith(".tgz"):
            continue
        log.info("scanning %s", info.filename)
        with zipfile.ZipFile(zip_path) as archive, archive.open(info) as raw:
            with tarfile.open(fileobj=raw, mode="r|gz") as tar:
                for member in tar:
                    # .jpg only. Fourteen of the frame directories carry a stray Windows
                    # desktop.ini, and counting those as frames is exactly the mistake
                    # that made metadata.csv's Frames column wrong in the first place.
                    if member.isfile() and member.name.lower().endswith(".jpg"):
                        counts[Path(member.name).parent.name] += 1
    return dict(counts)


def compare_frame_counts(
    counts: dict[str, int], dataset: C.CanonicalDataset
) -> tuple[list[str], int]:
    """Check every video's JPEG count against the length the annotations tile."""
    problems: list[str] = []
    for video_id in dataset.video_ids:
        annotated = dataset.num_frames(video_id)
        on_disk = counts.get(video_id)
        if on_disk is None:
            problems.append(f"{video_id}: no frames directory on disk")
        elif on_disk != annotated:
            problems.append(
                f"{video_id}: {on_disk} frames on disk vs {annotated} tiled by annotations"
            )
    for video_id in sorted(set(counts) - set(dataset.video_ids)):
        problems.append(f"{video_id}: frames on disk but no annotations")
    return problems, sum(counts.values())


# ----------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    config_utils.add_config_args(parser)
    parser.add_argument(
        "--zips",
        type=Path,
        default=None,
        help=f"directory holding the {ZIP_GLOB} download; omit if --raw is already unpacked",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="IPN_Hand source tree; defaults to <paths.raw_root>/IPN_Hand",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="canonical output directory; defaults to <paths.data_root>/<dataset.name>",
    )
    parser.add_argument(
        "--unpack",
        nargs="*",
        default=[ANNOTATIONS_DIR],
        metavar="GROUP",
        help=(
            f"which groups to extract from the zips (default: {ANNOTATIONS_DIR}). "
            f"Stage 3 wants '{FRAMES_DIR}', which is ~30 GB."
        ),
    )
    parser.add_argument(
        "--verify-frames",
        action="store_true",
        help="cross-check per-video JPEG counts (streams the archives if frames/ is absent)",
    )
    args = parser.parse_args()

    cfg = config_utils.load_config(args.config, args.overrides)
    if str(cfg.dataset.adapter) != IPNHandAdapter.name:
        log.error(
            "config selects dataset.adapter=%s; run with --set dataset=ipn_hand",
            cfg.dataset.adapter,
        )
        return 1

    raw_root = args.raw or Path(cfg.paths.raw_root) / "IPN_Hand"
    out_root = args.out or Path(cfg.paths.data_root) / str(cfg.dataset.name)

    if args.zips is not None:
        unpack(args.zips, raw_root, list(args.unpack))
    elif not raw_root.is_dir():
        log.error("%s does not exist and --zips was not given", raw_root)
        return 1

    # -- convert ------------------------------------------------------------------
    options = config_utils.to_dict(cfg.dataset)
    adapter = IPNHandAdapter(
        AdapterConfig(source_root=raw_root, output_root=out_root, options=options)
    )
    log.info("converting %s -> %s", raw_root, out_root)
    # Features arrive at Stage 3; until then the dataset is annotation-only and says so.
    report = adapter.convert(
        check_arrays=bool(cfg.validation.check_arrays),
        require_features=bool(cfg.validation.require_features),
    )

    computed = dict(adapter._stats)  # noqa: SLF001 - the adapter is the one that counted
    expected = config_utils.to_dict(cfg.dataset)["expected"]
    rows, figures_ok = verification_table(computed, expected)
    table = render_verification_table(rows)

    # Table II: the per-class check that pins down the class mapping, not just the counts.
    class_rows, classes_ok = per_class_table(
        C.read_annotations(out_root),
        C.read_classes(out_root),
        expected.get("per_class") or [],
        duration_tolerance=float(expected.get("duration_tolerance") or 1.0),
    )
    class_table = render_per_class_table(class_rows)

    # -- frame cross-check --------------------------------------------------------
    frame_problems: list[str] = []
    frames_total: int | None = None
    if args.verify_frames:
        counts = count_frames_on_disk(raw_root)
        source = f"{raw_root / FRAMES_DIR}"
        if counts is None:
            if args.zips is None:
                log.error("--verify-frames needs either an unpacked frames/ or --zips")
                return 1
            counts = count_frames_streaming(args.zips)
            source = "streamed from the zip parts"
        dataset = C.CanonicalDataset(out_root)
        frame_problems, frames_total = compare_frame_counts(counts, dataset)
        log.info("frame counts read from %s", source)

    # -- report -------------------------------------------------------------------
    print()
    print("Stage 1 gate -- IPN Hand published figures vs computed")
    print("=" * 78)
    print(table)
    print()
    print(f"none rows emitted:                      {computed['num_none_instances']}")
    print(f"  of which added by fill_none_spans:    {computed['num_filled_none_rows']}")
    print(f"frames per metadata.csv:                {computed['num_declared_frames']}")
    print(
        f"  videos where it overcounts by 1:      "
        f"{len(computed['videos_with_inflated_frame_count'])} "
        "(stray desktop.ini counted as a frame; annotations used instead)"
    )
    print("Per-class, against Table II of the paper")
    print("=" * 78)
    print(class_table)
    print()
    if args.verify_frames:
        print(f"frames counted on disk: {frames_total}")
        if frame_problems:
            print(f"frame-count problems ({len(frame_problems)}):")
            for problem in frame_problems[:10]:
                print(f"  {problem}")
            if len(frame_problems) > 10:
                print(f"  ... {len(frame_problems) - 10} more")
        else:
            print("every video's JPEG count matches the frames its annotations tile")
        print()
    print(report.render())

    # -- provenance (Hard Rule 4) -------------------------------------------------
    run = create_run(cfg, name="stage1-ipn-hand")
    run.write_text("verification.txt", f"{table}\n\n{class_table}\n")
    run.write_metrics(
        "stage1_verification",
        {
            "expected": dict(cfg.dataset.expected),
            "computed": computed,
            "figures_match": figures_ok,
            "per_class_match": classes_ok,
            "per_class": [
                {
                    "class": r[0], "index_ok": r[1],
                    "instances": r[2], "published_instances": r[3],
                    "mean_duration": r[4], "published_mean_duration": r[5],
                    "std_duration": r[6], "published_std_duration": r[7],
                    "ok": r[8],
                }
                for r in class_rows
            ],
            "subjects": sorted({subject_of(v) for v in C.read_annotations(out_root)["video_id"]}),
            "validation_ok": report.ok,
            "validation_errors": [str(i) for i in report.errors],
            "validation_warning_codes": sorted({i.code for i in report.warnings}),
            "frames_verified": args.verify_frames,
            "frames_total": frames_total,
            "frame_problems": frame_problems,
        },
    )
    log.info("run directory: %s", run.path)

    ok = figures_ok and classes_ok and report.ok and not frame_problems
    if not ok:
        log.error("Stage 1 gate NOT met -- investigate the mismatches above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
