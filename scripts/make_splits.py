"""Stage 2: generate subject-disjoint splits and freeze them to disk.

    python scripts/make_splits.py --set dataset=ipn_hand split=ipn_official
    python scripts/make_splits.py --show          # print the frozen split, change nothing

The gate for this stage is that no ``subject_id`` appears in more than one split, asserted
by ``tests/test_splits.py`` rather than by eye.

Freezing is a one-way door. A split that gets regenerated is not frozen, and two numbers
produced either side of a regeneration are not comparable -- so this refuses to overwrite
an existing split file unless ``--force`` is passed, and says what it would have changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import canonical as C  # noqa: E402
from src.data import splits as S  # noqa: E402
from src.utils import config as config_utils  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.run import create_run  # noqa: E402

log = get_logger("splits")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    config_utils.add_config_args(parser)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="dataset directory; defaults to <paths.data_root>/<dataset.name>",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the existing frozen split and exit without writing anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing frozen split (see the warning in the module docstring)",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="freeze the split file but leave annotations.csv's split column alone",
    )
    args = parser.parse_args()

    cfg = config_utils.load_config(args.config, args.overrides)
    root = args.root or Path(cfg.paths.data_root) / str(cfg.dataset.name)
    if not root.is_dir():
        log.error("dataset root does not exist: %s", root)
        return 1

    dataset = C.CanonicalDataset(root)
    spec = S.SplitSpec.from_config(config_utils.to_dict(cfg.split))
    path = S.split_path(root, spec.name)

    # -- show mode -----------------------------------------------------------------
    if args.show:
        if not path.is_file():
            log.error("no frozen split at %s", path)
            return 1
        assignment = S.load(root, spec.name, dataset=dataset)
        print(S.summarise(assignment, dataset))
        problems = S.verify(assignment, dataset)
        print()
        print(f"verification: {'OK' if not problems else 'FAILED'}")
        for problem in problems:
            print(f"  {problem}")
        return 0 if not problems else 1

    # -- refuse to silently re-freeze ----------------------------------------------
    if path.is_file() and not args.force:
        existing = S.load(root, spec.name)
        fresh = S.generate(dataset, spec)
        changed = sorted(
            s
            for s in set(existing.subject_split) | set(fresh.subject_split)
            if existing.subject_split.get(s) != fresh.subject_split.get(s)
        )
        log.error("%s already exists and splits are frozen; pass --force to overwrite", path)
        if changed:
            log.error(
                "regenerating would move %d subject(s), e.g. %s",
                len(changed),
                {s: (existing.subject_split.get(s), fresh.subject_split.get(s)) for s in changed[:3]},
            )
            log.error(
                "every number already reported against this split would stop being "
                "comparable. Work out what changed in the dataset first."
            )
        else:
            log.info("the regenerated split would be identical, so there is nothing to do")
        return 1

    # -- generate, verify, freeze ---------------------------------------------------
    assignment = S.generate(dataset, spec)
    problems = S.verify(assignment, dataset)

    summary = S.summarise(assignment, dataset)
    print()
    print(summary)
    print()

    if problems:
        log.error("split failed verification -- nothing written")
        for problem in problems:
            print(f"  {problem}")
        return 1

    frozen = S.freeze(assignment, root)
    log.info("froze split to %s", frozen)

    if not args.no_apply:
        rows = S.apply(root, assignment)
        log.info("wrote the split column for %d annotation rows", rows)

    report = C.validate_dataset(
        root,
        check_arrays=bool(cfg.validation.check_arrays),
        require_features=bool(cfg.validation.require_features),
        require_splits=not args.no_apply,
    )
    print(report.render())

    run = create_run(cfg, name=f"stage2-{spec.name}")
    run.write_text("split_summary.txt", summary + "\n")
    run.write_metrics(
        "stage2_split",
        {
            "spec": assignment.to_dict()["spec"],
            "counts": {
                split: len(assignment.subjects(split)) for split in C.VALID_SPLITS
            },
            "subject_split": assignment.to_dict()["subject_split"],
            "folds": assignment.to_dict()["folds"],
            "subjects_digest": assignment.subjects_digest,
            "verification_ok": not problems,
            "validation_ok": report.ok,
            "validation_errors": [str(i) for i in report.errors],
        },
    )
    log.info("run directory: %s", run.path)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
