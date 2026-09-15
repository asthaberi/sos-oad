"""Validate a dataset against the canonical format.

    python scripts/validate_dataset.py --root tests/fixtures/synthetic
    python scripts/validate_dataset.py --root data/ipn_hand --require-splits

Exit code 0 means admissible, 1 means it is not. Run this after every adapter change and
before every training run -- it is the cheapest leakage check in the project.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import canonical as C  # noqa: E402
from src.utils import config as config_utils  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


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
        "--no-check-arrays",
        action="store_true",
        help="skip loading .npy files (fast structural check only)",
    )
    parser.add_argument(
        "--require-poses", action="store_true", help="treat a missing pose stream as an error"
    )
    parser.add_argument(
        "--require-splits", action="store_true", help="treat unassigned splits as an error"
    )
    args = parser.parse_args()

    log = get_logger("validate")
    cfg = config_utils.load_config(args.config, args.overrides)

    root = args.root or Path(cfg.paths.data_root) / str(cfg.dataset.name)
    if not root.exists():
        log.error("dataset root does not exist: %s", root)
        return 1

    report = C.validate_dataset(
        root,
        check_arrays=not args.no_check_arrays and bool(cfg.validation.check_arrays),
        require_poses=args.require_poses or bool(cfg.validation.require_poses),
        require_splits=args.require_splits or bool(cfg.validation.require_splits),
    )

    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
