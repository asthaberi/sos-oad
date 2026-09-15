"""Generate the synthetic canonical-format dataset used as a test fixture.

    python scripts/make_fixture.py --out tests/fixtures/synthetic
    python scripts/make_fixture.py --config configs/base.yaml --set dataset.num_videos=32

The test suite does not read this output -- it generates its own copy in a tmp directory,
so `pytest` works on a clean checkout with nothing downloaded and nothing generated. This
script is for looking at the format by hand, and for giving downstream stages a real
canonical dataset to point at before IPN Hand is available.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.adapters.synthetic import make_synthetic_dataset  # noqa: E402
from src.utils import config as config_utils  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    config_utils.add_config_args(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/synthetic"),
        help="output directory for the canonical dataset",
    )
    parser.add_argument(
        "--force", action="store_true", help="delete the output directory if it exists"
    )
    args = parser.parse_args()

    log = get_logger("make-fixture")
    cfg = config_utils.load_config(args.config, args.overrides)

    if args.out.exists():
        if not args.force:
            log.error("%s already exists; pass --force to overwrite", args.out)
            return 1
        shutil.rmtree(args.out)

    options = config_utils.to_dict(cfg.dataset)
    log.info("generating synthetic dataset at %s", args.out)
    report = make_synthetic_dataset(args.out, **options)

    print()
    print(report.render())

    if not report.ok:
        log.error("generated fixture failed validation -- the generator is wrong")
        return 1
    log.info("fixture written and validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
