"""Stage the IPN Hand frames for upload to Kaggle as a private dataset.

    python scripts/prepare_kaggle_upload.py
    kaggle datasets create -p raw/kaggle_upload --dir-mode tar

Builds ``raw/kaggle_upload/`` containing exactly what the Kaggle extraction notebook needs
and nothing else:

* the five ``frames0N.tgz`` archives -- **five files, not 800k JPEGs**. Kaggle handles a
  few large files far better than hundreds of thousands of small ones, and the notebook
  untars only the shard it is working on.
* the ``annotations/`` folder, so the notebook can regenerate ``annotations.csv`` with the
  adapter rather than trusting a copied one.

The archives are **hard-linked**, not copied, so staging 9 GB costs no extra disk and takes
no time. They are on the same volume, which is what makes that possible.

Licensing: IPN Hand is CC BY 4.0, which permits redistribution with attribution, and the
metadata written here carries that attribution. The dataset is still created **private** by
default -- ``kaggle datasets create`` does not publish unless ``--public`` is passed. There
is no reason to mirror someone else's dataset publicly just to run a job on it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger  # noqa: E402

log = get_logger("kaggle-upload")

SLUG = "ipn-hand-frames"
CREDENTIALS = Path.home() / ".kaggle" / "kaggle.json"


def kaggle_username() -> str | None:
    """Read the username from the Kaggle credentials file, if it is there yet."""
    if not CREDENTIALS.is_file():
        return None
    try:
        return str(json.loads(CREDENTIALS.read_text(encoding="utf-8"))["username"])
    except Exception:  # noqa: BLE001 - a malformed file is not worth a traceback
        return None


def link_or_copy(source: Path, target: Path) -> str:
    """Hard-link if the filesystem allows it, else copy. Returns which happened."""
    if target.exists():
        return "present"
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, default=Path("raw/IPN_Hand"), help="IPN Hand source tree"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("raw/kaggle_upload"), help="staging directory"
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Kaggle username; read from ~/.kaggle/kaggle.json when omitted",
    )
    args = parser.parse_args()

    archives = sorted((args.raw / "frames").glob("frames*.tgz"))
    annotations = args.raw / "annotations"
    if not archives:
        log.error("no frames*.tgz under %s", args.raw / "frames")
        return 1
    if not annotations.is_dir():
        log.error("no annotations directory at %s", annotations)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for archive in archives:
        action = link_or_copy(archive, args.out / archive.name)
        size = archive.stat().st_size
        total += size
        log.info("%-8s %s (%.2f GB)", action, archive.name, size / 1e9)

    shutil.copytree(annotations, args.out / "annotations", dirs_exist_ok=True)
    log.info("copied annotations/ (%d files)", len(list(annotations.iterdir())))

    username = args.username or kaggle_username()
    metadata = {
        "title": "IPN Hand frames (extraction input)",
        "id": f"{username or 'YOUR_KAGGLE_USERNAME'}/{SLUG}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": (
            "Extracted RGB frames and annotations for IPN Hand, staged as tar archives for "
            "GPU feature extraction.\n\n"
            "Source: Benitez-Garcia et al., 'IPN Hand: A Video Dataset and Benchmark for "
            "Real-Time Continuous Hand Gesture Recognition', ICPR 2020. CC BY 4.0. "
            "https://gibranbenitez.github.io/IPN_Hand/\n\n"
            "Not a redistribution for general use; this exists as the input to "
            "https://github.com/asthaberi/sos-oad"
        ),
    }
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"staged {len(archives)} archive(s) + annotations in {args.out}  ({total / 1e9:.1f} GB)")
    print()
    if username:
        print("Credentials found. Upload with:")
    else:
        print("No ~/.kaggle/kaggle.json yet. Once the CLI is set up, re-run this script")
        print("(it fills in your username), then upload with:")
    print()
    print(f"    kaggle datasets create -p {args.out} --dir-mode tar")
    print()
    print("Creates a PRIVATE dataset. Resumes if interrupted. For a later revision:")
    print(f"    kaggle datasets version -p {args.out} -m 'note' --dir-mode tar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
