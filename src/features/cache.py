"""Writing the feature cache, and proving a copy of it is the one that was extracted.

CLAUDE.md section 4: the training hardware for this project is not decided, so features
get extracted on whatever machine has a GPU and the ``.npy`` cache is then shipped to
wherever training happens. That makes the cache a *transferred artefact*, and a
transferred artefact that cannot be verified is a place for a silent discrepancy to live:
a half-finished upload, a re-extraction with a different snippet stride, a cache from an
older commit sitting in the directory the new run reads.

So every array is checksummed as it is written, and the digests plus the full extraction
settings go into ``meta.yaml``. :func:`verify_cache` re-reads them. A number in the thesis
can then be traced to a config, a seed, a split file, a git SHA -- and now also to the
exact bytes of the features it was computed from.

Note which settings are recorded. ``batch_size`` is in there alongside ``snippet_stride``
and ``snippet_length``, because float32 matmul is not batch-size invariant: re-extracting
with a different batch size produces numerically different bytes. See
``tests/test_causality.py``.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from src.data import canonical as C
from src.utils.run import git_provenance

#: meta.yaml key holding the per-video digests and the settings that produced them.
CACHE_KEY = "feature_cache"


def write_stream(
    root: Path | str,
    video_id: str,
    array: np.ndarray,
    *,
    stream: str = "features",
) -> str:
    """Write one video's array and return its SHA-256.

    ``stream`` is ``features`` or ``poses``; both go through the canonical writers so the
    dtype and shape rules are enforced in exactly one place.
    """
    root = Path(root)
    if stream == "features":
        C.write_features(root, video_id, array)
        path = C.feature_path(root, video_id)
    elif stream == "poses":
        C.write_poses(root, video_id, array)
        path = C.pose_path(root, video_id)
    else:
        raise ValueError(f"stream must be 'features' or 'poses', got {stream!r}")
    return C.checksum_file(path)


def record_cache(
    root: Path | str,
    *,
    stream: str,
    digests: dict[str, str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Merge a stream's digests and extraction settings into meta.yaml.

    Merges rather than overwrites, so extracting the pose stream does not erase the RGB
    stream's provenance. ``feature_dim`` is promoted to a top-level meta key because the
    canonical format's validator requires it there.
    """
    root = Path(root)
    meta = C.read_meta(root)
    cache = dict(meta.get(CACHE_KEY) or {})
    cache[stream] = {
        **settings,
        "num_videos": len(digests),
        "extracted_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git_provenance(),
        "sha256": dict(sorted(digests.items())),
    }
    meta[CACHE_KEY] = cache

    if stream == "features" and "feature_dim" in settings:
        meta["feature_dim"] = settings["feature_dim"]
    if stream == "features" and "backbone" in settings:
        meta["feature_backbone"] = settings["backbone"]
    if stream == "poses" and "backbone" in settings:
        meta["pose_backbone"] = settings["backbone"]

    C.write_meta(root, meta)
    return meta


def verify_cache(
    root: Path | str,
    *,
    stream: str = "features",
    progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Re-checksum the cache against meta.yaml. Returns a list of problems; empty is good.

    Run this after copying a cache between machines, and before any run that will produce
    a number worth reporting.
    """
    root = Path(root)
    meta = C.read_meta(root)
    recorded = (meta.get(CACHE_KEY) or {}).get(stream)
    if not recorded:
        return [f"meta.yaml records no {stream} cache to verify"]

    digests: dict[str, str] = dict(recorded.get("sha256") or {})
    if not digests:
        return [f"meta.yaml's {stream} cache entry has no checksums"]

    path_for = C.feature_path if stream == "features" else C.pose_path
    problems: list[str] = []
    total = len(digests)
    for index, (video_id, expected) in enumerate(sorted(digests.items()), start=1):
        path = path_for(root, video_id)
        if not path.is_file():
            problems.append(f"{video_id}: recorded in meta.yaml but missing from {stream}/")
        elif (actual := C.checksum_file(path)) != expected:
            problems.append(
                f"{video_id}: checksum mismatch "
                f"(meta {expected[:12]}, file {actual[:12]}). This file is not the one the "
                "recorded settings produced."
            )
        if progress is not None:
            progress(index, total)

    on_disk = {p.stem for p in (root / (stream if stream == "poses" else "features")).glob("*.npy")}
    for video_id in sorted(on_disk - set(digests)):
        problems.append(f"{video_id}: present in {stream}/ but not recorded in meta.yaml")

    return problems


def missing_videos(
    root: Path | str, video_ids: Iterable[str], *, stream: str = "features"
) -> list[str]:
    """Which videos still need extracting. Lets a long run resume after an interruption."""
    path_for = C.feature_path if stream == "features" else C.pose_path
    return [v for v in video_ids if not path_for(Path(root), v).is_file()]


__all__ = ["CACHE_KEY", "write_stream", "record_cache", "verify_cache", "missing_videos"]
