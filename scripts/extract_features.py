"""Stage 3: extract a feature stream into the canonical cache.

    # locally, no GPU, dependency-free backbone -- proves the pipeline, not the features
    python scripts/extract_features.py --set dataset=ipn_hand features=projection

    # on a GPU box (Kaggle), the real thing, one shard of four
    python scripts/extract_features.py --set dataset=ipn_hand features=videomaev2 \
        device=cuda --shard 0/4

    # afterwards, anywhere, including after copying the cache between machines
    python scripts/extract_features.py --verify --set dataset=ipn_hand

Sharding and resuming are not conveniences. Kaggle caps a GPU session at twelve hours and
IPN Hand is 800k frames, so an extraction *will* be interrupted; ``--shard i/n`` splits
the videos deterministically by sorted order, and by default the run skips any video whose
``.npy`` already exists. Nothing is recomputed and nothing is half-written -- each array is
written whole, then checksummed.

The gate for this stage is ``--verify`` passing over all 200 videos.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import canonical as C  # noqa: E402
from src.features import cache, get_backbone  # noqa: E402
from src.features.base import JpegDirectorySource, SnippetExtractor  # noqa: E402
from src.utils import config as config_utils  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.run import create_run  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

log = get_logger("extract")


def shard_of(video_ids: list[str], shard: str | None) -> list[str]:
    """``"i/n"`` -> the i-th of n contiguous shards of the sorted video list.

    Contiguous rather than interleaved so a shard's videos are adjacent in the frames
    archive, which matters when a session untars only the part it needs.
    """
    if not shard:
        return video_ids
    try:
        index, count = (int(part) for part in shard.split("/"))
    except ValueError:
        raise SystemExit(f"--shard must look like i/n, got {shard!r}") from None
    if not 0 <= index < count:
        raise SystemExit(f"--shard index {index} outside [0, {count})")
    # Balanced, not ceil-sized: the first `remainder` shards take one extra video each.
    # Ceil-sizing leaves the last shard short by up to count-1 videos, which on Kaggle means
    # a session quietly gets a different slice of work than the sharding implies -- and the
    # one that matters is the session that gets *more* than planned and hits the 12h cap.
    size, remainder = divmod(len(video_ids), count)
    start = index * size + min(index, remainder)
    end = start + size + (1 if index < remainder else 0)
    return video_ids[start:end]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    config_utils.add_config_args(parser)
    parser.add_argument("--root", type=Path, default=None, help="canonical dataset directory")
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="directory of per-video frame folders; defaults to <raw_root>/IPN_Hand/frames",
    )
    parser.add_argument("--shard", default=None, metavar="i/n", help="process shard i of n")
    parser.add_argument(
        "--force", action="store_true", help="re-extract videos that are already cached"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N videos (for a timing probe)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="checksum the existing cache against meta.yaml and exit",
    )
    args = parser.parse_args()

    cfg = config_utils.load_config(args.config, args.overrides)
    root = args.root or Path(cfg.paths.data_root) / str(cfg.dataset.name)
    if not root.is_dir():
        log.error("dataset root does not exist: %s", root)
        return 1

    stream = str(cfg.features.stream)

    # -- verify mode ----------------------------------------------------------------
    if args.verify:
        dataset = C.CanonicalDataset(root)
        problems = cache.verify_cache(root, stream=stream)
        missing = cache.missing_videos(root, dataset.video_ids, stream=stream)
        print(f"videos in dataset:   {len(dataset.video_ids)}")
        print(f"still to extract:    {len(missing)}")
        print(f"checksum problems:   {len(problems)}")
        for problem in problems[:10]:
            print(f"  {problem}")
        if len(problems) > 10:
            print(f"  ... {len(problems) - 10} more")
        ok = not problems and not missing
        print("\nSTAGE 3 GATE: " + ("PASS" if ok else "NOT MET"))
        return 0 if ok else 1

    # -- extraction -------------------------------------------------------------------
    seed_report = set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))

    dataset = C.CanonicalDataset(root)
    frames_root = args.frames_root or Path(cfg.paths.raw_root) / "IPN_Hand" / "frames"
    if not frames_root.is_dir():
        log.error("frames directory does not exist: %s", frames_root)
        return 1

    video_ids = shard_of(dataset.video_ids, args.shard)
    if not args.force:
        todo = cache.missing_videos(root, video_ids, stream=stream)
        if len(todo) < len(video_ids):
            log.info("skipping %d already-cached video(s)", len(video_ids) - len(todo))
        video_ids = todo
    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    if not video_ids:
        log.info("nothing to do; every video in this shard is already cached")
        return 0

    options = config_utils.to_dict(cfg.features).get("backbone_options") or {}
    # Device is config-driven, never hardcoded (CLAUDE.md section 4). Backbones that do
    # not run on an accelerator simply ignore it.
    backbone = get_backbone(str(cfg.features.backbone), **options, device=str(cfg.device))
    extractor = SnippetExtractor(
        backbone,
        snippet_length=int(cfg.features.snippet_length),
        stride=int(cfg.features.snippet_stride),
        batch_size=int(cfg.features.batch_size),
    )
    log.info("backbone=%s device=%s dim=%d", backbone.name, cfg.device, backbone.dim)
    log.info("extracting %d video(s) into %s/%s", len(video_ids), root, stream)

    digests: dict[str, str] = {}
    started = time.time()
    total_frames = 0

    for position, video_id in enumerate(video_ids, start=1):
        num_frames = dataset.num_frames(video_id)
        source = JpegDirectorySource(frames_root / video_id, num_frames)
        clock = time.time()
        features = extractor.extract(source)

        if features.shape[0] != num_frames:
            log.error(
                "%s: produced %d rows for %d annotated frames",
                video_id,
                features.shape[0],
                num_frames,
            )
            return 1

        digests[video_id] = cache.write_stream(root, video_id, features, stream=stream)
        total_frames += num_frames
        elapsed = time.time() - clock
        log.info(
            "[%d/%d] %s  %d frames  %.1fs  %.0f fps",
            position,
            len(video_ids),
            video_id,
            num_frames,
            elapsed,
            num_frames / max(elapsed, 1e-6),
        )

    # Merge this shard's digests into whatever meta.yaml already records, so shards run in
    # separate sessions accumulate instead of overwriting each other.
    existing = (C.read_meta(root).get(cache.CACHE_KEY) or {}).get(stream) or {}
    merged = {**(existing.get("sha256") or {}), **digests}
    cache.record_cache(root, stream=stream, digests=merged, settings=extractor.describe())

    wall = time.time() - started
    log.info(
        "done: %d video(s), %d frames, %.1f min, %.0f fps overall",
        len(video_ids),
        total_frames,
        wall / 60,
        total_frames / max(wall, 1e-6),
    )

    run = create_run(cfg, name=f"stage3-{cfg.features.name}", seed_report=seed_report)
    run.write_metrics(
        "stage3_extraction",
        {
            "settings": extractor.describe(),
            "shard": args.shard,
            "videos": list(video_ids),
            "num_videos": len(video_ids),
            "num_frames": total_frames,
            "wall_seconds": wall,
            "frames_per_second": total_frames / max(wall, 1e-6),
            "cached_total": len(merged),
        },
    )
    log.info("run directory: %s", run.path)

    remaining = cache.missing_videos(root, dataset.video_ids, stream=stream)
    log.info("%d video(s) still uncached across the whole dataset", len(remaining))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
