"""Stage 3: the feature cache and its provenance.

Causality lives in ``tests/test_causality.py``; this file covers the other half of Stage 3
-- that a cache written on one machine can be shown, on another machine, to be the cache
the recorded settings produced. CLAUDE.md section 4 makes the cache a transferred
artefact, and an unverifiable transferred artefact is where a silent discrepancy hides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import canonical as C
from src.data.adapters.synthetic import make_synthetic_dataset
from src.features import cache, get_backbone
from src.features.base import ArrayFrameSource, ProjectionBackbone, SnippetExtractor


@pytest.fixture
def cached(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A small canonical dataset with a freshly written, recorded feature cache."""
    root = tmp_path / "ds"
    assert make_synthetic_dataset(root, num_videos=4, num_subjects=4, seed=3).ok

    dataset = C.CanonicalDataset(root)
    extractor = SnippetExtractor(ProjectionBackbone(dim=8), snippet_length=8, stride=4)

    digests: dict[str, str] = {}
    rng = np.random.default_rng(0)
    for video_id in dataset.video_ids:
        frames = rng.integers(
            0, 256, size=(dataset.num_frames(video_id), 8, 8, 3), dtype=np.uint8
        )
        features = extractor.extract(ArrayFrameSource(frames))
        digests[video_id] = cache.write_stream(root, video_id, features)

    cache.record_cache(root, stream="features", digests=digests, settings=extractor.describe())
    return root, digests


# ----------------------------------------------------------------------------------
# The cache verifies
# ----------------------------------------------------------------------------------


def test_a_freshly_written_cache_verifies(cached):
    root, _ = cached
    assert cache.verify_cache(root) == []


def test_cache_matches_the_annotated_frame_counts(cached):
    """One row per frame. The canonical format's own rule, checked end to end."""
    root, _ = cached
    dataset = C.CanonicalDataset(root)
    for video_id in dataset.video_ids:
        features = dataset.load_features(video_id)
        assert features.shape[0] == dataset.num_frames(video_id)
        assert features.dtype == np.float32


def test_dataset_validates_with_features_required(cached):
    """The Stage 3 strictness ratchet: require_features on, and the dataset passes."""
    root, _ = cached
    report = C.validate_dataset(root, require_features=True)
    assert report.ok, report.render()


# ----------------------------------------------------------------------------------
# ...and a corrupted or stale one does not
# ----------------------------------------------------------------------------------


def test_a_truncated_transfer_is_detected(cached):
    """The half-finished upload. The file exists and loads; it is simply not the one."""
    root, _ = cached
    video_id = C.CanonicalDataset(root).video_ids[0]
    features = np.load(C.feature_path(root, video_id))
    np.save(C.feature_path(root, video_id), features[: len(features) // 2])

    problems = cache.verify_cache(root)
    assert any("checksum mismatch" in p for p in problems), problems


def test_a_single_flipped_value_is_detected(cached):
    """Silent corruption, the kind no shape or dtype check would ever notice."""
    root, _ = cached
    video_id = C.CanonicalDataset(root).video_ids[0]
    features = np.load(C.feature_path(root, video_id))
    features[0, 0] = np.float32(features[0, 0] + 1e-3)
    np.save(C.feature_path(root, video_id), features)

    assert any("checksum mismatch" in p for p in cache.verify_cache(root))


def test_a_missing_file_is_detected(cached):
    root, _ = cached
    video_id = C.CanonicalDataset(root).video_ids[0]
    C.feature_path(root, video_id).unlink()
    assert any("missing from" in p for p in cache.verify_cache(root))


def test_an_unrecorded_extra_file_is_detected(cached):
    """A leftover from a previous extraction, sitting in the directory the run reads."""
    root, _ = cached
    C.write_features(root, "stowaway", np.zeros((10, 8), dtype=np.float32))
    assert any("not recorded" in p for p in cache.verify_cache(root))


def test_verifying_an_unextracted_dataset_says_so(tmp_path):
    root = tmp_path / "bare"
    assert make_synthetic_dataset(root, num_videos=2, num_subjects=2, seed=1).ok
    assert cache.verify_cache(root) == ["meta.yaml records no features cache to verify"]


# ----------------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------------


def test_meta_records_what_a_cache_would_need_to_be_reproduced(cached):
    root, _ = cached
    recorded = C.read_meta(root)[cache.CACHE_KEY]["features"]
    for key in ("backbone", "feature_dim", "snippet_length", "snippet_stride", "batch_size"):
        assert key in recorded, f"meta.yaml does not record {key}"
    assert "git" in recorded
    assert recorded["num_videos"] == 4
    assert "hold-last" in recorded["per_frame_expansion"]


def test_feature_dim_is_promoted_to_the_top_level(cached):
    """The canonical validator requires feature_dim at the top of meta.yaml."""
    root, _ = cached
    meta = C.read_meta(root)
    assert meta["feature_dim"] == 8
    assert meta["feature_backbone"] == "projection"


def test_recording_a_second_stream_keeps_the_first(cached):
    """Extracting poses must not erase the RGB stream's provenance."""
    root, _ = cached
    cache.record_cache(
        root, stream="poses", digests={"v": "abc"}, settings={"backbone": "mediapipe"}
    )
    recorded = C.read_meta(root)[cache.CACHE_KEY]
    assert set(recorded) == {"features", "poses"}
    assert recorded["features"]["num_videos"] == 4
    assert C.read_meta(root)["pose_backbone"] == "mediapipe"


def test_missing_videos_supports_resuming(cached):
    """An 800k-frame extraction will be interrupted at some point."""
    root, _ = cached
    dataset = C.CanonicalDataset(root)
    assert cache.missing_videos(root, dataset.video_ids) == []
    victim = dataset.video_ids[1]
    C.feature_path(root, victim).unlink()
    assert cache.missing_videos(root, dataset.video_ids) == [victim]


# ----------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------


def test_backbone_registry_returns_a_configured_instance():
    backbone = get_backbone("projection", dim=12, seed=7)
    assert backbone.dim == 12
    assert backbone.describe()["seed"] == 7


def test_unknown_backbone_is_refused():
    with pytest.raises(KeyError, match="unknown backbone"):
        get_backbone("videomaev2-b")


# ----------------------------------------------------------------------------------
# Sharding -- a bug here silently leaves videos unextracted
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 13])
def test_shards_partition_the_videos_exactly(count):
    """Every video in exactly one shard, no gaps, no duplicates.

    A shard that quietly drops videos would show up only as a cache that is short a few
    files -- and `--verify` would catch it, but after an hour of GPU time rather than
    before. The awkward counts are there because 200 does not divide by 3, 7 or 13.
    """
    from scripts.extract_features import shard_of

    video_ids = [f"v{i:03d}" for i in range(200)]
    shards = [shard_of(video_ids, f"{i}/{count}") for i in range(count)]
    flat = [v for shard in shards for v in shard]
    assert sorted(flat) == video_ids
    assert len(flat) == len(set(flat))
    sizes = [len(s) for s in shards if s]
    assert max(sizes) - min(sizes) <= 1, sizes


def test_shards_are_contiguous():
    """Contiguous, so a shard's videos cluster within a few frame archives.

    Interleaving would force every Kaggle session to stream all five tarballs.
    """
    from scripts.extract_features import shard_of

    video_ids = [f"v{i:03d}" for i in range(200)]
    first = shard_of(video_ids, "0/4")
    assert first == video_ids[: len(first)]


def test_no_shard_returns_everything():
    from scripts.extract_features import shard_of

    video_ids = [f"v{i:03d}" for i in range(10)]
    assert shard_of(video_ids, None) == video_ids


@pytest.mark.parametrize("bad", ["4/4", "5/4", "-1/4", "notashard"])
def test_bad_shard_spec_is_refused(bad):
    from scripts.extract_features import shard_of

    with pytest.raises(SystemExit):
        shard_of([f"v{i}" for i in range(10)], bad)


# ----------------------------------------------------------------------------------
# Concurrent shards and re-run adapters must not lose recorded checksums
# ----------------------------------------------------------------------------------


def test_concurrent_shards_all_keep_their_digests(cached):
    """Shards finishing at the same moment each merge into meta.yaml; none may be lost."""
    from concurrent.futures import ThreadPoolExecutor

    root, digests = cached
    shards = {f"shard{i}": f"{i:064x}" for i in range(16)}

    def record(item):
        video_id, digest = item
        cache.record_cache(root, stream="poses", digests={video_id: digest}, settings={})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, shards.items()))

    recorded = C.read_meta(root)[cache.CACHE_KEY]["poses"]["sha256"]
    assert recorded == shards
    assert not (root / (C.META_FILE + ".lock")).exists()


def test_a_held_lock_times_out_loudly_instead_of_being_broken(cached):
    root, _ = cached
    (root / (C.META_FILE + ".lock")).write_text("12345\n")
    with pytest.raises(TimeoutError, match="delete the file"):
        with cache.meta_lock(root, timeout=0.2):
            pass


def test_rerunning_the_adapter_keeps_the_recorded_cache(cached):
    """Re-running the dataset conversion must not erase Stage 3's checksums."""
    root, digests = cached
    assert make_synthetic_dataset(root, num_videos=4, num_subjects=4, seed=3).ok
    recorded = C.read_meta(root)[cache.CACHE_KEY]["features"]["sha256"]
    assert recorded == dict(sorted(digests.items()))
    # Keys the adapter owns are still the adapter's: the synthetic adapter rewrites its
    # own 16-dim features, and meta must describe what is on disk now.
    assert C.read_meta(root)["feature_dim"] == 16
