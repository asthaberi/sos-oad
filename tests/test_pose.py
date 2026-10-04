"""Stage 3b: the RTMW pose backbone's own rules.

Causality is in ``tests/test_causality.py`` with every other temporal module. This file
covers what the backbone decides per frame -- which person, what happens when there is
none, what the output means -- with fake models, so it runs without onnxruntime. One test
at the end runs the real models on real frames, and only where both are available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import canonical as C
from src.data.adapters.synthetic import make_synthetic_dataset
from src.features import cache
from src.features.backbones.rtmw import RTMWPoseBackbone, select_largest
from src.features.base import ArrayFrameSource, SnippetExtractor


class Recorder:
    """A fake detector + estimator pair that records what it was shown."""

    def __init__(self, boxes: np.ndarray, joints: int = 7) -> None:
        self.boxes = boxes
        self.joints = joints
        self.images: list[np.ndarray] = []
        self.chosen: list[np.ndarray] = []

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        return self.boxes

    def estimate(self, bgr: np.ndarray, boxes: np.ndarray):
        self.images.append(bgr)
        self.chosen.append(boxes)
        keypoints = np.arange(self.joints * 2, dtype=float).reshape(1, self.joints, 2)
        return keypoints, np.full((1, self.joints), 0.5)

    def backbone(self) -> RTMWPoseBackbone:
        return RTMWPoseBackbone(detector=self.detect, estimator=self.estimate)


def one_frame(height: int = 24, width: int = 32) -> np.ndarray:
    frame = np.zeros((1, 1, height, width, 3), np.uint8)
    frame[..., 0] = 200  # red
    return frame


def test_largest_box_wins():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 30, 20], [5, 5, 20, 20]], dtype=float)
    assert np.array_equal(select_largest(boxes), [[0, 0, 30, 20]])


def test_no_box_selects_nothing():
    assert select_largest(np.zeros((0, 4))) is None


def test_equal_boxes_resolve_to_the_first():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    assert np.array_equal(select_largest(boxes), [[0, 0, 10, 10]])


def test_the_estimator_gets_the_largest_box():
    rec = Recorder(np.array([[0, 0, 4, 4], [1, 1, 20, 20]], dtype=float))
    rec.backbone().forward_batch(one_frame())
    assert np.array_equal(rec.chosen[-1], [[1, 1, 20, 20]])


def test_no_detection_falls_back_to_the_whole_frame_and_is_counted():
    rec = Recorder(np.zeros((0, 4)))
    backbone = rec.backbone()
    backbone.forward_batch(np.concatenate([one_frame(), one_frame()]))
    assert np.array_equal(rec.chosen[-1], [[0, 0, 32, 24]])
    stats = backbone.run_stats()
    # The joint-count probe at load time is not a real frame and must not be counted.
    assert stats["frames_seen"] == 2
    assert stats["frames_without_detection"] == 2
    assert stats["frame_sizes_seen"] == [[32, 24]]


def test_models_receive_bgr_because_the_frame_sources_yield_rgb():
    rec = Recorder(np.array([[0, 0, 32, 24]], dtype=float))
    rec.backbone().forward_batch(one_frame())
    image = rec.images[-1]
    assert image[0, 0, 2] == 200 and image[0, 0, 0] == 0


def test_output_is_x_y_confidence_per_joint():
    rec = Recorder(np.array([[0, 0, 32, 24]], dtype=float), joints=7)
    row = rec.backbone().forward_batch(one_frame())[0].reshape(7, 3)
    assert np.array_equal(row[:, :2], np.arange(14).reshape(7, 2))
    assert np.all(row[:, 2] == 0.5)


def test_joint_count_is_probed_not_assumed():
    backbone = Recorder(np.zeros((0, 4)), joints=7).backbone()
    assert backbone.num_joints == 7 and backbone.dim == 21


def test_a_pose_cache_written_through_the_extractor_validates(tmp_path: Path):
    """End to end on the synthetic fixture: extract, reshape to [T, J, 3], write, record,
    and the validator accepts it with the pose stream required."""
    root = tmp_path / "ds"
    assert make_synthetic_dataset(root, num_videos=4, num_subjects=4, seed=3).ok
    dataset = C.CanonicalDataset(root)
    rec = Recorder(np.array([[0, 0, 8, 8]], dtype=float), joints=5)
    extractor = SnippetExtractor(rec.backbone(), snippet_length=1, stride=1, batch_size=1)

    rng = np.random.default_rng(0)
    digests = {}
    for video_id in dataset.video_ids:
        frames = rng.integers(0, 256, (dataset.num_frames(video_id), 8, 8, 3), dtype=np.uint8)
        flat = extractor.extract(ArrayFrameSource(frames))
        poses = flat.reshape(flat.shape[0], -1, 3)  # what extract_features.py does
        digests[video_id] = cache.write_stream(root, video_id, poses, stream="poses")
    cache.record_cache(root, stream="poses", digests=digests, settings=extractor.describe())

    assert cache.verify_cache(root, stream="poses") == []
    report = C.validate_dataset(root, require_poses=True)
    assert report.ok, report
    assert C.read_meta(root)["pose_backbone"] == "rtmw-wholebody"


# ----------------------------------------------------------------------------------
# The real thing: real models on real frames, where both exist
# ----------------------------------------------------------------------------------

REAL_ROOT = Path("data/ipn_hand")
FRAMES_ROOT = Path("raw/IPN_Hand/frames")


@pytest.mark.slow
@pytest.mark.causality
@pytest.mark.skipif(
    not (REAL_ROOT / C.ANNOTATIONS_FILE).is_file() or not FRAMES_ROOT.is_dir(),
    reason="needs data/ipn_hand and raw/IPN_Hand/frames",
)
def test_real_rtmw_on_real_frames_is_causal_and_deterministic():
    pytest.importorskip("rtmlib")
    pytest.importorskip("onnxruntime")
    import yaml

    from src.features import get_backbone
    from src.features.base import JpegDirectorySource

    config = yaml.safe_load(Path("configs/features/rtmw.yaml").read_text(encoding="utf-8"))
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = get_backbone(config["backbone"], **config["backbone_options"], device=device)
    extractor = SnippetExtractor(backbone, snippet_length=1, stride=1, batch_size=1)

    dataset = C.CanonicalDataset(REAL_ROOT)
    video_id = dataset.video_ids_for_split("train")[0]
    source = JpegDirectorySource(FRAMES_ROOT / video_id, 24)
    full = extractor.extract(source)
    again = extractor.extract(source)
    prefix = extractor.extract(source.truncated(9))

    assert np.array_equal(full, again), "two runs of the same frames differ"
    assert np.array_equal(full[:9], prefix), "a truncated stream changed earlier frames"
    poses = full.reshape(24, -1, 3)
    assert poses.shape[1] == 133
    assert np.isfinite(poses).all()
