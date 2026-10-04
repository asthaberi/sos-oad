"""RTMW whole-body pose as a per-frame backbone: the Stage 3 pose stream.

Top-down, one frame at a time: a person detector (YOLOX) finds the subject, then RTMW
estimates 133 COCO-WholeBody keypoints inside that box -- 17 body, 6 feet, 68 face, 21 per
hand. Each frame's output depends on that frame alone, so the stream is causal by
construction; it is still put through the truncation test like every other module.

Runtime: rtmlib + onnxruntime, not MMPose
-----------------------------------------
CLAUDE.md's first preference was RTMPose via MMPose. On the GPU server that stack does not
install: OpenMMLab publishes no ``mmcv`` builds for torch 2.5/2.6, its Windows builds stop
at Python 3.11, and ``mmpose`` pulls ``chumpy``, which fails to build. rtmlib runs the same
RTMPose-family ONNX exports (RTMW is from the RTMPose project) through onnxruntime with no
compiled extensions. The model files are pinned by URL in config and their SHA-256 digests
go into ``meta.yaml``.

Choices that are recorded rather than assumed
---------------------------------------------
**Person selection: the largest detected box.** IPN Hand has one subject per video, but the
detector occasionally returns a duplicate or partial second box. Largest-area is decided
from the current frame alone -- no tracking state, nothing carried between frames.

**No detection: estimate on the whole frame.** Rather than emitting zeros, which a model
could read as a real pose at the image origin. The confidence channel then carries the
model's own uncertainty, and the count of such frames is reported per run.

**Output: ``[J, 3]`` per frame = (x, y, confidence),** x and y in pixels of the *source*
frame (origin top-left, x right, y down). No normalisation here: per-person normalisation
(root-relative, scaled by torso or hand length) is a per-frame function applied at training
time, where it can be ablated, and the frame size needed to interpret pixels is in
``meta.yaml``.

**Deterministic sessions.** rtmlib builds onnxruntime sessions with default options; here
they are rebuilt with ``use_deterministic_compute`` and a heuristic (not benchmarked)
cuDNN convolution search, so algorithm selection cannot vary between runs. Rule 1's test
asserts bit-identity, and so does the cache's checksum.

**A requested GPU that silently falls back to CPU is an error.** onnxruntime-gpu built for a
different CUDA major version loads, logs an error, and quietly runs on CPU at ~1/5 the
speed -- found the hard way with 1.30 (CUDA 13) against torch's CUDA 12 DLLs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.features.base import SnippetBackbone

#: (BGR image) -> boxes ``[N, 4]`` as x1, y1, x2, y2.
Detector = Callable[[np.ndarray], np.ndarray]
#: (BGR image, boxes ``[1, 4]``) -> (keypoints ``[1, J, 2]``, scores ``[1, J]``).
Estimator = Callable[[np.ndarray, np.ndarray], "tuple[np.ndarray, np.ndarray]"]

#: COCO-WholeBody index ranges, for consumers that want a part of the skeleton.
WHOLEBODY_PARTS = {
    "body": (0, 17),
    "feet": (17, 23),
    "face": (23, 91),
    "left_hand": (91, 112),
    "right_hand": (112, 133),
}


def select_largest(boxes: np.ndarray) -> np.ndarray | None:
    """The largest-area box as ``[1, 4]``, or None if there are none. Ties: the first."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(np.argmax(areas))][None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RTMWPoseBackbone(SnippetBackbone):
    """Frozen YOLOX + RTMW. One frame in, ``J * 3`` numbers out.

    Used with ``snippet_length: 1`` and ``snippet_stride: 1``, so the extractor hands it
    every frame on its own. Given a longer clip it estimates the clip's *last* frame -- the
    current one -- and never looks at the others.

    ``detector`` / ``estimator`` can be injected, which is how the tests exercise the
    selection, fallback and output layout without onnxruntime, weights or a GPU.
    """

    name = "rtmw-wholebody"

    def __init__(
        self,
        *,
        det_model: str = "",
        det_input_size: tuple[int, int] | list[int] = (640, 640),
        pose_model: str = "",
        pose_input_size: tuple[int, int] | list[int] = (192, 256),
        device: str = "cpu",
        detector: Detector | None = None,
        estimator: Estimator | None = None,
    ) -> None:
        self.det_model = det_model
        self.det_input_size = tuple(int(v) for v in det_input_size)
        self.pose_model = pose_model
        self.pose_input_size = tuple(int(v) for v in pose_input_size)
        self.device = device

        self._detector = detector
        self._estimator = estimator
        self._loaded = False
        self._num_joints: int | None = None
        self._runtime: dict[str, Any] = {}

        self._frames_seen = 0
        self._fallback_frames = 0
        self._frame_sizes: set[tuple[int, int]] = set()

    # -- loading ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        if self._detector is None or self._estimator is None:
            self._load_rtmlib()
        self._loaded = True

        # Probe the joint count on a blank frame rather than hardcoding 133. The probe is
        # not a real frame, so it is kept out of the run statistics.
        _, scores = self._estimate(np.zeros((256, 256, 3), np.uint8))
        self._num_joints = int(scores.shape[0])
        self._frames_seen = self._fallback_frames = 0
        self._frame_sizes.clear()

    def _load_rtmlib(self) -> None:
        # torch first: on Windows its import puts its CUDA 12 / cuDNN 9 DLLs on the search
        # path, which is what onnxruntime-gpu's CUDA provider links against.
        import torch  # noqa: F401
        import onnxruntime as ort
        import rtmlib
        from rtmlib import YOLOX, RTMPose

        # Built on CPU (cheap), then their sessions are replaced with deterministic ones.
        det = YOLOX(self.det_model, model_input_size=self.det_input_size,
                    backend="onnxruntime", device="cpu")
        pose = RTMPose(self.pose_model, model_input_size=self.pose_input_size,
                       backend="onnxruntime", device="cpu")
        for tool in (det, pose):
            tool.session = self._session(ort, tool.onnx_model)

        self._detector = det
        self._estimator = lambda image, boxes: pose(image, bboxes=boxes)
        self._runtime = {
            "runtime": f"onnxruntime {ort.__version__} via rtmlib {rtmlib.__version__}"
            if hasattr(rtmlib, "__version__") else f"onnxruntime {ort.__version__} via rtmlib",
            "execution_provider": det.session.get_providers()[0],
            "det_model_sha256": _sha256(Path(det.onnx_model)),
            "pose_model_sha256": _sha256(Path(pose.onnx_model)),
        }

    def _session(self, ort: Any, model_path: str) -> Any:
        options = ort.SessionOptions()
        options.use_deterministic_compute = True
        options.log_severity_level = 3  # errors only; node-placement notices are noise
        if self.device.startswith("cuda"):
            device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            providers = [(
                "CUDAExecutionProvider",
                {"device_id": device_id, "cudnn_conv_algo_search": "HEURISTIC"},
            )]
        else:
            providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        if self.device.startswith("cuda") and session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"device={self.device!r} was requested but onnxruntime fell back to "
                f"{session.get_providers()}. Usually an onnxruntime-gpu build for a different "
                "CUDA major version than the DLLs torch ships; see requirements-cuda.txt."
            )
        return session

    @property
    def dim(self) -> int:
        self._load()
        assert self._num_joints is not None
        return self._num_joints * 3

    @property
    def num_joints(self) -> int:
        self._load()
        assert self._num_joints is not None
        return self._num_joints

    # -- inference ----------------------------------------------------------------

    def _estimate(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One BGR frame -> (keypoints ``[J, 2]``, scores ``[J]``) for the selected person."""
        assert self._detector is not None and self._estimator is not None
        box = select_largest(self._detector(bgr))
        if box is None:
            height, width = bgr.shape[:2]
            box = np.array([[0.0, 0.0, float(width), float(height)]])
            self._fallback_frames += 1
        keypoints, scores = self._estimator(bgr, box)
        return np.asarray(keypoints)[0], np.asarray(scores)[0]

    def forward_batch(self, clips: np.ndarray) -> np.ndarray:
        self._load()
        clips = np.asarray(clips)
        if clips.ndim != 5 or clips.shape[-1] != 3:
            raise ValueError(f"clips must be [B, L, H, W, 3], got {clips.shape}")

        rows = []
        for clip in clips:
            frame = clip[-1]  # the current frame; earlier ones are never read
            bgr = np.ascontiguousarray(frame[:, :, ::-1])  # frame sources yield RGB
            keypoints, scores = self._estimate(bgr)
            rows.append(np.concatenate([keypoints, scores[:, None]], axis=1).reshape(-1))
            self._frames_seen += 1
            self._frame_sizes.add((int(frame.shape[1]), int(frame.shape[0])))
        return np.stack(rows).astype(np.float32)

    # -- provenance ---------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "backbone": self.name,
            "det_model": self.det_model,
            "det_input_size": list(self.det_input_size),
            "pose_model": self.pose_model,
            "pose_input_size": list(self.pose_input_size),
            "num_joints": self.num_joints,
            "feature_dim": self.dim,
            "layout": "[T, J, 3] = (x, y, confidence); x, y in source-frame pixels, "
                      "origin top-left; COCO-WholeBody joint order",
            "parts": {name: list(span) for name, span in WHOLEBODY_PARTS.items()},
            "person_selection": "largest detected box, decided per frame",
            "no_detection": "estimate on the whole frame",
            "deterministic": "use_deterministic_compute=True, cudnn_conv_algo_search=HEURISTIC",
            "device_used": self.device,
            "frozen": True,
            **self._runtime,
        }

    def run_stats(self) -> dict[str, Any]:
        """Per-run counts, kept out of :meth:`describe` because that one identifies the
        cache and is shared by every shard; these differ shard to shard."""
        return {
            "frames_seen": self._frames_seen,
            "frames_without_detection": self._fallback_frames,
            "frame_sizes_seen": sorted(list(size) for size in self._frame_sizes),
        }


__all__ = ["RTMWPoseBackbone", "select_largest", "WHOLEBODY_PARTS"]
