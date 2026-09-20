"""VideoMAEv2-Base as a frozen snippet backbone.

Stage 3's first-preference RGB stream (CLAUDE.md). Checkpoint ``OpenGVLab/VideoMAEv2-Base``
on the HuggingFace Hub, which expects 16 frames at 3x224x224 and wants its pixel tensor
permuted to ``(B, C, T, H, W)``.

torch and transformers are imported lazily inside :meth:`_load`, so this module stays
importable -- and the whole test suite stays runnable -- on a machine with neither
installed. That matters because the author's laptop has no CUDA and the causal machinery
must remain testable there (CLAUDE.md section 4).

Choices that are recorded rather than assumed
---------------------------------------------
**Feature dimension is probed, not hardcoded.** The model card does not document the
output shape. Assuming 768 and being wrong would produce a cache of plausible-looking
garbage, so the dimension is measured on a dummy clip at load time and written into
``meta.yaml``.

**Frames are resized full-frame, not centre-cropped.** The usual eval transform is
resize-short-side-then-centre-crop, which on 640x480 keeps only the middle ~75% of the
width. Four of the thirteen gesture classes are ``throw_left`` / ``throw_right`` /
``zoom_in`` / ``zoom_out`` -- lateral hand motion that reaches toward the frame edges --
so a centre crop can literally cut the gesture out of the clip. Squashing the aspect ratio
is the lesser evil here and it is config-exposed (``crop: center`` restores the usual
behaviour) so the choice can be ablated rather than argued about.

**Preprocessing is done with tensor ops, not the HF image processor.** The processor's
per-frame PIL path is slow, and decode is already the bottleneck. The normalisation
constants are read *from* the processor so they match the checkpoint, and the exact
resize/normalise recipe is written into ``meta.yaml``.

**No dropout, no train mode, no autocast by default.** Rule 1's test asserts bit-identity,
and a cache that changes between runs cannot back a reproducible number either.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.features.base import SnippetBackbone

#: Frames the checkpoint expects per clip. The snippet length must match this.
CLIP_FRAMES = 16
#: Spatial size the checkpoint expects.
INPUT_SIZE = 224


class VideoMAEv2Backbone(SnippetBackbone):
    """Frozen VideoMAEv2-Base. Clip in, one pooled feature vector out."""

    name = "videomaev2-base"

    def __init__(
        self,
        *,
        checkpoint: str = "OpenGVLab/VideoMAEv2-Base",
        device: str = "cpu",
        input_size: int = INPUT_SIZE,
        crop: str = "resize",
        dtype: str = "float32",
    ) -> None:
        if crop not in ("resize", "center"):
            raise ValueError(f"crop must be 'resize' or 'center', got {crop!r}")
        self.checkpoint = checkpoint
        self.device = device
        self.input_size = int(input_size)
        self.crop = crop
        self.dtype = dtype

        self._model = None
        self._mean = None
        self._std = None
        self._dim: int | None = None

    # -- loading ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

        config = AutoConfig.from_pretrained(self.checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.checkpoint, config=config, trust_remote_code=True
        )
        model.eval()
        # Frozen: this is a feature extractor, not something being trained. Disabling grad
        # on the parameters as well as using no_grad keeps memory down on a 16 GB GPU.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(self.device, dtype=getattr(torch, self.dtype))
        self._model = model

        # Normalisation constants come from the checkpoint's own processor so they match
        # what it was trained with, even though the resize is done with tensor ops.
        processor = VideoMAEImageProcessor.from_pretrained(self.checkpoint)
        self._mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1, 1)
        self._std = torch.tensor(processor.image_std).view(1, 3, 1, 1, 1)
        self._mean = self._mean.to(self.device, dtype=getattr(torch, self.dtype))
        self._std = self._std.to(self.device, dtype=getattr(torch, self.dtype))

        # Probe the output dimension rather than trusting a number from a blog post.
        dummy = np.zeros((1, CLIP_FRAMES, self.input_size, self.input_size, 3), np.uint8)
        self._dim = int(self.forward_batch(dummy).shape[1])

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    # -- inference ----------------------------------------------------------------

    def _preprocess(self, clips: np.ndarray):
        """``[B, L, H, W, 3]`` uint8 -> ``[B, 3, L, S, S]`` normalised tensor."""
        import torch
        import torch.nn.functional as F

        batch, length, height, width, _ = clips.shape
        tensor = torch.from_numpy(np.ascontiguousarray(clips)).to(self.device)
        # -> [B, 3, L, H, W], float, 0..1
        tensor = tensor.permute(0, 4, 1, 2, 3).to(getattr(torch, self.dtype)).div_(255.0)

        size = self.input_size
        # interpolate wants [N, C, H, W], so collapse batch and time into one axis.
        frames = tensor.permute(0, 2, 1, 3, 4).reshape(batch * length, 3, height, width)
        if self.crop == "center":
            scale = size / min(height, width)
            frames = F.interpolate(
                frames,
                size=(round(height * scale), round(width * scale)),
                mode="bilinear",
                align_corners=False,
            )
            top = (frames.shape[2] - size) // 2
            left = (frames.shape[3] - size) // 2
            frames = frames[:, :, top : top + size, left : left + size]
        else:
            frames = F.interpolate(
                frames, size=(size, size), mode="bilinear", align_corners=False
            )

        tensor = frames.reshape(batch, length, 3, size, size).permute(0, 2, 1, 3, 4)
        return (tensor - self._mean) / self._std

    def forward_batch(self, clips: np.ndarray) -> np.ndarray:
        import torch

        self._load()
        clips = np.asarray(clips)
        if clips.ndim != 5 or clips.shape[-1] != 3:
            raise ValueError(f"clips must be [B, L, H, W, 3], got {clips.shape}")
        if clips.shape[1] != CLIP_FRAMES:
            raise ValueError(
                f"{self.name} expects {CLIP_FRAMES} frames per clip, got {clips.shape[1]}. "
                "Set features.snippet_length to match the checkpoint."
            )

        with torch.no_grad():
            pixels = self._preprocess(clips)
            outputs = self._model(pixel_values=pixels)
            features = self._pool(outputs)
        return features.float().cpu().numpy().astype(np.float32)

    @staticmethod
    def _pool(outputs: Any):
        """Reduce whatever the model returned to ``[B, D]``.

        The remote checkpoint code may return a bare tensor or an output object, and it
        may return per-token features. Mean-pooling over tokens is a *spatiotemporal*
        reduction within a single clip, and that clip contains only frames at or before
        its own end frame -- so it does not cross the causal boundary. Pooling across
        clips would; that is what the hold-last expansion deliberately avoids.
        """
        tensor = getattr(outputs, "last_hidden_state", None)
        if tensor is None:
            tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if tensor.ndim == 3:  # [B, tokens, D]
            tensor = tensor.mean(dim=1)
        if tensor.ndim != 2:
            raise ValueError(
                f"could not reduce model output of shape {tuple(tensor.shape)} to [B, D]"
            )
        return tensor

    # -- provenance ---------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "backbone": self.name,
            "checkpoint": self.checkpoint,
            "feature_dim": self.dim,
            "clip_frames": CLIP_FRAMES,
            "input_size": self.input_size,
            "crop": self.crop,
            "dtype": self.dtype,
            "device_used": self.device,
            "preprocessing": (
                f"uint8 -> /255 -> bilinear {'full-frame resize' if self.crop == 'resize' else 'short-side resize + centre crop'} "
                f"to {self.input_size}x{self.input_size} -> (x - mean) / std with the "
                "checkpoint's VideoMAEImageProcessor constants"
            ),
            "frozen": True,
            "pooling": "mean over output tokens, within a single clip",
        }


__all__ = ["VideoMAEv2Backbone", "CLIP_FRAMES", "INPUT_SIZE"]
