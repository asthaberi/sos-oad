"""Stage 3 feature extraction: RGB and pose streams, both causal-safe.

The pieces that Rule 1 depends on -- the snippet grid and the per-frame expansion -- live
in :mod:`src.features.causal` and are testable with nothing installed beyond numpy. The
pieces that need weights, torch or a GPU sit behind :class:`SnippetBackbone` and are
imported only when a config asks for them, so ``pytest`` runs on a clean checkout.
"""

from __future__ import annotations

from typing import Any, Type

from src.features.base import (
    ArrayFrameSource,
    FrameSource,
    JpegDirectorySource,
    ProjectionBackbone,
    SnippetBackbone,
    SnippetExtractor,
)

#: name -> backbone class. Real pretrained backbones register here as they land; the
#: projection backbone is registered as a real one so the fixture path and the production
#: path are the same code.
BACKBONES: dict[str, Type[SnippetBackbone]] = {
    ProjectionBackbone.name: ProjectionBackbone,
}


def get_backbone(name: str, **kwargs: Any) -> SnippetBackbone:
    """Build a backbone by config name.

    Pretrained backbones resolve through a lazy factory so that asking for the projection
    backbone never imports torch or transformers -- which is what keeps the test suite
    runnable on a machine that has neither.
    """
    if name in BACKBONES:
        return BACKBONES[name](**kwargs)

    from src.features.backbones import LAZY_BACKBONES

    if name in LAZY_BACKBONES:
        return LAZY_BACKBONES[name](**kwargs)

    raise KeyError(
        f"unknown backbone {name!r}; registered: {sorted([*BACKBONES, *LAZY_BACKBONES])}"
    )


__all__ = [
    "BACKBONES",
    "get_backbone",
    "FrameSource",
    "ArrayFrameSource",
    "JpegDirectorySource",
    "SnippetBackbone",
    "ProjectionBackbone",
    "SnippetExtractor",
]
