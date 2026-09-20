"""Pretrained snippet backbones.

Each module here imports torch (and whatever else it needs) *lazily*, inside the class,
so that importing this package costs nothing and the test suite runs on a machine with
none of it installed. The registry below maps config names to classes without importing
the heavy modules until one is actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from src.features.base import SnippetBackbone


def load_videomaev2(**kwargs: Any) -> "SnippetBackbone":
    from src.features.backbones.videomae import VideoMAEv2Backbone

    return VideoMAEv2Backbone(**kwargs)


#: name -> factory. Factories, not classes, so the import stays deferred.
LAZY_BACKBONES = {
    "videomaev2-base": load_videomaev2,
}

__all__ = ["LAZY_BACKBONES", "load_videomaev2"]
