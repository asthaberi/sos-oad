"""Dataset adapter registry.

The only dataset-aware package in the repository. Adding a dataset means adding one
module here and one YAML under ``configs/dataset/`` -- nothing downstream changes.
See CLAUDE.md Rule 5.
"""

from __future__ import annotations

from typing import Type

from src.data.adapters.base import AdapterConfig, DatasetAdapter
from src.data.adapters.ipn_hand import IPNHandAdapter
from src.data.adapters.synthetic import SyntheticAdapter

#: name -> adapter class. Phase 2 registers the author's own SOS recordings here and
#: stops there.
ADAPTERS: dict[str, Type[DatasetAdapter]] = {
    SyntheticAdapter.name: SyntheticAdapter,
    IPNHandAdapter.name: IPNHandAdapter,
}


def get_adapter(name: str) -> Type[DatasetAdapter]:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown adapter {name!r}; registered adapters: {sorted(ADAPTERS)}"
        ) from None


__all__ = [
    "ADAPTERS",
    "AdapterConfig",
    "DatasetAdapter",
    "IPNHandAdapter",
    "SyntheticAdapter",
    "get_adapter",
]
