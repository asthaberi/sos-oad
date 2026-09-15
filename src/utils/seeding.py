"""Seeding and determinism.

Thesis numbers have to be reproducible, and where they cannot be, the run log has to say
so rather than imply a precision that is not there. Stage 8 reports a seed-variance
estimate, which only means something if a fixed seed actually pins the run down.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SeedReport:
    """What was actually pinned, for the run log."""

    seed: int
    deterministic_requested: bool
    torch_available: bool
    cudnn_deterministic: bool = False
    torch_deterministic_algorithms: bool = False
    caveats: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.caveats is None:
            self.caveats = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "deterministic_requested": self.deterministic_requested,
            "torch_available": self.torch_available,
            "cudnn_deterministic": self.cudnn_deterministic,
            "torch_deterministic_algorithms": self.torch_deterministic_algorithms,
            "caveats": list(self.caveats),
        }


def set_seed(seed: int, deterministic: bool = True) -> SeedReport:
    """Seed every RNG we use and, optionally, ask for deterministic kernels.

    Returns a report rather than nothing, so the caller can write into the run log what
    was genuinely pinned. Determinism on GPU is best-effort: some ops have no
    deterministic implementation, and forcing one where it exists can cost real speed.
    Where we cannot guarantee it, the caveat is recorded instead of glossed over.
    """
    report = SeedReport(seed=seed, deterministic_requested=deterministic, torch_available=False)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        report.caveats.append("torch not installed; only random and numpy were seeded")
        return report

    report.torch_available = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not deterministic:
        report.caveats.append("deterministic mode disabled by config")
        return report

    # cuBLAS needs this set before the first CUDA context to be reproducible in matmul.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import torch.backends.cudnn as cudnn

        cudnn.deterministic = True
        cudnn.benchmark = False
        report.cudnn_deterministic = True
    except Exception as exc:  # noqa: BLE001
        report.caveats.append(f"could not configure cuDNN determinism: {exc}")

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        report.torch_deterministic_algorithms = True
        report.caveats.append(
            "torch.use_deterministic_algorithms(warn_only=True): ops without a "
            "deterministic kernel will warn and run non-deterministically"
        )
    except Exception as exc:  # noqa: BLE001
        report.caveats.append(f"could not enable deterministic algorithms: {exc}")

    return report


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``.

    Without this, each worker process reseeds numpy from entropy and the augmentation /
    clip-sampling stream stops being reproducible even with a fixed global seed.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = ["SeedReport", "set_seed", "seed_worker"]
