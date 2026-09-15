"""Run directories and provenance capture.

Hard Rule 4: a result with no run directory behind it does not exist.

Every run gets a timestamped directory containing the resolved config, the git commit it
was produced at (with a dirty flag), the seeding report, and the environment. The point is
viva-defensibility: for any number in the thesis there is exactly one directory that says
which code, which config, which seed and which split produced it.

    runs/<timestamp>-<name>/
      config.yaml       resolved config, after group composition and CLI overrides
      provenance.json   git sha, dirty flag, seeds, platform, library versions
      metrics/          metric dumps written by the eval harness
      tb/               TensorBoard event files
      <checkpoints, figures, ...>
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.utils import config as config_utils


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git absent or not a repo; provenance degrades gracefully
        return None


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Commit, branch and dirty state.

    ``dirty`` matters more than the sha: a run made from an uncommitted working tree is
    not reproducible from the sha alone, and the run log should admit that rather than
    record a commit that does not contain the code that ran.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    status = _git("status", "--porcelain", cwd=root)
    return {
        "commit": _git("rev-parse", "HEAD", cwd=root),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def environment_provenance() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpus"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info


@dataclass
class RunDirectory:
    """A single run's output directory."""

    path: Path
    name: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def tb_dir(self) -> Path:
        return self._sub("tb")

    @property
    def metrics_dir(self) -> Path:
        return self._sub("metrics")

    @property
    def figures_dir(self) -> Path:
        return self._sub("figures")

    @property
    def checkpoints_dir(self) -> Path:
        return self._sub("checkpoints")

    def _sub(self, name: str) -> Path:
        sub = self.path / name
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def write_metrics(self, name: str, metrics: dict[str, Any]) -> Path:
        target = self.metrics_dir / f"{name}.json"
        target.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        return target

    def write_text(self, name: str, text: str) -> Path:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target


def create_run(
    cfg: DictConfig,
    name: str | None = None,
    *,
    runs_root: Path | str | None = None,
    seed_report: Any = None,
) -> RunDirectory:
    """Create a timestamped run directory and capture everything needed to defend it."""
    root = Path(runs_root or config_utils.require(cfg, "paths.runs_root"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = name or str(cfg.get("run_name") or "run")
    path = root / f"{stamp}-{run_name}"
    path.mkdir(parents=True, exist_ok=True)

    run = RunDirectory(path=path, name=run_name)

    config_utils.save_config(cfg, path / "config.yaml")

    run.provenance = {
        "run_name": run_name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git": git_provenance(),
        "environment": environment_provenance(),
        "seeding": seed_report.as_dict() if seed_report is not None else None,
    }
    run.write_json("provenance.json", run.provenance)

    return run


__all__ = [
    "RunDirectory",
    "create_run",
    "git_provenance",
    "environment_provenance",
]
