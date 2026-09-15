"""Console and metric logging.

Two separate concerns behind one small surface:

* ``get_logger`` -- ordinary stdlib console logging, so scripts print consistently.
* ``MetricLogger`` -- the Hard Rule 4 sink. TensorBoard by default, W&B optional, and a
  no-op backend so that unit tests and quick CLI runs do not litter the disk with event
  files. Every backend also mirrors scalars to a JSONL file in the run directory, because
  event files are awkward to read back and the thesis needs the raw numbers in a form
  that a plotting script can consume months later.

The backend is chosen by config (``logging.backend``), never by an import-time guess.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "sos-oad", level: int = logging.INFO) -> logging.Logger:
    """A console logger that does not double-print when called twice."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


class MetricLogger:
    """Backend-agnostic scalar sink.

    Args:
        backend: ``tensorboard`` | ``wandb`` | ``none``.
        run_dir: run directory; scalars are mirrored to ``metrics/scalars.jsonl``.
        run_name: display name for the experiment tracker.
        config: resolved config, recorded by the tracker where supported.
    """

    def __init__(
        self,
        backend: str = "tensorboard",
        run_dir: Path | str | None = None,
        *,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = (backend or "none").lower()
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.log = get_logger("metrics")
        self._writer: Any = None
        self._wandb: Any = None
        self._jsonl: Path | None = None

        if self.run_dir is not None:
            metrics_dir = self.run_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl = metrics_dir / "scalars.jsonl"

        if self.backend == "tensorboard":
            self._init_tensorboard()
        elif self.backend == "wandb":
            self._init_wandb(run_name, config)
        elif self.backend != "none":
            raise ValueError(
                f"unknown logging backend {backend!r}; expected tensorboard, wandb or none"
            )

    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "logging.backend=tensorboard requires tensorboard: pip install -r requirements.txt"
            ) from exc
        tb_dir = (self.run_dir / "tb") if self.run_dir else Path("runs/tb")
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(tb_dir))

    def _init_wandb(self, run_name: str | None, config: Mapping[str, Any] | None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "logging.backend=wandb requires wandb: pip install wandb"
            ) from exc
        self._wandb = wandb
        wandb.init(
            project="sos-oad",
            name=run_name,
            config=dict(config or {}),
            dir=str(self.run_dir) if self.run_dir else None,
        )

    # -- writing ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log_scalars({tag: value}, step)

    def log_scalars(self, values: Mapping[str, float], step: int) -> None:
        clean = {k: float(v) for k, v in values.items() if v is not None}
        if not clean:
            return

        if self._writer is not None:
            for tag, value in clean.items():
                self._writer.add_scalar(tag, value, step)
        if self._wandb is not None:
            self._wandb.log(dict(clean), step=step)

        if self._jsonl is not None:
            with self._jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": step, **clean}) + "\n")

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        if self._writer is not None:
            self._writer.add_text(tag, text, step)
        if self._wandb is not None:
            self._wandb.log({tag: self._wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def log_figure(self, tag: str, figure: Any, step: int = 0) -> None:
        if self._writer is not None:
            self._writer.add_figure(tag, figure, step)
        if self._wandb is not None:
            self._wandb.log({tag: self._wandb.Image(figure)}, step=step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["get_logger", "MetricLogger"]
