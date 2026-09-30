"""Console and metric logging.

Two separate concerns behind one small surface:

* ``get_logger`` -- ordinary stdlib console logging, so scripts print consistently.
* ``MetricLogger`` -- the Hard Rule 4 sink. W&B by default, TensorBoard as an
  alternative, and a no-op backend so that unit tests and quick CLI runs do not litter the
  disk with event files.

The tracker is a dashboard, not the record
------------------------------------------
Every backend *also* mirrors scalars to ``metrics/scalars.jsonl`` in the run directory,
and that mirror is the authoritative copy. Rule 4 says a result with no run directory
behind it does not exist, and that wording is deliberate: a hosted tracker can be deleted,
rate-limited, moved between accounts, or simply be unreachable from the machine marking
the thesis. The run directory holds the resolved config, the git commit, the seed report
and the raw scalars, and it survives all of that. W&B makes the numbers pleasant to look
at while the work is happening; it is not what the thesis rests on.

The backend is chosen by config (``logging.backend``), never by an import-time guess, and
nothing about the tracker -- project, entity, mode -- is hardcoded here.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
        backend: ``wandb`` | ``tensorboard`` | ``none``.
        run_dir: run directory; scalars are mirrored to ``metrics/scalars.jsonl``.
        run_name: display name for the experiment tracker.
        config: resolved config, recorded by the tracker where supported.
        project, entity, mode, group, tags: W&B settings, all supplied from config.
        provenance: git and seed information, merged into the tracker's own config so a
            run in the dashboard can be traced to a commit without leaving the page.
    """

    def __init__(
        self,
        backend: str = "wandb",
        run_dir: Path | str | None = None,
        *,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        project: str = "sos-oad",
        entity: str | None = None,
        mode: str = "online",
        group: str | None = None,
        tags: Sequence[str] | None = None,
        provenance: Mapping[str, Any] | None = None,
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
            self._init_wandb(
                run_name,
                config,
                project=project,
                entity=entity,
                mode=mode,
                group=group,
                tags=list(tags or []),
                provenance=provenance,
            )
        elif self.backend != "none":
            raise ValueError(
                f"unknown logging backend {backend!r}; expected wandb, tensorboard or none"
            )

    @classmethod
    def from_config(
        cls, cfg: Any, run: Any = None, *, group: str | None = None
    ) -> "MetricLogger":
        """Build from a resolved config and a :class:`~src.utils.run.RunDirectory`.

        ``group`` is how the LOSGO folds stay legible: the five folds of one experiment
        share a group, so the dashboard aggregates them rather than showing five unrelated
        curves. It is passed here rather than written into the YAML because it varies per
        run while the YAML does not.
        """
        from omegaconf import OmegaConf

        section = cfg.get("logging", {}) if hasattr(cfg, "get") else {}
        resolved = (
            OmegaConf.to_container(cfg, resolve=True)
            if OmegaConf.is_config(cfg)
            else dict(cfg)
        )

        def setting(key: str, default: Any = None) -> Any:
            value = section.get(key, default) if hasattr(section, "get") else default
            return default if value is None else value

        return cls(
            backend=str(setting("backend", "wandb")),
            run_dir=getattr(run, "path", None),
            run_name=getattr(run, "name", None),
            config=resolved,
            project=str(setting("project", "sos-oad")),
            entity=setting("entity", None),
            mode=str(setting("mode", "online")),
            group=group,
            tags=list(setting("tags", []) or []),
            provenance=getattr(run, "provenance", None),
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

    def _init_wandb(
        self,
        run_name: str | None,
        config: Mapping[str, Any] | None,
        *,
        project: str,
        entity: str | None,
        mode: str,
        group: str | None,
        tags: list[str],
        provenance: Mapping[str, Any] | None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "logging.backend=wandb requires wandb: pip install wandb"
            ) from exc
        if mode not in ("online", "offline", "disabled"):
            raise ValueError(
                f"logging.mode must be online, offline or disabled, got {mode!r}"
            )

        # Commit, dirty flag and seed go into the tracker's own config, so a run in the
        # dashboard answers "which code produced this?" without leaving the page. Rule 4
        # asks for exactly that; a dashboard that cannot answer it is a pretty chart.
        merged: dict[str, Any] = dict(config or {})
        if provenance:
            git_info = dict(provenance.get("git") or {})
            merged["git_commit"] = git_info.get("commit")
            merged["git_branch"] = git_info.get("branch")
            merged["git_dirty"] = git_info.get("dirty")
            seeding = provenance.get("seeding")
            if isinstance(seeding, Mapping):
                merged["seed"] = seeding.get("seed")

        self._wandb = wandb
        settings = dict(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags or None,
            config=merged,
            dir=str(self.run_dir) if self.run_dir else None,
        )
        try:
            wandb.init(mode=mode, **settings)
        except Exception as exc:  # noqa: BLE001 - see below
            # An unauthenticated or unreachable tracker must not take a training run with
            # it. W&B raises here when no API key is configured, which on a twelve-hour
            # GPU session would mean losing the run to a missing credential -- and the
            # scalars are mirrored to the run directory regardless, so there is nothing to
            # gain by failing. Degrade to offline, say so loudly, and carry on; the run can
            # be pushed later with `wandb sync`.
            if mode != "online":
                raise
            self.log.warning(
                "W&B could not start in online mode (%s: %s). Falling back to offline -- "
                "metrics are still written to the run directory, and the buffered run can "
                "be uploaded later with `wandb sync`.",
                type(exc).__name__,
                exc,
            )
            wandb.init(mode="offline", **settings)

        if merged.get("git_dirty"):
            # Worth saying out loud: the commit recorded beside these metrics does not
            # contain the code that produced them.
            self.log.warning(
                "run logged from a dirty working tree -- the commit recorded in W&B does "
                "not contain the code that ran"
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
