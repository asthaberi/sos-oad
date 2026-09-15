"""Configuration loading.

Config-driven means config-driven: nothing that affects a reported number is allowed to
live as a literal in Python. Seeds, paths, thresholds, window lengths, class counts,
frame rates -- all of it comes from YAML, and the resolved config is written into the run
directory so any figure in the thesis can be traced back to the exact settings that made
it.

This is a small ``defaults:``-style composer over OmegaConf rather than full Hydra. It
gives the useful part of Hydra -- compose a base config with named group files, override
from the command line -- in about fifty lines, without Hydra taking ownership of the
process's working directory and logging, which fights with the run-directory scheme in
``src/utils/run.py``.

A config file may declare::

    defaults:
      dataset: ipn_hand      # loads configs/dataset/ipn_hand.yaml into cfg.dataset

Command-line overrides use OmegaConf dotlist syntax::

    python scripts/foo.py --config configs/base.yaml --set seed=7 dataset.fps=25
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

#: Sentinel for a value that must be supplied by a group file or an override.
MISSING = "???"


def load_config(
    config_path: Path | str,
    overrides: Sequence[str] | None = None,
    *,
    config_dir: Path | str | None = None,
) -> DictConfig:
    """Compose a config from a base YAML, its ``defaults:`` groups, and CLI overrides.

    Args:
        config_path: path to the base YAML.
        overrides: OmegaConf dotlist entries, e.g. ``["seed=7", "dataset.fps=25"]``.
        config_dir: root for resolving ``defaults:`` groups. Defaults to the base
            config's own directory.

    Returns:
        The resolved config, with ``???`` still present for anything left unspecified so
        that a missing required value fails loudly at access rather than silently
        defaulting.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    root = Path(config_dir) if config_dir is not None else config_path.parent
    base = OmegaConf.load(config_path)
    if not isinstance(base, DictConfig):
        raise TypeError(f"{config_path} must contain a mapping at the top level")

    defaults = base.pop("defaults", None)
    groups: dict[str, Any] = (
        dict(OmegaConf.to_container(defaults)) if defaults is not None else {}
    )

    # A bare `--set dataset=ipn_hand` selects a config *group*, not a scalar. Pull those
    # out before resolution, otherwise cfg.dataset would end up as the string
    # "ipn_hand" instead of the loaded group file -- a silent, confusing failure.
    value_overrides: list[str] = []
    for item in overrides or []:
        key, sep, value = str(item).partition("=")
        if sep and key.strip() in groups:
            groups[key.strip()] = value.strip()
        else:
            value_overrides.append(item)

    merged = OmegaConf.create({})
    for group, choice in groups.items():
        if choice is None:
            continue
        group_path = root / str(group) / f"{choice}.yaml"
        if not group_path.is_file():
            raise FileNotFoundError(
                f"config group {group}={choice} not found at {group_path}"
            )
        loaded = OmegaConf.load(group_path)
        merged = OmegaConf.merge(merged, OmegaConf.create({group: loaded}))

    merged = OmegaConf.merge(merged, base)

    if value_overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(value_overrides))

    return merged  # type: ignore[return-value]


def require(cfg: DictConfig, key: str) -> Any:
    """Fetch a config value, failing clearly if it is missing or still ``???``."""
    value = OmegaConf.select(cfg, key, default=None)
    if value is None or value == MISSING:
        raise KeyError(
            f"required config key {key!r} is not set. Supply it in the YAML or with "
            f"--set {key}=<value>"
        )
    return value


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def add_config_args(parser: argparse.ArgumentParser, default: str = "configs/base.yaml") -> None:
    """Attach the standard ``--config`` / ``--set`` pair to a script's parser."""
    parser.add_argument(
        "--config", type=Path, default=Path(default), help="path to the base YAML config"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides in dotlist form, e.g. --set seed=7 dataset.fps=25",
    )


def config_from_args(args: argparse.Namespace) -> DictConfig:
    return load_config(args.config, args.overrides)


__all__ = [
    "MISSING",
    "load_config",
    "require",
    "to_dict",
    "save_config",
    "add_config_args",
    "config_from_args",
]
