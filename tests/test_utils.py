"""Tests for config composition, seeding and run provenance.

These are the machinery behind Hard Rule 4 -- if config composition or provenance capture
is wrong, every run directory lies about what produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.utils import config as config_utils
from src.utils.logging import MetricLogger
from src.utils.run import create_run, environment_provenance, git_provenance
from src.utils.seeding import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================================
# Config
# ======================================================================================


@pytest.fixture
def config_tree(tmp_path: Path) -> Path:
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "toy.yaml").write_text(
        yaml.safe_dump({"name": "toy", "fps": 30, "feature_dim": 8}), encoding="utf-8"
    )
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "defaults": {"dataset": "toy"},
                "seed": 1337,
                "device": "cpu",
                "paths": {"data_root": "data", "runs_root": "runs"},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_defaults_group_is_composed(config_tree: Path) -> None:
    cfg = config_utils.load_config(config_tree / "base.yaml")
    assert cfg.dataset.name == "toy"
    assert cfg.dataset.fps == 30
    assert cfg.seed == 1337
    assert "defaults" not in cfg


def test_cli_overrides_win(config_tree: Path) -> None:
    cfg = config_utils.load_config(
        config_tree / "base.yaml", ["seed=7", "dataset.fps=25", "device=cuda"]
    )
    assert cfg.seed == 7
    assert cfg.dataset.fps == 25
    assert cfg.device == "cuda"
    assert cfg.dataset.name == "toy"          # untouched keys survive


def test_missing_group_fails_loudly(config_tree: Path) -> None:
    (config_tree / "base.yaml").write_text(
        yaml.safe_dump({"defaults": {"dataset": "nonexistent"}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="config group"):
        config_utils.load_config(config_tree / "base.yaml")


def test_missing_config_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        config_utils.load_config(tmp_path / "absent.yaml")


def test_require_rejects_unset(config_tree: Path) -> None:
    cfg = config_utils.load_config(config_tree / "base.yaml")
    assert config_utils.require(cfg, "paths.data_root") == "data"
    with pytest.raises(KeyError, match="not set"):
        config_utils.require(cfg, "model.hidden_dim")


def test_save_config_round_trips(config_tree: Path, tmp_path: Path) -> None:
    cfg = config_utils.load_config(config_tree / "base.yaml", ["seed=99"])
    out = tmp_path / "saved" / "config.yaml"
    config_utils.save_config(cfg, out)
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["seed"] == 99


def test_group_override_selects_a_group_not_a_string(config_tree: Path) -> None:
    """`--set dataset=other` must load the group file, not set cfg.dataset to a string."""
    (config_tree / "dataset" / "other.yaml").write_text(
        yaml.safe_dump({"name": "other", "fps": 60}), encoding="utf-8"
    )
    cfg = config_utils.load_config(config_tree / "base.yaml", ["dataset=other", "seed=3"])
    assert cfg.dataset.name == "other"
    assert cfg.dataset.fps == 60
    assert cfg.seed == 3


def test_group_override_still_allows_dotted_keys(config_tree: Path) -> None:
    """A dotted key under a group name is a value override, not a group selection."""
    cfg = config_utils.load_config(config_tree / "base.yaml", ["dataset.fps=12"])
    assert cfg.dataset.name == "toy"
    assert cfg.dataset.fps == 12


def test_shipped_configs_load() -> None:
    """The real configs must compose, not just the synthetic ones built in tests."""
    cfg = config_utils.load_config(REPO_ROOT / "configs" / "base.yaml")
    assert cfg.dataset.name == "synthetic"
    assert cfg.dataset.gesture_classes == ["wave", "point", "fist"]
    assert cfg.seed == 1337
    assert cfg.device == "cpu"

    ipn = config_utils.load_config(
        REPO_ROOT / "configs" / "base.yaml", ["dataset=ipn_hand"]
    )
    assert ipn.dataset.name == "ipn_hand"
    assert ipn.dataset.fps == 30.0


def test_ipn_hand_config_declares_published_targets() -> None:
    """Stage 1's gate lives in config, not as literals in Python."""
    cfg = config_utils.load_config(
        REPO_ROOT / "configs" / "base.yaml", [], config_dir=REPO_ROOT / "configs"
    )
    ipn = config_utils.load_config(REPO_ROOT / "configs" / "dataset" / "ipn_hand.yaml")
    assert len(ipn.gesture_classes) == 13
    assert "none" not in ipn.gesture_classes
    assert ipn.expected.num_videos == 200
    assert ipn.expected.num_subjects == 50
    assert ipn.expected.num_gesture_instances == 4218
    assert ipn.expected.num_none_instances == 1431


# ======================================================================================
# Seeding
# ======================================================================================


def test_seed_makes_numpy_reproducible() -> None:
    set_seed(123)
    first = np.random.rand(10)
    set_seed(123)
    assert np.array_equal(first, np.random.rand(10))


def test_different_seeds_differ() -> None:
    set_seed(1)
    first = np.random.rand(10)
    set_seed(2)
    assert not np.array_equal(first, np.random.rand(10))


def test_seed_report_records_caveats() -> None:
    """Determinism on GPU is best-effort; the report must admit that rather than imply
    a precision that is not there."""
    report = set_seed(42, deterministic=True)
    payload = report.as_dict()
    assert payload["seed"] == 42
    assert payload["deterministic_requested"] is True
    assert isinstance(payload["caveats"], list)
    if payload["torch_available"]:
        assert payload["cudnn_deterministic"] is True


def test_non_deterministic_mode_is_recorded() -> None:
    report = set_seed(42, deterministic=False)
    assert any("deterministic mode disabled" in c for c in report.caveats)


# ======================================================================================
# Run directory / provenance
# ======================================================================================


def test_create_run_captures_provenance(config_tree: Path, tmp_path: Path) -> None:
    cfg = config_utils.load_config(config_tree / "base.yaml", ["seed=5"])
    seed_report = set_seed(5)
    run = create_run(cfg, name="unit-test", runs_root=tmp_path / "runs", seed_report=seed_report)

    assert (run.path / "config.yaml").is_file()
    provenance = json.loads((run.path / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["run_name"] == "unit-test"
    assert provenance["seeding"]["seed"] == 5
    assert "python" in provenance["environment"]
    assert "git" in provenance
    assert yaml.safe_load((run.path / "config.yaml").read_text(encoding="utf-8"))["seed"] == 5


def test_run_subdirectories_are_created_on_demand(config_tree: Path, tmp_path: Path) -> None:
    cfg = config_utils.load_config(config_tree / "base.yaml")
    run = create_run(cfg, name="dirs", runs_root=tmp_path / "runs")
    for path in (run.tb_dir, run.metrics_dir, run.figures_dir, run.checkpoints_dir):
        assert path.is_dir()


def test_run_writes_metrics(config_tree: Path, tmp_path: Path) -> None:
    cfg = config_utils.load_config(config_tree / "base.yaml")
    run = create_run(cfg, name="metrics", runs_root=tmp_path / "runs")
    path = run.write_metrics("val", {"per_frame_map": 0.42, "false_alarms_per_hour": 3.1})
    assert json.loads(path.read_text(encoding="utf-8"))["per_frame_map"] == 0.42


def test_git_provenance_reports_dirty_state() -> None:
    """`dirty` matters more than the sha: a run from an uncommitted tree is not
    reproducible from the sha alone."""
    provenance = git_provenance(REPO_ROOT)
    assert set(provenance) == {"commit", "branch", "dirty", "dirty_files"}
    if provenance["commit"] is not None:
        assert len(provenance["commit"]) == 40
        assert isinstance(provenance["dirty"], bool)


def test_environment_provenance_has_python() -> None:
    info = environment_provenance()
    assert "python" in info
    assert "platform" in info


# ======================================================================================
# Metric logging -- Hard Rule 4
# ======================================================================================


def test_metric_logger_none_backend_still_mirrors_to_jsonl(tmp_path: Path) -> None:
    """Event files are awkward to read back; the raw numbers must survive in a form a
    plotting script can consume months later."""
    with MetricLogger(backend="none", run_dir=tmp_path) as logger:
        logger.log_scalars({"loss": 1.5, "acc": 0.2}, step=0)
        logger.log_scalar("loss", 1.0, step=1)

    lines = (tmp_path / "metrics" / "scalars.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 0, "loss": 1.5, "acc": 0.2}
    assert json.loads(lines[1]) == {"step": 1, "loss": 1.0}


def test_metric_logger_tensorboard_writes_events(tmp_path: Path) -> None:
    with MetricLogger(backend="tensorboard", run_dir=tmp_path) as logger:
        logger.log_scalar("loss", 0.5, step=0)
    assert any((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_metric_logger_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown logging backend"):
        MetricLogger(backend="mlflow", run_dir=tmp_path)


def test_metric_logger_skips_none_values(tmp_path: Path) -> None:
    with MetricLogger(backend="none", run_dir=tmp_path) as logger:
        logger.log_scalars({"a": None, "b": 1.0}, step=0)
    line = json.loads((tmp_path / "metrics" / "scalars.jsonl").read_text(encoding="utf-8"))
    assert line == {"step": 0, "b": 1.0}
