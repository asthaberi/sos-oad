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
from types import SimpleNamespace

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


# ----------------------------------------------------------------------------------
# W&B backend
#
# These run the real wandb code path with mode="disabled", which needs no network and no
# login. Mocking wandb.init would only assert that we call a function we wrote the call
# for; letting the library validate its own arguments is the part worth testing.
# ----------------------------------------------------------------------------------


def test_wandb_backend_still_mirrors_scalars_to_the_run_directory(tmp_path: Path) -> None:
    """The mirror is the record. Rule 4 does not rely on a hosted service being reachable."""
    with MetricLogger(backend="wandb", run_dir=tmp_path, mode="disabled") as logger:
        logger.log_scalars({"loss": 1.5, "map": 0.25}, step=0)
        logger.log_scalar("loss", 1.25, step=1)

    lines = (tmp_path / "metrics" / "scalars.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 0, "loss": 1.5, "map": 0.25}
    assert json.loads(lines[1]) == {"step": 1, "loss": 1.25}


def test_wandb_mode_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="online, offline or disabled"):
        MetricLogger(backend="wandb", run_dir=tmp_path, mode="sometimes")


def test_from_config_reads_the_tracker_settings(tmp_path: Path) -> None:
    """Nothing about the tracker may be hardcoded; it all arrives from config."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "seed": 1337,
            "logging": {
                "backend": "wandb",
                "project": "thesis-project",
                "entity": None,
                "mode": "disabled",
                "tags": ["stage4"],
            },
        }
    )
    run = SimpleNamespace(path=tmp_path, name="b1-baseline", provenance={})
    with MetricLogger.from_config(cfg, run) as logger:
        assert logger.backend == "wandb"
        logger.log_scalar("loss", 0.5, step=0)
    assert (tmp_path / "metrics" / "scalars.jsonl").is_file()


def test_from_config_defaults_to_wandb_when_logging_is_absent(tmp_path: Path) -> None:
    """No logging section still resolves the declared default rather than losing runs."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"seed": 1, "logging": {"mode": "disabled"}})
    logger = MetricLogger.from_config(cfg, SimpleNamespace(path=tmp_path, name="x", provenance={}))
    assert logger.backend == "wandb"
    logger.close()


def test_missing_api_key_degrades_to_offline_instead_of_killing_the_run(
    tmp_path: Path, monkeypatch
) -> None:
    """An unauthenticated tracker must not take a twelve-hour training run with it.

    W&B raises when no API key is configured. Failing there would lose a GPU session to a
    missing credential, and the scalars are mirrored to the run directory regardless, so
    there is nothing to gain by failing. It degrades to offline and says so.
    """
    import wandb

    real_init = wandb.init
    calls: list[str] = []

    def flaky(**kwargs):
        calls.append(kwargs["mode"])
        if kwargs["mode"] == "online":
            raise wandb.errors.UsageError("No API key configured.")
        return real_init(**kwargs)

    monkeypatch.setattr(wandb, "init", flaky)

    # get_logger sets propagate=False so scripts do not double-print, which also means
    # caplog's root handler never sees these records. Attach one directly.
    import logging as _logging

    records: list[_logging.LogRecord] = []

    class Capture(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            records.append(record)

    handler = Capture()
    _logging.getLogger("metrics").addHandler(handler)
    try:
        MetricLogger(backend="wandb", run_dir=tmp_path, mode="online").close()
    finally:
        _logging.getLogger("metrics").removeHandler(handler)

    assert calls == ["online", "offline"]
    assert any("falling back to offline" in r.getMessage().lower() for r in records)


def test_an_explicitly_offline_failure_is_not_swallowed(tmp_path: Path, monkeypatch) -> None:
    """The fallback exists for the online case only. If offline itself fails, that is a
    real problem with the run directory or the install, and hiding it would be worse."""
    import wandb

    def always_fails(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(wandb, "init", always_fails)
    with pytest.raises(RuntimeError, match="disk full"):
        MetricLogger(backend="wandb", run_dir=tmp_path, mode="offline")


def test_provenance_is_carried_into_the_tracker_config(tmp_path: Path, monkeypatch) -> None:
    """A dashboard that cannot answer 'which commit produced this?' is a pretty chart.

    Rule 4 wants config, seed and git SHA for every run, so the commit and seed are pushed
    into W&B's own config rather than living only in provenance.json.
    """
    captured: dict = {}

    import wandb

    real_init = wandb.init

    def spy(**kwargs):
        captured.update(kwargs)
        return real_init(**kwargs)

    monkeypatch.setattr(wandb, "init", spy)

    provenance = {
        "git": {"commit": "abc1234", "branch": "main", "dirty": False},
        "seeding": {"seed": 1337},
    }
    MetricLogger(
        backend="wandb",
        run_dir=tmp_path,
        mode="disabled",
        run_name="fold0",
        group="m1-losgo",
        tags=["stage5"],
        config={"model": "m1"},
        provenance=provenance,
    ).close()

    assert captured["config"]["git_commit"] == "abc1234"
    assert captured["config"]["git_branch"] == "main"
    assert captured["config"]["seed"] == 1337
    assert captured["config"]["model"] == "m1"
    assert captured["group"] == "m1-losgo"
    assert captured["tags"] == ["stage5"]


def test_losgo_folds_can_share_a_group(tmp_path: Path) -> None:
    """Five folds of one experiment belong together in the dashboard, not as five
    unrelated curves. The group is per-run, so it is passed rather than configured."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"logging": {"backend": "wandb", "mode": "disabled"}})
    for fold in range(3):
        run = SimpleNamespace(path=tmp_path / f"f{fold}", name=f"fold{fold}", provenance={})
        with MetricLogger.from_config(cfg, run, group="m1-losgo") as logger:
            logger.log_scalar("val/map", 0.4 + fold / 100, step=0)
    for fold in range(3):
        assert (tmp_path / f"f{fold}" / "metrics" / "scalars.jsonl").is_file()
