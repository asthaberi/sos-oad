"""Stage 4: train a baseline and score it with the complete metric set.

    # on the GPU server, the real thing
    python scripts/train_baseline.py --set dataset=ipn_hand split=ipn_official model=b0 device=cuda
    python scripts/train_baseline.py --set dataset=ipn_hand split=ipn_official model=b1 device=cuda

    # anywhere, on the synthetic fixture: proves the pipeline, never a reportable number
    python scripts/train_baseline.py --set model=b1 logging.backend=none model.epochs=2

Trains on the ``train`` split, picks the epoch with the best validation mcAP, then scores
``eval.split`` (validation unless deliberately changed at final reporting). Everything goes
into one run directory: config, provenance, per-epoch history, the checkpoint with its
standardisation statistics, the per-frame validation scores (so Stage 6 can study the
decision layer without retraining), and the full metric dump.

Hard Rule 2 is enforced here: if any headline ratio exceeds 0.97, the run is marked and the
script says so loudly. The number is not a result until the leak hunt is done.
"""

from __future__ import annotations

import argparse
import dataclasses

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import canonical as C  # noqa: E402
from src.data import splits as S  # noqa: E402
from src.data.datasets import InputSpec, fit_standardiser, load_split  # noqa: E402
from src.eval.evaluate import TRIPWIRE, evaluate_split, headline, tripwire  # noqa: E402
from src.models.train import TrainSettings, predict_split, train  # noqa: E402
from src.utils import config as config_utils  # noqa: E402
from src.utils.logging import MetricLogger, get_logger  # noqa: E402
from src.utils.run import create_run  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

log = get_logger("stage4")

#: Keys every result must carry for the Stage 4 gate: "both baselines produce the complete
#: metric set".
REQUIRED = {
    "frame": ("mAP", "mcAP", "balanced_accuracy", "per_class_AP", "per_class_cAP"),
    "point": ("by_tolerance", "mean"),
    "alarm": (
        "event_precision",
        "event_recall",
        "event_f1",
        "false_alarms_per_hour",
        "delay_median_s",
        "delay_p95_s",
    ),
    "efficiency": ("fps_model_on_cached_features", "peak_gpu_memory_mb"),
}


def missing_metrics(result: dict) -> list[str]:
    return [f"{group}.{key}" for group, keys in REQUIRED.items() for key in keys if key not in result.get(group, {})]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    config_utils.add_config_args(parser)
    parser.add_argument("--root", type=Path, default=None, help="canonical dataset directory")
    args = parser.parse_args()

    cfg = config_utils.load_config(args.config, args.overrides)
    root = args.root or Path(cfg.paths.data_root) / str(cfg.dataset.name)
    device = str(cfg.device)
    if device == "cuda" and not torch.cuda.is_available():
        log.error("device=cuda requested but CUDA is not available; refusing a silent CPU run")
        return 1

    # Training consumes features and poses, so it is the stage that holds the dataset to the
    # full ratchet: every array present, finite, the right shape, splits assigned.
    report = C.validate_dataset(
        root,
        check_arrays=bool(cfg.validation.check_arrays),
        require_poses=bool(cfg.validation.require_poses) and "poses" in list(cfg.inputs.streams),
        require_splits=bool(cfg.validation.require_splits),
        require_features=bool(cfg.validation.require_features),
    )
    if not report.ok:
        print(report.render())
        log.error("dataset failed validation; refusing to train on it")
        return 1

    seed_report = set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    dataset = C.CanonicalDataset(root)
    # Refuses a split file built from a different subject set (src/data/splits.py).
    S.load(root, str(cfg.split.name), dataset=dataset)

    eval_split = str(cfg.eval.split)
    if eval_split == "test":
        log.warning("eval.split=test: this spends the test set. Only at final reporting.")
    train_ids = dataset.video_ids_for_split("train")
    eval_ids = dataset.video_ids_for_split(eval_split)
    val_ids = dataset.video_ids_for_split("val")
    if not train_ids or not eval_ids or not val_ids:
        log.error("empty split: train=%d val=%d %s=%d", len(train_ids), len(val_ids), eval_split, len(eval_ids))
        return 1

    spec = InputSpec.from_config(cfg.inputs, cfg.dataset.pose_normalisation)
    settings = TrainSettings.from_config(cfg.model)
    signals = (
        None
        if cfg.eval.signals is None
        else [dataset.class_to_index[str(n)] for n in cfg.eval.signals]
    )

    run = create_run(cfg, name=f"stage4-{settings.kind}", seed_report=seed_report)
    logger = MetricLogger.from_config(cfg, run, group=f"stage4-{cfg.dataset.name}")
    log.info("run directory: %s", run.path)

    clock = time.time()
    train_data, train_raw = load_split(dataset, train_ids, spec, standardiser=None)
    standardiser = fit_standardiser(train_raw)
    del train_raw
    train_data.inputs = {v: standardiser.apply(a) for v, a in train_data.inputs.items()}
    val_data, _ = load_split(dataset, val_ids, spec, standardiser)
    eval_data = val_data if eval_split == "val" else load_split(dataset, eval_ids, spec, standardiser)[0]
    in_dim = next(iter(train_data.inputs.values())).shape[1]
    log.info(
        "inputs %s, dim %d; train %d videos / %d frames; %s %d videos / %d frames; loaded in %.0fs",
        list(spec.streams), in_dim, len(train_ids), train_data.num_frames,
        eval_split, len(eval_ids), eval_data.num_frames, time.time() - clock,
    )

    def on_epoch(epoch: int, row: dict) -> None:
        logger.log_scalars({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        log.info(
            "epoch %d  loss %.4f  val mAP %.4f  mcAP %.4f  bal.acc %.4f",
            epoch, row["train_loss"], row["val_mAP"], row["val_mcAP"], row["val_balanced_accuracy"],
        )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = train(
        settings,
        train_data,
        val_data,
        num_classes=dataset.num_classes,
        class_names=dataset.classes,
        fps=dataset.fps,
        device=device,
        seed=int(cfg.seed),
        on_epoch=on_epoch,
    )
    log.info("best epoch %d (val %s %.4f)", result.best_epoch, settings.selection_metric, result.best_score)

    # Time inference on cached features. Synchronised so GPU work is actually counted.
    if device == "cuda":
        torch.cuda.synchronize()
    clock = time.time()
    scores = predict_split(settings, result.model, eval_data, fps=dataset.fps, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - clock
    efficiency = {
        "fps_model_on_cached_features": eval_data.num_frames / max(seconds, 1e-9),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        ),
        "note": "excludes feature extraction; see the Stage 3 run directories for that cost",
    }

    metrics = evaluate_split(
        scores,
        eval_data.labels,
        fps=dataset.fps,
        class_names=dataset.classes,
        signals=signals,
        threshold=float(cfg.eval.threshold),
        match_pre_s=float(cfg.eval.match_pre_s),
        match_post_s=float(cfg.eval.match_post_s),
        point_tolerances_s=[float(t) for t in cfg.eval.point_tolerances_s],
        efficiency=efficiency,
    )
    metrics["model"] = {"kind": settings.kind, "best_epoch": result.best_epoch, "inputs": list(spec.streams), "in_dim": in_dim}
    metrics["split"] = eval_split
    missing = missing_metrics(metrics)
    tripped = tripwire(metrics)
    metrics["rule2_tripwire"] = tripped

    run.write_metrics(eval_split, metrics)
    run.write_json("metrics/history.json", {"history": result.history})
    np.savez_compressed(run.metrics_dir / f"{eval_split}_scores.npz", **scores)
    torch.save(
        {
            "model_state": result.model.state_dict(),
            "settings": dataclasses.asdict(settings),
            "standardiser": standardiser.state(),
            "input_spec": {"streams": list(spec.streams), "root": list(spec.pose_root_joints), "scale": list(spec.pose_scale_joints)},
            "classes": dataset.classes,
            "in_dim": in_dim,
            "best_epoch": result.best_epoch,
        },
        run.checkpoints_dir / "best.pt",
    )
    final = headline(metrics)
    logger.log_scalars({f"{eval_split}/{k}": v for k, v in final.items()}, step=settings.epochs + 1)
    logger.close()

    print(f"\n{settings.kind.upper()} on {eval_split} ({len(eval_ids)} videos, best epoch {result.best_epoch})")
    for key, value in final.items():
        print(f"  {key:<24} {value:.4f}")
    print(f"  {'fps (cached features)':<24} {efficiency['fps_model_on_cached_features']:.0f}")
    if tripped:
        run.write_text("RULE2_TRIPWIRE.txt", "Metrics above %.2f: %s\nNot a result until investigated.\n" % (TRIPWIRE, ", ".join(tripped)))
        print(f"\n*** RULE 2 TRIPWIRE: {', '.join(tripped)} above {TRIPWIRE}. Investigate before reporting. ***")
    print(f"\ncomplete metric set: {'YES' if not missing else 'NO, missing ' + ', '.join(missing)}")
    print(f"run directory: {run.path}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
