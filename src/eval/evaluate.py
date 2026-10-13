"""One call that scores a method on one split with the complete metric set.

Every script that reports a number goes through :func:`evaluate_split`, so B0, B1 and M1
can only differ in their scores, never in how those scores are counted.

The full set reported every time (CLAUDE.md, Stage 7):

* frame level: mAP, mcAP, balanced accuracy, per-class AP and cAP;
* point level: start-point mAP at 1-10 s tolerance (ODAS);
* alarm level, at the operating threshold: event precision / recall / F1, false alarms per
  hour of non-gesture video, median and p95 onset-to-alarm delay;
* the delay vs false-alarm curve, swept over the alarm threshold;
* efficiency: frames per second of the method on cached features, peak GPU memory.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.data.canonical import NONE_INDEX
from src.decision.onset import onset_alarms, threshold_grid
from src.eval import metrics as M


def evaluate_split(
    scores: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    fps: float,
    class_names: Sequence[str],
    signals: Sequence[int] | None,
    threshold: float,
    match_pre_s: float,
    match_post_s: float,
    point_tolerances_s: Sequence[float],
    sweep: Sequence[float] | None = None,
    efficiency: Mapping[str, float | None] | None = None,
) -> dict:
    """Score per-video ``[T, C]`` probabilities against per-video ``[T]`` labels."""
    videos = sorted(scores)
    missing = [v for v in videos if v not in labels]
    if missing:
        raise KeyError(f"no labels for {missing[:3]}")
    for v in videos:
        if len(scores[v]) != len(labels[v]):
            raise ValueError(f"{v}: {len(scores[v])} score rows for {len(labels[v])} labels")

    num_classes = int(next(iter(scores.values())).shape[1])
    sig = M.signal_classes(num_classes, signals)
    all_scores = np.concatenate([scores[v] for v in videos])
    all_labels = np.concatenate([labels[v] for v in videos])
    instances = [i for v in videos for i in M.instances_from_labels(v, labels[v]) if i.cls in sig]
    background = M.frames_of_class({v: labels[v] for v in videos}, NONE_INDEX)

    frame = M.frame_metrics(all_scores, all_labels, class_names=class_names, signals=sig)

    starts = [d for v in videos for d in M.predicted_starts(v, scores[v]) if d.cls in sig]
    point = M.point_level_map(
        starts, instances, fps=fps, tolerances_s=point_tolerances_s, signals=sig
    )

    def alarms_at(th: float) -> list[M.Detection]:
        return [d for v in videos for d in onset_alarms(v, scores[v], th) if d.cls in sig]

    kwargs = dict(fps=fps, background_frames=background, pre_s=match_pre_s, post_s=match_post_s)
    alarm = M.alarm_metrics(alarms_at(threshold), instances, **kwargs)
    curve = []
    for th in threshold_grid(sweep):
        row = M.alarm_metrics(alarms_at(th), instances, **kwargs)
        curve.append(
            {
                "threshold": th,
                "false_alarms_per_hour": row["false_alarms_per_hour"],
                "delay_median_s": row["delay_median_s"],
                "delay_p95_s": row["delay_p95_s"],
                "event_recall": row["event_recall"],
                "event_precision": row["event_precision"],
                "event_f1": row["event_f1"],
            }
        )

    return {
        "videos": len(videos),
        "signal_classes": [class_names[c] for c in sig],
        "num_instances": len(instances),
        "frame": frame,
        "point": point,
        "alarm": {**alarm, "threshold": threshold, "match_pre_s": match_pre_s, "match_post_s": match_post_s},
        "curve": curve,
        "efficiency": dict(efficiency or {}),
    }


def headline(result: dict) -> dict[str, float]:
    """The scalars logged to the tracker and printed at the gate."""
    a, f, p = result["alarm"], result["frame"], result["point"]
    return {
        "mAP": f["mAP"],
        "mcAP": f["mcAP"],
        "balanced_accuracy": f["balanced_accuracy"],
        "point_mAP_mean": p["mean"],
        "event_precision": a["event_precision"],
        "event_recall": a["event_recall"],
        "event_f1": a["event_f1"],
        "false_alarms_per_hour": a["false_alarms_per_hour"],
        "delay_median_s": a["delay_median_s"],
        "delay_p95_s": a["delay_p95_s"],
    }


#: Hard Rule 2. On a subject-disjoint online detection task a value above this is treated
#: as a symptom of leakage, not as a result.
TRIPWIRE = 0.97

_RATIO_METRICS = (
    "mAP",
    "mcAP",
    "balanced_accuracy",
    "point_mAP_mean",
    "event_precision",
    "event_recall",
    "event_f1",
)


def tripwire(result: dict) -> list[str]:
    """Names of headline ratio metrics above :data:`TRIPWIRE`. Empty means no alarm.

    Scripts refuse to call such a run a result: they print the offending metrics and mark
    the run directory, and the number is not reported until the leak hunt is done.

    The rule is applied literally, to every ratio, with no exemptions. One known non-leak
    cause is worth knowing before the hunt starts: ``event_recall`` alone exceeds the line
    when alarms flood -- random scores fire almost every frame and so "catch" every gesture
    (``tests/test_eval.py`` shows it). The false-alarm rate in the same result tells the two
    apart; a leak raises precision and recall together.
    """
    values = headline(result)
    return [k for k in _RATIO_METRICS if not np.isnan(values[k]) and values[k] > TRIPWIRE]


__all__ = ["TRIPWIRE", "evaluate_split", "headline", "tripwire"]
