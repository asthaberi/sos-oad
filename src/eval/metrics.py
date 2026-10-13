"""Stage 7 metrics, built at Stage 4 so that every method is scored by one implementation.

Three families, each answering a different question:

* **Frame level** -- how well are the per-frame scores ranked? Per-class AP, mAP, calibrated
  AP (cAP, De Geest et al. 2016) and balanced accuracy. These are what the online action
  detection literature reports, so they keep the thesis comparable with it.
* **Point level** -- how well are action *starts* detected? AP over predicted start points,
  matched to true starts within a time tolerance (the ODAS protocol of Shou et al. 2018),
  swept over tolerances of 1-10 s.
* **Alarm level** -- how does the system behave as an alarm? Event precision, recall and F1,
  false alarms per hour of non-gesture video, and the delay from signal onset to alarm.
  These are the measures the thesis is about; see :mod:`src.decision` for how alarms are
  produced from scores.

Conventions shared by every function here:

* ``scores`` is ``[T, C]`` (higher = more likely), ``labels`` is ``[T]`` of class indices.
  Class 0 is ``none`` (:data:`src.data.canonical.NONE_INDEX`).
* mAP and mcAP average over the *signal* classes only, never over ``none`` -- the usual
  OAD convention, and the honest one: averaging in the largest, easiest class would
  inflate the number.
* Nothing here is fitted, thresholded on, or tuned against the data it scores. Thresholds
  live in the decision layer and are chosen on validation only (Hard Rule 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.data.canonical import NONE_INDEX

# --------------------------------------------------------------------------------------
# Average precision
# --------------------------------------------------------------------------------------


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    """Non-interpolated average precision: the mean of precision at each true positive.

    ``targets`` is boolean (or 0/1). Ties are broken by a stable sort on the negated score,
    so the result is deterministic. Returns ``nan`` when there are no positives -- AP is
    undefined then, and silently returning 0 or 1 would bias any average it enters.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets).astype(bool).ravel()
    if scores.shape != targets.shape:
        raise ValueError(f"scores {scores.shape} and targets {targets.shape} differ")
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = targets[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / positives)


def calibrated_average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    """Calibrated AP (De Geest et al. 2016).

    Plain precision depends on how many negatives a test set happens to contain, and
    continuous video is mostly negatives. Calibrated precision rescales true positives by
    ``w = negatives / positives``, i.e. it reports precision as if the classes were equally
    frequent: ``cPrec = w*TP / (w*TP + FP)``. cAP is the mean of cPrec at each true positive.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets).astype(bool).ravel()
    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if positives == 0:
        return float("nan")
    if negatives == 0:
        return 1.0
    w = negatives / positives
    order = np.argsort(-scores, kind="stable")
    hits = targets[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    cprec = (w * tp) / (w * tp + fp)
    return float(cprec[hits].sum() / positives)


def signal_classes(num_classes: int, signals: Sequence[int] | None = None) -> list[int]:
    """The classes that count as signals: all but ``none`` unless a subset is given."""
    if signals is None:
        return [c for c in range(num_classes) if c != NONE_INDEX]
    bad = [c for c in signals if c == NONE_INDEX or not 0 <= c < num_classes]
    if bad:
        raise ValueError(f"invalid signal classes {bad}: none or out of range")
    return sorted(set(int(c) for c in signals))


# --------------------------------------------------------------------------------------
# Frame level
# --------------------------------------------------------------------------------------


def frame_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Sequence[str],
    signals: Sequence[int] | None = None,
) -> dict:
    """Per-frame metrics over frames pooled from every video of a split.

    Pooling across videos (rather than averaging per-video APs) is the OAD convention
    and avoids short videos with one gesture dominating the average.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = scores.shape[1]
    if len(class_names) != num_classes:
        raise ValueError("class_names does not match the score width")
    sig = signal_classes(num_classes, signals)

    per_class_ap: dict[str, float] = {}
    per_class_cap: dict[str, float] = {}
    for c in range(num_classes):
        per_class_ap[class_names[c]] = average_precision(scores[:, c], labels == c)
        per_class_cap[class_names[c]] = calibrated_average_precision(scores[:, c], labels == c)

    sig_ap = [per_class_ap[class_names[c]] for c in sig]
    sig_cap = [per_class_cap[class_names[c]] for c in sig]

    predicted = scores.argmax(axis=1)
    recalls = []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            recalls.append(float((predicted[mask] == c).mean()))

    return {
        "mAP": float(np.nanmean(sig_ap)) if sig_ap else float("nan"),
        "mcAP": float(np.nanmean(sig_cap)) if sig_cap else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "accuracy": float((predicted == labels).mean()),
        "per_class_AP": per_class_ap,
        "per_class_cAP": per_class_cap,
        "num_frames": int(labels.size),
    }


# --------------------------------------------------------------------------------------
# Segments: ground-truth instances and predicted starts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One ground-truth gesture instance, closed interval ``[start, end]`` in frames."""

    video: str
    cls: int
    start: int
    end: int


def instances_from_labels(video: str, labels: np.ndarray) -> list[Instance]:
    """Runs of the same non-``none`` label become instances.

    Two adjacent annotations of the same class would merge here; the canonical tiling
    does not produce them for IPN Hand, and a merged pair would only make detection easier
    to count, never inflate a false-alarm rate.
    """
    labels = np.asarray(labels, dtype=np.int64)
    out: list[Instance] = []
    if labels.size == 0:
        return out
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [labels.size - 1]])
    for s, e in zip(starts, ends):
        c = int(labels[s])
        if c != NONE_INDEX:
            out.append(Instance(video, c, int(s), int(e)))
    return out


@dataclass(frozen=True)
class Detection:
    """A predicted start point or an alarm: a class at a frame, with a confidence."""

    video: str
    cls: int
    frame: int
    score: float = 1.0


def predicted_starts(video: str, scores: np.ndarray) -> list[Detection]:
    """ODAS start points: every frame where the argmax switches *into* a signal class.

    The score of a start is the probability of the new class at that frame. Uses only the
    current and previous frame, so it is causal.
    """
    scores = np.asarray(scores)
    top = scores.argmax(axis=1)
    out: list[Detection] = []
    previous = NONE_INDEX
    for t, c in enumerate(top.tolist()):
        if c != NONE_INDEX and c != previous:
            out.append(Detection(video, int(c), t, float(scores[t, c])))
        previous = c
    return out


# --------------------------------------------------------------------------------------
# Point level (ODAS)
# --------------------------------------------------------------------------------------


def point_level_ap(
    starts: Sequence[Detection],
    instances: Sequence[Instance],
    cls: int,
    tolerance_frames: float,
) -> float:
    """AP of predicted starts for one class, ODAS-style.

    Starts are ranked by score. A start is a true positive if an unmatched true start of the
    same class, in the same video, lies within ``tolerance_frames`` of it (either side);
    each true start can be matched once, to the nearest eligible prediction in rank order.
    """
    gts = [g for g in instances if g.cls == cls]
    if not gts:
        return float("nan")
    preds = sorted((d for d in starts if d.cls == cls), key=lambda d: -d.score)
    by_video: dict[str, list[int]] = {}
    for i, g in enumerate(gts):
        by_video.setdefault(g.video, []).append(i)
    used = np.zeros(len(gts), dtype=bool)
    hits = np.zeros(len(preds), dtype=bool)
    for k, d in enumerate(preds):
        best, best_gap = -1, np.inf
        for i in by_video.get(d.video, []):
            if used[i]:
                continue
            gap = abs(d.frame - gts[i].start)
            if gap <= tolerance_frames and gap < best_gap:
                best, best_gap = i, gap
        if best >= 0:
            used[best] = True
            hits[k] = True
    if not len(preds):
        return 0.0
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(preds) + 1)
    return float(precision[hits].sum() / len(gts))


def point_level_map(
    starts: Sequence[Detection],
    instances: Sequence[Instance],
    *,
    fps: float,
    tolerances_s: Iterable[float],
    signals: Sequence[int],
) -> dict:
    """Point-level mAP over signal classes, at each tolerance and averaged."""
    out: dict[str, float] = {}
    for tol in tolerances_s:
        aps = [point_level_ap(starts, instances, c, tol * fps) for c in signals]
        out[f"{tol:g}s"] = float(np.nanmean(aps)) if aps else float("nan")
    values = [v for v in out.values() if not np.isnan(v)]
    return {"by_tolerance": out, "mean": float(np.mean(values)) if values else float("nan")}


# --------------------------------------------------------------------------------------
# Alarm level
# --------------------------------------------------------------------------------------


@dataclass
class AlarmMatch:
    true_positives: list[tuple[Detection, Instance]] = field(default_factory=list)
    false_alarms: list[Detection] = field(default_factory=list)
    missed: list[Instance] = field(default_factory=list)


def match_alarms(
    alarms: Sequence[Detection],
    instances: Sequence[Instance],
    *,
    pre_frames: float,
    post_frames: float,
) -> AlarmMatch:
    """Match alarms to true signals, one to one.

    An alarm of class ``c`` at frame ``a`` can claim a true instance of class ``c`` in the
    same video if ``start - pre_frames <= a <= end + post_frames``. Alarms are processed in
    time order and each claims the earliest unclaimed instance it can, so the *first* alarm
    for a gesture is the one whose delay is measured.

    Everything an alarm does not justify counts against it:

    * a second alarm for an already-claimed gesture is a **false alarm** -- staff would be
      called twice for one signal;
    * an alarm of the wrong class is a **false alarm**, and the gesture stays **missed**
      unless a correct alarm also claims it.
    """
    result = AlarmMatch()
    claimed: set[int] = set()
    ordered = sorted(alarms, key=lambda d: (d.video, d.frame))
    by_video: dict[str, list[int]] = {}
    for i, g in enumerate(instances):
        by_video.setdefault(g.video, []).append(i)
    for d in ordered:
        chosen = -1
        for i in sorted(by_video.get(d.video, []), key=lambda j: instances[j].start):
            g = instances[i]
            if i in claimed or g.cls != d.cls:
                continue
            if g.start - pre_frames <= d.frame <= g.end + post_frames:
                chosen = i
                break
        if chosen >= 0:
            claimed.add(chosen)
            result.true_positives.append((d, instances[chosen]))
        else:
            result.false_alarms.append(d)
    result.missed = [g for i, g in enumerate(instances) if i not in claimed]
    return result


def alarm_metrics(
    alarms: Sequence[Detection],
    instances: Sequence[Instance],
    *,
    fps: float,
    background_frames: int,
    pre_s: float,
    post_s: float,
) -> dict:
    """Event precision / recall / F1, false alarms per hour and onset-to-alarm delay.

    ``background_frames`` is the number of ``none`` frames scored: false alarms are
    reported per hour of non-gesture activity, the quantity the thesis argues a monitoring
    alarm should be judged on.
    """
    m = match_alarms(alarms, instances, pre_frames=pre_s * fps, post_frames=post_s * fps)
    tp, fa, missed = len(m.true_positives), len(m.false_alarms), len(m.missed)
    precision = tp / (tp + fa) if tp + fa else float("nan")
    recall = tp / (tp + missed) if tp + missed else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if tp and not np.isnan(precision) and not np.isnan(recall)
        else 0.0
    )
    hours = background_frames / fps / 3600.0
    delays = np.array([(d.frame - g.start) / fps for d, g in m.true_positives], dtype=np.float64)
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "alarms": tp + fa,
        "true_alarms": tp,
        "false_alarms": fa,
        "missed": missed,
        "background_hours": hours,
        "false_alarms_per_hour": float(fa / hours) if hours > 0 else float("nan"),
        "delay_median_s": float(np.median(delays)) if delays.size else float("nan"),
        "delay_p95_s": float(np.percentile(delays, 95)) if delays.size else float("nan"),
    }


def frames_of_class(labels: Mapping[str, np.ndarray], cls: int) -> int:
    return int(sum(int((np.asarray(v) == cls).sum()) for v in labels.values()))


__all__ = [
    "AlarmMatch",
    "Detection",
    "Instance",
    "alarm_metrics",
    "average_precision",
    "calibrated_average_precision",
    "frame_metrics",
    "frames_of_class",
    "instances_from_labels",
    "match_alarms",
    "point_level_ap",
    "point_level_map",
    "predicted_starts",
    "signal_classes",
]
