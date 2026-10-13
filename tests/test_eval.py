"""The evaluation harness, checked against answers worked out by hand.

Every metric here ends up in the thesis, so each is pinned to a small case whose value can
be computed on paper. A metric that has only been run on real data has not been tested.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.decision.onset import active_classes, onset_alarms
from src.eval import metrics as M
from src.eval.evaluate import evaluate_split, headline, tripwire

FPS = 30.0


# ----------------------------------------------------------------------------------
# Average precision
# ----------------------------------------------------------------------------------


def test_ap_by_hand():
    # ranked: + - + -   precision at the hits: 1/1, 2/3
    assert M.average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == pytest.approx(5 / 6)


def test_ap_perfect_and_worst():
    assert M.average_precision([0.9, 0.1], [1, 0]) == 1.0
    assert M.average_precision([0.1, 0.9], [1, 0]) == pytest.approx(0.5)


def test_ap_without_positives_is_nan_not_zero():
    assert math.isnan(M.average_precision([0.3, 0.2], [0, 0]))


def test_cap_equals_ap_when_balanced():
    s, t = [0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]
    assert M.calibrated_average_precision(s, t) == pytest.approx(M.average_precision(s, t))


def test_cap_by_hand_under_imbalance():
    # 2 positives, 3 negatives, w = 1.5. Ranked + - - - +.
    # hit 1: TP=1 FP=0 -> 1.  hit 2: TP=2 FP=3 -> 3/(3+3) = 0.5.  cAP = 0.75.
    # Plain AP would be (1 + 2/5)/2 = 0.7: calibration removes the penalty for the negatives
    # simply being more numerous.
    s, t = [0.9, 0.8, 0.7, 0.6, 0.5], [1, 0, 0, 0, 1]
    assert M.calibrated_average_precision(s, t) == pytest.approx(0.75)
    assert M.average_precision(s, t) == pytest.approx(0.7)


def test_map_ignores_the_none_class():
    # none is perfectly ranked; the one signal class is ranked worst-first.
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.0]])
    labels = np.array([0, 0, 1])
    out = M.frame_metrics(scores, labels, class_names=["none", "wave"])
    assert out["mAP"] == pytest.approx(1 / 3)
    assert out["per_class_AP"]["none"] == pytest.approx(1.0)


def test_balanced_accuracy_is_mean_recall_over_classes():
    scores = np.eye(2)[[0, 0, 0, 0, 1]]  # predicts none 4x then wave
    labels = np.array([0, 0, 0, 1, 1])  # one wave frame missed
    out = M.frame_metrics(scores, labels, class_names=["none", "wave"])
    assert out["balanced_accuracy"] == pytest.approx((1.0 + 0.5) / 2)
    assert out["accuracy"] == pytest.approx(4 / 5)


# ----------------------------------------------------------------------------------
# Instances and start points
# ----------------------------------------------------------------------------------


def test_instances_are_runs_of_signal_labels():
    got = M.instances_from_labels("v", np.array([0, 0, 1, 1, 0, 2, 2, 2]))
    assert got == [M.Instance("v", 1, 2, 3), M.Instance("v", 2, 5, 7)]


def test_predicted_starts_fire_on_switch_into_a_signal():
    scores = np.eye(3)[[0, 1, 1, 2, 0, 2]]
    starts = M.predicted_starts("v", scores)
    assert [(d.cls, d.frame) for d in starts] == [(1, 1), (2, 3), (2, 5)]


def test_point_ap_within_and_outside_tolerance():
    gts = [M.Instance("v", 1, 100, 150)]
    near = [M.Detection("v", 1, 110, 0.9)]
    far = [M.Detection("v", 1, 200, 0.9)]
    assert M.point_level_ap(near, gts, 1, tolerance_frames=30) == 1.0
    assert M.point_level_ap(far, gts, 1, tolerance_frames=30) == 0.0


def test_point_ap_matches_each_true_start_once():
    gts = [M.Instance("v", 1, 100, 150)]
    two = [M.Detection("v", 1, 101, 0.9), M.Detection("v", 1, 102, 0.8)]
    # ranked: hit, miss -> precision at the hit is 1, one GT -> AP 1.0; the duplicate
    # costs nothing in AP but is counted as a false alarm by the alarm metrics.
    assert M.point_level_ap(two, gts, 1, tolerance_frames=30) == 1.0


# ----------------------------------------------------------------------------------
# Alarm matching and alarm metrics
# ----------------------------------------------------------------------------------


GT = [M.Instance("v", 1, 100, 160), M.Instance("v", 2, 300, 360)]


def test_a_correct_alarm_inside_the_gesture_is_a_true_positive():
    m = M.match_alarms([M.Detection("v", 1, 115)], GT, pre_frames=0, post_frames=0)
    assert len(m.true_positives) == 1 and not m.false_alarms
    assert [g.cls for g in m.missed] == [2]


def test_a_second_alarm_for_the_same_gesture_is_a_false_alarm():
    alarms = [M.Detection("v", 1, 110), M.Detection("v", 1, 130)]
    m = M.match_alarms(alarms, GT, pre_frames=0, post_frames=0)
    assert len(m.true_positives) == 1
    assert [d.frame for d in m.false_alarms] == [130]


def test_a_wrong_class_alarm_is_false_and_the_gesture_stays_missed():
    m = M.match_alarms([M.Detection("v", 2, 120)], GT, pre_frames=0, post_frames=0)
    assert len(m.false_alarms) == 1
    assert {g.cls for g in m.missed} == {1, 2}


def test_pre_and_post_tolerance_widen_the_window():
    early = [M.Detection("v", 1, 95)]
    late = [M.Detection("v", 1, 170)]
    assert not M.match_alarms(early, GT, pre_frames=0, post_frames=0).true_positives
    assert M.match_alarms(early, GT, pre_frames=10, post_frames=0).true_positives
    assert M.match_alarms(late, GT, pre_frames=0, post_frames=15).true_positives


def test_alarms_in_other_videos_never_match():
    m = M.match_alarms([M.Detection("w", 1, 115)], GT, pre_frames=0, post_frames=0)
    assert len(m.false_alarms) == 1


def test_false_alarms_per_hour_and_delay_by_hand():
    one_hour = int(FPS * 3600)
    alarms = [
        M.Detection("v", 1, 115),  # 15 frames = 0.5 s after onset
        M.Detection("v", 2, 330),  # 30 frames = 1.0 s after onset
        M.Detection("v", 1, 1000),  # nothing there: false alarm
        M.Detection("v", 2, 2000),  # false alarm
    ]
    out = M.alarm_metrics(alarms, GT, fps=FPS, background_frames=one_hour, pre_s=0, post_s=0)
    assert out["false_alarms_per_hour"] == pytest.approx(2.0)
    assert out["event_precision"] == pytest.approx(0.5)
    assert out["event_recall"] == pytest.approx(1.0)
    assert out["event_f1"] == pytest.approx(2 / 3)
    assert out["delay_median_s"] == pytest.approx(0.75)


def test_no_alarms_gives_zero_recall_and_zero_false_alarms():
    out = M.alarm_metrics([], GT, fps=FPS, background_frames=1000, pre_s=0, post_s=0)
    assert out["event_recall"] == 0.0 and out["false_alarms"] == 0 and out["event_f1"] == 0.0
    assert math.isnan(out["delay_median_s"])


# ----------------------------------------------------------------------------------
# The Stage 4 alarm rule
# ----------------------------------------------------------------------------------


def test_threshold_demotes_unconfident_frames_to_none():
    scores = np.array([[0.2, 0.8, 0.0], [0.5, 0.45, 0.05]])
    assert active_classes(scores, 0.9).tolist() == [0, 0]
    assert active_classes(scores, 0.5).tolist() == [1, 0]


def test_one_alarm_per_run_of_the_same_class():
    scores = np.eye(3)[[0, 1, 1, 1, 0, 1, 2, 2]]
    alarms = onset_alarms("v", scores, threshold=0.5)
    assert [(d.cls, d.frame) for d in alarms] == [(1, 1), (1, 5), (2, 6)]


def test_alarm_rule_is_causal_under_truncation():
    """Rule 1 for the decision rule: alarms before frame t do not depend on frames after t."""
    rng = np.random.default_rng(0)
    raw = rng.random((200, 4))
    scores = raw / raw.sum(axis=1, keepdims=True)
    full = onset_alarms("v", scores, 0.3)
    for t in (1, 17, 50, 133, 199):
        partial = onset_alarms("v", scores[:t], 0.3)
        assert partial == [d for d in full if d.frame < t], f"t={t}"


# ----------------------------------------------------------------------------------
# The whole harness
# ----------------------------------------------------------------------------------


def _harness(scores, labels, **kw):
    defaults = dict(
        fps=FPS,
        class_names=["none", "wave", "point"],
        signals=None,
        threshold=0.5,
        match_pre_s=0.5,
        match_post_s=1.0,
        point_tolerances_s=[1, 2],
    )
    defaults.update(kw)
    return evaluate_split(scores, labels, **defaults)


def test_perfect_scores_trip_rule_2():
    labels = {"v": np.array([0] * 40 + [1] * 30 + [0] * 40 + [2] * 30 + [0] * 40)}
    scores = {"v": np.eye(3)[labels["v"]]}
    out = _harness(scores, labels)
    h = headline(out)
    assert h["mAP"] == 1.0 and h["event_recall"] == 1.0 and h["false_alarms_per_hour"] == 0.0
    assert h["delay_median_s"] == 0.0
    assert set(tripwire(out)) >= {"mAP", "event_f1"}


def test_random_scores_flood_alarms_and_trip_only_recall():
    """Random scores fire almost every frame, so they 'catch' every gesture: recall alone
    crosses the Rule 2 line, with a huge false-alarm rate and near-zero precision. The rule
    still trips -- it is never relaxed -- and this is the signature that tells a flood
    apart from a leak, which raises precision and recall together."""
    rng = np.random.default_rng(1)
    labels = {"v": np.array([0] * 400 + [1] * 60 + [0] * 400 + [2] * 60 + [0] * 400)}
    raw = rng.random((len(labels["v"]), 3))
    out = _harness({"v": raw / raw.sum(axis=1, keepdims=True)}, labels)
    assert tripwire(out) == ["event_recall"]
    assert out["alarm"]["event_precision"] < 0.05
    assert out["alarm"]["false_alarms_per_hour"] > 1000
    assert len(out["curve"]) > 5


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        _harness({"v": np.zeros((5, 3))}, {"v": np.zeros(6, dtype=int)})


def test_signal_subset_restricts_every_metric():
    labels = {"v": np.array([0] * 40 + [1] * 30 + [0] * 40 + [2] * 30 + [0] * 40)}
    scores = {"v": np.eye(3)[labels["v"]]}
    out = _harness(scores, labels, signals=[2])
    assert out["signal_classes"] == ["point"] and out["num_instances"] == 1
