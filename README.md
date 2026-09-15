# sos-oad — Online Temporal Action Detection for distress hand signals

M.Tech thesis system. Given a **stream** of video frames, emit a label for the **current** frame
and decide when to fire an alarm.

This is **online temporal action detection**, not action recognition on pre-trimmed clips and not
offline temporal localisation. The model is **strictly causal**: it never reads a frame from the
future. That constraint is enforced by a test, not by convention — see
[Causality](#causality-is-tested) below.

## Phases

| Phase | Data | Status |
|---|---|---|
| 1 | **IPN Hand** as the engineering vehicle — 13 touchless-interface gestures + a non-gesture class. Not distress signals; a deliberate stand-in with the right shape. | in progress |
| 2 | Author's own recorded SOS data, swapped in behind the dataset adapter. No downstream code changes. | future |

## Canonical data format

Every dataset is converted by a small adapter into exactly this. The training code sees nothing
else — that is what makes Phase 2 a drop-in.

```
data/<dataset>/
  features/<video_id>.npy      # [T, D]    float32  — one row per frame
  poses/<video_id>.npy         # [T, J, 3] float32  — optional skeleton stream
  annotations.csv              # video_id,class,start_frame,end_frame,subject_id,split
  classes.txt                  # one class per line; index 0 is ALWAYS "none"
  meta.yaml                    # fps, feature dim, backbone id, extraction settings, checksums
```

Class index 0 is always `none`, the non-gesture class. It is the largest class in the dataset and
the whole detection problem depends on it. It is never dropped as background.

## Setup

Requires **Python 3.11**. (3.8 caps PyTorch at 2.4; 3.13 has no mmcv / MediaPipe wheels.)

```powershell
git clone <repo-url> sos-oad
cd sos-oad
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA training hardware, additionally:

```powershell
pip install -r requirements-cuda.txt
```

The device is config-driven (`cpu` / `cuda` / `mps`) and never hardcoded, so the same commands run
on a laptop and on a cluster.

## Reproduction

Commands are added here as each stage lands, so this section always reproduces the current state
of the thesis.

<!-- STAGE 0 --> _(pending)_

## Causality is tested

`tests/test_causality.py` feeds a truncated stream of length `t` and asserts that outputs for
frames `0..t` are **bit-identical** to the corresponding slice of a full-stream run. Any module
that fails is not causal, whatever its docstring says.

```powershell
pytest tests/ -v
```

## Evaluation

Every reported result comes from a **subject-disjoint** split and carries the full metric set:
per-frame mAP, point-level mAP (ODAS protocol, onset tolerance swept 1–10 s), event-level
precision / recall / F1, **false alarms per hour**, median and p95 onset-to-alarm latency,
end-to-end FPS, and peak GPU memory.

Splits are frozen to disk and the test set is never tuned against.

## Project contract

[`CLAUDE.md`](CLAUDE.md) holds the full operating contract: hard rules, the canonical format spec,
the stage plan with gates, and the environment assumptions.

## Dataset

IPN Hand — Benitez-Garcia et al., *IPN Hand: A Video Dataset and Benchmark for Real-Time
Continuous Hand Gesture Recognition* (ICPR 2020). CC BY 4.0.
<https://gibranbenitez.github.io/IPN_Hand/>

The dataset is **not** redistributed in this repository. `scripts/` documents how to obtain it and
the adapter converts it to the canonical format locally.
