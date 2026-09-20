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

Class index 0 is always `none`, the non-gesture class. The whole detection problem depends on it,
and it is never dropped as background. (On IPN Hand it is the largest class by instance count but
only the third largest by frame count — see the Stage 1 notes in `CLAUDE.md`.)

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

### Stage 0 — canonical format

Generate the synthetic fixture (a small known-good dataset in the canonical format) and validate
it. This needs no dataset download and is the Stage 0 gate:

```powershell
python scripts/make_fixture.py --out tests/fixtures/synthetic --force
python scripts/validate_dataset.py --root tests/fixtures/synthetic
pytest tests/ -v
```

### Stage 1 — IPN Hand

Download IPN Hand (CC BY 4.0, https://gibranbenitez.github.io/IPN_Hand/). The Google Drive copy
arrives as 16 independent zips totalling ~28 GB. Point the script at the directory holding them;
it unpacks only the annotation text files (a few hundred KB), converts to the canonical format,
and prints the published figures beside the computed ones. That table is the Stage 1 gate:

```powershell
python scripts/prepare_ipn_hand.py --zips $HOME\Downloads --set dataset=ipn_hand
```

Add `--verify-frames` to also count the ~800k JPEGs, streamed straight out of the archives with
nothing written to disk (~2 min). Re-check the gate at any time with:

```powershell
pytest tests/test_ipn_hand.py -m slow
```

### Stage 2 — frozen subject-disjoint splits

Generate and freeze the split. The test set is IPN's own 13 subjects; the remaining 37 are
partitioned into 5 LOSGO folds, and fold 0 is promoted to be the primary validation set:

```powershell
python scripts/make_splits.py --set dataset=ipn_hand split=ipn_official
python scripts/make_splits.py --set dataset=ipn_hand split=ipn_official --show
pytest -m splits
```

The frozen split lands at `data/ipn_hand/splits/ipn_official.json` and **is committed** — it is
the evidence that the split behind any reported number never moved. Re-running refuses to
overwrite it without `--force`, and says which subjects would change side.

### Stage 3 — feature extraction

Causality first. The truncation test from Hard Rule 1, plus three deliberately non-causal
extractors the harness must reject:

```powershell
pytest -m causality
```

Extraction runs on a Kaggle T4 (this laptop has no CUDA). Upload the five `frames*.tgz`
archives and the `annotations/` folder as a Kaggle dataset, then run
`notebooks/kaggle_extract_ipn.ipynb`, which clones this repo at a pinned commit so every run
is traceable to a SHA. One shard per session:

```python
SHARD = '0/4'        # 12-hour session cap; re-running skips what is already cached
```

Locally, the same pipeline with the dependency-free backbone — proves the plumbing, never the
features:

```powershell
python scripts/extract_features.py --set dataset=ipn_hand features=projection --limit 2
```

After the shards are merged back, the Stage 3 gate is:

```powershell
python scripts/extract_features.py --verify --set dataset=ipn_hand features=videomaev2
```

On a fresh clone, `annotations.csv` is regenerated by the adapter but the frozen split is
applied, never re-drawn:

```powershell
python scripts/make_splits.py --set dataset=ipn_hand split=ipn_official --apply-frozen
```

Validate any canonical dataset, with strictness that ratchets up per stage:

```powershell
python scripts/validate_dataset.py --root data/ipn_hand                     # Stage 2 strictness
python scripts/validate_dataset.py --root data/ipn_hand --require-poses `
                                   --require-features                       # Stage 3
```

Configs compose a base with a dataset group, overridable from the command line:

```powershell
python scripts/validate_dataset.py --config configs/base.yaml --set dataset=ipn_hand seed=7
```

<!-- STAGE 1 --> _(pending)_

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
