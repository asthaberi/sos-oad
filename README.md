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

## Where things live

Open this first if the folder looks confusing. Two rules explain almost all of it:

1. **Code is committed; data is not.** Anything under `data/`, `raw/`, `runs/` or
   `reports/samples/` is either downloaded or regenerated, so it is deliberately kept out of git.
   A fresh clone is ~2 MB.
2. **Nothing is lost by deleting a regenerable folder** — the command that rebuilds it is in the
   Reproduction section below.

### Committed — this is the project

| Folder | What it is |
|---|---|
| `src/` | The library. All the real logic lives here |
| `scripts/` | Commands you run from a terminal. Thin wrappers around `src/` |
| `configs/` | Settings, as YAML. Nothing important is hardcoded outside here |
| `tests/` | Automatic checks. `pytest` runs them all |
| `notebooks/` | The Kaggle extraction notebook |
| `reports/` | Progress report, panel report, presentation |
| `CLAUDE.md` | The project contract: rules, stage plan, decisions and why |
| `README.md` | This file — setup and reproduction commands |

### Not committed — downloaded or generated

| Folder | What it is | Size | Rebuild with |
|---|---|---|---|
| `data/<dataset>/` | The dataset in canonical format | ~2 GB | `scripts/prepare_ipn_hand.py` |
| `data/*/splits/*.json` | **The frozen split — this one IS committed** | 10 KB | never regenerate; apply with `--apply-frozen` |
| `raw/IPN_Hand/` | Extracted frames and annotations | ~20 GB | re-extract from the archives |
| `runs/` | One folder per execution: config, provenance, metrics | small | produced by each run |
| `reports/samples/` | Rendered sample clips | ~20 MB | `scripts/make_sample_clip.py` |
| `.venv/` | The Python environment | ~5 GB | `pip install -r requirements.txt` |
| `IPN hand Dataset/` | The original download, 16 zip archives | ~28 GB | re-download from the dataset authors |

### The one exception worth knowing

`data/` is gitignored **except** `data/*/splits/*.json`. The frozen split is the evidence behind
Hard Rule 3 — it is what proves the split behind a reported number never moved — so it belongs in
version control even though everything around it does not.

### Which machine has what

Code is identical on both (`git pull`). Data is not:

- **Laptop** — writes code, runs tests. Has the original download and the extracted frames from
  the Stage 1 work. Does not need them any more: extraction now happens on the GPU server.
- **GPU server** (`D:\ASTHA_BERI\sos-oad`) — has the dataset, the feature cache and the real run
  directories. This is where results come from.

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

Extraction runs on a Kaggle T4 (this laptop has no CUDA). Stage the upload — five `.tgz`
archives plus `annotations/`, hard-linked so it costs no extra disk — and create a **private**
Kaggle dataset from it:

```powershell
pip install kaggle          # then save your API token to ~/.kaggle/kaggle.json
python scripts/prepare_kaggle_upload.py
kaggle datasets create -p raw/kaggle_upload --dir-mode tar
```

Five files, not 800k JPEGs: Kaggle handles a few large archives far better than a huge file
count, and the notebook untars only the shard it needs.

Then run `notebooks/kaggle_extract_ipn.ipynb`, which clones this repo at a pinned commit so
every run is traceable to a SHA. One shard per session:

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

### Experiment tracking

Weights & Biases is the default backend; TensorBoard and `none` are also available. Log in once:

```powershell
wandb login
```

Without a key, runs degrade to **offline** with a warning rather than failing — useful on Kaggle,
where a session may have no stored credential. Upload buffered runs afterwards:

```powershell
wandb sync runs/<run>/wandb/offline-run-*
```

Switch backend or project per run:

```powershell
python scripts/<any>.py --set logging.backend=tensorboard
python scripts/<any>.py --set logging.mode=offline logging.project=sos-oad-dev
```

Whatever the backend, scalars are mirrored to `runs/<run>/metrics/scalars.jsonl` alongside the
resolved config and `provenance.json`. That directory, not the dashboard, is what a reported
number traces back to.

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
