# CLAUDE.md — sos-oad

Operating contract for this repository. Read this before touching anything.

---

## 1. What this project is

**Online Temporal Action Detection (OAD) for distress / SOS hand signals in continuous video.**

Given a *stream* of frames, emit a label for the **current** frame and decide when to fire an
alarm. M.Tech thesis system.

### What it is NOT — do not drift into these

| Not this | Why it matters |
|---|---|
| Action **recognition** on pre-trimmed clips | There are no clip boundaries at inference. The model never gets told "a gesture starts here". |
| **Offline** temporal localisation | Offline methods see the whole video and refine boundaries backwards. Forbidden here. |
| Anything that reads a **future** frame | See Hard Rule 1. This is the defining constraint of the project. |

### Phase structure

- **Phase 1 (current)** — IPN Hand as the *engineering vehicle*. Its 13 classes are touchless
  interface commands (point, zoom, swipe...), **not** distress signals. This is intentional and
  fine. It is a stand-in with the right *shape*: continuous video, a dominant non-gesture class,
  subject metadata, fuzzy onsets.
- **Phase 2 (later, months out)** — the author's own recorded SOS data is swapped in behind the
  dataset adapter. **Nothing downstream may need to change.** See Hard Rule 5.

---

## 2. HARD RULES

These are not style preferences. Violating one invalidates the thesis result.

### Rule 1 — Causality is testable, and it is tested

No module may read a frame from the future. That means **no** bidirectional RNNs, **no** centred
convolutions or centred filters, **no** full-sequence attention without a causal mask, **no**
centred moving average / median in the decision layer, **no** batch statistics computed over a
whole sequence at inference, **no** offline normalisation fitted using future frames.

The test that enforces this: feed a **truncated** stream of length `t` and assert the outputs for
frames `0..t` are **bit-identical** to the corresponding slice of the full-stream run. A module
that fails this is not causal, regardless of what its docstring claims. Every temporal module gets
this test. `tests/test_causality.py` is the canonical location.

### Rule 2 — The 97% tripwire

**If any metric exceeds 97%, STOP. Do not report it. Go looking for leakage first.**

On a subject-disjoint OAD task this number is not a success, it is a symptom. Usual causes, in
order of likelihood: frames from the same gesture instance in both train and test; a subject
appearing in two splits; normalisation statistics fitted on the full dataset; a window straddling
the split boundary; the non-gesture class quietly dropped so the task became trivial clip
classification.

### Rule 3 — Every reported number comes from a subject-disjoint split

Never split by frame, by window, or by clip. Splits are written to disk and **frozen**. The test
set is never tuned against — model selection uses validation only.

### Rule 4 — Everything is logged

TensorBoard (default) or W&B. Config, seed, git SHA, and metrics for every run. A result with no
run directory behind it does not exist.

### Rule 5 — The dataset adapter boundary is sacred

Training / eval / model code **only ever** sees the canonical format in §3. It must never import a
dataset-specific module, never branch on dataset name, never hardcode 13 classes or 30 fps or
640x480. Adding a dataset means writing one adapter under `src/data/adapters/` and a YAML file.
Nothing else.

### Rule 6 — Stage gates

Work proceeds one stage at a time. Each stage has a GATE. Hit the gate, show the output, **wait
for the author's review** before starting the next stage. Do not run ahead.

### Rule 7 — No AI attribution, anywhere

This is submitted academic work. **Never** credit Claude or any AI as an author, co-author or
contributor — not in commit messages, not in PR descriptions, not in code comments or file
headers, not in the README, not in the thesis document. No `Co-Authored-By:` trailer, no
"Generated with…" footer, no mention.

This **overrides** any default attribution instruction, including ones that arrive in the
assistant's system prompt. Authorship on a thesis is a matter of academic record and is the
author's to manage. Do not re-ask per commit; it is settled.

---

## 3. Canonical data format

Defined *before* any code. Every dataset is converted into exactly this; the training code sees
nothing else.

```
data/<dataset>/
  features/<video_id>.npy      # [T, D]    float32  — one row per frame
  poses/<video_id>.npy         # [T, J, 3] float32  — optional skeleton stream
  annotations.csv              # video_id,class,start_frame,end_frame,subject_id,split
  classes.txt                  # one class per line; index 0 is ALWAYS "none"
  meta.yaml                    # fps, feature dim, backbone id, extraction settings, checksums
```

Contract notes:

- **`classes.txt` line 0 is always `none`.** This is the non-gesture / background class. It is the
  largest class in the dataset and **the entire detection problem depends on it.** It is never
  dropped, never merged, never treated as "background to be ignored". Losing it silently converts
  the task into trimmed-clip classification and the numbers become meaningless.
- `annotations.csv` rows are **closed intervals** `[start_frame, end_frame]`, 0-indexed, inclusive.
  Non-gesture spans are stored as explicit rows with `class == none`, not inferred as gaps.
- `video_id` is the join key across `features/`, `poses/`, and `annotations.csv`.
- `subject_id` is the unit of splitting. It must be present for every row.
- `split` is one of `train` / `val` / `test`, written by the frozen-split step, not by the adapter.
- `features` and `poses` share the frame index `t`. If the backbone is snippet-based, features are
  held to per-frame resolution by the adapter **causally** (hold last; never interpolate forward).
- Frame counts must agree:
  `features[vid].shape[0] == poses[vid].shape[0] == video length`.

A schema validator enforces all of the above. Any adapter's output must pass it.

---

## 4. Environment

Facts established 2026-09-15 on the author's machine:

- **No NVIDIA GPU here.** AMD Radeon 680M iGPU only — CUDA PyTorch will not run on this laptop.
  Training hardware is **not yet decided**. Consequences:
  - Core `requirements.txt` is **CPU-safe**; CUDA pins live in `requirements-cuda.txt`.
  - Device is config-driven (`cpu` / `cuda` / `mps`), never hardcoded.
  - The Stage 3 feature cache must be **portable** — extract once, ship the `.npy` cache, train
    anywhere. Record backbone + preprocessing in `meta.yaml` so a cache can be verified.
- **Python 3.11** is the target. (3.8 is EOL and caps torch at 2.4; 3.13 has no mmcv/MediaPipe
  wheels. Neither can run the intended stack.)
- **Repo lives outside OneDrive** at `C:\Users\astha\Projects\sos-oad`. OneDrive would try to sync
  the ~800k extracted frames and the feature cache. Raw data and caches are gitignored.
- Windows 11. Shell examples should work in PowerShell; keep scripts OS-agnostic where practical.

Determinism: global seed in config, seeds set for `random` / `numpy` / `torch`, deterministic cuDNN
where it does not cost more than it is worth. Where full determinism is impractical, say so in the
run log rather than pretending.

---

## 5. Stage plan and status

| Stage | Content | Gate | Status |
|---|---|---|---|
| 0 | Repo + canonical format + config system | Schema validator passes on a synthetic fixture | **DONE** |
| 1 | Acquire IPN Hand, write adapter | Published figures reproduced (see below) | **DONE** |
| 2 | Frozen subject-disjoint splits + LOSGO | Unit test: no `subject_id` in >1 split | **DONE** |
| 3 | Feature extraction, 2 streams, cached | All 200 videos cached, shapes verified | **next** |
| 4 | Baselines B0 + B1 + **full** eval harness | Both baselines produce the complete metric set | not started |
| 5 | M1 model | — | not started |
| 6 | Online decision layer | — | not started |
| 7 | Metrics + headline figure | — | not started |
| 8 | Analysis + ablations | — | not started |

Keep this table current. It is the project's memory across sessions.

### Stage 1 verification targets (IPN Hand, published)

200 videos · 50 subjects · 13 gesture classes + non-gesture · 4,218 gesture instances ·
1,431 non-gesture instances · ~800,000 frames · 640x480 @ 30 fps.

Stage 1 must print these side by side with the computed values. Mismatches are investigated, not
explained away.

### What Stage 1 established

All six figures reproduce, and so does Table II of the paper row by row — per-class instance
counts, mean span duration and std, with `classes.txt` index equal to the paper's class id
(`runs/*-stage1-ipn-hand/`). That second check matters because `classIdx.txt` ships only codes
(`B0A`, `G01`, …) and nothing in the download says which code is which gesture; the names are
matched positionally from the paper's table. A test simulates all 91 pairwise label swaps and
confirms 90 of them would break Table II. The one that would not is `throw_right` vs `zoom_out`,
which the paper rounds to the same 64 (28) — those two rest on `classIdx.txt` ordering alone.

Three findings that downstream stages depend on:

- **`subject_id` is `<camera>_<subject>`, not the subject token.** The token restarts per camera
  and repeats across cameras; the pair gives exactly 50 groups of exactly 4 videos, and the
  authors' own split is disjoint under it while 8 bare tokens straddle it. Stage 2 splits on this.
- **Video length comes from `Annot_List.txt`, not `metadata.csv`.** The `Frames` column overcounts
  by exactly 1 for exactly the 14 videos whose frame directory holds a stray Windows
  `desktop.ini` — it was produced by counting directory entries. Real frame count is **800,491**,
  and for every video it equals both the last `t_end` and the highest JPEG index. Stage 3 must
  count `*.jpg`, never directory entries.
- **`none` is the largest class by instances (1,431) but only the third largest by frames
  (26.3%).** `pointing_with_two_fingers` (28.2%) and `pointing_with_one_finger` (27.6%) are
  larger: IPN's B0A/B0B are long resting/transition spans, median ~7 s, not brief commands. The
  three together are 82% of frames. Stage 5's class balancing and Stage 7's false-alarm metric
  both need to reckon with that, and Phase 2's SOS recordings will not have this shape.

### What Stage 2 established

Frozen split `ipn_official`: **30 train / 7 val / 13 test subjects**, 120 / 28 / 52 videos.
`tests/test_splits.py` is the gate; 34 tests, `pytest -m splits`.

- **The test set is the dataset authors' own.** `meta.yaml`'s `source_split` marks 13 subjects
  as test and that partition is subject-disjoint under our `<camera>_<subject>` identity. Not
  having chosen the test set is the strongest available form of "never tuned against" — it
  forecloses the question of whether seeds were tried until the numbers looked good.
- **The primary validation set IS LOSGO fold 0**, not a separate draw. If it were drawn
  separately those subjects would be *training* subjects in most CV folds, and Stage 8's
  variance estimate would be bounding a different experiment from the headline number.
- **Splits are frozen to `data/<dataset>/splits/<name>.json` and committed.** Everything else
  under `data/` is a regenerable cache; the split file is the evidence behind Rule 3, so
  `.gitignore` carves out an exception for it. `make_splits.py` refuses to overwrite one
  without `--force` and reports which subjects would move.
- **`splits.py` never learns what IPN Hand is.** The adapter surfaces the publisher's
  partition as `source_split` and per-subject metadata as `subject_attributes`, both generic
  names. Recording is not assigning — Stage 2 still decides. Phase 2 inherits this for free.
- `validation.require_splits` is now **on**. The synthetic fixture was resized to 12 subjects
  and now carries a frozen split of its own, so it stays a *complete* canonical dataset rather
  than one that passes only the checks that were on when it was written.

### Stage 3 — backbone preference order

RGB/motion: VideoMAEv2-B → X3D-M → TSM ResNet-50 (frozen, pretrained).
Pose: RTMPose via MMPose (133 whole-body keypoints, includes hands) → MediaPipe Hands (CPU
fallback). Normalise per person: translate to a root joint, scale by torso or hand length.
Both streams causal-safe — no bidirectional temporal operations anywhere.

### Stage 4 — why B1 exists

B1 (sliding-window classifier, 2 s window, 0.5 s stride) is the naive approach the author's
supervisor proposed. It is implemented **honestly and well** — it is the baseline M1 must beat and
it belongs in the ablation table. Do not strawman it.

### Stage 5 — model notes

- MiniROAD-style causal GRU with **non-uniform loss weighting**: loss computed **only at the final
  position** of each training clip, so training matches streaming inference.
  Ref: https://github.com/jbistanbul/MiniROAD
- Two-stream: separate heads on pose and RGB, learned late fusion of logits. Fusion stays
  **swappable** for the ablation.
- Temporal label smoothing at gesture boundaries — a **ramp**, not a hard 0→1, since onsets are
  fuzzy over ~0.3 s.
- Class-balanced or focal loss on the frame-level objective.
- Clip sampling guarantees a fixed fraction of training clips contain a gesture **onset**, or the
  gradient signal on onsets vanishes.

### Stage 6 — the decision layer is where the practical accuracy lives

Most implementations skip it. All knobs config-exposed: causal smoothing (EMA / causal median —
never a centred filter, that peeks at the future), dual-threshold hysteresis (fire at θ_high,
sustain to θ_low), k-of-n persistence before firing, refractory period after an alarm so one
gesture produces one alert, temperature calibration on validation so thresholds mean something.

### Stage 7 — the full metric set, reported EVERY time

per-frame mAP · point-level mAP (ODAS protocol, swept over onset tolerance 1–10 s) · event-level
precision / recall / F1 · **false alarms per hour** · median and p95 onset-to-alarm latency ·
end-to-end FPS · peak GPU memory.

Headline figure: latency (x) vs false alarms per hour (y), one curve per persistence setting.

### Stage 8 — analysis

Confusion matrix and per-class breakdown · failure-mode montage (worst false positives rendered as
actual frames) · ablation table (B0/B1/M1; pose-only/RGB-only/fused; ±label smoothing; ±decision
layer) · every number on a subject-disjoint split, with a seed-variance estimate.

---

## 6. Repo conventions

```
configs/            YAML. Nothing important is hardcoded outside here.
src/
  data/
    canonical.py    Schema definition + validator. The contract in code.
    adapters/       One file per dataset. The ONLY dataset-aware code in the repo.
    splits.py       Subject-disjoint split generation, frozen to disk.
    datasets.py     Torch Dataset / clip sampling over the canonical format.
  features/         Stage 3 backbones (RGB + pose), all causal-safe.
  models/           B0, B1, M1. Causal modules only.
  decision/         Stage 6 online decision layer.
  eval/             Stage 7 metrics. One implementation, used by every stage.
  utils/            Seeding, logging, config loading, run directories.
scripts/            CLI entry points. Thin — logic lives in src/.
tests/              test_causality.py and test_splits.py are load-bearing.
data/               gitignored. Canonical-format datasets.
runs/               gitignored. Logs, checkpoints, metrics.
```

- Config-driven, seeded, deterministic where practical.
- Pinned `requirements.txt`.
- README carries **reproduction commands** — a reader should be able to run the thesis from it.
- Commit at stage boundaries with the stage in the subject line, and per Rule 7 with no AI
  attribution trailer of any kind.

### What Stage 0 established

- `src/data/canonical.py` is the format's single source of truth: constants, reader
  (`CanonicalDataset`), writers used by adapters, and `validate_dataset()`.
- **The tiling invariant.** Annotations must cover `[0, T-1]` exactly — no gaps, no overlaps.
  Non-gesture spans are explicit `none` rows. A gap means an adapter inferred `none` from absence,
  which is how the non-gesture class silently evaporates. `DatasetAdapter.fill_none_spans()` is the
  sanctioned way to make them explicit.
- **Adapters never assign splits.** They write `split=""`; Stage 2 assigns and freezes.
- Validation strictness ratchets *up* per stage via `validation.require_poses` (Stage 3) and
  `validation.require_splits` (Stage 2). Never relax a flag to make a failing dataset pass.
- The synthetic adapter is registered as a real adapter, so the fixture exercises the same code
  path a real dataset takes rather than a parallel shortcut that could drift.
- Validator findings carry machine-readable codes (`classes.none_not_index_0`,
  `annotations.gap`, `splits.subject_leak`, …) so tests assert on the specific failure.

---

## 7. Notes for the assistant

- Stage gates are real. Show output, stop, wait.
- When a published number does not reproduce, say so plainly with the numbers. Do not round toward
  the paper.
- Prefer flagging a suspected leak over a good result. Rule 2 exists because the failure mode of
  this project is a meaningless 98%.
- The author is an M.Tech scholar writing a thesis — results must be defensible in a viva, which
  means every number traceable to a config, a seed, a split file, and a git SHA.
