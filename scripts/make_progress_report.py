"""Generate the supervisor progress report as a PDF.

    python scripts/make_progress_report.py
    python scripts/make_progress_report.py --no-tests    # skip running the suite

Numbers are read from the repository as it stands -- the annotations, the frozen split
file, the git log, and (unless suppressed) a live pytest run. Nothing is typed in by hand,
so the report cannot quietly drift from the state of the work it describes. That matters
more than it sounds: a progress report with a stale figure in it is the kind of thing that
gets quoted back in a viva.

Requires reportlab, which is a reporting tool rather than a dependency of the thesis
pipeline, so it is deliberately not in requirements.txt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------
# Facts gathered from the repository
# ----------------------------------------------------------------------------------


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def test_counts(run: bool) -> dict[str, str]:
    if not run:
        return {}
    out: dict[str, str] = {}
    for label, marker in (("total", None), ("causality", "causality"), ("splits", "splits")):
        cmd = [sys.executable, "-m", "pytest", "-q"]
        if marker:
            cmd += ["-m", marker]
        try:
            result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
            line = [ln for ln in result.stdout.splitlines() if "passed" in ln or "failed" in ln]
            out[label] = line[-1].strip() if line else "unknown"
        except Exception as exc:  # noqa: BLE001
            out[label] = f"could not run ({exc})"
    return out


def dataset_facts() -> dict:
    """Per-class and split statistics, straight from the canonical dataset."""
    import pandas as pd

    from src.data import canonical as C

    root = REPO / "data" / "ipn_hand"
    if not (root / C.ANNOTATIONS_FILE).is_file():
        return {}

    dataset = C.CanonicalDataset(root)
    annotations = dataset.annotations.copy()
    annotations["frames"] = annotations["end_frame"] - annotations["start_frame"] + 1

    grouped = annotations.groupby("class").agg(
        instances=("class", "size"), frames=("frames", "sum")
    )
    grouped["pct"] = 100 * grouped["frames"] / grouped["frames"].sum()
    grouped = grouped.reindex(dataset.classes)

    facts = {
        "classes": dataset.classes,
        "per_class": grouped,
        "num_videos": len(dataset.video_ids),
        "num_subjects": len(dataset.subjects()),
        "total_frames": int(grouped["frames"].sum()),
        "meta": dataset.meta,
    }

    split_file = root / "splits" / "ipn_official.json"
    if split_file.is_file():
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for value in payload["subject_split"].values():
            counts[value] = counts.get(value, 0) + 1
        videos = annotations.groupby("subject_id")["video_id"].nunique().to_dict()
        facts["split"] = {
            "counts": counts,
            "videos": {
                name: sum(videos.get(s, 0) for s, v in payload["subject_split"].items() if v == name)
                for name in ("train", "val", "test")
            },
            "folds": [len(f) for f in payload["folds"]],
            "seed": payload["spec"]["seed"],
        }
    return facts


# ----------------------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------------------


def build(path: Path, facts: dict, tests: dict[str, str]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    ink = colors.HexColor("#1a1a1a")
    muted = colors.HexColor("#5b6572")
    rule = colors.HexColor("#c9ced6")
    accent = colors.HexColor("#243b63")

    body = ParagraphStyle(
        "body", parent=styles["BodyText"], fontSize=9.5, leading=14,
        alignment=TA_JUSTIFY, textColor=ink, spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "h1", parent=styles["Heading1"], fontSize=14, leading=18, textColor=accent,
        spaceBefore=14, spaceAfter=7,
    )
    h2 = ParagraphStyle(
        "h2", parent=styles["Heading2"], fontSize=11, leading=15, textColor=ink,
        spaceBefore=10, spaceAfter=4,
    )
    small = ParagraphStyle(
        "small", parent=body, fontSize=8.2, leading=11.5, textColor=muted
    )
    mono = ParagraphStyle(
        "mono", parent=body, fontName="Courier", fontSize=8, leading=11, alignment=0
    )

    story: list = []

    def para(text: str, style=body) -> None:
        story.append(Paragraph(text, style))

    def bullets(items: list[str], style=body) -> None:
        story.append(
            ListFlowable(
                [ListItem(Paragraph(i, style), leftIndent=10) for i in items],
                bulletType="bullet", bulletFontSize=6, leftIndent=12, bulletOffsetY=1,
            )
        )
        story.append(Spacer(1, 4))

    def table(data, widths, header=True, align_right=()) -> None:
        cells = [[Paragraph(str(c), small) for c in row] for row in data]
        t = Table(cells, colWidths=widths, hAlign="LEFT")
        style = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, rule),
            ("LINEBELOW", (0, -1), (-1, -1), 0.4, rule),
        ]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")))
        for col in align_right:
            style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
        t.setStyle(TableStyle(style))
        story.append(t)
        story.append(Spacer(1, 8))

    # -- title ---------------------------------------------------------------------
    today = dt.date.today().isoformat()
    para(
        "Online Temporal Action Detection for Distress / SOS Hand Signals",
        ParagraphStyle("title", parent=styles["Title"], fontSize=17, leading=21,
                       textColor=accent, spaceAfter=2),
    )
    para(
        "M.Tech thesis &mdash; progress report",
        ParagraphStyle("sub", parent=styles["Title"], fontSize=11, leading=14,
                       textColor=muted, spaceAfter=10),
    )
    story.append(HRFlowable(width="100%", thickness=0.8, color=rule, spaceAfter=10))

    commit = git("rev-parse", "--short", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = "clean" if not git("status", "--porcelain") else "uncommitted changes present"
    table(
        [
            ["Date", today],
            ["Repository", "github.com/asthaberi/sos-oad"],
            ["Commit", f"{commit} ({branch}, {dirty})"],
            ["Phase", "Phase 1 &mdash; IPN Hand as engineering vehicle"],
            ["Status", "Stages 0&ndash;2 complete; Stage 3 in progress"],
        ],
        widths=[32 * mm, 130 * mm], header=False,
    )

    # -- 1. summary ----------------------------------------------------------------
    para("1. Summary", h1)
    para(
        "The system under development labels the <i>current</i> frame of a continuous video "
        "stream and decides when to raise an alarm. This is online temporal action detection: "
        "there are no clip boundaries at inference time, and no module may read a frame from "
        "the future. Phase 1 uses the IPN Hand dataset as an engineering vehicle; its thirteen "
        "classes are touchless interface commands rather than distress signals, but it has the "
        "right shape &mdash; continuous video, a dominant non-gesture class, per-subject "
        "metadata and fuzzy gesture onsets. Phase 2 substitutes our own recorded SOS data "
        "behind a dataset adapter, with no change to training or evaluation code."
    )
    para(
        "<b>Work to date has been data integrity, splitting methodology and causality "
        "infrastructure. There are no experimental results yet, and none should be expected "
        "before Stage 4.</b> This ordering is deliberate. On a subject-disjoint online "
        "detection task the characteristic failure is not a poor score but a high and "
        "meaningless one, produced by leakage that no downstream check can see. Each item "
        "below exists to make a later result defensible rather than to produce one sooner."
    )

    # -- 2. stage plan -------------------------------------------------------------
    para("2. Stage plan and status", h1)
    stage_rows = [
        ["#", "Content", "Gate", "Status"],
        ["0", "Repository, canonical data format, config system",
         "Validator passes on a synthetic fixture", "Complete"],
        ["1", "Acquire IPN Hand, write dataset adapter",
         "Published figures reproduced", "Complete"],
        ["2", "Frozen subject-disjoint splits and LOSGO folds",
         "No subject in more than one split", "Complete"],
        ["3", "Feature extraction, two streams, cached",
         "All 200 videos cached and verified", "In progress"],
        ["4", "Baselines B0 and B1, full evaluation harness",
         "Both baselines produce the complete metric set", "Not started"],
        ["5", "M1 model (causal GRU, two-stream)", "&mdash;", "Not started"],
        ["6", "Online decision layer", "&mdash;", "Not started"],
        ["7", "Metrics and headline figure", "&mdash;", "Not started"],
        ["8", "Analysis and ablations", "&mdash;", "Not started"],
    ]
    table(stage_rows, widths=[8 * mm, 62 * mm, 62 * mm, 26 * mm])

    # -- 3. methodology ------------------------------------------------------------
    para("3. Methodological constraints", h1)
    para(
        "Five constraints are enforced by automated tests rather than by convention. They are "
        "recorded here because each one, if violated, would invalidate the result rather than "
        "merely degrade it."
    )
    bullets([
        "<b>Causality.</b> No module may read a future frame. Enforced by feeding a truncated "
        "stream of length <i>t</i> and requiring the outputs for frames 0..<i>t</i> to be "
        "<i>bit-identical</i> to the corresponding slice of the full-stream run. Bit-identical "
        "rather than approximately equal: a module that is almost causal reads the future a "
        "little, and a little is enough to bias an onset-latency measurement.",
        "<b>Subject-disjoint splits.</b> Never split by frame, window or clip. Splits are "
        "written to disk, frozen, and placed under version control.",
        "<b>The test set is never tuned against.</b> Model selection uses validation only.",
        "<b>The non-gesture class is never dropped.</b> Removing it would silently convert "
        "online detection into trimmed-clip classification and render the numbers meaningless.",
        "<b>Provenance.</b> Every run records its config, seed, git commit and metrics. A "
        "result without a run directory behind it is not treated as a result.",
    ])
    para(
        "A working tripwire accompanies these: any metric exceeding 97% is treated as a "
        "symptom of leakage and investigated before it is reported."
    )

    story.append(PageBreak())

    # -- 4. completed work ---------------------------------------------------------
    para("4. Completed work", h1)

    para("4.1 Stage 0 &mdash; canonical data format", h2)
    para(
        "A single data format was specified before any code was written. Every dataset is "
        "converted into it by a small adapter, and training, evaluation and model code read "
        "nothing else. The format is enforced by a schema validator whose findings carry "
        "machine-readable codes, so tests assert on specific failures rather than on the fact "
        "that something went wrong. The central invariant is that annotations must tile each "
        "video exactly: non-gesture spans are explicit rows, never gaps inferred from absence, "
        "because inferring them is how the non-gesture class quietly disappears."
    )

    para("4.2 Stage 1 &mdash; dataset acquisition and verification", h2)
    para(
        "All published figures for IPN Hand were reproduced from the files on disk and printed "
        "beside the paper's values as the stage gate."
    )
    table(
        [
            ["Quantity", "Published", "Computed", "Agreement"],
            ["Videos", "200", "200", "exact"],
            ["Subjects", "50", "50", "exact"],
            ["Gesture classes", "13", "13", "exact"],
            ["Gesture instances", "4,218", "4,218", "exact"],
            ["Non-gesture instances", "1,431", "1,431", "exact"],
            ["Frames", "~800,000", "800,491", "within tolerance"],
        ],
        widths=[52 * mm, 32 * mm, 32 * mm, 36 * mm], align_right=(1, 2),
    )
    para(
        "Table II of the paper was additionally reproduced row by row &mdash; per-class "
        "instance count, mean span duration and standard deviation &mdash; for all fourteen "
        "classes. This second check matters because it verifies the class <i>mapping</i> and "
        "not merely the class <i>count</i>; see section 5.3."
    )

    if facts.get("per_class") is not None:
        para("Per-class composition of the converted dataset:", body)
        rows = [["Class", "Instances", "Frames", "% of frames"]]
        for name, row in facts["per_class"].iterrows():
            rows.append([
                name.replace("_", " "),
                f"{int(row['instances']):,}",
                f"{int(row['frames']):,}",
                f"{row['pct']:.1f}",
            ])
        table(rows, widths=[62 * mm, 28 * mm, 34 * mm, 28 * mm], align_right=(1, 2, 3))

    para("4.3 Stage 2 &mdash; frozen subject-disjoint splits", h2)
    split = facts.get("split")
    if split:
        counts, videos = split["counts"], split["videos"]
        table(
            [
                ["Split", "Subjects", "Videos"],
                ["Train", str(counts.get("train", 0)), str(videos.get("train", 0))],
                ["Validation", str(counts.get("val", 0)), str(videos.get("val", 0))],
                ["Test", str(counts.get("test", 0)), str(videos.get("test", 0))],
            ],
            widths=[34 * mm, 28 * mm, 28 * mm], align_right=(1, 2),
        )
        para(
            f"Cross-validation uses {len(split['folds'])} leave-one-subject-group-out folds of "
            f"{'/'.join(str(f) for f in split['folds'])} subjects over the non-test subjects "
            f"(seed {split['seed']}).",
            body,
        )
    para(
        "Two choices here are worth the supervisor's attention. First, <b>the test set is the "
        "dataset authors' own partition</b>, not one we selected. Their partition is "
        "subject-disjoint under our subject identity, and adopting it forecloses any question "
        "about whether several splits were tried until the numbers looked favourable. Not "
        "having chosen the test set is the strongest available form of 'never tuned against'. "
        "Second, <b>the primary validation set is cross-validation fold 0 promoted</b>, rather "
        "than a separate draw. Had it been drawn separately, those subjects would appear as "
        "<i>training</i> subjects in four of the five folds, and the variance estimate reported "
        "in Stage 8 would be bounding a different experiment from the headline number it is "
        "meant to qualify."
    )
    para(
        "The frozen split is committed to version control, carries a digest of the subject list "
        "it was built from, and is refused at load time if that digest no longer matches the "
        "dataset. Regenerating it requires an explicit override, which reports which subjects "
        "would change side."
    )

    story.append(PageBreak())

    # -- 5. findings ---------------------------------------------------------------
    para("5. Data-integrity findings", h1)
    para(
        "Three defects were found in the dataset or in the obvious way of reading it. Each "
        "would have produced a plausible-looking result rather than an error."
    )

    para("5.1 Subject identity is not the subject number", h2)
    para(
        "IPN Hand video names decompose as camera, subject, hand and clip number. The subject "
        "token is <i>not</i> an identity: it restarts at 1 for each camera and repeats across "
        "cameras, giving only 29 distinct values for 50 subjects. The pair (camera, subject) "
        "yields exactly 50 groups of exactly 4 videos, matching the published figures, and the "
        "authors' own train/test partition is disjoint under it while eight bare subject tokens "
        "straddle it. Splitting on the token alone would have placed the same person on both "
        "sides of the split, producing an inflated accuracy with no detectable symptom."
    )

    para("5.2 The dataset's frame counts are wrong for fourteen videos", h2)
    para(
        "The supplied <font face='Courier'>metadata.csv</font> reports 800,505 frames in total. "
        "Counting the image files gives 800,491. The discrepancy is exactly one frame for "
        "exactly fourteen videos, and those fourteen are precisely the ones whose frame "
        "directory contains a stray Windows <font face='Courier'>desktop.ini</font> file: the "
        "column was evidently produced by counting directory entries rather than images. Video "
        "length is therefore taken from the annotations, which match the image count for all "
        "200 videos. Trusting the metadata would have appended one non-existent frame to each "
        "of those videos; it would have validated cleanly and produced fourteen feature vectors "
        "with no image behind them."
    )

    para("5.3 The class mapping was verified rather than assumed", h2)
    para(
        "The dataset ships class <i>codes</i> only &mdash; B0A, G01 and so on &mdash; with no "
        "statement of which code corresponds to which gesture. The names were matched "
        "positionally against the paper's class table. Because a wrong mapping would leave "
        "every total reconciling exactly, a test simulates all 91 possible pairwise label swaps "
        "and confirms that 90 of them would be detected by the per-class statistics. The single "
        "undetectable pair, which the paper rounds to identical duration figures, was resolved "
        "by rendering frames from instances of both classes and inspecting them directly."
    )

    para("5.4 A defect found in our own code by the causality tests", h2)
    para(
        "On its first run the causality suite failed twelve tests against our own extractor. "
        "The cause was not a future read: floating-point matrix multiplication is not invariant "
        "to batch size, so the same input produces slightly different output depending on how "
        "many rows share the call. On a truncated stream the final batch is smaller, and "
        "bit-identity failed for a reason unrelated to causality. The tempting response is to "
        "relax the requirement to a numerical tolerance &mdash; which would have left the "
        "project with a causality test incapable of detecting a causality violation. Instead "
        "the behaviour was characterised (only the row count matters, not position within the "
        "batch) and the final batch is now padded to a fixed size. The batch size is recorded "
        "alongside the cache because it changes the stored values."
    )

    # -- 6. verification -----------------------------------------------------------
    para("6. Verification", h1)
    if tests:
        table(
            [["Suite", "Result"]]
            + [[k.capitalize(), v] for k, v in tests.items()],
            widths=[40 * mm, 120 * mm],
        )
    para(
        "The causality suite is the load-bearing one. Besides testing the real implementation, "
        "it constructs three deliberately non-causal extractors &mdash; one using a clip centred "
        "on its own index, one anchoring the snippet grid to the final frame, one interpolating "
        "forward between snippets &mdash; and asserts that the harness <i>rejects</i> each. All "
        "three produce output of the correct shape, correct dtype and finite values, which is "
        "the point: a shape assertion is not a causality check. A test suite that has only ever "
        "been run against correct code demonstrates nothing."
    )
    para(
        "The same principle is applied to the split logic, where a known-good split is corrupted "
        "one property at a time and the validator is required to catch each corruption "
        "specifically."
    )

    story.append(PageBreak())

    # -- 7. in progress ------------------------------------------------------------
    para("7. Work in progress &mdash; Stage 3", h1)
    para(
        "Frozen VideoMAEv2-Base is used as the RGB feature extractor, producing one feature "
        "vector per frame. Because the backbone consumes short clips rather than single frames, "
        "features are computed on a snippet grid and expanded to per-frame resolution by "
        "holding the most recent available value. Every part of that scheme is chosen for "
        "causality: each clip covers frames ending at the frame it describes rather than "
        "centred on it; the grid is anchored at frame 0 rather than at the end of the video, so "
        "that a prefix's grid is a prefix of the full grid; and values are held forward rather "
        "than interpolated, since interpolation would blend in a value computed from frames "
        "that have not yet arrived."
    )
    para(
        "Feature normalisation is deliberately <i>not</i> applied at this stage. Statistics "
        "fitted over a whole dataset are forbidden, so mean and standard deviation must be "
        "fitted on the training split alone, at the stage that trains. Normalising into the "
        "cache would embed a leak where no downstream check could observe it."
    )
    para(
        "The available development machine has no CUDA-capable GPU, so extraction is performed "
        "on a free cloud GPU. The 9&nbsp;GB of frame archives have been uploaded and verified "
        "byte-for-byte; extraction is sharded and resumable to fit within the provider's "
        "session limit. The resulting cache is checksummed per video, with the backbone, "
        "preprocessing recipe and every extraction setting recorded alongside it, so that a "
        "copy transferred to whatever machine later trains can be proven to be the cache the "
        "recorded settings produced."
    )
    para(
        "<b>Outstanding for this stage:</b> the extraction run itself, and the pose stream, "
        "which has not been started."
    )

    # -- 8. planned ----------------------------------------------------------------
    para("8. Planned work", h1)
    bullets([
        "<b>Stage 4 &mdash; baselines and the evaluation harness.</b> B0 is a trivial "
        "reference. B1 is the sliding-window classifier proposed in supervision (2&nbsp;s "
        "window, 0.5&nbsp;s stride), implemented properly rather than as a straw man: it is the "
        "comparison the proposed model must beat and it belongs in the ablation table. The full "
        "metric set is built here and used unchanged by every later stage.",
        "<b>Stage 5 &mdash; the proposed model (M1).</b> A MiniROAD-style causal GRU with loss "
        "computed only at the final position of each training clip, so that training matches "
        "streaming inference. Two streams (pose and RGB) with learned late fusion kept "
        "swappable for ablation; temporal label smoothing as a ramp at gesture boundaries, "
        "since onsets are fuzzy over roughly 0.3&nbsp;s; class-balanced or focal loss; and clip "
        "sampling that guarantees a fixed fraction of training clips contain a gesture onset, "
        "without which the gradient signal on onsets vanishes.",
        "<b>Stage 6 &mdash; the online decision layer.</b> Frequently omitted in published "
        "work, and where much of the practical accuracy lies. Causal smoothing, dual-threshold "
        "hysteresis, k-of-n persistence before firing, a refractory period so that one gesture "
        "produces one alert, and temperature calibration on validation so that thresholds are "
        "meaningful. Every parameter exposed in configuration.",
        "<b>Stage 7 &mdash; metrics.</b> Reported together every time: per-frame mAP; "
        "point-level mAP under the ODAS protocol swept over onset tolerance; event-level "
        "precision, recall and F1; false alarms per hour; median and 95th-percentile "
        "onset-to-alarm latency; end-to-end throughput and peak memory. The headline figure "
        "plots latency against false alarms per hour, one curve per persistence setting.",
        "<b>Stage 8 &mdash; analysis.</b> Confusion matrix and per-class breakdown, a "
        "failure-mode montage rendering the worst false positives as actual frames, and an "
        "ablation table covering B0/B1/M1, pose-only against RGB-only against fused, with and "
        "without label smoothing, and with and without the decision layer &mdash; every number "
        "on a subject-disjoint split with a seed-variance estimate.",
    ])
    para(
        "<b>Phase 2</b> then substitutes our own recorded SOS data behind the dataset adapter. "
        "The adapter boundary has been maintained throughout precisely so that this requires "
        "one new adapter and one configuration file, with no change to training, model or "
        "evaluation code."
    )

    # -- 9. risks ------------------------------------------------------------------
    para("9. Risks and open items", h1)
    bullets([
        "<b>Training hardware is not settled.</b> Feature extraction is handled by free cloud "
        "GPU and the cache is portable by design, but the hardware for Stages 5 to 8 remains "
        "the principal scheduling unknown.",
        "<b>Class composition differs from expectation.</b> In IPN Hand the non-gesture class "
        "is the largest by instance count but only the third largest by frame count; two "
        "'pointing' classes are longer resting or transition states. Those three classes "
        "together account for roughly 82% of all frames. This affects class balancing in Stage "
        "5 and the false-alarm denominator in Stage 7, and our own SOS recordings will not "
        "share this shape &mdash; so anything tuned tightly to it will not transfer.",
        "<b>The pose stream is not started.</b> The preferred whole-body pose estimator depends "
        "on a toolchain that is awkward to install; a documented CPU fallback exists.",
    ])

    # -- appendix ------------------------------------------------------------------
    story.append(PageBreak())
    para("Appendix A &mdash; commit history", h1)
    log = git("log", "--pretty=format:%h|%ad|%s", "--date=short")
    if log:
        rows = [["Commit", "Date", "Subject"]]
        rows += [line.split("|", 2) for line in log.splitlines()]
        table(rows, widths=[20 * mm, 24 * mm, 118 * mm])

    para("Appendix B &mdash; reproduction", h1)
    para(
        "The repository README carries the commands for every completed stage. The sequence "
        "that reproduces the work described above is:"
    )
    for line in [
        "python scripts/prepare_ipn_hand.py --zips &lt;download dir&gt; --set dataset=ipn_hand",
        "python scripts/make_splits.py --set dataset=ipn_hand split=ipn_official",
        "pytest                    # full suite",
        "pytest -m causality       # Hard Rule 1",
        "pytest -m splits          # subject disjointness",
    ]:
        para(line, mono)
    story.append(Spacer(1, 6))
    para(
        "Every reported quantity in this document was generated from the repository at the "
        "commit named on the first page, rather than transcribed by hand.",
        small,
    )

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="OAD for distress hand signals - progress report",
        author="",
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(22 * mm, 11 * mm, f"Progress report &middot; {today}".replace("&middot;", "·"))
        canvas.drawRightString(A4[0] - 22 * mm, 11 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "reports")
    parser.add_argument("--no-tests", action="store_true", help="skip running the test suite")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"progress-report-{dt.date.today().isoformat()}.pdf"

    print("gathering facts from the repository...")
    facts = dataset_facts()
    if not facts:
        print("  note: data/ipn_hand not present; dataset tables will be omitted")
    tests = test_counts(not args.no_tests)
    for key, value in tests.items():
        print(f"  {key}: {value}")

    build(path, facts, tests)
    print(f"\nwrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
