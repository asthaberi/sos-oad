const pptx = require("pptxgenjs");
const path = require("path");

const REPO = "C:/Users/astha/Projects/sos-oad";
const OUT = path.join(REPO, "reports", "first-review-presentation.pptx");

// Palette: navy for surveillance/video, signal red for alarm. Not generic blue.
const DEEP = "131A3C";
const NAVY = "1E2761";
const ICE = "CADCFC";
const RED = "D7263D";
const PAPER = "FFFFFF";
const TINT = "F2F5FA";
const INK = "1A1A1A";
const MUTED = "5B6572";

const HEAD = "Cambria";
const BODY = "Calibri";

const W = 13.33;
const M = 0.62;

const deck = new pptx();
deck.layout = "LAYOUT_WIDE";
deck.author = "";
deck.title = "Online Temporal Action Detection for Distress Hand Signals";

const shadow = () => ({ type: "outer", blur: 10, offset: 2, angle: 90, color: "9AA5B5", opacity: 0.3 });

function dark(slide) {
  slide.background = { color: DEEP };
}

function titleOn(slide, text, kicker) {
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: M, y: 0.42, w: 9, h: 0.26, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11, bold: true, color: RED, charSpacing: 2,
    });
  }
  slide.addText(text, {
    x: M, y: kicker ? 0.70 : 0.5, w: W - 2 * M, h: 0.98, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 32, bold: true, color: NAVY,
  });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(deck.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || TINT }, line: { color: "E2E8F2", width: 0.75 },
    shadow: shadow(),
  });
}

function numDot(slide, x, y, n, colour) {
  slide.addShape(deck.ShapeType.ellipse, {
    x, y, w: 0.42, h: 0.42, fill: { color: colour || NAVY }, line: { color: colour || NAVY },
  });
  slide.addText(String(n), {
    x, y, w: 0.42, h: 0.42, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle",
  });
}

function note(slide, text) { slide.addNotes(text); }

/* ---------------------------------------------------------------- 1. Title */
{
  const s = deck.addSlide(); dark(s);
  s.addText("Online Temporal Action Detection for", {
    x: M, y: 2.0, w: 11.5, h: 0.6, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 30, color: ICE,
  });
  s.addText("Distress Hand Signals in Continuous Video", {
    x: M, y: 2.62, w: 11.5, h: 0.8, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 38, bold: true, color: "FFFFFF",
  });
  s.addShape(deck.ShapeType.ellipse, { x: M, y: 3.78, w: 0.17, h: 0.17, fill: { color: RED }, line: { color: RED } });
  s.addText("M.Tech Thesis  ·  First Review Presentation", {
    x: M + 0.32, y: 3.68, w: 9, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, color: ICE,
  });
  s.addText("Name  ·  Roll number  ·  Supervisor  ·  Department, Institution", {
    x: M, y: 5.9, w: 11.5, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, color: "8FA3C4", italic: true,
  });
  note(s, "Replace the last line with your actual details before presenting.");
}

/* ------------------------------------------------------------ 2. Motivation */
{
  const s = deck.addSlide();
  titleOn(s, "A person in danger often cannot speak", "Motivation");
  const items = [
    ["Under observation", "The person threatening them is present and watching."],
    ["No privacy", "A public space where a phone call would be overheard."],
    ["Physically prevented", "Unable to reach a phone or speak freely."],
  ];
  items.forEach(([h, b], i) => {
    const y = 2.0 + i * 1.18;
    card(s, M, y, 6.6, 1.0);
    numDot(s, M + 0.26, y + 0.29, i + 1);
    s.addText(h, { x: M + 0.86, y: y + 0.16, w: 5.5, h: 0.32, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 15, bold: true, color: NAVY });
    s.addText(b, { x: M + 0.86, y: y + 0.5, w: 5.5, h: 0.36, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12, color: MUTED });
  });
  card(s, 7.6, 2.0, 5.1, 3.36, NAVY);
  s.addText("A silent hand gesture\nis the only channel left", {
    x: 7.9, y: 2.35, w: 4.5, h: 1.0, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 21, bold: true, color: "FFFFFF", lineSpacing: 28,
  });
  s.addText("The \u201cSignal for Help\u201d gesture was introduced in 2020 as a one-handed, speech-free indication of domestic distress.", {
    x: 7.9, y: 3.5, w: 4.5, h: 1.0, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, color: ICE,
  });
  s.addText("Verify and cite a source for this before presenting.", {
    x: 7.9, y: 4.72, w: 4.5, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10, color: "8FA3C4", italic: true,
  });
  note(s, "Keep this short. The panel does not need convincing that distress happens; they need to see you have a specific, well-defined signal to detect.");
}

/* ------------------------------------------------------------------ 3. Gap */
{
  const s = deck.addSlide();
  titleOn(s, "The signal only works if someone is watching", "The problem");
  s.addText("Its effectiveness depends entirely on a human being present, looking in the right direction, and knowing what the gesture means.", {
    x: M, y: 1.78, w: 11.9, h: 0.6, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 16, color: INK,
  });
  const stats = [
    ["Already installed", "Camera infrastructure exists in the environments where such a signal would be made."],
    ["Not interpreted", "What is absent is the ability to recognise the signal automatically, in real time."],
  ];
  stats.forEach(([h, b], i) => {
    const x = M + i * 6.2;
    card(s, x, 2.7, 5.8, 1.9);
    s.addText(h, { x: x + 0.34, y: 2.98, w: 5.1, h: 0.36, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 16, bold: true, color: i ? RED : NAVY });
    s.addText(b, { x: x + 0.34, y: 3.42, w: 5.1, h: 0.9, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, color: MUTED });
  });
  s.addText("This project addresses that gap: automatic detection of distress hand signals in a continuous video stream, in real time.", {
    x: M, y: 5.1, w: 11.9, h: 0.6, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: NAVY,
  });
}

/* ----------------------------------------------- 4. Online vs offline (KEY) */
{
  const s = deck.addSlide();
  titleOn(s, "Detection in a stream, not classification of a clip", "Problem definition");
  const cols = [
    ["Action recognition", "G I V E N   A   C L I P", ["Boundaries are given", "\u201cWhat gesture is this?\u201d", "Well studied"], "97A3B6"],
    ["Offline localisation", "G I V E N   T H E   W H O L E   V I D E O", ["May refine using later frames", "Cannot wait for the video to end", "Not deployable"], "97A3B6"],
    ["Online detection", "G I V E N   A   S T R E A M", ["No boundaries exist", "\u201cIs it happening right now?\u201d", "This work"], RED],
  ];
  cols.forEach(([h, sub, pts, colour], i) => {
    const x = M + i * 4.08;
    const isOurs = i === 2;
    card(s, x, 1.95, 3.75, 3.6, isOurs ? NAVY : TINT);
    s.addText(h, { x: x + 0.26, y: 2.18, w: 3.2, h: 0.36, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 15, bold: true, color: isOurs ? "FFFFFF" : NAVY });
    s.addText(sub, { x: x + 0.26, y: 2.56, w: 3.3, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 8.5, bold: true, color: isOurs ? ICE : colour });
    s.addText(pts.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < pts.length - 1 } })), {
      x: x + 0.26, y: 2.95, w: 3.25, h: 2.3, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12.5, color: isOurs ? ICE : MUTED, paraSpaceAfter: 8,
    });
  });
  s.addText("The model is never told where a gesture starts. Establishing that is most of the difficulty.", {
    x: M, y: 5.78, w: 11.9, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 16, bold: true, color: NAVY,
  });
  note(s, "This is the single most important slide. If the panel understands only one thing, make it this distinction.");
}

/* ----------------------------------------------------------- 5. Causality */
{
  const s = deck.addSlide();
  titleOn(s, "The defining constraint: no peeking at the future", "Why it is hard");
  s.addText("During development the whole video file is available, so it is easy to write code that uses frame 510 to decide what is happening at frame 500. That model reports excellent accuracy and is useless in deployment \u2014 in a live camera, frame 510 has not happened yet.", {
    x: M, y: 1.8, w: 11.9, h: 0.8, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14.5, color: INK,
  });
  // timeline
  const tx = M, ty = 3.15, tw = 11.9;
  s.addShape(deck.ShapeType.rect, { x: tx, y: ty, w: tw * 0.62, h: 0.5, fill: { color: NAVY }, line: { color: NAVY } });
  s.addShape(deck.ShapeType.rect, { x: tx + tw * 0.62, y: ty, w: tw * 0.38, h: 0.5, fill: { color: "E8EDF5" }, line: { color: "D5DEEC" } });
  s.addText("frames already seen  \u2014  may be used", { x: tx + 0.2, y: ty, w: tw * 0.62 - 0.4, h: 0.5, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12.5, bold: true, color: "FFFFFF", valign: "middle" });
  s.addText("future frames  \u2014  forbidden", { x: tx + tw * 0.62 + 0.2, y: ty, w: tw * 0.38 - 0.4, h: 0.5, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12.5, bold: true, color: MUTED, valign: "middle" });
  s.addShape(deck.ShapeType.line, { x: tx + tw * 0.62, y: ty - 0.22, w: 0, h: 0.94, line: { color: RED, width: 2.5 } });
  s.addText("now", { x: tx + tw * 0.62 - 0.4, y: ty + 0.74, w: 0.8, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 11, bold: true, color: RED, align: "center" });
  card(s, M, 4.42, 11.9, 1.5);
  s.addText("How it is enforced", { x: M + 0.34, y: 4.62, w: 5, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 14, bold: true, color: NAVY });
  s.addText("Feed the model a truncated stream of length t, and require the outputs for frames 0\u2026t to be bit-identical to the same frames from a full-length run. If they differ, the model used information from later frames. Enforced by automated test, not by inspection.", {
    x: M + 0.34, y: 4.96, w: 11.2, h: 0.8, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, color: MUTED,
  });
}

/* ---------------------------------------------------------- 6. Objectives */
{
  const s = deck.addSlide();
  titleOn(s, "Objectives", "Scope");
  const obj = [
    "Design a strictly causal, two-stream (appearance and pose) online detector for hand gestures in continuous video.",
    "Implement an explicit online decision layer governing when an alarm is raised, rather than treating classifier output as an alarm.",
    "Evaluate under an alarm-appropriate protocol \u2014 false alarms per hour and onset-to-alarm latency, not accuracy alone.",
    "Establish results on a subject-disjoint split, every number traceable to a config, seed, split file and code revision.",
    "Build so the distress dataset can be substituted later with no change to model, training or evaluation code.",
  ];
  obj.forEach((t, i) => {
    const y = 1.92 + i * 0.86;
    numDot(s, M, y, i + 1, i === 4 ? RED : NAVY);
    s.addText(t, { x: M + 0.7, y: y - 0.02, w: 11.3, h: 0.62, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 14, color: INK });
  });
}

/* ------------------------------------------------------ 7. Related work I */
{
  const s = deck.addSlide();
  titleOn(s, "Where this task sits", "Related work");
  const rows = [
    ["Task", "Input at inference", "Future frames", "Examples"],
    ["Action recognition", "Pre-trimmed clip", "Available", "I3D, TSM"],
    ["Offline localisation", "Whole video", "Available", "BSN, BMN"],
    ["Online action detection", "Stream, prefix only", "Unavailable", "TRN, OadTR, LSTR, MiniROAD"],
    ["Action anticipation", "Stream, predicts ahead", "Unavailable", "RED"],
  ];
  s.addTable(
    rows.map((r, i) => r.map((c) => ({
      text: c,
      options: {
        bold: i === 0 || (i === 3 && true),
        color: i === 0 ? "FFFFFF" : (i === 3 ? NAVY : MUTED),
        fill: { color: i === 0 ? NAVY : (i === 3 ? "EAF0FA" : "FFFFFF") },
        fontSize: i === 0 ? 12.5 : 13,
      },
    }))),
    { x: M, y: 2.0, w: 11.9, colW: [3.0, 3.1, 2.2, 3.6], rowH: 0.52, border: { pt: 0.75, color: "DCE3EE" }, fontFace: BODY, valign: "middle", margin: 8 }
  );
  s.addText("Online action detection was formalised by De Geest et al. (2016), who showed that methods performing well on trimmed clips degrade substantially when boundaries are withheld.", {
    x: M, y: 5.3, w: 11.9, h: 0.6, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13.5, color: INK,
  });
}

/* ----------------------------------------------------- 8. Related work II */
{
  const s = deck.addSlide();
  titleOn(s, "How the field has moved", "Related work");
  const items = [
    ["Recurrent", "RED (2017), TRN (2019)", "Predict future representations and feed them back \u2014 inferred, not observed, so causality holds."],
    ["Transformer", "OadTR (2021), LSTR (2021)", "Longer effective context by separating long- and short-term memory."],
    ["Efficient streaming", "TeSTra (2022)", "Constant cost per frame via temporal smoothing kernels."],
    ["Back to RNN", "MiniROAD (2023)", "Much of the transformer gain came from a training/inference mismatch. Fixing the loss weighting lets a minimal GRU compete at a fraction of the cost."],
  ];
  items.forEach(([tag, who, what], i) => {
    const y = 1.95 + i * 1.02;
    const last = i === 3;
    card(s, M, y, 11.9, 0.88, last ? "EAF0FA" : TINT);
    s.addText(tag, { x: M + 0.3, y: y + 0.1, w: 2.1, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 11, bold: true, color: last ? RED : MUTED });
    s.addText(who, { x: M + 0.3, y: y + 0.42, w: 2.4, h: 0.32, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13.5, bold: true, color: NAVY });
    s.addText(what, { x: M + 2.95, y: y + 0.14, w: 8.7, h: 0.66, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12.5, color: last ? INK : MUTED });
  });
  s.addText("MiniROAD's finding motivates the architecture adopted here.", {
    x: M, y: 6.12, w: 11.9, h: 0.35, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, italic: true, color: NAVY,
  });
}

/* ------------------------------------------------------------- 9. The gap */
{
  const s = deck.addSlide();
  titleOn(s, "Three gaps this work addresses", "Research gap");
  const gaps = [
    ["Distress signals are unstudied", "The Signal for Help is documented in public-safety literature, but computational work on detecting it in continuous video is minimal."],
    ["The decision layer is skipped", "Published work reports per-frame metrics and omits the mechanism turning scores into alarms \u2014 thresholding, persistence, refractory behaviour \u2014 despite it determining practical performance."],
    ["Alarm metrics are not reported", "False alarms per hour and onset latency are seldom given, making published methods hard to assess for deployment."],
  ];
  gaps.forEach(([h, b], i) => {
    const x = M + i * 4.08;
    card(s, x, 2.0, 3.75, 3.5);
    numDot(s, x + 0.28, 2.26, i + 1, RED);
    s.addText(h, { x: x + 0.28, y: 2.85, w: 3.2, h: 0.72, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 15, bold: true, color: NAVY });
    s.addText(b, { x: x + 0.28, y: 3.62, w: 3.2, h: 1.7, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12, color: MUTED });
  });
  s.addText("Gaps 2 and 3 are where this thesis expects to contribute most.", {
    x: M, y: 5.78, w: 11.9, h: 0.4, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY,
  });
}

/* -------------------------------------------------------- 10. Two phases */
{
  const s = deck.addSlide();
  titleOn(s, "Two phases: build now, swap the data later", "Approach");
  s.addText("Recording a distress-signal dataset requires ethics approval and months of work. The system is therefore developed on a stand-in with the correct structure.", {
    x: M, y: 1.78, w: 11.9, h: 0.5, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 14, color: INK,
  });
  card(s, M, 2.5, 5.8, 3.3, TINT);
  s.addText("PHASE 1  \u2014  current", { x: M + 0.32, y: 2.74, w: 5, h: 0.28, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 10.5, bold: true, color: RED, charSpacing: 1 });
  s.addText("IPN Hand", { x: M + 0.32, y: 3.04, w: 5, h: 0.42, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 22, bold: true, color: NAVY });
  const p1 = ["200 videos, 50 subjects, ~800k frames", "13 touchless interface gestures \u2014 not distress signals", "Continuous video with a dominant non-gesture class", "Per-subject metadata for subject-disjoint evaluation"];
  s.addText(p1.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < p1.length - 1 } })), {
    x: M + 0.32, y: 3.56, w: 5.2, h: 2.0, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12.5, color: MUTED, paraSpaceAfter: 7,
  });
  card(s, 7.1, 2.5, 5.6, 3.3, NAVY);
  s.addText("PHASE 2  \u2014  later", { x: 7.42, y: 2.74, w: 5, h: 0.28, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 10.5, bold: true, color: ICE, charSpacing: 1 });
  s.addText("Recorded SOS data", { x: 7.42, y: 3.04, w: 5, h: 0.42, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 22, bold: true, color: "FFFFFF" });
  s.addText("Substituted behind a dataset adapter. Adding a dataset means one adapter file and one settings file \u2014 no change to model, training or evaluation code.", {
    x: 7.42, y: 3.6, w: 5.0, h: 1.1, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, color: ICE,
  });
  s.addText("Needs institutional ethics approval \u2014 application to begin now.", {
    x: 7.42, y: 4.85, w: 5.0, h: 0.6, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12, bold: true, color: "FFC9CF",
  });
  s.addText("Why a stand-in is legitimate: the research contribution is the detection method and its evaluation, neither of which depends on which gesture is being detected.", {
    x: M, y: 6.05, w: 11.9, h: 0.5, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, italic: true, color: NAVY,
  });
}

/* ------------------------------------------------ 11. Dataset + verification */
{
  const s = deck.addSlide();
  titleOn(s, "The dataset, verified against the paper", "Progress  \u00b7  Stage 1");
  s.addImage({ path: path.join(REPO, "reports/samples/1CM1_1_R_217_3367-3814_sheet.png"), x: M, y: 1.82, w: 11.9, h: 1.96 });
  s.addText("Ground-truth labels drawn onto the frames. The label changes exactly when the hand starts moving \u2014 a visual check that the annotation indexing is correct.", {
    x: M, y: 3.88, w: 11.9, h: 0.36, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTED,
  });
  const rows = [
    ["Quantity", "Published", "Computed"],
    ["Videos", "200", "200"],
    ["Subjects", "50", "50"],
    ["Gesture instances", "4,218", "4,218"],
    ["Non-gesture instances", "1,431", "1,431"],
    ["Frames", "~800,000", "800,491"],
  ];
  s.addTable(
    rows.map((r, i) => r.map((c) => ({
      text: c,
      options: { bold: i === 0, color: i === 0 ? "FFFFFF" : (i ? NAVY : MUTED), fill: { color: i === 0 ? NAVY : "FFFFFF" }, fontSize: 12 },
    }))),
    { x: M, y: 4.42, w: 6.5, colW: [2.9, 1.8, 1.8], rowH: 0.36, border: { pt: 0.75, color: "DCE3EE" }, fontFace: BODY, valign: "middle", margin: 6 }
  );
  card(s, 7.5, 4.42, 5.2, 2.16, "EAF0FA");
  s.addText("Table II also reproduced", { x: 7.78, y: 4.64, w: 4.7, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 14, bold: true, color: NAVY });
  s.addText("Per-class instance count, mean span duration and standard deviation, for all 14 classes. This verifies the class mapping, not merely the class count \u2014 the dataset ships codes (B0A, G01) without stating which gesture each denotes.", {
    x: 7.78, y: 5.0, w: 4.7, h: 1.4, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12, color: MUTED,
  });
}

/* -------------------------------------------------------- 12. Imbalance */
{
  const s = deck.addSlide();
  titleOn(s, "A finding that shapes the method", "Progress  \u00b7  Stage 1");
  s.addText("82%", { x: M, y: 1.95, w: 3.4, h: 1.3, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 76, bold: true, color: RED });
  s.addText("of all frames belong to just three classes", { x: M, y: 3.22, w: 3.6, h: 0.6, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 14, bold: true, color: NAVY });
  const bars = [
    ["pointing, two fingers", 28.2], ["pointing, one finger", 27.6], ["none (no gesture)", 26.3], ["the other 11 gestures", 17.9],
  ];
  bars.forEach(([label, pct], i) => {
    const y = 2.05 + i * 0.78;
    s.addText(label, { x: 4.5, y, w: 3.0, h: 0.32, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12.5, color: INK });
    s.addShape(deck.ShapeType.rect, { x: 7.6, y: y + 0.03, w: (pct / 30) * 4.3, h: 0.26, fill: { color: i === 3 ? "97A3B6" : NAVY }, line: { color: i === 3 ? "97A3B6" : NAVY } });
    s.addText(pct + "%", { x: 12.0, y, w: 0.8, h: 0.32, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12, bold: true, color: MUTED });
  });
  card(s, M, 5.42, 11.9, 1.12);
  s.addText("The two \u201cpointing\u201d classes are long resting states with a median duration of about seven seconds, not brief commands. This drives class balancing in the model and the false-alarm denominator in evaluation \u2014 and the recorded distress data will not share this shape.", {
    x: M + 0.32, y: 5.62, w: 11.3, h: 0.75, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, color: INK,
  });
}

/* --------------------------------------------------------- 13. Pipeline */
{
  const s = deck.addSlide();
  titleOn(s, "System pipeline", "Methodology");
  const steps = [
    ["Feature\nextraction", "Frozen VideoMAEv2 turns each frame into numbers, computed causally"],
    ["Temporal\nmodel", "Causal GRU keeps state across the stream, emits per-frame scores"],
    ["Decision\nlayer", "Smoothing, hysteresis, persistence, refractory period \u2192 discrete alarms"],
    ["Evaluation", "Events scored against ground truth: latency vs false alarms per hour"],
  ];
  steps.forEach(([h, b], i) => {
    const x = M + i * 3.12;
    card(s, x, 2.15, 2.78, 2.5, i === 2 ? NAVY : TINT);
    numDot(s, x + 0.24, 2.38, i + 1, i === 2 ? RED : NAVY);
    s.addText(h, { x: x + 0.24, y: 2.95, w: 2.3, h: 0.72, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 15, bold: true, color: i === 2 ? "FFFFFF" : NAVY });
    s.addText(b, { x: x + 0.24, y: 3.72, w: 2.3, h: 0.82, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 11.5, color: i === 2 ? ICE : MUTED });
    if (i < 3) {
      s.addShape(deck.ShapeType.rightArrow, { x: x + 2.84, y: 3.26, w: 0.22, h: 0.26, fill: { color: "A9B6CA" }, line: { color: "A9B6CA" } });
    }
  });
  s.addText("Every temporal operation in stages 1\u20133 is strictly causal, and each is covered by the truncation test.", {
    x: M, y: 4.95, w: 11.9, h: 0.4, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13.5, color: INK,
  });
  card(s, M, 5.5, 11.9, 1.05, "EAF0FA");
  s.addText("The decision layer is the contribution most often skipped in published work \u2014 it is treated here as a first-class, ablatable component with every parameter exposed in configuration.", {
    x: M + 0.32, y: 5.7, w: 11.3, h: 0.7, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, color: NAVY,
  });
}

/* -------------------------------------------------------- 14. Splitting */
{
  const s = deck.addSlide();
  titleOn(s, "Splitting by person, not by video", "Methodology");
  s.addText("If the same person appears in training and test, the model recognises the individual rather than the gesture. The score is inflated and the failure is invisible.", {
    x: M, y: 1.8, w: 11.9, h: 0.5, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 14.5, color: INK,
  });
  const sp = [["Train", "30 subjects", "120 videos", NAVY], ["Validation", "7 subjects", "28 videos", "47608E"], ["Test", "13 subjects", "52 videos", RED]];
  sp.forEach(([h, a, b, colour], i) => {
    const x = M + i * 4.08;
    card(s, x, 2.55, 3.75, 1.5, colour);
    s.addText(h, { x: x + 0.3, y: 2.75, w: 3.1, h: 0.34, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 15, bold: true, color: "FFFFFF" });
    s.addText(a + "   \u00b7   " + b, { x: x + 0.3, y: 3.16, w: 3.15, h: 0.52, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 16, bold: true, color: "FFFFFF" });
  });
  const pts = [
    ["The test set is the dataset authors\u2019 own partition, not one we chose.", "Nobody can ask whether several splits were tried until one looked favourable."],
    ["The split is frozen to disk and kept under version control.", "It carries a digest of the subject list and is refused if the dataset changes."],
    ["Cross-validation uses five leave-one-subject-group-out folds.", "The primary validation set is one of those folds promoted, so CV is a strict superset of the headline experiment."],
  ];
  pts.forEach(([h, b], i) => {
    const y = 4.3 + i * 0.78;
    s.addShape(deck.ShapeType.ellipse, { x: M + 0.04, y: y + 0.08, w: 0.13, h: 0.13, fill: { color: RED }, line: { color: RED } });
    s.addText(h, { x: M + 0.38, y, w: 11.4, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13.5, bold: true, color: NAVY });
    s.addText(b, { x: M + 0.38, y: y + 0.3, w: 11.4, h: 0.34, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12, color: MUTED });
  });
}

/* ------------------------------------------------------- 15. Stage status */
{
  const s = deck.addSlide();
  titleOn(s, "Where the work stands", "Progress");
  const rows = [
    ["Stage", "Content", "Status"],
    ["0", "Canonical data format and configuration system", "Complete"],
    ["1", "Dataset acquisition, adapter, verification", "Complete"],
    ["2", "Frozen subject-disjoint splits and LOSGO folds", "Complete"],
    ["3", "Two-stream feature extraction and caching", "In progress"],
    ["4", "Baselines and full evaluation harness", "Not started"],
    ["5", "Proposed model", "Not started"],
    ["6", "Online decision layer", "Not started"],
    ["7", "Metrics and headline figure", "Not started"],
    ["8", "Ablations and failure analysis", "Not started"],
  ];
  s.addTable(
    rows.map((r, i) => {
      const done = ["Complete"].includes(r[2]);
      const now = r[2] === "In progress";
      return r.map((c, j) => ({
        text: c,
        options: {
          bold: i === 0 || (j === 2 && (done || now)),
          color: i === 0 ? "FFFFFF" : (j === 2 ? (done ? "1E7B4F" : now ? RED : "97A3B6") : NAVY),
          fill: { color: i === 0 ? NAVY : (now ? "FDEEF0" : "FFFFFF") },
          fontSize: 12.5,
        },
      }));
    }),
    { x: M, y: 1.95, w: 11.9, colW: [1.1, 7.6, 3.2], rowH: 0.4, border: { pt: 0.75, color: "DCE3EE" }, fontFace: BODY, valign: "middle", margin: 7 }
  );
  s.addText("No model has been trained and no detection results exist yet. Stages 0\u20133 are foundations \u2014 and each defect on the next slide would have produced a plausible but wrong number rather than a visible error.", {
    x: M, y: 6.25, w: 11.9, h: 0.55, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13.5, color: INK,
  });
  note(s, "Expect 'why no results yet?'. Answer with the next slide.");
}

/* ------------------------------------------------------- 16. The findings */
{
  const s = deck.addSlide();
  titleOn(s, "Three defects found before they reached a result", "Progress  \u00b7  key outcome");
  const f = [
    ["Subject identity", "The subject number restarts for each camera and repeats across cameras \u2014 29 distinct values for 50 people. Splitting on it would have put the same person in training and test."],
    ["Frame counts", "The dataset\u2019s own metadata overcounts by one frame for exactly 14 videos \u2014 those whose folder contains a stray desktop.ini. It was produced by counting directory entries, not images."],
    ["Class mapping", "The dataset ships codes without saying which gesture each denotes. All 91 pairwise label swaps were simulated: 90 would be detected. The last was resolved by inspecting video frames."],
  ];
  f.forEach(([h, b], i) => {
    const y = 1.95 + i * 1.42;
    card(s, M, y, 11.9, 1.26);
    numDot(s, M + 0.3, y + 0.42, i + 1, RED);
    s.addText(h, { x: M + 0.92, y: y + 0.2, w: 2.6, h: 0.34, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 15, bold: true, color: NAVY });
    s.addText(b, { x: M + 3.6, y: y + 0.18, w: 8.0, h: 0.95, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12.5, color: MUTED });
  });
  s.addText("Each would have produced a confident, wrong number rather than an error message.", {
    x: M, y: 6.3, w: 11.9, h: 0.4, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 16, bold: true, color: RED,
  });
  note(s, "This is your strongest slide. It shows diligence rather than output, which is exactly what a first review should demonstrate.");
}

/* ------------------------------------------------------ 17. Verification */
{
  const s = deck.addSlide();
  titleOn(s, "How correctness is established", "Verification");
  const stats = [["211", "automated tests"], ["41", "causality tests"], ["36", "split tests"]];
  stats.forEach(([n, l], i) => {
    const x = M + i * 2.6;
    card(s, x, 1.95, 2.3, 1.35);
    s.addText(n, { x: x + 0.2, y: 2.06, w: 1.9, h: 0.72, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 36, bold: true, color: NAVY, align: "center" });
    s.addText(l, { x: x + 0.2, y: 2.8, w: 1.9, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 11.5, color: MUTED, align: "center" });
  });
  card(s, 8.5, 1.95, 4.2, 1.35, NAVY);
  s.addText("Every run records its config, seed, code revision and metrics.", {
    x: 8.78, y: 2.2, w: 3.7, h: 0.9, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, color: ICE,
  });
  s.addText("The causality suite proves it can detect a violation", {
    x: M, y: 3.55, w: 11.9, h: 0.36, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 16, bold: true, color: NAVY,
  });
  s.addText("Three deliberately non-causal implementations are built alongside the real one, and the test harness is required to reject each:", {
    x: M, y: 3.95, w: 11.9, h: 0.34, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, color: INK,
  });
  const neg = [
    ["Centred clip", "Uses frames either side of the one it describes"],
    ["End-anchored grid", "Results shift depending on how long the video turns out to be"],
    ["Forward interpolation", "Blends in a value computed from frames not yet arrived"],
  ];
  neg.forEach(([h, b], i) => {
    const x = M + i * 4.08;
    card(s, x, 4.4, 3.75, 1.2, "FDEEF0");
    s.addText(h, { x: x + 0.28, y: 4.58, w: 3.2, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13.5, bold: true, color: RED });
    s.addText(b, { x: x + 0.28, y: 4.9, w: 3.2, h: 0.6, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 11.5, color: MUTED });
  });
  s.addText("All three produce output of the correct shape, type and range. A test suite that has only ever been run against correct code establishes nothing.", {
    x: M, y: 5.78, w: 11.9, h: 0.5, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, italic: true, color: INK,
  });
}

/* ----------------------------------------------------------- 18. Closing */
{
  const s = deck.addSlide(); dark(s);
  s.addText("Next steps", {
    x: M, y: 0.62, w: 11.9, h: 0.6, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 32, bold: true, color: "FFFFFF",
  });
  const next = [
    ["Immediate", "Complete feature extraction on cloud GPU; begin baselines and the evaluation harness."],
    ["Then", "Proposed model, online decision layer, full metric set, ablations."],
    ["In parallel", "Begin the ethics approval application for Phase 2 data recording."],
  ];
  next.forEach(([h, b], i) => {
    const y = 1.72 + i * 1.08;
    s.addShape(deck.ShapeType.roundRect, { x: M, y, w: 7.4, h: 0.92, rectRadius: 0.08, fill: { color: "1C2550" }, line: { color: "2E3C72", width: 0.75 } });
    s.addText(h, { x: M + 0.3, y: y + 0.13, w: 6.8, h: 0.28, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 11, bold: true, color: RED, charSpacing: 1 });
    s.addText(b, { x: M + 0.3, y: y + 0.42, w: 6.8, h: 0.4, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 13, color: ICE });
  });
  s.addShape(deck.ShapeType.roundRect, { x: 8.3, y: 1.72, w: 4.4, h: 2.44, rectRadius: 0.08, fill: { color: "3A1620" }, line: { color: "6B2733", width: 0.75 } });
  s.addText("PRINCIPAL RISKS", { x: 8.6, y: 1.94, w: 3.9, h: 0.3, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 11, bold: true, color: "FF9AA6", charSpacing: 1 });
  const risks = ["Training hardware is not yet settled", "Ethics approval may take several months", "Pose stream toolchain is awkward to install"];
  s.addText(risks.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < risks.length - 1 } })), {
    x: 8.6, y: 2.34, w: 3.9, h: 1.7, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12.5, color: "FFD8DD", paraSpaceAfter: 9,
  });
  s.addText("Thank you", {
    x: M, y: 5.35, w: 6, h: 0.6, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 28, bold: true, color: "FFFFFF",
  });
  s.addText("Questions", {
    x: M, y: 5.95, w: 6, h: 0.4, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 15, color: ICE,
  });
}

deck.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
