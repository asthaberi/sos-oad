# Online Temporal Action Detection for Distress Hand Signals in Continuous Video

## Interim Progress Report

*M.Tech Thesis — Panel Review*

---

## 1. Introduction

### 1.1 Motivation

A person in danger is frequently unable to call for help by voice. They may be under
surveillance by the person threatening them, in a public space where speech is not private,
or physically prevented from speaking. A silent, deliberate hand gesture is in such cases
the only channel available.

This observation is not hypothetical. The "Signal for Help" hand gesture, introduced in
2020 as a one-handed, speech-free indication of domestic distress, has since been used in
real incidents and recognised by bystanders. Its effectiveness, however, depends entirely
on a human being present, looking in the right direction, and knowing what the gesture
means. Camera infrastructure is already widespread in the environments where such a signal
would be made; what is absent is the ability to interpret it automatically.

This project addresses that gap: the automatic detection of distress hand signals in a
continuous video stream, in real time.

### 1.2 Problem statement

The task is **online temporal action detection (OAD)**. Given a video stream arriving frame
by frame, the system must assign a label to the **current** frame using only frames that
have already arrived, and decide when to raise an alarm.

This is distinct from two superficially similar and much better-studied problems:

- **Action recognition** operates on pre-trimmed clips. The clip boundaries are given. In a
  deployed system no such boundaries exist, and establishing them is most of the difficulty.
- **Offline temporal action localisation** has access to the complete video and may refine a
  predicted boundary using later frames. A deployed alarm system cannot wait for the video
  to end.

The defining constraint of this work is therefore **causality**: no component may use
information from a future frame. This constraint is easy to violate accidentally during
development, because during development the entire video file is available. A model that
peeks forward reports excellent accuracy and is useless in deployment.

### 1.3 Objectives

1. Design a strictly causal, two-stream (appearance and pose) online detector for hand
   gestures in continuous video.
2. Implement an explicit online decision layer governing when an alarm is raised, rather
   than treating a per-frame classifier output as an alarm.
3. Evaluate under a protocol appropriate to an alarm system — including false alarms per
   hour and onset-to-alarm latency, not accuracy alone.
4. Establish the result on a subject-disjoint split, with every reported number traceable
   to a configuration, a random seed, a frozen split file and a source-code revision.
5. Build the system so that the distress-signal dataset can be substituted later without
   modification to the model, training or evaluation code.

### 1.4 Scope and two-phase structure

Recording a purpose-built distress-signal dataset is a substantial undertaking and is
deferred. The work is therefore structured in two phases.

**Phase 1 (current)** develops and validates the complete system using the **IPN Hand**
dataset as an engineering vehicle. Its thirteen classes are touchless interface commands —
pointing, clicking, swiping, zooming — and are explicitly *not* distress signals. It is
used because it has the correct structural properties: continuous unsegmented video, a
dominant non-gesture class, per-subject metadata permitting subject-disjoint evaluation,
and gesture onsets that are genuinely ambiguous rather than cleanly annotated.

**Phase 2** substitutes purpose-recorded distress-signal data behind a dataset adapter
boundary, with no change required to downstream code.

This separation allows the methodological contributions to be developed and tested
rigorously before the domain data exists.

### 1.5 Expected contributions

1. A strictly causal two-stream online detector, with causality enforced by automated test
   rather than asserted in prose.
2. An online decision layer treated as a first-class component and systematically ablated.
3. An evaluation protocol for alarm-oriented detection, reporting the latency / false-alarm
   trade-off as a curve rather than a single operating point.
4. A dataset-agnostic implementation permitting substitution of the target domain data.

---

## 2. Literature Review

### 2.1 Problem taxonomy

Video understanding tasks differ principally in what information is available at inference.

| Task | Input at inference | Future frames | Representative work |
|---|---|---|---|
| Action recognition | Pre-trimmed clip | Available | Two-Stream, I3D, TSM |
| Offline temporal localisation | Whole video | Available | BSN, BMN |
| **Online action detection** | **Stream, prefix only** | **Unavailable** | **TRN, OadTR, LSTR, MiniROAD** |
| Action anticipation | Stream, predicts ahead | Unavailable | RED |

Online action detection was formalised by De Geest et al. (2016), who introduced the task
alongside the TVSeries dataset and demonstrated that methods performing well on trimmed
clips degrade substantially when boundaries are withheld.

### 2.2 Online action detection methods

Early approaches adapted recurrent architectures. **RED** (Gao et al., 2017) used a
reinforced encoder–decoder for anticipation, establishing that modelling the near future
improves present-frame classification. **TRN** (Xu et al., 2019) made this explicit by
predicting future representations and feeding them back to inform the current decision —
notably, without violating causality, since the predicted future is inferred rather than
observed.

Subsequent work addressed the limited temporal horizon of recurrent models. **IDN** (Eun et
al., 2020) learned to discriminate relevant from irrelevant accumulated information.
**OadTR** (Wang et al., 2021) and **LSTR** (Xu et al., 2021) introduced transformer
architectures, the latter separating long-term and short-term memory to extend effective
context. **TeSTra** (Zhao and Krähenbühl, 2022) addressed the computational cost of
attention over long histories using temporal smoothing kernels permitting streaming
inference at constant cost per frame.

Most relevant to this work, **MiniROAD** (An et al., 2023) observed that much of the
reported advantage of transformer approaches arises from a training/inference mismatch in
recurrent baselines: during training, loss is typically applied at every position of a
clip, whereas at inference only the final position is ever used. Applying non-uniform loss
weighting — concentrating the objective at the final position — a minimal GRU attains
competitive accuracy at a fraction of the computational cost. This finding directly
motivates the architecture adopted here.

### 2.3 Online hand gesture detection

Work specific to continuous hand gesture streams is comparatively sparse. Molchanov et al.
(2016) addressed online detection and classification of dynamic hand gestures with a
recurrent 3D CNN and connectionist temporal classification, introducing the NVGesture
dataset. Köpüklü et al. (2019) proposed a two-model architecture in which a lightweight
detector gates a heavier classifier, explicitly addressing the single-time activation
problem — ensuring one gesture produces exactly one output rather than a burst.

Benitez-Garcia et al. (2020) introduced **IPN Hand**, the dataset used in Phase 1 of this
work, specifically to support continuous rather than trimmed evaluation. The dataset
provides long sequences containing multiple gestures separated by labelled non-gesture
intervals.

### 2.4 Feature representations

Online detectors conventionally operate on features extracted by a frozen, pretrained
backbone rather than on raw pixels, which decouples representation learning from temporal
modelling and makes experimentation tractable.

**TSM** (Lin et al., 2019) achieves temporal modelling at 2D CNN cost through channel
shifting, and supports a causal (uni-directional) shift variant. **X3D** (Feichtenhofer,
2020) provides an efficient 3D architecture family. **VideoMAE** (Tong et al., 2022) and
**VideoMAE V2** (Wang et al., 2023) apply masked autoencoding to video, yielding strong
transferable representations; VideoMAE V2 is adopted here as the appearance backbone.

For the pose stream, **RTMPose** (Jiang et al., 2023) provides real-time whole-body keypoint
estimation including hand keypoints, with **MediaPipe Hands** (Zhang et al., 2020) as a
CPU-viable alternative.

### 2.5 Evaluation protocols

Per-frame mean average precision is the conventional OAD metric but is a poor proxy for
alarm-system utility: it rewards correctly labelling the long interior of a gesture, which
is of little operational interest once an alarm has been raised.

Point-level evaluation under the ODAS protocol instead scores whether a detection occurs
within a tolerance window of the gesture onset, which corresponds more closely to the
practical requirement. For an alarm system, two further quantities are decisive and are
frequently omitted: the **false-alarm rate per unit time**, which determines whether the
system is tolerable in deployment, and the **onset-to-alarm latency**, which determines
whether it is useful. These trade off against one another, and reporting either in
isolation is uninformative.

### 2.6 Research gap

Three gaps motivate this work.

1. **Automatic distress-signal detection is largely unaddressed.** The Signal for Help
   gesture is documented in public-safety literature, but computational work on detecting
   it in continuous video is minimal.
2. **The decision layer is routinely neglected.** Published OAD work typically reports
   per-frame metrics and omits the mechanism converting a per-frame score sequence into
   discrete alarms — thresholding, persistence, refractory behaviour — despite this
   mechanism determining practical performance.
3. **Alarm-oriented metrics are under-reported.** False alarms per hour and onset latency
   are seldom given, making published methods difficult to assess for deployment.

---

## 3. Methodology

### 3.1 System overview

The pipeline comprises four stages:

1. **Feature extraction.** Each frame is represented by a feature vector from a frozen
   pretrained backbone, computed causally.
2. **Temporal model.** A causal recurrent network maintains state across the stream and
   emits per-frame class scores.
3. **Decision layer.** Per-frame scores are converted into discrete alarm events by causal
   smoothing, hysteresis thresholding, persistence and refractory logic.
4. **Evaluation.** Detected events are scored against ground truth under the protocol in
   §3.8.

### 3.2 Dataset and canonical format

Phase 1 uses IPN Hand: 200 videos, 50 subjects, 13 gesture classes plus a non-gesture
class, 640×480 at 30 fps.

To permit later substitution of the distress-signal dataset, every dataset is converted by
a small adapter into a single **canonical format** — per-frame feature arrays, an
annotation table of closed frame intervals with subject identifiers, a class list, and a
metadata file. All model, training and evaluation code reads only this format and never
references a dataset by name. Adding a dataset requires one adapter and one configuration
file.

A schema validator enforces the format. Its central invariant is that annotations must tile
each video exactly: non-gesture intervals are stored as explicit labelled rows, never
inferred from absence. This prevents the non-gesture class from silently disappearing,
which would reduce the task to trimmed-clip classification.

### 3.3 The causality constraint

No component may use information from a future frame. This excludes bidirectional recurrent
networks, centred convolutions and filters, unmasked full-sequence attention, and
normalisation statistics computed over a complete sequence.

The constraint is enforced by an automated test rather than by inspection. The test
supplies a truncated stream of length *t* and requires the outputs for frames 0…*t* to be
**bit-identical** to the corresponding slice of a full-stream run. Bit-identity rather than
numerical closeness is required, since a module that is approximately causal is one that
uses future information slightly, which is sufficient to bias a latency measurement.

To establish that the test itself has discriminative power, three deliberately non-causal
implementations are constructed and the harness is required to **reject** each: a clip
centred on its own index, a snippet grid anchored to the final frame of the video, and
forward interpolation between snippets. All three produce output of correct shape, correct
type and finite value — a test suite that has only ever been exercised against correct code
establishes nothing.

### 3.4 Subject-disjoint splitting

All evaluation uses subject-disjoint splits: no subject appears in more than one of train,
validation and test. Splitting by frame, window or clip would place frames of the same
gesture instance on both sides of the split, and the resulting metric would measure
memorisation of individuals rather than generalisation to new ones.

Splits are generated once, written to disk, and placed under version control. The test
partition adopted is the one published by the dataset authors, which was verified to be
subject-disjoint under the subject identity established in §4.3. Adopting rather than
selecting the test partition removes the possibility that several candidate splits were
evaluated before one was retained.

Cross-validation uses leave-one-subject-group-out (LOSGO) over the non-test subjects. The
primary validation set is defined as one of these folds, promoted, rather than drawn
separately — otherwise those subjects would act as *training* subjects in the remaining
folds and the variance estimate would characterise a different experiment from the headline
result it is intended to qualify.

### 3.5 Feature extraction

Two parallel streams are extracted.

The **appearance stream** uses frozen VideoMAE V2 (Base). As the backbone consumes short
clips rather than single frames, features are computed on a snippet grid and expanded to
per-frame resolution. Each element of this scheme is dictated by causality: each clip spans
frames *ending at* the frame it describes rather than centred upon it; the snippet grid is
anchored at frame 0 rather than at the end of the video, so that the grid for a stream
prefix is a prefix of the grid for the full stream; and values are **held forward** between
grid positions rather than interpolated, since interpolation would incorporate a value
computed from frames not yet arrived.

The **pose stream** extracts hand and upper-body keypoints, normalised per person by
translation to a root joint and scaling by a reference length, so that the representation is
invariant to subject distance from the camera.

Feature normalisation is deliberately *not* applied during extraction. Statistics fitted
over a complete dataset constitute a form of leakage; mean and standard deviation are
therefore fitted on the training split alone, at the stage that trains.

### 3.6 Proposed model

The proposed model (M1) is a causal gated recurrent network applied to the fused feature
streams, following the MiniROAD finding that **non-uniform loss weighting** — applying the
training objective only at the final position of each clip — aligns training with streaming
inference and recovers most of the advantage otherwise attributed to transformer
architectures.

Further design elements:

- **Two-stream late fusion.** Separate heads on pose and appearance features with learned
  fusion of logits, with the fusion mechanism kept substitutable for ablation.
- **Temporal label smoothing.** Gesture onsets are ambiguous over approximately 0.3 s;
  targets are therefore ramped across the boundary rather than stepped.
- **Class-balanced objective.** The class distribution is severely skewed (§4.4).
- **Onset-aware clip sampling.** A fixed proportion of training clips is required to contain
  a gesture onset; without this constraint, onsets constitute a vanishing fraction of
  sampled positions and receive negligible gradient.

### 3.7 Online decision layer

Converting per-frame scores into alarms is treated as a distinct, ablatable component with
all parameters exposed in configuration:

- **Causal smoothing** by exponential moving average or causal median. Centred filters are
  excluded, as they incorporate future frames.
- **Dual-threshold hysteresis** — an alarm is raised at a high threshold and sustained to a
  lower one, preventing oscillation near a single threshold.
- **k-of-n persistence** before firing, suppressing isolated spurious frames.
- **Refractory period** following an alarm, so that one gesture yields one alert.
- **Temperature calibration** on validation data, so that thresholds carry consistent
  meaning across models.

### 3.8 Evaluation protocol

The following are reported together for every configuration:

- per-frame mean average precision;
- point-level mean average precision (ODAS protocol), swept over onset tolerance from 1 to
  10 seconds;
- event-level precision, recall and F1;
- **false alarms per hour**;
- **median and 95th-percentile onset-to-alarm latency**;
- end-to-end throughput and peak memory.

The headline figure plots latency against false alarms per hour, one curve per persistence
setting, since no single operating point characterises an alarm system.

### 3.9 Reproducibility

Every run writes a directory recording the fully resolved configuration, the source-code
revision and working-tree cleanliness, the random seeds, the software environment and the
resulting metrics. Experiment tracking mirrors scalars to this directory, so that results
remain traceable independently of any hosted service. No quantity affecting a result is
hard-coded outside configuration files.

A working safeguard is applied throughout: any metric exceeding 97% is treated as evidence
of leakage and investigated before it is reported. On a subject-disjoint online detection
task such a value is implausible, and the characteristic failure mode of this class of
project is not a poor result but an excellent and meaningless one.

---

## 4. Results and Evaluation

### 4.1 Status and scope of this section

Work to date has established the data foundations, the splitting methodology and the
causality infrastructure. **Model training has not yet commenced, and no detection
performance figures are available.** This section reports what has been evaluated: the
integrity of the dataset, the correctness of the split, and the validity of the causality
guarantee.

This ordering is deliberate. Each of the defects reported in §4.3 would have produced a
plausible but incorrect performance figure rather than a visible error.

### 4.2 Dataset verification

Every published figure for IPN Hand was recomputed from the distributed files and compared
against the values reported by Benitez-Garcia et al. (2020).

| Quantity | Published | Computed | Agreement |
|---|---|---|---|
| Videos | 200 | 200 | exact |
| Subjects | 50 | 50 | exact |
| Gesture classes | 13 | 13 | exact |
| Gesture instances | 4,218 | 4,218 | exact |
| Non-gesture instances | 1,431 | 1,431 | exact |
| Frames | ~800,000 | 800,491 | within tolerance |

Table II of the source publication was additionally reproduced row by row — per-class
instance count, mean span duration and standard deviation — for all fourteen classes. This
verifies the class *mapping* and not merely the class *count*; see §4.3.

### 4.3 Data integrity findings

Three defects were identified in the dataset or in the conventional means of reading it.

**(i) Subject identity is not the subject index.** IPN Hand video identifiers decompose into
camera, subject, hand and clip fields. The subject field is not a global identity: it
restarts at 1 for each camera and repeats across cameras, yielding only 29 distinct values
for 50 subjects. The pair (camera, subject) yields exactly 50 groups of exactly 4 videos,
consistent with the published figures, and the authors' own partition is disjoint under it
while eight bare subject indices straddle it. Splitting on the index alone would have placed
the same individual on both sides of the split.

**(ii) The distributed frame counts are incorrect for fourteen videos.** The supplied
metadata reports 800,505 frames; the image files number 800,491. The discrepancy is exactly
one frame for exactly fourteen videos, and those fourteen are precisely those whose frame
directory contains a stray operating-system metadata file. The count was evidently produced
by enumerating directory entries rather than images. Video length is therefore taken from
the annotations, which agree with the image count for all 200 videos.

**(iii) The class-code mapping required verification.** The dataset distributes class
*codes* without stating which code denotes which gesture; names were matched positionally
against the published class table. Because an incorrect mapping would leave every aggregate
count consistent, all 91 pairwise label exchanges were simulated, confirming that 90 would
be detected by the per-class statistics. The single undetectable pair, for which the
published duration figures round identically, was resolved by direct inspection of video
frames.

A fourth defect was identified in the implementation itself. On first execution the
causality suite rejected the feature extractor. The cause was not a future read: floating
point matrix multiplication is not invariant to batch size, and a truncated stream produces
a smaller final batch. The available responses were to relax the bit-identity requirement to
a numerical tolerance, or to characterise and eliminate the dependency. The former would
have left a causality test incapable of detecting a causality violation; the latter was
adopted.

### 4.4 Dataset composition

The converted dataset comprises 800,491 frames across 200 videos and 50 subjects.

| Class | Instances | Frames | % of frames |
|---|---|---|---|
| none | 1,431 | 210,542 | 26.3 |
| pointing with one finger | 1,010 | 221,193 | 27.6 |
| pointing with two fingers | 1,007 | 225,683 | 28.2 |
| click with one finger | 200 | 11,197 | 1.4 |
| click with two fingers | 200 | 11,935 | 1.5 |
| throw up | 200 | 12,400 | 1.5 |
| throw down | 201 | 13,095 | 1.6 |
| throw left | 200 | 13,226 | 1.7 |
| throw right | 200 | 12,761 | 1.6 |
| open twice | 200 | 15,126 | 1.9 |
| double click with one finger | 200 | 13,525 | 1.7 |
| double click with two fingers | 200 | 13,910 | 1.7 |
| zoom in | 200 | 13,033 | 1.6 |
| zoom out | 200 | 12,865 | 1.6 |

The distribution has a consequence for method design. The non-gesture class is the largest
by instance count but only the third largest by frame count: the two pointing classes are
long resting or transitional states with a median duration of approximately seven seconds,
rather than brief commands. Together these three classes account for roughly 82% of all
frames, leaving the eleven command gestures approximately 18%. This bears directly on class
balancing and on the false-alarm denominator, and the distress-signal data of Phase 2 is not
expected to share this structure.

### 4.5 Split composition

| Split | Subjects | Videos |
|---|---|---|
| Train | 30 | 120 |
| Validation | 7 | 28 |
| Test | 13 | 52 |

Cross-validation uses five LOSGO folds of 7–8 subjects over the 37 non-test subjects. The
split is frozen to disk, carries a digest of the subject list from which it was derived, and
is refused at load time should that digest cease to match the dataset.

### 4.6 Verification status

| Suite | Tests | Purpose |
|---|---|---|
| Causality | 41 | Hard constraint of §3.3, including three negative controls |
| Splits | 36 | Subject disjointness and fold integrity |
| Full suite | 211 | All of the above plus format, adapter and cache validation |

All tests pass at the revision reported on the title page.

---

## 5. Remaining Work

| Stage | Content | Status |
|---|---|---|
| 0 | Canonical data format and configuration system | Complete |
| 1 | Dataset acquisition, adapter, verification | Complete |
| 2 | Frozen subject-disjoint splits and LOSGO folds | Complete |
| 3 | Two-stream feature extraction and caching | In progress |
| 4 | Baselines and full evaluation harness | Not started |
| 5 | Proposed model (M1) | Not started |
| 6 | Online decision layer | Not started |
| 7 | Metrics and headline figure | Not started |
| 8 | Ablations and failure analysis | Not started |

Stage 4 includes a sliding-window classifier baseline, implemented to a standard permitting
fair comparison rather than as a strawman, since it represents the natural first approach to
the problem and belongs in the ablation table.

**Principal risk.** Hardware for model training is not yet settled. Feature extraction is
performed on a free cloud GPU service and the resulting feature cache is portable by design
and verified by checksum; the hardware for Stages 5 to 8 remains the primary scheduling
uncertainty.

---

## 6. References

1. De Geest, R., Gavves, E., Ghodrati, A., Li, Z., Snoek, C., Tuytelaars, T. (2016). Online
   Action Detection. *ECCV*.
2. Gao, J., Yang, Z., Nevatia, R. (2017). RED: Reinforced Encoder-Decoder Networks for
   Action Anticipation. *BMVC*.
3. Xu, M., Gao, M., Chen, Y.-T., Davis, L., Crandall, D. (2019). Temporal Recurrent Networks
   for Online Action Detection. *ICCV*.
4. Eun, H., Moon, J., Park, J., Jung, C., Kim, C. (2020). Learning to Discriminate
   Information for Online Action Detection. *CVPR*.
5. Wang, X., Zhang, S., Qing, Z., Shao, Y., Zuo, Z., Gao, C., Sang, N. (2021). OadTR: Online
   Action Detection with Transformers. *ICCV*.
6. Xu, M., Xiong, Y., Chen, H., Li, X., Xia, W., Tu, Z., Soatto, S. (2021). Long Short-Term
   Transformer for Online Action Detection. *NeurIPS*.
7. Zhao, Y., Krähenbühl, P. (2022). Real-Time Online Video Detection with Temporal Smoothing
   Transformers. *ECCV*.
8. An, J., Kang, H., Han, S.H., Yang, M.-H., Kim, S.J. (2023). MiniROAD: Minimal RNN
   Framework for Online Action Detection. *ICCV*.
9. Molchanov, P., Yang, X., Gupta, S., Kim, K., Tyree, S., Kautz, J. (2016). Online Detection
   and Classification of Dynamic Hand Gestures with Recurrent 3D Convolutional Neural
   Networks. *CVPR*.
10. Köpüklü, O., Gunduz, A., Kose, N., Rigoll, G. (2019). Real-time Hand Gesture Detection
    and Classification Using Convolutional Neural Networks. *FG*.
11. Benitez-Garcia, G., Olivares-Mercado, J., Sanchez-Perez, G., Yanai, K. (2020). IPN Hand:
    A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition. *ICPR*.
12. Lin, J., Gan, C., Han, S. (2019). TSM: Temporal Shift Module for Efficient Video
    Understanding. *ICCV*.
13. Feichtenhofer, C. (2020). X3D: Expanding Architectures for Efficient Video Recognition.
    *CVPR*.
14. Tong, Z., Song, Y., Wang, J., Wang, L. (2022). VideoMAE: Masked Autoencoders are
    Data-Efficient Learners for Self-Supervised Video Pre-Training. *NeurIPS*.
15. Wang, L., Huang, B., Zhao, Z., Tong, Z., He, Y., Wang, Y., Wang, Y., Qiao, Y. (2023).
    VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking. *CVPR*.
16. Jiang, T., Lu, P., Zhang, L., Ma, N., Han, R., Lyu, C., Li, Y., Chen, K. (2023).
    RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. *arXiv*.
17. Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L.,
    Grundmann, M. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. *arXiv*.
