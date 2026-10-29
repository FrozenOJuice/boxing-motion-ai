# Pose-observation reliability study

Development-source investigation, 2026-10-29; starting revision `fb961a9`.

**Decision:** retain production extraction. Cropping has selective experimental benefits, but
no tested variant establishes reliable far-arm recovery or clean combination separation.

## Scope and controls

This study compares the exact 19 cases used for the second punch-instance baseline:
the V6 six-second upload, V4 eight- and 29-second uploads, existing blank no-person control, V4-bg-1/2/3
with the existing context, and V4/V6-motion-1 through -6. The 11 human V4 events are
comparison references only. No V4-bg-4/5, V5/V7/V8/V10 motion or V9 is used. Original
poses, recordings, annotations, reviews, checkpoints, features and production extraction
remain unchanged. This is neither independent validation nor a new training experiment.

The diagnostic-only runner is `vision/observation_study.py`, version
`pose-observation-study-1.0`. It is not wired to the upload worker, dataset preparation or
production inference. Fresh ignored output directories are required. Existing v2 diagnostic
inputs, frame clocks and frozen segmentation code/policy are hash-bound. Native V4/V6 source
SHA-256 and decoded integer PTS are checked; uploads reuse their exact lossless, oriented,
960-pixel prepared frames. No workbook is needed for inference.

| Variant | Processing |
| --- | --- |
| Stored | Original immutable MediaPipe IMAGE poses, the v2 baseline |
| IMAGE 960 | Fresh Full MediaPipe IMAGE, uncropped, width capped at 960 |
| IMAGE 1920 | Same, width capped at 1920; no upscaling |
| Crop IMAGE | Fixed whole-case baseline valid-landmark union plus 15% margin, cap 960 |
| Crop VIDEO | Identical crop, MediaPipe VIDEO tracking, cap 960 |
| RTMPose crop | Existing body7 medium ONNX, same fixed crop, RGB 192×256 affine model input |

Fresh MediaPipe resizing uses Pillow RGB bilinear. Original native extraction used PyAV
YUV resizing, so fresh IMAGE 960 is a preprocessing control, not a promised bit-identical
copy of stored native poses. The uploads need no resize at either width cap; their high-cap
comparison cannot recover detail already discarded during preparation. Original native pixels
are available for the native-resolution comparison. MediaPipe model SHA-256 remains
`5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`;
RTMPose remains `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`.
No model or dependency is downloaded implicitly.

The crop uses the whole interval's **poses, never event labels**. This is an offline assisted
localization upper bound requiring a prior full-frame pose pass. It is not an implemented
causal boxer detector or identity tracker. Every pose is mapped back into original normalized
image coordinates, including crop-width scaling for MediaPipe Z. Raw crop landmarks/scores
are retained separately. Out-of-crop predictions cannot gain validity simply by mapping inside
the source canvas: their derived scores are zeroed; the untouched original scores remain in
`raw-landmarks.npz`. No coordinate clipping, interpolation or last-pose carry is performed.

VIDEO state restarts at case boundaries, baseline source/model discontinuities, scene/segment changes
and source timestamp gaps. It has access only to past/current frames, but its internal temporal
estimates are not proof of a presently visible joint. No additional smoothing is applied.

All segmentation arms use **unchanged V2 policy and original frozen activity bouts and frame observability**. This
isolates pose effects; it is not an end-to-end revalidation of the binary detector on a new
pose distribution. Changed MediaPipe masks still enforce the original 0.5 visibility/presence
rule, normalization and per-arm missingness. The segmenter never receives references or class
scores. The downstream six-class classifier is deliberately not run on experimental poses:
coordinate scaling back to the same schema does not establish distribution compatibility.

RTMPose scores have no calibrated equivalent to MediaPipe visibility/presence. Its raw coordinates
and scores are retained without inventing such a threshold. The segmentation-only sensitivity
comparison substitutes common-joint XY **only where original MediaPipe masks are valid** and
RTMPose is finite/in-source. Original confidence/masks are retained; invalid/outside RTMPose
slots retain baseline coordinates. This hybrid is explicitly **not a recovered-pose output**,
not a measure of RTMPose coverage, and never a classifier or production input. Raw RTMPose
visuals show finite estimates without claiming validity. It cannot demonstrate recovery of
missing joints; it asks whether changing coordinates alone helps the current segmentation.

## Findings and decision

**Do not replace production extraction.** Higher resolution did not improve reviewed matching.
Cropping and tracking have selective near-arm benefits and raise temporal matches from 2 to 4
of 11, but neither establishes accurate continuous far-arm recovery. RTMPose's mask-matched
coordinate ablation matches 3 of 11 and retains a background proposal. Keep the versioned
crop/tracking outputs as experimental diagnostic artifacts only; do not add a production pose
adapter, change validity thresholds, fill occluded joints, or tune further variants on this set.
The study implements separate, source-provenance-preserving experimental extraction; it does
not certify any variant for recognition.

There were **2,635 frames × five fresh variants**, plus stored-pose comparisons:
2,610 V4/V6 frames and 25 frames of the existing synthetic blank no-person control per variant. Native V4 is
oblique bag work, not a pure shadowboxing benchmark. V6 is the clearer shadowboxing example
but has no exhaustive positive instance reference here. All model calls were real local inference.

### Frozen V2 segmentation and reviewed timing

Matches use the existing one-to-one exact-time IoU ≥ 0.5 rule inside exhaustively reviewed cores.
Counts below refer to candidates intersecting those cores; unreviewed context is not background.
“Merge risk” means at least 25% overlap with two reviewed punches, not verified separation.

| Pose variant | V6 / V4-8s / V4-29s upload candidates | Native V4 candidates | Matches / 11 | Misses | Background proposals | Merge risks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stored baseline | 3 / 1 / 3 | 4 | 2 | 9 | 1 | 2 |
| Fresh IMAGE 960 | 3 / 1 / 3 | 3 | 2 | 9 | 0 | 2 |
| IMAGE 1920 | 3 / 1 / 3 | 3 | 2 | 9 | 0 | 2 |
| Crop IMAGE | 4 / 1 / 2 | 5 | 4 | 7 | 0 | 3 |
| Crop VIDEO | 4 / 2 / 4 | 5 | 4 | 7 | 0 | 4 |
| RTMPose XY ablation | 4 / 0 / 6 | 4 | 3 | 8 | 1 | 2 |

No split risks were found under the existing multiple-peaks-within-one-reference diagnostic;
this does not establish correct counting. All 12 verified negative intervals (1,151 frames,
48.006 s) produce zero final candidates in every comparison, **with the original activity gate
held fixed**. These negative results do not establish new pose specificity or a false-positive
rate. Geometrically supported phases rejected only by the activity gate number 6 / 6 / 6 /
7 / 9 / 9 for stored / IMAGE 960 / IMAGE 1920 / crop IMAGE / crop VIDEO / RTMPose ablation.
These are motion phases, not verified punches. The no-person control remains unobservable,
not confirmed non-punch exposure.

The fresh IMAGE control reproduces stored coordinates, scores, masks and clocks **exactly on
all three uploads**. The 1920 cap gives identical arrays there too: no extra pixels exist.
For native sources, the RGB-vs-historical-YUV resize change alone removes the baseline's
background proposal and changes one matched boundary. Attribute neither effect to cropping.

Neither crop method matches any of the three reviewed **lead hooks**, or the one lead uppercut.
Crop IMAGE matches the V4-bg-2 jab and two rear uppercuts plus one V4-bg-3 rear uppercut.
Crop VIDEO instead matches one V4-bg-1, one V4-bg-2 and two V4-bg-3 rear uppercuts.
Do not union these outputs into a retrospectively selected ensemble. The scalar reach-phase
baseline, phase ambiguity and activity gating remain limitations beyond missing landmarks.

Matched full-reference boundary errors below are `(start, stop)` in source frames; positive is
late. Matching itself uses the clipped reviewed core, while these errors retain full known
human event bounds. V4-bg-2's final event still extends four frames beyond the exhaustive core.

| Variant | Case / reference | IoU in reviewed core | Boundary error frames |
| --- | --- | ---: | --- |
| Stored baseline | V4-bg-2 rear_uppercut [81,95) | 0.667 | (-2, +5) |
| Stored baseline | V4-bg-3 rear_uppercut [92,102) | 0.667 | (-1, +4) |
| Fresh IMAGE 960 | V4-bg-2 rear_uppercut [81,95) | 0.500 | (-6, -4) |
| Fresh IMAGE 960 | V4-bg-3 rear_uppercut [92,102) | 0.667 | (-1, +4) |
| IMAGE 1920 | V4-bg-2 rear_uppercut [81,95) | 0.538 | (-2, +10) |
| IMAGE 1920 | V4-bg-3 rear_uppercut [92,102) | 0.667 | (-1, +4) |
| Crop IMAGE | V4-bg-2 jab [64,80) | 0.550 | (-4, -5) |
| Crop IMAGE | V4-bg-2 rear_uppercut [81,95) | 0.538 | (-2, +10) |
| Crop IMAGE | V4-bg-2 rear_uppercut [111,122) | 0.636 | (-4, +0) |
| Crop IMAGE | V4-bg-3 rear_uppercut [92,102) | 0.769 | (+0, +3) |
| Crop VIDEO | V4-bg-1 rear_uppercut [42,66) | 0.727 | (+0, +9) |
| Crop VIDEO | V4-bg-2 rear_uppercut [81,95) | 0.519 | (-3, +10) |
| Crop VIDEO | V4-bg-3 rear_uppercut [92,102) | 0.714 | (-1, +3) |
| Crop VIDEO | V4-bg-3 rear_uppercut [108,115) | 0.500 | (-3, +6) |
| RTMPose XY ablation | V4-bg-2 rear_uppercut [81,95) | 0.538 | (-2, +10) |
| RTMPose XY ablation | V4-bg-3 jab [82,92) | 0.538 | (-3, -3) |
| RTMPose XY ablation | V4-bg-3 rear_uppercut [92,102) | 0.714 | (-1, +3) |

Ranges in that table are case-local; add source offsets 17,098 / 22,860 / 11,368 for
V4-bg-1/2/3 respectively. Exact PTS/timebases remain in the bound video/pose records.
For example, crop IMAGE's V4-bg-2 jab proposal [60,75) peaks at frame 63, before the
reviewed [64,80) onset. Its matched rear-uppercut [79,105) extends ten frames beyond
[81,95), into the adjacent hook. Passing IoU is not proof of clean individual isolation.
The tracker adds a V4-bg-1 rear-uppercut match [42,75), but that also includes nine frames
of the following hook. These are substantial boundary failures, not recovered complete punches.

### Mask support versus accurate recovery

Counts below use the original validity/normalization policy, original frame observability and
source resets. They are **mask-supported upper bounds**, not verified anatomically usable joints.
They include diagnostic context. Longest complete runs use `[first PTS, next PTS after last)`
support, including the last frame's duration; v2's earlier last-center-span figures were one frame
shorter. V2 tracking-jump resets also break runs. Right/left are anatomical channels, not certified
punching-arm or lead/rear assignments.

| Variant | Case | L wrist / elbow / complete frames | R wrist / elbow / complete frames | Longest L / R complete run (s) |
| --- | --- | --- | --- | --- |
| Stored baseline | V4-bg-1 | 120 / 143 / 120 | 62 / 70 / 62 | 2.336 / 1.084 |
| Stored baseline | V4-bg-2 | 138 / 142 / 138 | 78 / 69 / 67 | 5.005 / 1.418 |
| Stored baseline | V4-bg-3 | 120 / 142 / 120 | 65 / 50 / 48 | 4.129 / 0.584 |
| Stored baseline | V6-upload | 144 / 144 / 144 | 140 / 138 / 138 | 6.005 / 4.087 |
| Fresh IMAGE 960 | V4-bg-1 | 123 / 143 / 123 | 59 / 69 / 57 | 2.336 / 0.792 |
| Fresh IMAGE 960 | V4-bg-2 | 138 / 142 / 138 | 80 / 71 / 67 | 3.878 / 1.418 |
| Fresh IMAGE 960 | V4-bg-3 | 119 / 143 / 119 | 64 / 52 / 51 | 4.129 / 0.584 |
| Fresh IMAGE 960 | V6-upload | 144 / 144 / 144 | 140 / 138 / 138 | 6.005 / 4.087 |
| IMAGE 1920 | V4-bg-1 | 122 / 143 / 122 | 64 / 74 / 62 | 2.336 / 1.251 |
| IMAGE 1920 | V4-bg-2 | 139 / 142 / 139 | 83 / 69 / 68 | 5.005 / 1.418 |
| IMAGE 1920 | V4-bg-3 | 120 / 143 / 120 | 64 / 51 / 50 | 4.129 / 0.584 |
| IMAGE 1920 | V6-upload | 144 / 144 / 144 | 140 / 138 / 138 | 6.005 / 4.087 |
| Crop IMAGE | V4-bg-1 | 132 / 143 / 132 | 54 / 48 / 46 | 4.505 / 0.668 |
| Crop IMAGE | V4-bg-2 | 139 / 141 / 139 | 74 / 66 / 62 | 5.380 / 1.293 |
| Crop IMAGE | V4-bg-3 | 129 / 141 / 129 | 64 / 57 / 57 | 4.547 / 0.751 |
| Crop IMAGE | V6-upload | 144 / 144 / 144 | 140 / 140 / 140 | 6.005 / 4.128 |
| Crop VIDEO | V4-bg-1 | 135 / 143 / 135 | 37 / 34 / 28 | 4.755 / 0.459 |
| Crop VIDEO | V4-bg-2 | 142 / 142 / 142 | 70 / 57 / 57 | 5.922 / 1.710 |
| Crop VIDEO | V4-bg-3 | 128 / 143 / 128 | 47 / 23 / 23 | 5.297 / 0.751 |
| Crop VIDEO | V6-upload | 144 / 144 / 144 | 144 / 144 / 144 | 6.005 / 6.005 |

Cropping extends V4-bg-1's near/left complete run from 2.336 to 4.505 s, while far/right
complete frames fall from 62 to 46. Tracking raises that near run to 4.755 s but leaves only
28 far/right complete frames. In V4-bg-3 tracking leaves only 23 right-elbow frames of 143.
Fewer mask passes could be appropriate abstention from a hidden arm, rather than worse
localization; more passes could be hallucination. Neither sign alone proves accuracy.
V6 tracking extends the right complete run to all 6.005 s, but does not correct every visible
elbow. RTMPose coverage is deliberately not tabulated as recovery: its ablation retains the
baseline masks and confidence, and therefore cannot demonstrate filling missing observations.
Per-reviewed-event wrist/elbow fractions and every reset/extremum remain in comparison.json.

### Visual evidence, including counterexamples

Inspected the raw/overlay panels at all 11 V4 event midpoints and six evenly spaced frames
from each of the three uploads (29 distinct frame rows), plus six raw enlarged frames before
variant comparison, enlarged V6 frame 111, and the no-person control. Cyan is left, magenta right,
orange indicates a failed MediaPipe mask. RTMPose panels explicitly show raw estimates without
validity claims. These are **assistant visual observations**, not new human reviews or keypoint GT.

| Exact source example | Observed result and limitation |
| --- | --- |
| V6 upload frame 111, fast extension | All variants still place the right elbow very near the glove/wrist rather than following the visible upper-arm bend. Stored elbow visibility .995/presence .999 and crop elbow .988/.997 both pass. Tracking's added continuity does not establish correct recovery. |
| V6 upload frames 36, 61, 86 | Several visible bent-arm segments remain broadly similar under crop/tracking/RTMPose. Gloves overlap at 61; offhand precision and hidden-side identity cannot be certified. No uniform correction is visible. |
| V4-bg-1 frame 53 = source 17151 | Near wrist/elbow are visible; far chain is hidden behind the torso/arm. RTMPose draws its other wrist near the same foreground hand. A second finite chain is not evidence of a recovered far wrist. |
| V4-bg-2 frame 102 = source 22962, hook | Crop moves the other wrist toward the lowered glove compared with stored MediaPipe; the far elbow remains hidden and masks remain unreliable. This is a selective visual change, not a certified continuous correct-arm recovery. RTMPose also has ambiguous overlapping chains. |
| V4-bg-3 frames 104/111 = source 11472/11479 | During hook/uppercut overlap, gloves blur and the far elbow is not visible. Variant estimates disagree or cluster on the foreground arm. No true hidden elbow coordinate can be certified. |
| V4 eight-second upload frames 80/184 | The extended limb is visible, but cropped VIDEO rejects more far-arm observations than IMAGE. Increased near-arm stability does not solve far-side coverage. |

No percentage of **accurate** wrist/elbow recovery is reported: there is no independent visible-joint
reference set here. The evidence is enough to reject visibility-as-accuracy and a blanket estimator
replacement, but not to estimate localization error or prove the 2→4 match gain generalizes.
No scores from the six-class model were generated on changed representations. Original manual
and automatic recognition remain on the original pinned pose pipeline.

### Cost and integration implications

Sequential local CPU passes, one pass per variant; these are descriptive timings, not a randomized
repeat performance benchmark. Model calls include estimator preprocessing/inference/postprocessing;
case processing also includes PNG reads, hashes, crops/resizing, resets and serialization. Source
preparation/decoding and existing full-frame crop-discovery pass are additional costs. OS cache,
initialization and local load differ across passes; the apparent 1920-vs-960 speed difference is not
a speed claim. The diagnostic PNG workflow is not production upload latency. MediaPipe initialization/resets
are included in case processing; RTMPose's one-time ONNX constructor load precedes its case timer.

| Variant | Model-call median / p95 (ms) | Total case processing for 2,635 frames (s) |
| --- | --- | ---: |
| Fresh IMAGE 960 | 35.3 / 70.0 | 347.8 |
| IMAGE 1920 | 29.9 / 42.3 | 290.0 |
| Crop IMAGE | 29.3 / 40.6 | 265.3 |
| Crop VIDEO | 21.6 / 61.1 | 367.8 |
| RTMPose XY ablation | 35.1 / 94.8 | 317.0 |

A fixed crop is not free: discovery consumes original poses and future whole-case extent. A
real upload implementation would need a separately validated person/crop tracker, failure policy,
cut resets and local cache key. Cropped IMAGE changes detection/mask patterns; VIDEO also changes
motion dynamics/phase lag. Both may shift normalized velocities and classifier training distribution
even though coordinates map to the same 33-joint schema. RTMPose has different topology and score
semantics; it has no validated MediaPipe-compatible mask contract. Existing binary/classifier
checkpoints cannot be assumed valid for any of these new distributions. Do not silently reuse a
production pose cache or overwrite its original arrays.

## Interpretation and next-step requirements

A mask pass measures availability, not anatomical accuracy. Visual judgments in this study
are assistant observations, not human-certified keypoint ground truth. Hidden joints stay
unknown, and glove centers cannot establish precise wrist anatomy. Lower jitter and more
continuous trajectories can reflect lag or confident hallucinations rather than recovery.
The prior [pose-estimator audit](POSE_ESTIMATOR_AUDIT.md) already demonstrated confident
MediaPipe errors and RTMPose counterexamples; it does not authorize tuning on its other sources.

## Practical architecture after this study

If pose-only separation remains weak, the next modeling target should be **class-agnostic
individual-instance localization**, with video appearance as evidence and pose as an optional,
mask-aware branch. Do not train another six-class clip classifier to create boundaries.
A concrete bounded architecture is a native-rate RGB/motion branch (per-frame appearance plus
adjacent-frame optical-flow features), concatenated with a masked pose/velocity embedding,
followed by a small temporal convolutional head predicting activity and onset/end evidence. Keep appearance-only, pose-only
and fused ablations; do not silently treat missing poses as background. Declare an offline
context window first, with exact frame timestamps and no time warping. A live causal variant
would need a separately tested state/reset and actual-emission contract.

Boundary proposals need an explicit unresolved/overlap output. A per-arm head is a later option
**only if arm-specific supervision exists**; the present reference types and MediaPipe sides do
not supply verified arm identities, onset/apex/recovery phases or occluded joint targets.
Keep six-class recognition conditional on a supported isolated clip and report localization
separately from conditional class recognition. Current models remain frozen references.

This must acknowledge the existing [RGB feasibility pilot](LEVEL1_RGB_FEASIBILITY.md):
frozen MobileNetV3-Small single-image features sampled around 10 Hz plus a pooled TCN did
not outperform pose classification. Do not repeat that setup unchanged or call appearance a
proven solution. Reuse its exact-PTS decoding, crop/redaction and source-shortcut controls,
not its multi-source pool or trained heads. Native-rate motion evidence and dense localization
are explicit new hypotheses; run appearance-only/motion-only controls before assuming fusion
helps. Source HUD and punch-number text must never become features. Optical flow remains
image-plane motion evidence, not recovered hidden joints.

Existing authorized supervision can legitimately support a loader and tiny **training-source
overfit/engineering check**: 11 human V4 events, the reviewed V4 exhaustive domains and 12
V4/V6 negative intervals. Unknown context must have no foreground/background or boundary loss;
retain original censoring and the V4-bg-2 event extending beyond the reviewed core. Adjacent
frames must not become an independent validation split. There are no cross or rear-hook events
in the frozen continuous manifest; four-class event coverage cannot support a six-class claim.
V6 supplies only negatives in that manifest, creating an appearance/source shortcut as well as
the existing pose shortcut. It cannot establish video-fusion generalization.

Eligible existing human-confirmed **training-source** sparse anchors, such as V6:46, could be
considered in a separately approved, hash-bound dataset version, leaving surrounding footage
Unknown. The 1,123-example historical classifier pool is not an automatic continuous-detection
training set; it includes pilot evaluation sources, provisional boundaries and different review
states. Preserve V1/V6 grouping, all historical exclusions and the V6:264/266 holds. V8/V10 cross
anchors cannot fill this training deficit. V5/V7/V8/V10 and V9 stay outside this milestone.
No new localization model can be called reliable using only these 11 positives; no training
or broader annotation request is part of this change.

## Reproduction and verification

With existing local pose/train/dashboard extras, pinned MediaPipe and the prior optional
RTMPose setup (`.cache/pose-audit-deps`, `weights/pose-audit/rtmpose-m-body7.onnx`), run from
the repository root. The required retained baseline is
`reports/boxingvi/punch-instances-v2-repeat/`; the v2 diagnostic runner can prepare that baseline
if absent, but never overwrite an existing experiment. No evaluation-source audit command is run.

```sh
.venv/bin/python -m boxing_motion_ai.vision.observation_study prepare \
  --output reports/boxingvi/pose-observation-new
for variant in image960 image1920 crop960 crop960-video rtmpose-crop; do
  PYTHONPATH=.cache/pose-audit-deps .venv/bin/python \
    -m boxing_motion_ai.vision.observation_study extract \
    --output reports/boxingvi/pose-observation-new --variant "$variant"
done
.venv/bin/python -m boxing_motion_ai.vision.observation_study evaluate \
  --output reports/boxingvi/pose-observation-new
.venv/bin/python -m boxing_motion_ai.vision.observation_visuals \
  --output reports/boxingvi/pose-observation-new
# Enlarged, exact case-local frame (optional):
.venv/bin/python -m boxing_motion_ai.vision.observation_visuals \
  --output reports/boxingvi/pose-observation-new --case V6-upload --frame 111
```

MediaPipe still requires macOS graphics initialization even for this CPU delegate; normal OS
access was used as in the existing extraction pipeline. Model imports/inference are separate
from PyAV decoding. Output preparation and each variant require fresh paths and fail on changed
source/model/pixel/policy hashes. Missing inputs never trigger synthetic substitutes.

Actual ignored outputs are under `reports/boxingvi/pose-observation-study-v1-complete/`:
`manifest.json`, per-case `input.json`, raw landmarks/scores, source-coordinate poses,
model/timing provenance, `environment.json`, `comparison.json`, `summary.json` and six visual sheets.
The measured environment is Python 3.11.17, MediaPipe 0.10.35, NumPy 2.4.6, Pillow 13.0.0,
rtmlib 0.0.14 and OpenCV 5.0.0 on macOS arm64. Exact-frame
panels may be opened directly in VS Code. No generated media, arrays or model weights are committed.
An interrupted initial preparation remains separately preserved at `pose-observation-study-v1/`.
The executed extraction source snapshot is retained with its actual hash; the subsequent report-only
run-span correction adds V2 tracking-jump resets to coverage accounting. Initial reports are
preserved under `pre-run-span-accounting/`; model poses, candidate counts and matches did not change.

Verified **674 Python tests, 99 frontend tests**, Ruff, formatting, TypeScript and production
build. The existing Starlette/httpx deprecation warning remains. Focused tests cover crop-coordinate
inversion and depth scale, out-of-crop abstention, source/reset handling, jump-broken support runs,
RTMPose mask/score preservation and held-source/policy guards. Actual all-case inference, report
generation and visual rendering succeeded. Fresh upload-control poses replayed exactly; no new
browser upload was needed because production preparation/inference/UI were not modified.

The **5,197-file** preservation check found no changed retained source, model, review or experiment
artifact. V9 was not accessed. Production manual classification and automatic V2 detection remain
unchanged and their regression tests pass. No classifier/detector training, threshold search,
checkpoint replacement, new human annotation, dashboard change or prototype migration occurred.

The most practical next implementation is a **hash-bound native-rate RGB/motion + masked-pose
cache and continuous-supervision loader**, reusing the existing 11 events/12 negatives and Unknown
masks, with no training yet. Verify exact timing, source redaction, motion sensitivity and a
small targeted visible-joint audit on already inspected frames before authorizing a localization
experiment. It must preserve the present models as controls and cannot turn this small,
training-source collection into an independent six-class validation set.
