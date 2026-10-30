# Native-motion punch-instance localization: bounded smoke experiment

Starting revision `f223da3`. This is a separately authorized class-agnostic training-source
engineering experiment, not a production model, independent validation or six-class result.

**Outcome:** boundary heads separate seen combinations better than V2, but all learned arms
fail to return individual candidates on the V6 punching upload. Preserve all three as research
artifacts; no model is selected or promoted.
The existing binary detector, causal decoder, V2 automatic analyzer, classifier and poses remain
unchanged. No reserved V5/V7/V8/V10 motion or V9 is loaded for training or tuning.

## Actual results — training-source diagnostics only

**The engineering baseline can fit and separate reviewed training combinations, but does not
merit production promotion.** There is no selected winner. All three final checkpoints and
all failures remain separate. The experiment ran once at the frozen settings, with exact
repeats solely for determinism; no settings were changed after inspecting learned outputs.

The frozen V2 comparison reuses its original binary activity outputs and heuristic policy.
It is an existing-reference comparison, not a matched retraining of that older detector:
V2's binary smoke fit used V4, while these three arms also optimize the authorized V6 negatives.
Between the three learned arms, data, supervision, initialization and budget are identical.

Matches use the existing exact-time one-to-one IoU ≥ .5 rule, maximizing count then total IoU,
inside the reviewed V4 cores. Counts include censored-reference cases as before. Candidate
counts below are those intersecting the scored core; wholly unreviewed context is not negative.

| Method | V4 candidates | Matches / 11 | Misses | Background proposals | Merge risks | Split risks | Candidates on 12 verified negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 | 4 | 2 | 9 | 1 | 2 | 0 | 0 |
| pose | 9 | 8 | 3 | 1 | 0 | 0 | 0 |
| rgb_motion | 7 | 7 | 4 | 0 | 0 | 0 | 0 |
| fused | 11 | 10 | 1 | 1 | 0 | 0 | 0 |

Zero negative proposals are **training fit**, not a measured generalization false-positive
rate. Split risk means multiple candidate activity maxima inside one reference; merge risk
means ≥25% overlap with two references. Zero under these rules does not establish perfect
counting or isolation. Pose/fused both add a tail proposal [11483,11492): [11483,11487) is
verified background, while [11487,11492) lies outside exhaustive coverage and remains Unknown.
This is an incorrect extra recovery/tail instance even though the formal split-risk rule is zero.
The old V2 background proposal instead overlaps [11391,11410) of known background.

Pose misses the V4-bg-1 lead uppercut, V4-bg-2 hook and V4-bg-3 jab. RGB/motion misses the
V4-bg-2 jab, hook and final rear uppercut, plus V4-bg-3's final rear uppercut. Fused misses the
V4-bg-3 jab. These are **reference types for error analysis**, not model type predictions.
The three hooks that V2 misses are represented in the training labels: pose/RGB each match
two, fused matches all three. This demonstrates representational engineering fit, not transfer.

### Source boundaries and combination example

Median absolute boundary error among matched events is zero frames in every learned arm.
Misses remain in the table above; they are not removed from the task. The nonzero conditional
errors are:

| Arm | Matched events | Largest absolute onset error | Largest absolute ending error |
| --- | ---: | --- | --- |
| pose | 8 | 1 frames / 42 ms | 2 frames / 83 ms |
| rgb_motion | 7 | 1 frames / 41 ms | 1 frames / 42 ms |
| fused | 10 | 2 frames / 83 ms | 2 frames / 83 ms |

Actual fused source ranges for the ten matched events:

| Case / reference type | Human [start,stop) | Predicted [start,stop) | Start / stop error frames |
| --- | --- | --- | --- |
| V4-bg-1 lead_uppercut | [17121,17138) | [17119,17137) | -2 / -1 |
| V4-bg-1 rear_uppercut | [17140,17164) | [17140,17164) | +0 / +0 |
| V4-bg-1 lead_hook | [17164,17179) | [17164,17178) | +0 / -1 |
| V4-bg-2 jab | [22924,22940) | [22925,22940) | +1 / +0 |
| V4-bg-2 rear_uppercut | [22941,22955) | [22941,22954) | +0 / -1 |
| V4-bg-2 lead_hook | [22955,22971) | [22955,22971) | +0 / +0 |
| V4-bg-2 rear_uppercut | [22971,22982) | [22971,22984) | +0 / +2 |
| V4-bg-3 rear_uppercut | [11460,11470) | [11460,11470) | +0 / +0 |
| V4-bg-3 lead_hook | [11470,11476) | [11470,11476) | +0 / +0 |
| V4-bg-3 rear_uppercut | [11476,11483) | [11476,11483) | +0 / +0 |

In V4-bg-3, fused returns **[11460,11470), [11470,11476), [11476,11483)** while the minimum
activity score throughout these 23 contiguous frames is **0.9870**. Separate onset/end evidence,
not activity valleys or fixed-duration splitting, creates the three ranges. All three match the
reviewed boundaries. It also proposes the incorrect tail above, so this is not a correct-count claim.

The final V4-bg-2 event's core-clipped IoU is 1 for pose/fused despite predicting stop 22984
instead of full-reference 22982. Its ending is outside the exhaustive core and was not trained;
the clipped match must not be described as a verified perfect ending. Full source-PTS and signed
boundary errors are retained for every match in `records.json` and the generated report.

### Unresolved motion and the V6 failure

No model returns an individual candidate on the existing six-second V6 punching upload.
The clip has no exhaustive positive inventory here, so a numerical V6 recall is unavailable,
but prior/raw-frame inspection shows punching. Zero proposals is a meaningful qualitative failure:

- Pose activity reaches .991 and ending .963, but onset never exceeds .193: no supported pair.
- RGB/motion finds onset and ending at the same frame 111. The zero-duration pair is rejected.
- Fused likewise pairs both heads at frame 107 (activity reaches .996). It remains unresolved;
  the decoder does not stretch it into a fabricated punch or fall back to manual boundaries.

V4-bg-3's fused jab miss exposes a second failure: an earlier onset at local frame 15 in unknown
context remains pending until the ending at 91. The real jab onset at 82 then becomes
`multiple_onsets_before_end`; no pair is forced. The present decoder does not expire pending
onsets during a quiet/background gap; maximum span is checked when pairing. This conservative
baseline can therefore let an earlier unsupported boundary contaminate a later action. **It was
not repaired using these results.** A future decoder version should test evidence expiry and
background resets on synthetic cases before a new frozen comparison, not tune the current counts.

| Arm | Unresolved records across 19 streams | Unique unresolved frames / seconds | Unobservable frames |
| --- | ---: | --- | ---: |
| pose | 81 | 796 / 33.200 | 43 |
| rgb_motion | 92 | 870 / 36.285 | 43 |
| fused | 72 | 601 / 25.067 | 43 |

“Unique” here unions overlapping unresolved records **within each diagnostic stream**, then
sums streams. Some recordings/uploads overlap; this is not independent exposure. Unresolved
ranges include inconsistent boundary-only evidence and quiet gaps, not time certified as punching.
Each learned arm has 43 unobservable frames: 25 from the existing blank control plus the first
motion-less frame of each of the 18 real diagnostic streams. Original wrist/elbow missingness
continues separately in the pose masks. None of these abstentions is confirmed background.

Upload candidate counts (unverified, not punch counts):

| Method | V6 six-second | V4 eight-second | V4 29-second |
| --- | ---: | ---: | ---: |
| v2 | 3 | 1 | 3 |
| pose | 0 | 3 | 6 |
| rgb_motion | 0 | 2 | 9 |
| fused | 0 | 5 | 12 |

### Effects of the inputs

At identical settings, learned training-source matches are 8 pose / 7 motion / 10 fused.
This is not enough evidence to select fused for future users. Inference-only value-removal
controls help verify wiring, not establish causal feature importance or substitute for separately
trained controls. Zeroing dynamic visual values while keeping availability bits leaves the
pose arm unchanged, removes all native V4 candidates from RGB/motion, and also removes them
from fused. Replacing pose values/confidences with just their masks leaves RGB/motion unchanged;
fused then returns 0 / 0 / 3 raw-context candidates on V4-bg-1/2/3. Thus the fused fit uses both
value streams, but source/mask/kinematic shortcuts and overfitting remain possible. No extra
mask-only/background classifier was trained in this bounded three-arm experiment.

## Runtime, memory and reproducibility

Each arm has **14,819 parameters**, 59,276 bytes of float32 parameters. All use the same
[15,142,162] full batch, approximately **1.41 MB** of input/target/loss-mask tensors, plus
activation/gradient/optimizer state. Actual process high-water RSS is about **381 MiB**, including
PyTorch, source validation and caches; it is cumulative process memory, not an isolated per-arm
peak. No GPU or learned visual encoder is used. The 19 compressed feature files total 1,139,773
bytes. Feature-loop work took **167.61 s**, including per-case hashing/PNG reads/crops/flow;
initial review validation and preservation hashing are additional. Existing poses were reused.

| Arm | Primary fit (s) | Exact-repeat fit (s) | 2,635-frame model inference (s) | Instance decoding (s) | Initial → final weighted loss |
| --- | ---: | ---: | ---: | ---: | --- |
| pose | 2.179 | 1.137 | 0.0069 | 0.1247 | 0.8440 → 0.0423 |
| rgb_motion | 1.137 | 1.088 | 0.0062 | 0.1484 | 0.8940 → 0.0409 |
| fused | 1.142 | 1.114 | 0.0064 | 0.0868 | 0.8414 → 0.0319 |

These are single local CPU measurements, not a randomized latency benchmark. Model inference
excludes feature generation, validation and control/reload checks. The 120-update repeats have
identical tensors, curves and source-frame predictions; checkpoint reload predictions are exact.
PyTorch loss weighting is 9.0423 for activity and 40 for each boundary head. All arms see 1,426
eligible activity frames and 1,375 per boundary head; triangle target masses are 142 / 19.996 /
20.008. Those masses are not additional event annotations. The 12 first negative-frame motion
abstentions account for the change from 1,438 exhaustive to 1,426 loss-eligible activity frames.

## Frozen experiment

`configs/punch_localization_smoke_v1.json` fixes the protocol before any training output:

- Three input arms: pose-only, RGB/motion-only, fused. Same 162 input slots, hidden width 32,
  three residual temporal convolution blocks (kernel 3, dilations 1/2/4), per-frame channel
  LayerNorm, ReLU and dropout .1, then three logits: activity, onset, end. All arms have
  identical parameters, initialization, eligible frames, targets, seed and optimizer budget;
  unused input slots are zero. There is no clip pooling or batch normalization.
- CPU, two threads, deterministic PyTorch, seed 20261029. Exactly **120 full-batch AdamW
  updates** per arm, learning rate .003, decay .0001, gradient clipping 1. No sampler,
  augmentation, architecture search, best-epoch selection, decoder tuning or extra model arm.
  Every epoch is retained and the final fixed epoch is used even if the method fails.
- Binary cross entropy per head, each normalized by its supervised positive/negative weight
  mass, then equal head averaging. Positive weights derive only from eligible training
  targets and are capped at 40. No test/development metric selects any setting.
- An exact same-budget repeat verifies state tensors, curves and predictions; this is a
  reproducibility check, not another seed or search. Checkpoint reload is also verified.

The network is **offline**, with seven past/seven future native samples (RF 15). Source cuts,
gaps and observation abstentions split sequences. Padding is masked at the input and after
every block, so batch padding cannot change real predictions. Unknown-label context may provide
pixels/poses but has no loss. Native rates/timestamps are retained; RF 15 is not rate invariance.
Boundary peak suppression uses additional future context, and the fixed crop uses whole-case
pose extent. There is no streaming, actual-emission or live-latency claim.

## Reused data and provenance

The loader adapts the existing frozen `continuous-dataset-v1` and its human-reviewed
`supervision_runs`; it does not create a second dataset/review system. The existing validator
rechecks live human references, source hashes and review backups. Manifest SHA remains
`c7993318ee8f8a16e2cd89f04ad0acaf7ce95c0cad8aabad0c70079b683b2d59`.
Its historical `detector_training_authorized=false` field is preserved: this user's new bounded
training authorization is recorded separately in the new experiment config and provenance.

Only V4-bg-1/2/3 and V4/V6-motion-1 through -6 optimize weights: **11 reviewed V4 events**
and **12 exhaustively reviewed negative intervals**, including their verified mixed-interval
background. There are 1,438 exhaustive frames (142 active, 1,296 background), about 59.976 s.
V6 supplies only negatives; this is a substantial recording/activity shortcut. V1/V6 remain
one source group, V4-bg-4/5 stay excluded, and no historical classifier pool is imported.

Diagnostic inference uses the same 19-case inventory as V2/pose observation: those 15 native
cases, three existing V4/V6 uploads, plus the 25-frame synthetic blank no-person control.
The uploads have no exhaustive positive references and **never optimize weights**. There are
2,610 V4/V6 frames plus the blank control. Context is not exhaustively labeled. The three
native positive cases preserve their original context and global source-index offsets.

The feature cache binds original videos, lossless source pixels, original pose arrays, exact
integer PTS/timebases, decoded frame order, label/review manifest, crop/redaction policy,
feature code and dependency versions. Every source pixel is checked before extraction. Cache
reuse verifies feature bytes, source/metadata hashes and replays targets/masks from the frozen
manifest. Native pose-model metadata and the original task hash are recorded; no extraction
checkpoint or existing feature definition is changed. There is no implicit download.

## Feature contract

`native-rgb-motion-masked-pose-1.2` uses every native frame:

1. Reuse the original 65-channel torso-normalized XY, visibility/presence, joint masks and
   exact-time backward XY velocities. Missing joints and derivatives remain zero sentinels
   with false masks. No interpolation, smoothing or invented coordinates. Storage retains
   these features unchanged; the network divides only numerical velocity values by 10 for
   fixed conditioning, retaining all mask/confidence bits.
2. Reuse the prior RGB study's 4.5-torso crop geometry and HUD rectangles, with a **fixed
   whole-case median crop** so camera/crop movement does not manufacture flow. Crop discovery
   uses original poses, never annotation locations. RGB-only is therefore pose-guided and
   shares the same person-observation gate; it is an input ablation, not an independent tracker.
3. Expand fixed HUD masks by 1% of source dimensions, mask the visible V4 bag-logo region,
   and mask fixed whole-case shirt/lower-body regions. The head/guard workspace is retained.
   Masks are static across each case; no moving blackout-box trajectory becomes motion.
   Outside-source padding and redacted pixels are excluded, with eroded pair support.
4. Resize the redacted crop to 160×160. For **each adjacent source pair**, compute absolute
   RGB difference, signed/RMS grayscale difference and Farneback optical flow. Parameters:
   pyramid scale .5, two levels, window 15, two iterations, polynomial 5/sigma1.1. Subtract
   median supported flow to reduce common camera motion. Divide by exact elapsed source time;
   flow units are crop fractions/s, not physical glove speed. Pool six dynamic quantities
   into a 4×4 grid, fixed tanh scales, plus a motion-valid bit: **97 channels**.
5. There are **no static RGB/color/semantic embeddings**, image identifiers, absolute clocks,
   filenames, event types, source rows, crop coordinates or annotation text in model tensors.
   Identical frames yield exactly zero dynamic values regardless of static texture/color.
   Tests verify HUD/identifier-region changes are removed before motion computation.

This is a compact handcrafted visual-motion baseline, not a test of every video-appearance
encoder. It deliberately avoids repeating the earlier unsuccessful frozen single-image RGB
classifier pilot unchanged. Motion outside masks can still carry recording-specific dynamics;
V6-negative-only supervision cannot prove removal of statistical source shortcuts. No source-
independence claim follows from masking metadata/HUD/branding and static appearance.

**Availability and labels are separate.** Known targets survive missing observations. All three
arms use the same eligible observation gate: reliable person/torso plus a real adjacent native
motion pair. The first frame after a start/cut/gap lacks motion and is an abstention, not
background. Missing wrists alone do not remove a reliable torso frame. Unknown footage has
no target loss even when fully visible. Unobservable frames remain in the reference inventory
and therefore can remain missed punches.

### Pre-training QA correction

The first cache (`...features-v1`, feature policy 1.0) masked head plus torso too broadly.
Visual inspection showed it obscured guard-hand motion and emphasized legs/background. **No
model was trained on that cache**. It is preserved with all receipts. Before training, a second QA cache (1.1) retained
head/guard motion, excluded lower-body
identifiers and added the bag-logo mask. A remaining top shirt-text strip prompted the final
1.2 shirt mask to begin at the shoulder line. Both QA caches are preserved and neither trained
a model. This was pixel/input QA, not selection using
training predictions. No feature/decoder revision was made to improve learned result counts.

## Continuous supervision

Activity is the union of human event spans within the exhaustive domain. Only explicitly
reviewed non-event spans become background. Onset evidence peaks at the reviewed first frame;
end evidence peaks at the **last included** frame (`stop-1`), not an impact/apex guess.
Each uses a triangular target within 1/12 s of that boundary using exact PTS. This is a declared
engineering support width, not calibrated biomechanical uncertainty or new phase annotation.

Boundary absence at domain edges is unknown: a radius-sized edge band has no boundary loss.
Events touching/crossing the review edge do not create that censored boundary target. V4-bg-1's
first onset and V4-bg-2's final ending are censored; full known event bounds remain in metadata.
The latter extends four frames beyond exhaustive coverage; those frames have no training loss.
Overlapping known event spans may supervise shared activity and separate boundary evidence,
but never supply per-arm pairing or a fabricated six-class label.

## Deterministic instance decoder

`offline-boundary-pair-decoder-1.0` is separate from the preserved production activity decoder:

- Fixed activity and onset/end thresholds .5. Onset/end peaks use exact-time 80 ms local
  neighborhoods, one earliest anchor per equal-score plateau, deterministic ties and same-head
  suppression. No duration grid splits activity.
- Sweep onset/end evidence in source order. Exactly one supported onset paired with an end
  can become a candidate; span 80 ms–2 s and at least half its source-time support must exceed
  the activity threshold. End is last included source frame; storage stop is end+1.
- Multiple onsets before an end are unresolved/overlap-ambiguous, not arbitrarily paired.
  Missing starts/ends and observation-edge boundaries remain censored/unresolved. No candidate
  crosses a cut, source gap or unobservable run. Remaining activity is explicitly unresolved;
  unobservable source ranges are not background. Interruptions retain original clock anchors.
- Consecutive punches can have separate boundary pairs while activity stays high throughout;
  a synthetic regression verifies this. A maximum activity-score frame is only a diagnostic
  matching anchor, **not impact**. All scores are uncalibrated.

No six-class classifier is called. Candidates, possible activity, boundary evidence, unresolved
motion and human-reference comparisons remain distinct. Changed localization outputs never
enter production analysis or human reviews.

## Decision and genuinely necessary next data

**Merits further development as a research baseline; not a production candidate.** The loader,
masked losses, boundary heads and exact-time pairing work, and seen combinations no longer
require radial arm cycles. But 10/11 on optimized examples cannot establish improvement for
new footage. The V6 failure, extra recovery proposal, pending-onset contamination and narrow
single-source positives are more important than choosing the highest training match count.
All three final models remain experiments; there is no “winning model” pointer.

A credible subsequent milestone needs:

1. **Source-balanced positive and negative continuous shadowboxing.** Add authorized V6/new
   training-source punches as well as matched guard, footwork, recovery and non-punch movement
   from those same recordings. Otherwise the current V4-positive/V6-negative shortcut remains.
   Reuse eligible existing human-confirmed sparse training anchors in a separately versioned
   manifest where suitable; surrounding unreviewed footage stays Unknown. Do not automatically
   import the historical classifier pool, held V4-bg-4/5 or excluded V6:264/266 examples.
2. **Exhaustive individual-instance development references**, not only pre-cut positives or
   activity-bout labels, from an authorized separate recording/session group. Mark first/last
   included action frames, uncertainty, incomplete coverage and overlaps. Exact impact or arm
   identity is unnecessary. Current short curated intervals cannot supply realistic false-alarm
   exposure across full recordings. No blanket new annotation project is required for this
   completed engineering check; this describes the evidence needed before credible model claims.
3. A frozen evaluation protocol on separate authorized source groups after development choices
   are fixed. Preserve V1/V6 and V7/V8 grouping and collect athlete/session identity metadata
   where available. Existing V7/V8/V10 activity-review blocks are not automatically exhaustive
   **individual-punch** ground truth. V5/V7/V8/V10 were not opened in this milestone; V9 remains
   untouched. New access, supervision and training would need their own approved scope.

Subsequent reporting should include boundary errors and misses for censored/unobservable
references, event matches, duplicate/tail proposals, unresolved overlap and false alarms per
**fully reviewed** minute. Keep localization separate from conditional six-class recognition.
The present supervised set lacks two six-class types, but this class-agnostic model does
not require subtype labels to learn localization.

Future engineering hypotheses are explicit: test stale-onset expiry/low-activity resets in a
new decoder version, improve the compact motion representation and crop support without hiding
hands crossing torso branding, and test broader motion-aware video features. Those are not
changes made to these results. The current handcrafted grid, gray masks, whole-case cropping
and limited viewpoints may lose important hand evidence. Do not lower thresholds or extend
training until these same 11 examples look perfect.

## Reproduce and review

Requires the existing local `pose`, `train`, `dashboard` dependencies and OpenCV (measured here
as 5.0.0), the accepted continuous dataset/reviews, original pose artifacts and the exact-pixel
stage from `vision.observation_study`. No new packages/weights were downloaded for this experiment.
The pixel stage needs no experimental pose re-extraction; this cache uses original poses.

From the repository root, using fresh ignored paths:

```sh
.venv/bin/python -m boxing_motion_ai.detection.localization_data \
  --output reports/boxingvi/localization-features-new
# Inspect each redaction-preview.png before training; these images are never model tensors.
.venv/bin/python -m boxing_motion_ai.detection.localization_smoke \
  --cache reports/boxingvi/localization-features-new \
  --output reports/boxingvi/localization-smoke-new
MPLCONFIGDIR=/private/tmp/boxing-localization-mpl \
.venv/bin/python -m boxing_motion_ai.detection.localization_report \
  --cache reports/boxingvi/localization-features-new \
  --run reports/boxingvi/localization-smoke-new \
  --output reports/boxingvi/localization-smoke-new/figures
```

The actual verified run is `reports/boxingvi/punch-localization-smoke-v1/`, with feature cache
`reports/boxingvi/punch-localization-features-v1-2/`. Each arm contains its final checkpoint,
exact-repeat checkpoint, per-frame activity/onset/end scores, training curves, source-timed
candidates/unresolved ranges, full reviewed comparisons, removal controls and summary. Source
snapshots, fixed config, original pose-model provenance, code/cache/dataset hashes and runtime
versions are preserved. Candidate frame indices are case-local plus the record's explicit
source offset; rational PTS/timebases are original source timing. Human types occur only in
reference comparisons. No model output contains six-class predictions.

Open `figures-context-v1/V4-bg-3-boundaries.png` for the contiguous combination and
`figures-context-v1/V6-upload-boundaries.png` for the failed shadowboxing transfer. Gray marks
human events, green marks candidates, hatched native context has no exhaustive labels. Uploads
have no exhaustive reference track. Earlier unhatched figures remain preserved too.
`figures-context-v1/details.json` records exact boundary errors, unioned unresolved support and
its accounting scope. `artifact-hashes.json` binds the saved input/result artifacts used for
these figures. All generated images, feature arrays, reports and weights stay outside Git.

## Verification and preservation

- **692 Python tests**, including 18 new contract regressions; **99 frontend tests**, TypeScript,
  Prettier, Ruff, changed-Python formatting and production build passed. The existing
  Starlette/httpx deprecation warning remains. No frontend or production inference code changed.
- Tests cover source-role/group/budget guards, Unknown and censored losses, exact frame clocks,
  missing wrists without relabeling background, HUD/static-image suppression, adjacent motion,
  padded-context isolation, deterministic training, missing motion, source resets, continuous-
  activity combinations, ambiguous overlaps, plateau suppression and edge censoring.
- Actual cache verification aligned all 2,635 clocks, checked masked pose/velocity sentinels,
  replayed six native motion pairs byte-for-byte and rejected a deliberately modified cache
  copy. The modified verification copy is separate and was never trained or used for inference.
- Every actual arm's repeat has identical state tensors, loss/gradient curves and predictions;
  checkpoint reload is exact across all 19 cases. No test or failure caused a post-result
  threshold, feature, epoch or architecture change.
- **8,644 protected files** remained byte-identical, including original source/review/model and
  pose-study artifacts plus the two untrained preprocessing QA caches. Original checkpoints,
  datasets, review ledgers, source-group restrictions and V9 protections remain intact.

This milestone stops at the bounded experiment. Production Analyze, automatic V2 detection,
manual six-class classification, dashboard navigation and the separate prototype are unchanged.
