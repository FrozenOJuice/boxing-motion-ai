# Blue-reference localization experiment v1

Starting revision: `57092b7`. This is the explicitly authorized, fixed-budget **blue-marker
subset experiment**, not production promotion or an independent final test.

**Outcome:** V6 training-reference matching improves, but transfer remains weak. The new
RGB/motion arm matches 2/8 V5 development references, pose/fused match 0/8, and frozen V2 matches
4/8. Original V4 training fit regresses. No model is selected, promoted or substituted into Analyze.
The existing activity detector, V2 analyzer and six-class classifier/checkpoints remain unchanged.

## Frozen evidence and selection

The latest six append-only review histories contain **40 active saved human events**. The
experiment selects **22** and excludes **18** non-blue manual events without deleting any.

| Queue interval | Selected human blue events | Excluded other manual events | Role | Full references eligible for primary comparison |
| --- | ---: | ---: | --- | ---: |
| V6-motion-101 | 3 | 5 | Training | 2 |
| V6-motion-102 | 2 | 4 | Training | 2 |
| V6-motion-103 | 3 | 0 | Training | 3 |
| V6-motion-104 | 2 | 8 | Training | 0 |
| V6-motion-105 | 4 | 1 | Training | 4 |
| V5-motion-101 | 8 | 0 | Development only | 8 |

Association is a **unique exact native [start,stop) match** between an active saved human punch
and a prepared source suggestion, backed by the actual manual upsert and inspection of both
included endpoint frames. The editor did not record an explicit blue-marker click ID. This
association is therefore a documented deterministic inference from saved provenance and exact
bounds, not a claim that a click audit exists. No nearest-range, label or model-score matching
was used. Missing, ambiguous, duplicate or uninspected associations fail closed. A source mark
without a corresponding human save cannot become a target.

The selected V6 ranges are:

- 101: [486,497), [541,550), [566,577).
- 102: [6468,6485), [6566,6589).
- 103: [8306,8323), [8368,8386), [8447,8467).
- 104: [25094,25109), [25187,25203).
- 105: [30509,30532), [30535,30558), [30563,30586), [30594,30616).

V5: [8140,8146), [8146,8160), [8221,8229), [8230,8247), [8307,8315), [8316,8330),
[8392,8400), [8400,8413). Source row 389 is provisionally rear_hook, but its saved human type is
cross. Its exact boundary association is eligible for this **class-agnostic** experiment; the
source type does not replace the human field or train a class label.

The unchanged original continuous v1 supplies **11 V4 events**, three exhaustive mixed cores,
and **12 explicitly verified negative intervals**. Two previously approved, uncontested V4
sparse anchors are retained: V4:144 [10404,10411) and V4:343 [18192,18202). Their surroundings
remain Unknown. V6:7 and V6:46 exactly agree with the new selected blue events and are linked
as earlier evidence, not counted twice. V4-bg-4/5, V4:497, V6:613, V6:264/266 and all prior holds
stay excluded. V4 and the V1/V6 group are training-only. V5 is development-only; no V7/V8/V10 frames are decoded or used for training/evaluation.
Inherited preservation checks may hash old reserved artifacts, but no V9 path is opened.

### Excluded motion, contradictions and loss masks

Earlier no-punch completions in V6-103 and V5-101 are preserved in history. Subsequent actual
human upserts and completions supersede their *current* empty state. This does not authorize
reinterpreting or modifying the historical records.

All new review cores are **positive-reference-only for this selective experiment**, including
V5. Existing completion flags are preserved, but are not used to infer absence between the
selectively verified blue suggestions. Excluded manual events, uncertain regions and gaps are
Unknown, not background. This deliberate experiment mask restriction does not rewrite the
human completion fields. Input context may contain excluded motion, but its excluded labels
and boundaries never enter loss or the evaluation reference list.

Three blue events overlap excluded manual events:

| Selected blue event | Excluded manual range | Policy |
| --- | --- | --- |
| V6 [486,497) | [496,501) | [496,497) Unknown; disputed end not supervised/scored |
| V6 [25094,25109) | [25108,25113) | [25108,25109) Unknown; disputed end not supervised/scored |
| V6 [25187,25203) | [25202,25208) | [25202,25203) Unknown; disputed end not supervised/scored |

These three retain uncontested positive frames/onsets for loss, but are excluded as *full*
primary instance references. Boundary support within 1/12 s of excluded frames is also masked.
Their original full bounds, UUIDs and all conflicting revisions remain in provenance. This is
an explicit scoped partial-evidence policy, not silent priority over a human annotation or a
relaxation of the earlier exhaustive-snapshot validator. No contradictory background is imported.
The new dataset has 27 training positive records (13 V4, 14 V6), of which 24 are fully scorable;
three are partial/conflicted. V5 has eight development positive references.

Sparse activity targets are positive only on selected, unconflicted frames. Boundary triangles
use the same 1/12 s source-time radius, first/last-included convention and three-head layout;
support is restricted to the selected instance where it is known. No boundary negatives are
created in unreviewed gaps. Original V4/negative targets remain byte-identical to the prior cache.
After the shared observation gate, loss uses **1,688 activity** and **1,630 frames per boundary
head**, with activity positive mass 404 and boundary masses 42.56 / 39.508. Missing pose/motion
observations remain abstentions and are never converted into background labels.

### Immutable artifacts

- Dataset pin: `configs/blue_localization_dataset_v1.json`.
- Private snapshot: `reports/boxingvi/blue-localization-dataset-v1/manifest.json`.
- Dataset SHA256: `af3a30e400609cd5c1d3c5c7718f6b3e5c49f85dfede25aa95ba3d7211d5dd26`.
- Protocol: `configs/punch_localization_blue_v1.json`.
- Final cache: `reports/boxingvi/blue-localization-features-v1-redacted/`.
- Run/checkpoints: `reports/boxingvi/punch-localization-blue-v1/`.
- Preflight review backup/inventory: `reports/boxingvi/blue-subset-preflight-20261030T055423Z/`.

The adapter composes the existing continuous validator, exact timing, source-group policies and
human-review history replay. Review bytes and receipts have durable snapshot backups. Revalidation
rejects changed source inputs, review heads, selected ranges, configs, feature code or cached targets.
The old continuous/evidence pins and all original artifacts remain unchanged.

## Feature and training protocol

The architecture and optimizer are unchanged: **14,819 parameters**, 162 input channels,
32-channel centered TCN, dilations 1/2/4, per-frame LayerNorm, dropout .1, activity/onset/end logits.
All arms start from the same seed **20261029**, use CPU/two threads, deterministic PyTorch,
**120 full-batch AdamW updates**, LR .003, decay .0001, gradient clip 1. Data-derived head weights
use the existing formula/cap 40: 3.1782 / 37.2989 / 40. Final epoch only; no search, best epoch,
threshold change, development optimization or model selection. Exact same-budget repeats are
verification, not additional settings. Reload and repeat predictions agree on every case.

Pose-only, RGB/motion-only and fused use identical training cases, observations and masks.
V5 has zero optimization sequences; a regression test perturbs its values/targets without
changing training sequences. No statistics are fitted on V5. Labels/types/IDs, source rows,
absolute clocks and annotation geometry are absent from model tensors.

The original **65-channel masked torso-normalized pose/velocity** and **97-channel adjacent
native RGB-difference/flow** features are reused. The 19-case v1 cache is retained unchanged;
eight new native contexts are added (five V6, one V5, two V4 anchors), for **27 inference cases**.
All new pixels are decoded at the exact pose-array PTS; per-frame RGB hashes, native source-index
maps, source/model hashes, crop/mask policy and label receipts are recorded. No pose extraction,
interpolation, smoothing or six-class inference is performed.

`native-blue-localization-features-1.1` extends the original fixed-crop/redaction implementation
to V5. It uses V5's existing counter/HUD mask plus fixed flags/header, wall-poster and gym-sign
identifier masks established by pre-training pixel QA. The same torso/lower-body masking,
160×160 image, 4×4 motion grid, flow parameters, scales and observation gate are retained.
Redaction removes useful torso/occluded-hand evidence too; it is not a proof of source invariance
or a learned appearance encoder. Mask footprints and recording-specific motion can still leak
recording properties. No encoder/crop/mask setting was chosen using localization outputs.

Two incomplete timing-adapter preflights and an untrained native-clock QA cache are preserved
under their separate `blue-localization-features-v1*` paths. Missing integer PTS fields were
corrected before fitting. Pixel QA then found readable V5 background identifiers; fixed masks
were added before the only bounded experiment. No model trained on the incomplete/unredacted
caches. Local PyAV/OpenCV emit duplicate AVFoundation-class registration warnings; native
software decoding completed and every decoded PTS was checked. This run makes no camera-capture
or platform-stability claim from those warnings.

Processing remains **offline**: RF15 (seven past/seven future native samples), whole-context
fixed crop and future-aware peak suppression. The 14-gap receptive-field span is about .584 s
at V4/V6's native rate versus .467 s at V5's rate. There is no rate-invariance or live-latency
claim. These differences were not adjusted after observing the results.

## Actual comparison

The previous smoke checkpoints are evaluated unchanged on all 27 current cases. Predictions on
the original 19 cases are **exactly equal** to their stored v1 scores. Frozen V2 uses its unchanged
binary detector, causal decoder, activity results/policy and V2 segmenter. It is a legacy baseline,
not a matched retraining. On the new cases its activity inference uses the same cached original
poses and native clock. No classifier scores manufacture instances.

Matching retains the existing one-to-one maximum-count/IoU assignment at **IoU ≥ .5**. New sparse
cases use full candidate/reference ranges over context, never clipping broad predictions to
known positive spans. The original V4 exhaustive-core comparison keeps its historical clipping
and censor policy for comparability; conditional full-boundary errors remain recorded.

| Method | Original V4 matches /11 | Sparse V4 /2 | Uncontested blue V6 /11 | V5 development /8 | V5 candidates | Proposals on 12 verified negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen V2 | 2 | 0 | 2 | 4 | 6 | 0 |
| Previous pose | 8 | 0 | 1 | 0 | 0 | 0 |
| Previous RGB/motion | 7 | 0 | 0 | 0 | 0 | 0 |
| Previous fused | 10 | 0 | 0 | 0 | 0 | 0 |
| New pose | 1 | 1 | 6 | 0 | 2 | 0 |
| New RGB/motion | 1 | 1 | 2 | 2 | 14 | 1 |
| New fused | 5 | 1 | 8 | 0 | 0 | 0 |

V4/V6 results are **training-source fit**, including the original 12 negative intervals. New
V5 is recording-separated development transfer, not a pristine final test: V5 influenced prior
classifier development. Counts over 11/8 are positive-reference matching, not full detection
recall. No precision, full recall or false-positive rate is calculated on the sparse new domains.
Extra proposals there may be real excluded/unmarked punches. Three conflicted V6 blue references
are retained as unscored; they are not quietly counted as correct or missed full instances.

### V6 adaptation and remaining errors

| New arm | V6 candidates | Matches /11 | Missed selected references | Unmatched proposals (unlabeled) | Split risks | Merge risks | Median absolute onset/stop error (frames) | Max absolute onset/stop error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Pose | 13 | 6 | 5 | 7 | 1 | 0 | 1 / 1 | 6 / 4 |
| RGB/motion | 30 | 2 | 9 | 28 | 5 | 0 | 4 / 4 | 8 / 7 |
| Fused | 12 | 8 | 3 | 4 | 0 | 0 | 0 / 0.5 | 0 / 1 |

Split/merge risks are the existing overlap/peak diagnostics, not independently verified counts.
Zero risks does not establish perfect segmentation. Fused matches per V6 interval are 1, 1, 3,
0, 3. Its misses are [566,577), [6468,6485) and [30594,30616); the three
conflicted events are separately unscored. Source confidence-mask wrist/elbow availability in
V6-105's far arm is about 56% / 53% over context. These are validity-mask fractions, not measured
landmark accuracy. The extra V4 anchor fit is 1/2 for every new arm; added positives do not imply
complete training memorization. All arms lose original V4 matches under the enlarged dataset,
partial losses and fixed update budget. This was training from scratch, not checkpoint fine-tuning.

Only the RGB/motion arm adds a proposal on an original verified no-punch interval: V6-motion-5,
local [30,38). That is an actual training-background error. Its new sparse-case unmatched
proposals are not added to this false-positive count. No-punch exposure remains the original
48.006 seconds of the 12 negative intervals; it is not independent held-out exposure.

### V5: limited transfer, not reliable recognition

RGB/motion's two matched full source ranges are:

| Human [start,stop) | Candidate | IoU | Start error | Stop error |
| --- | --- | --- | --- | --- |
| [8230,8247) | [8231,8240) | 100/189 ≈ .529 | +1 frame / +33 ms | −7 frames / −234 ms |
| [8400,8413) | [8402,8410) | 267/434 ≈ .615 | +2 frames / +67 ms | −3 frames / −100 ms |

Those are two human-referenced instances, **not predicted cross labels**. RGB/motion misses the
other six references and returns 12 unmatched proposals, with one split risk. For example,
it fragments [8230,8247) into [8231,8240), [8241,8245) and an overlapping tail [8246,8255).
Fused preserves its long ambiguous combination spans as unresolved, rather than emitting them
as a supposedly merged individual punch. Pose returns two
unmatched candidates [8209,8218), [8407,8419) and misses all eight. Fused returns zero candidates
and misses all eight. Frozen V2's 4/8 matching is better on this small reference set, so the new
learned model has not established a better practical alternative.

This is not solely a person-visibility abstention: V5 has one common unobservable frame, outside
the selected references. Wrist validity fractions are about 94% / 100%; those values do not prove
accurate wrists. Fused V5 activity/onset/end scores reach .995 / .934 / .842, yet only six onset
peaks and three end peaks survive the frozen decoder. It produces unresolved spans [8138,8215)
and [8220,8307) with `multiple_onsets_before_end`, followed by [8392,8491) with no observed end.
Pose similarly collapses consecutive onsets into ambiguous pairs. RGB/motion has many noisy
boundary peaks and both fragmented and short candidates. Unresolved motion is preserved;
no arbitrary split, six-class assignment or calibrated-confidence claim fills these gaps.

### Previously tested uploads

Candidate counts, **without exhaustive upload reference inventories**:

| Method | V6 six-second upload | V4 eight-second upload | V4 29-second upload |
| --- | ---: | ---: | ---: |
| Frozen V2 | 3 | 1 | 3 |
| Previous pose / RGB / fused | 0 / 0 / 0 | 3 / 2 / 5 | 6 / 9 / 12 |
| New pose / RGB / fused | 1 / 1 / 2 | 2 / 1 / 3 | 4 / 20 / 4 |

New fused V6-upload ranges are local [25,41), [103,123); pose [24,42), RGB/motion [24,31).
These are real model proposals, not verified punch counts. V6 now appears in positive training,
so a different upload from V6 is a same-recording transfer check, not unfamiliar-recording
validation. V5 is the relevant new recording-separated check. The blank no-person control has
zero candidates and 25 unobservable frames for every arm; no missing footage is called background.

### Unresolved and unobservable support

Unique unresolved frames are unioned within each case, then summed per group. They are not
independent exposure and may include quiet/unknown footage.

| New arm | V6 unresolved seconds | V6 common unobservable seconds | V5 unresolved seconds | V5 common unobservable seconds |
| --- | ---: | ---: | ---: | ---: |
| Pose | 28.573 | 2.253 | 4.671 | .033 |
| RGB/motion | 22.150 | 2.253 | 9.244 | .033 |
| Fused | 24.190 | 2.253 | 9.942 | .033 |

Common unobservable means the shared person/adjacent-motion gate used by the three ablations;
V2 has additional arm-specific limitations in its own outputs. Unresolved records overlap;
their naive summed durations must not be treated as distinct punching time.

## Cost, reproducibility and verification

Final new-case feature loops took **97.67 s**, excluding preflight attempts, inherited cache
validation and artifact hashing. That run overlapped verification work and is a single local
CPU observation, not an isolated latency benchmark. Inherited native pixels/poses/features
were reused. The fit batch is [23,240,162], with **3,659,760 bytes** of input/target/mask tensors
plus presence/activations/optimizer memory. Parameter bytes remain 59,276. Cumulative process
peak RSS is about **435 MiB**, including caches and baseline comparisons.

| Arm | Primary fit seconds | 27-case inference seconds | Decoder seconds | Initial → final weighted loss |
| --- | ---: | ---: | ---: | --- |
| Pose | 4.045 | .0217 | .4411 | .9203 → .0870 |
| RGB/motion | 5.333 | .0133 | .5635 | .9464 → .1326 |
| Fused | 3.918 | .0136 | .2174 | .9102 → .0628 |

All exact-repeat fits have identical state hashes, curves and predictions. Checkpoint reload
is exact; original-model scores are exact on the inherited 19 cases. No output changed the
architecture, optimizer budget, thresholds, masks or feature settings after training began.
Tests cover exact human association, missing/ambiguous/uninspected events, excluded overlap
masks, partial comparison without clipping, Unknown context, V5 role/optimization isolation,
source restrictions and HUD/identifier suppression. The shared existing suite covers native
PTS, missing poses, deterministic training and decoding. **716 Python tests and 111 frontend
tests pass**, as do Ruff, changed-Python formatting, TypeScript, Prettier and production build
(one existing Starlette/httpx warning). No frontend or production code changed.

**9,125 pre-existing artifact files remain byte-identical**, including all current human
revisions, original sources/poses, old datasets/checkpoints/experiments and historical review
backups. Preservation receipts and current review-head revalidation pass. New checkpoints,
feature caches, scores, figures and raw review backups are Git-ignored. No push, new human
labeling request or automatic production promotion occurs.

## Recommended next improvement

The result supports continuing research, **not deploying automatic punch labeling**. The
recording/activity shortcut is reduced enough to fit V6 instances, but the same model does
not reliably isolate V5 punches and degrades V4 fit. Fusion is not a winner because its V6
training matches are higher. RGB/motion's two V5 hits and V2's four do not establish accuracy.

The next bounded engineering step should diagnose **boundary supervision and pairing** before
adding architecture complexity: source/event-balanced loss sampling under the same truthful
Unknown masks, followed by a separately versioned decoder test for competing onsets, evidence
expiry and missing intermediate ends. Begin with synthetic controls and replay frozen scores;
do not lower thresholds or force combination splits to improve these eight V5 matches. A new
fixed-budget replication can then compare the unchanged TCN under a predeclared protocol.
Native-rate context differences and useful motion removed by identifier/torso masks are
specific input limitations to measure, not assumptions to correct using V5 scores. Keep V5 as
development, preserve all outputs, and require broader recording-separated evidence before any
reliability or six-class application claim. This milestone requests no further annotations.

## Reproduction

```sh
# Immutable snapshot and live provenance check:
.venv/bin/python -m boxing_motion_ai.data.blue_localization --check

# Fresh paths only; never overwrite the retained caches or run:
.venv/bin/python -m boxing_motion_ai.detection.localization_blue_features \
  --output reports/boxingvi/blue-localization-features-REPEAT \
  --preflight reports/boxingvi/blue-subset-preflight-20261030T055423Z
.venv/bin/python -m boxing_motion_ai.detection.localization_blue_experiment \
  --cache reports/boxingvi/blue-localization-features-REPEAT \
  --output reports/boxingvi/punch-localization-blue-REPEAT
```
