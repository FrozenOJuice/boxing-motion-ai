# Punch-instance segmentation v2: real-motion diagnostics

Follow-up: the [pose-observation study](POSE_OBSERVATION_STUDY.md) compares resolution,
cropping, tracking and the previously investigated RTMPose model with this frozen policy.
Its experimental outputs do not replace production poses or change the results below.

2026-10-29. V2 now proposes real source-frame ranges and sends eligible ranges to the
**unchanged trained six-class classifier**. It replaces neither trained model and does not
change the frozen causal activity decoder. V1 code, policy and historical outputs remain intact.

This is progress beyond v1's zero real candidates, **not reliable automatic punch labeling**.
The native V4 comparison has substantial misses, overextended boundaries, a false background
proposal and a wrong class prediction. No generalization, calibrated-confidence, punch-count
accuracy or coaching claim follows from these development/training-source checks.

## What was inspected

Both methods received the same poses, timing and real binary activity outputs:

- The existing V6-derived six-second upload and V4-derived eight- and 29-second uploads.
- Original cached V4 poses for V4-bg-1/2/3, using their existing 23-frame before/after context.
  This provides exact native-frame correspondence to the 11 existing human punch records.
- The 12 exhaustively verified V4-motion-1 through -6 and V6-motion-1 through -6 negative intervals, without
  extra unreviewed context: 1,151 frames, **48.006 seconds** total.
- The existing no-person upload, which remains unobservable rather than verified background.

No V9 media or V5/V7/V8/V10 evaluation footage was used. V4-bg-4/5 stay held. V1/V6 grouping,
V7/V8 grouping, original source roles, reviews, datasets, model parameters and checkpoints are
unchanged. Neither annotations nor classifier outputs enter candidate generation. The diagnostic
runner reads reference events only after both segmenters have returned.

The resized/re-encoded uploads have their own validated clocks, so they are integration examples,
not silently mapped to original human frame indices. The native V4 cases retain the original
integer PTS/timebase and explicit source-index offset. Only the exhaustively reviewed core is used
for matching/background judgments. A reference extending beyond its reviewed block is clipped
explicitly for that comparison; the final rear uppercut in V4-bg-2 remains in the 11-event inventory.
Context outside the core is unknown, not automatically negative. Native-context activity inference
uses the same frozen model/decoder for both methods; its input context differs from the older
core-only smoke demonstration, so its events need not equal that historical stream.

## Why v1 failed

All three original uploads have 100% torso observability. That does not establish arm visibility:

| Input | Frames | Left complete arm | Right complete arm | Left/right quiet runs |
| --- | ---: | ---: | ---: | ---: |
| V6 upload | 144 | 144 | 138 | 17 / 9 |
| V4 eight-second upload | 192 | 167 | 94 | 13 / 7 |
| V4 29-second upload | 695 | 588 | 367 | 40 / 31 |
| Native V4-bg-1 context | 143 | 120 | 62 | 9 / 5 |
| Native V4-bg-2 context | 142 | 138 | 67 | 5 / 2 |
| Native V4-bg-3 context | 143 | 120 | 48 | 6 / 3 |

“Complete arm” requires the original shoulder/elbow/wrist masks, normalization and pose-status gate.
Left/right mean MediaPipe anatomical sides, not an assumed lead/rear stance. Missing wrists are a
larger problem than missing elbows alone. For example, the eight-second clip has 96 observable
right-wrist frames but only 94 complete-arm frames; dropping the elbow prerequisite cannot recover
the missing wrist trajectory. Native V4-bg-3's right arm has only a 0.542-second longest complete
run, versus 1.209 seconds of continuous wrist evidence.

During the reviewed rear uppercuts, complete right-arm visibility ranges from **9.1% to 50%**.
The last rear uppercut in V4-bg-3 has only 28.6% wrist and 14.3% elbow visibility on that side.
There is insufficient evidence for reliable far-arm separation in many of these frames. V2 does
not fill those positions or claim to solve occlusion.

V1's quiet-boundary requirement also excludes real motion before its other tests can run.
There is no qualifying quiet run within any of V4-bg-3's four reviewed punches. In V4-bg-2,
only the jab contains a short left-arm quiet run; the following three have no left-arm quiet
run. The hook has a brief right-arm quiet run, but no usable activity-gated pair results.
A quiet run during extension is not necessarily a guard boundary.

The following is a census of possible same-observable-run quiet-boundary pairs with peaks inside
activity. Conditions are evaluated **independently**, so failure counts overlap:

| Input | Gated pairs | Guard ratio fails | Excursion fails | Duration fails | Return fails | Radial variation fails | Phase efficiency fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V6 upload | 20 | 0 | 5 | 0 | 13 | 17 | 19 |
| V4 eight-second upload | 15 | 2 | 15 | 6 | 7 | 9 | 15 |
| V4 29-second upload | 53 | 10 | 39 | 14 | 26 | 22 | 47 |
| Native V4-bg-1 | 6 | 0 | 6 | 1 | 2 | 5 | 6 |
| Native V4-bg-2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Native V4-bg-3 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |

These are diagnostic window opportunities, **not punches or final proposals**. V1 may skip some
alternate pairs after consuming an earlier cycle outside activity. Its actual ordered first-failure
counts differ: V6 has 5 excursion, 12 return and 3 irregular-phase rejections; the eight-second V4
clip has 13 excursion and 2 guard-ratio rejections; the 29-second clip has 25 excursion, 10 return
and 10 guard-ratio rejections. No case reaches v1's overlap rejection stage with an otherwise
accepted candidate. Overlap was therefore not the immediate cause of these zero outputs.

Activity gating matters too. V1 has 8 possible quiet pairs in native V4-bg-2 but **zero** with
peaks inside activity; V4-bg-3 has 12 pairs but only one gated peak. One accepted v2 V6 motion peak
occurs one source frame before a bout's start. Requiring that single point to fall inside the bout
can reject a phase whose recovery overlaps model activity. The frozen detector itself is limited
and can also fire during reviewed background; v2 must not turn its bouts into verified punches.

Detailed per-window measurements, independent failures, visibility fractions and complete
first-rejection logs are in the local `comparison.json` and per-case `diagnostics.json`.

## The versioned v2 method

`detection/instances_v2.py` implements `pose-reach-phase-candidates-2.0`;
`configs/punch_instances_v2.json` records its engineering policy. This is one development baseline
informed by the observed trajectory failures, not a parameter sweep or held-out selection. The
same diagnostic footage informed its design and its reported results.

1. Reuse the original aspect-correct torso normalization and masks. Analyze each wrist relative
   to its own shoulder. A missing elbow does not erase an observable wrist, but elbow coverage
   is retained and checked before six-class interpretation.
2. Break each arm's trajectory at missing wrist/shoulder/torso observations, source/model resets,
   cuts, scene/segment changes and timestamp gaps. A backward speed above 40 torso lengths/s
   introduces a tracking-jump reset. No missing landmark is filled and no gap is smoothed across.
3. Within an observable run, average observed positions with a **centered triangular 60 ms
   radius** using native elapsed timestamps. This is a derived segmentation signal only;
   cached poses and classifier inputs are unchanged. It is not frame resampling or interpolation.
4. Compute wrist-to-shoulder reach. Extract alternating minima/maxima confirmed by at least
   **0.18 torso lengths** of prominence. Small noise extrema do not each become a punch.
5. A candidate needs an extension and recovery around a maximum. Its start is the last observed
   pre-peak frame below 10% of the local reach rise; its last included frame is the first reaching
   90% of the local recovery. The return minimum can differ from the starting position: there is
   no exact coordinate-return requirement, compact-guard ratio or wrist-quiet prerequisite.
6. Require 80 ms of observed pre-onset context, at least 40 ms on each phase, a trimmed span of
   120 ms–1.5 s and a net trajectory turn of at least 60 degrees. A terminal minimum additionally
   needs 80 ms of recovery evidence with a remaining reach range at most half the prominence
   threshold; an internal recovery minimum is confirmed by a subsequent prominent rebound. These are estimates of motion support, not verified physical punch boundaries.
7. Require at least **80 ms of source-time overlap** with real model activity, rather than a
   peak-only gate. Context may begin before activity, but an unsupported phase outside all
   activity is rejected. Decoder activation/deactivation thresholds are unchanged.
8. Deduplicate by arm and peak. Preserve distinct arm phases rather than trimming them at
   arbitrary temporal midpoints. If a clip contains another candidate's peak, or peaks are
   less than 80 ms apart, withhold that clip's classification and retain unresolved activity.
   A shared trough frame between consecutive same-arm candidates is not itself a duplicate.
9. Also withhold classification if the candidate arm's elbow coverage is below 70%, or the
   phase crosses multiple separate activity bouts. Remaining candidates pass through the original
   classifier's **additional unchanged** mask-aware70, duration, torso and timing requirements.

New quantities are not equivalent to simply lowering v1's thresholds: local shoulder-relative
reach prominence replaces displacement from a fixed guard coordinate, and phase turns replace
quiet guard-pair enumeration. No class score selects a phase, moves a boundary or creates a split.
A scalar reach cycle can nevertheless correspond to guarding, defense, tracking noise or a
combination. Hooks with little reach change and occluded punches can remain undetectable.

This method is **offline**. Centered averaging, extrema and recovery evidence use later frames.
`motion_evidence_through` conservatively records the end of the used observable run, not a causal
instance emission. The parent activity record's actual first-emission times remain unchanged.
No instance latency or live-streaming claim is made. All exported bounds use observed native
source frames, integer PTS/timebases and exact rational timestamps, not nominal FPS.

## Actual comparison

| Footage | V1 candidates | V2 candidates | Six-class results available |
| --- | ---: | ---: | ---: |
| V6 six-second upload | 0 | 3 | 3 |
| V4 eight-second upload | 0 | 1 | 1 |
| V4 29-second upload | 0 | 3 | 3 |
| Native V4-bg-1 context | 0 | 0 | 0 |
| Native V4-bg-2 context | 0 | 1 | 1 |
| Native V4-bg-3 context | 0 | 3 | 3 |
| 12 verified no-punch intervals | 0 | 0 | 0 |
| No-person upload | 0 | 0 | 0; footage unobservable |

All seven uploaded-clip candidates reached the real pinned classifier. V6 produced:

| Zero-based upload range | Predicted type |
| --- | --- |
| [30,38) | cross |
| [82,109) | lead uppercut |
| [108,116) | lead hook |

The first range is approximately **1.251–1.585 s**; its cross score is **0.601662**, an uncalibrated
six-class score, not correctness probability. The V4 eight-second proposal was predicted lead
uppercut. The 29-second proposals were predicted lead hook, lead hook and lead uppercut. These
are genuine model outputs, not confirmed class labels or counts. V6 has no exhaustive instance
reference used here, so no V6 recall or recognition accuracy is calculated.

For native V4, references and candidates are explicitly clipped to the reviewed core. Matching
uses exact-time IoU ≥ 0.5, maximum cardinality then total IoU with deterministic ties. No first
emission is fabricated to reuse a latency metric. V1 matches zero of the 11 reference events;
v2 matches **2**, leaving **9 unmatched at that criterion**. Six references have some temporal
intersection with a candidate; mere intersection is not a successful match.

| Native source | Candidate range | Matched reference | Exact IoU | Predicted type |
| --- | --- | --- | --- | --- |
| V4-bg-2 | [22939,22960) | rear uppercut [22941,22955) | 2/3 | rear uppercut |
| V4-bg-3 | [11459,11474) | rear uppercut [11460,11470) | 417/625 | lead uppercut — wrong type |

Important failures:

- V4-bg-1's three reviewed punches remain unresolved. Fragmented observations and unestablished
  recovery boundaries still prevent proposals; v2's three phase attempts are boundary-censored.
- The other V4-bg-2 jab/hook/rear-uppercut references and three V4-bg-3 references do not achieve
  one-to-one IoU matches. A jab proposal in V4-bg-3 starts too early: [11441,11457), IoU about 0.37
  against [11450,11460). The last rear uppercuts in both blocks have no candidate overlap.
- Both matched ranges extend into at least 25% of an adjacent hook reference. They therefore
  have **two merge/overextended-recovery risks**, not demonstrated perfect individual isolation.
  The algorithm cannot use those annotations to repair the boundaries retrospectively.
- V4-bg-3 proposes [11387,11410), including [11391,11410) inside exhaustively reviewed background.
  This is an **incorrect proposal**, not a missed annotation inferred from a high class score.
- No reference contains multiple candidate peaks under the diagnostic split-risk rule. This does
  not establish correct counting: misses, a background proposal and boundary contamination remain.
- The negative intervals emit no final candidates because their frozen activity streams emit
  no bouts. Six geometrically supported phase proposals there fail only the activity-overlap
  gate. Thus kinematics alone are not punch-specific, and zero output on these small selected
  negatives is not a general false-positive-rate estimate.

V2 rejection counts on the three uploads are: V6, two duration, two boundary and one activity
rejection; eight-second V4, two boundary rejections; 29-second V4, five duration and three boundary
rejections. The source report retains every rejection and observed-run boundary, exposing
missing arm support without filling it.

## Integration, verification and reproduction

The existing **Analyze automatically** action now defaults to v2. Production records use
`automatic-punch-candidates-2.0`, bind the classifier pin and parent activity provenance, and include
the full policy, implementation hashes,
source bounds, arm coverage, overlap/eligibility flags, classification or unavailable reason,
rejected proposals and unresolved activity. The worker retains v1 policy-schema handling while default inputs select v2;
the diagnostic runner directly compares the separate implementations.

Fingerprints now bind the v2 policy and code. Old v1 records remain on disk; pressing Analyze
automatically recomputes segmentation/classification under a new fingerprint while reusing valid
poses and activity. No manual prediction, annotation or research record is overwritten. There is
no dashboard redesign or prototype migration.

Fresh Safari verification uploaded the real V6-derived video into isolated local storage on
port 8001, leaving all four existing library videos intact. Decode/MediaPipe ran once; Auto ran
without manual boundaries, returned the three ranges and real scores, and candidate replay froze
at zero-based frame 37 for [30,38). Both manual boundaries remained unset. A subsequent manual
API classification of that automatically generated range reproduced all six scores exactly.
Independent recomputation was byte-identical, and repeated requests returned the same cached
result without changing poses, preview or activity bytes/mtime. Preparation took about 14.31 s;
automatic processing including its initial activity scan took about 4.10 s, **not accuracy metrics**.

All v1/v2 result bytes also matched across **19 independently prepared diagnostic cases**.
Five compact real-motion fixtures provide regression checks without distributing source videos,
checkpoints or full pose datasets. Synthetic tests cover moving guards, repeated arms, overlap
withheld from classification, missing wrists/elbows, source gaps/cuts/resets, tracking jumps,
noise/body translation, delayed activity gating and exact variable timestamps.

From the repository root with existing pose/train/dashboard extras and local artifacts present:

```sh
.venv/bin/python -m boxing_motion_ai.detection.instance_diagnostics \
  --prepare --output reports/boxingvi/punch-instance-v2-new-run
# Optional fresh fixture output; never overwrite a retained fixture:
.venv/bin/python -m boxing_motion_ai.detection.instance_diagnostics \
  --output reports/boxingvi/punch-instance-v2-new-run \
  --fixture-out /private/tmp/punch-motion-v2-new.json
```

The runner uses an explicit V4/V6 inventory and fails if required retained local inputs are
missing. It never substitutes synthetic data. Per-case reports, motion plots, repeatability receipts
and actual browser captures stay ignored under `reports/boxingvi/punch-instances-v2-diagnostics/`
and `punch-instances-v2-repeat/`. The committed fixture contains only masked normalized trajectories,
source hashes, timing, cached activity bounds and clearly separate comparison references.

Verification: **666 Python tests, 99 frontend tests**, Ruff, changed-Python formatting, TypeScript,
Prettier and production build passed. The existing Starlette/httpx deprecation warning remains.
The 5,043-file preservation inventory, including prior experiment outputs and saved input/result
artifacts, is unchanged. V1 code/policy and both checkpoints are unchanged; V9 was not opened.

## What still needs work

Useful proposals are demonstrated, but robust combination separation is not. The present 2D poses
cannot support many occluded-arm instances. Improve person-crop/temporal pose reliability under
an explicit versioned contract, and collect authorized per-arm onset/apex/recovery references.
Evaluate boundary error and overlaps separately from conditional class recognition on a held-out
protocol. A visibility-aware temporal boundary/instance model is a reasonable next modeling
approach after that evidence exists and training is separately approved. Do not retune this tiny
pool until its counts look good or convert unobservable arms into inferred punches.
