# Automatic punch candidates: conservative offline baseline

**Historical v1 assessment.** The application now defaults to the
[v2 reach-phase baseline](PUNCH_INSTANCE_SEGMENTATION_V2.md), which produces real proposals
but remains unvalidated. V1 implementation, policy and the findings below are preserved.

2026-10-29. The production upload workflow now has **Analyze automatically**. It reuses the
existing binary activity scan, attempts per-arm motion segmentation, and runs the unchanged
six-class classifier on accepted candidates. Manual selection/classification remains available.
There is no prototype migration, dashboard redesign, new model, retraining or decoder tuning.

**The pipeline works, but useful automatic individual-punch labeling has NOT been demonstrated
on the tested real recordings.** All tested real punching clips remained unresolved under this
conservative policy. Synthetic complete cycles exercise automatic classification/replay. This
is an auditable baseline and a negative real-footage finding, not a solved detector or generalization
result. Do not use candidate totals as punch counts or class scores as proof of punching.

## Using it

Use the existing [local setup](ASSISTED_PUNCH_RECOGNITION.md#run-locally), including the local
binary and Level 1 checkpoints, MediaPipe task, FFmpeg and dashboard/pose/train extras:

```sh
npm --prefix web run build
uv run --locked --extra dashboard --extra pose --extra train python -m boxing_motion_ai.api
```

Open http://127.0.0.1:8000. Upload a short continuous single-boxer video and wait for preparation.
Choose **Analyze automatically** below the existing timeline. There is no manual selection prerequisite.
Missing activity results are computed first; unchanged prepared poses and valid activity results
are reused. Processing status is visible. Errors can be retried. The existing upload limits remain
30 seconds, 150 MiB, 60 FPS, 1800 frames and four saved local videos.

The workspace distinguishes:

- **Possible punching activity:** unchanged training-only binary detector bouts. A bout may
  contain zero, one or several physical punches; it is not an individual classifier input.
- **Automatic punch candidates:** heuristic complete arm cycles. Accepted candidates get
  a predicted type, six expandable **uncalibrated** scores or a classification-unavailable reason.
  Names mean predicted type among jab, cross, lead/rear hook and lead/rear uppercut, not verified labels.
- **Unresolved activity:** portions of activity without an accepted individual candidate.
  They remain replayable and receive no six-class prediction. Missing poses are separately
  retained as unobservable time, never confirmed background.
- **Manual classification:** independent start/end controls and saved selections remain available.

Replay candidate/unresolved controls use their stored source range, bring the existing player into
view, and stop at the last included frame. They do not manufacture manual punch boundaries or
save a human review. The candidate's physical movement boundaries remain estimates even though
the selected source-frame inventory and timestamps are exact. The positive candidate UI was
verified with synthetic component tests; the real browser demonstrations showed unresolved results.

## Segmentation mechanism

`detection/instances.py` and `configs/punch_instances_v1.json` define
`pose-arm-cycle-candidates-1.0`. This is a **pose heuristic**, not a learned instance detector.
Its parameters are explicit engineering defaults, with synthetic edge-case checks; there was no
threshold sweep or parameter fitting against annotations, no reserved-source inspection and no
changes to the frozen binary decoder. The examined real failures were not forced into positive
outputs by relaxing the policy.

1. Validate original masks, pose arrays and exact timing. Reuse the existing aspect-correct,
   hip-centered, torso-scaled 2D features. For each arm, subtract its shoulder from its wrist;
   rigid body translation cannot itself create a wrist excursion. The elbow is also required
   for an arm to be observable and for the boundary flexion check.
2. Split motion at unreliable torso/arm poses, missing data, the existing activity resets,
   scene/segment changes, declared cuts or excessive timestamp gaps. No interpolation,
   smoothing, temporal filling or guessed source frames is used.
3. Find observed quiet wrist-motion runs lasting at least **2/25 s**, with shoulder-relative
   wrist speed at most **1 torso length/s**. Quiet does not automatically mean guard: boundary
   shoulder-to-wrist reach must be at most **0.8 of the projected upper-arm + forearm length**.
4. Starting at the last frame of a quiet run, look for the earliest subsequent quiet boundary
   that establishes a complete excursion and return. An intermediate pause during extension
   is not automatically an end or a second punch. A qualifying cycle must have:

   | Check | Engineering default |
   | --- | --- |
   | Boundary-frame elapsed time | 4/25 to 3/2 s |
   | Peak displacement from starting wrist position | at least 0.45 torso lengths |
   | Return distance from that starting position | at most 0.20 torso lengths |
   | Radial total variation / ideal out-and-back variation | at most 1.25 |
   | Outward and inward displacement / phase path length | each at least 0.55 |

   The variation/efficiency checks reject irregular or multi-peak movement instead of slicing
   it into a requested number of punches. Parameters are not physiological truths. The next
   cycle starts from the completed return run, rather than searching alternative boundaries
   for whichever classification score looks best.
5. Gate a cycle by its peak frame lying inside an existing observed activity bout. Motion
   context can extend outside a bout to establish onset/return, within the same contiguous
   observable segment. Reject a cycle crossing multiple distinct activity bouts.
6. Deduplicate identical proposals. Reject **every** proposal in an overlapping range cluster,
   including chained overlaps across arms. Do not trim them at arbitrary midpoints. Fast
   combinations often have overlapping recovery/extension and will consequently remain unresolved.
7. Use exact observed boundary frames: last pre-motion quiet frame through first post-return
   quiet frame, inclusive; storage is zero-based half-open `[start, stop)`. Residual parts of
   each activity bout remain unresolved. Rejected proposals and reasons are retained in the record.

A detected bout is never converted directly into one punch. Independent complete cycles can
produce several candidates within one bout, including repeated same-arm punches. Synthetic
sequences demonstrate this; the tested real combinations did not satisfy these requirements.
A feint, defense or another arm movement can still satisfy the heuristic if the binary model
also fires. Conversely, moving guards, occlusion, foreshortening, hooks/uppercuts without a clean
return, slow punches and overlapping combinations are likely to be missed or remain unresolved.

## Timing and classifier contract

This stage is **offline full-video analysis**. It reads complete quiet runs and requires later
return evidence. It does not claim causal instance emission, live streaming or measured instance
latency. `motion_evidence_through` records the last used post-return quiet-run frame (potentially
well after the candidate); it is not a backdated detection timestamp. Activity bouts keep the
unchanged causal decoder's actual `first_emission` and censoring fields. The source-clock median
gap metadata and streaming/batch guarantees of [activity scanning](UPLOADED_ACTIVITY_DETECTION.md)
are unchanged. There is no inference across unobservable regions.

Each accepted range goes through the original `analysis.inference.Classifier.classify`:
mask-aware70 quality policy, reliable torso normalization, exact-time backward velocities,
scene/segment/gap checks, original clip pooling and maximum three-second duration. There is no
arm-to-class lookup, No punch class, class-score threshold deciding whether a punch occurred,
new feature representation or trained model replacement. The pinned Level 1 checkpoint remains
`bd314c2f608a0f73b5b13bca912510a513e7491c6062a8c73446404f146bdbfa`.

Quality failures return unavailable without scores. Per-candidate model exceptions also return
unavailable, identify a retryable failure and allow another automatic-analysis request. A missing
or incompatible checkpoint fails the task clearly; no fallback scores or simulated predictions
exist in production. No classification score is used to generate, select or split candidates.

## API, artifacts and cache

`POST /api/analyze/videos/{id}/automatic` starts the task (202), returns the same in-progress
request, or reuses a valid finished result (200). `GET` on that path returns not_started,
processing, ready or failed. Existing local-origin/JSON security and opaque upload ID checks apply.
Only one analysis/inference task runs on the local executor; manual inference and removal are
blocked during automatic work. Activity scanning runs inline on that executor when necessary,
without queueing work and then deadlocking while waiting on itself.

Ignored `.analysis-local/<id>/automatic-<fingerprint>.json` contains:

- `automatic-punch-candidates-1.0`, upload identity, false human-review/verification flags;
- parent activity record fingerprint/hash, original bouts and unobservable intervals;
- heuristic version, full policy, implementation hashes and existing model/feature provenance;
- each candidate's arm, phase measurements, peak, start/exclusive-stop, native PTS/timebase
  source points, exact rational times, player-relative times and post-return evidence point;
- predicted class and all six uncalibrated scores, or an explicit unavailable reason;
- rejected motion proposals/reasons and replayable unresolved activity ranges.

The fingerprint binds current activity inputs/runtime, classifier pin/provenance/checkpoint,
segmentation policy and new inference implementation. Cache reads verify artifact hash, schema,
video identity and parent activity hash. Changed inputs, code, models or policy invalidate the
cache. Wall-clock timing is kept separately in `automatic-state.json`; volatile per-candidate
inference seconds are omitted from the deterministic record. Successful unchanged inputs reuse
inference; transient per-candidate failures can be retried, while quality-unavailable results stay
cached. Failed/interrupted jobs release busy locks and can be retried. This remains a single-server
local application, not distributed job coordination.

Automatic results do not enter manual prediction files, annotation ledgers, datasets or source
roles. Removing a local upload scopes cleanup to its own generated files. These upload-ID routes do not resolve registered recordings, including V9;
historical V1/V6, V7/V8 and development/evaluation exclusions remain.

## Actual verification and findings

All real checks used permitted local material, without training or changing thresholds:

| Upload | Frames | Activity bouts | Accepted individual candidates | Outcome |
| --- | ---: | ---: | ---: | --- |
| Existing six-second V6-derived diagnostic clip | 144 | 2 | 0 | Both bouts unresolved; stable complete returns not established |
| Fresh eight-second V4 excerpt around source 713 s | 192 | 2 | 0 | Both bouts unresolved; incomplete/weak cycles and arm visibility limitations |
| Fresh 28.987-second V4 excerpt around source 700 s | 695 | 8 | 0 | All eight bouts unresolved |
| Existing one-second no-person clip | 25 | 0 | 0 | Entire second unobservable, not confirmed background |

The first V4 excerpt includes the already reviewed combination around original frames
17121–17179 (lead uppercut, rear uppercut, lead hook). The activity model merges that sequence
into a broad region, and this baseline **does not separate it**. The new re-encoded upload has
its own validated frame clock; these original annotations were not supplied to inference or
used for new timing accuracy scores. Right-arm complete-pose coverage on that excerpt is only
about 49%, versus about 87% for the left. Torso observability alone is insufficient for reliable
per-arm segmentation. The longer clip was checked with the same policy, not used to fit it.
No incorrect individual splits were accepted in these examples because no individual candidates
were accepted; this is a failure to demonstrate useful extraction, not evidence of high precision.

Synthetic tests demonstrate isolated punches, two non-overlapping cycles in one bout, repeated
same-arm cycles, held extension, irregular/multi-peak motion, rigid body translation, overlapping
and duplicate proposals, missing poses/arms, gaps/cuts/resets, exact nonzero source origins,
classifier quality gates and deterministic six-score inference. Synthetic geometry is not real
boxing footage and establishes implementation behavior only.

A separate real-classifier-stage regression supplied the existing manual `[105,119)` range from
the V6 upload directly to the candidate-classification adapter. It reproduced the saved scores
exactly: lead hook 0.727813, jab 0.188512, lead uppercut 0.081364 (other scores below 0.001).
**Those bounds were manually supplied, not automatically detected.** This diagnostic was saved
only as a labeled verification receipt and never inserted into a production automatic result.
It verifies feature/model reuse, not successful automatic labeling.

Safari verified fresh uploads, pose preparation, automatic analysis without manual boundaries,
processing status, activity refresh, unresolved results, cached results and replay on the real
player. A range [284,300) replayed from its source-relative start and froze at its last included
frame after bringing the viewer into view. Candidate replay and six-score UI rendering are covered
by synthetic frontend tests, since no positive candidate existed in the real examples. Existing
manual-classification regression tests pass. Safari also reopened the saved manual [105,119)
range and completed its cached classification request independently of the automatic results;
the real classifier-stage check above preserves its scores.

Independent recomputation produced byte-identical records for every checked real upload,
including the archived eight-second example. Repeated successful API requests returned the same
cached request/result without changing the result mtime, cached poses or activity result bytes/mtime.
The longer clip took about 4.17 s for automatic work including its initial activity scan, separate
from 28.60 s of pose extraction. Those times and pose coverage are not recognition accuracy.

Receipts and screenshots stay ignored under `reports/boxingvi/automatic-punch-verification-v1/`.
The owned eight-second test upload was moved intact there to retain it while respecting the
four-video library cap; no pre-existing upload was removed. Existing research files were checked
against the prior 4,797-file preservation inventory, excluding V9. Both checkpoints, all frozen
feature/decoder files, datasets and human records remain unchanged. No evaluation-only V5/V7/V8/
V10 footage or V9 media was opened for this work.

Checks: **647 Python tests** and **99 frontend tests** passed, along with Ruff,
changed-Python formatting, TypeScript, Prettier and the production build. Python retains the existing Starlette/httpx deprecation warning.

## Remaining work for useful automatic labeling

1. Improve phase proposals against explicitly authorized instance-boundary references, with a
   predeclared development/validation protocol. Establish whether quiet guards/full returns can
   be replaced by stronger elbow/trajectory phase evidence without manufacturing midpoints.
2. Handle overlapping opposite-arm motion with an explicit uncertainty policy. Current whole-body
   classifier clips cannot be assumed to contain one punch when arm cycles overlap. Do not simply
   force disjoint clips or classify a complete combination as one punch.
3. Measure segmentation omissions, merges/splits and boundary error separately from conditional
   six-class accuracy. Audit pose visibility, frame rate and camera geometry. Activity recall itself
   limits candidate recall; the current binary model is still a tiny training-only smoke model.
4. Validate automatic clip distribution and scores on genuinely held-out authorized footage before
   claiming reliable automatic labels. No precision/recall, punch-count accuracy, calibrated
   confidence, boxer-independent performance or latency against human references is established here.

No additional training experiment, reserved-set inference or broad UI redesign was started.
