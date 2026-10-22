# Dataset expansion and path to the first PyTorch baseline

Historical pre-training snapshot. The current provisional reference is **V4-only XY +
validity-gated velocities + 50% mirror augmentation**, with V5 development and V9 reserved.
The [frozen transfer diagnostics](LEVEL1_VELOCITY_TRANSFER.md) do not establish consistent
benefits beyond V5 and recommend a controlled representation investigation before V6
expansion. V7/V8 remain one unresolved source group, and the V1/V6 guard remains.
The reference registry is `configs/level1_reference.yaml`. This supersedes the prospective
recommendations below; those findings remain historical.

This step performs data validation only. No temporal model has been implemented or trained.
Source punch labels, raw-index mappings and endpoint assumptions remain provisional. Pose
retention is a quality filter, not validation of the expected action.

## Recording decisions

- **V4/V5:** the first provisional development sources, with full independent-frame poses.
  Keep original masks/scores and all annotation exclusions. The user accepted their working
  raw-index mappings; this does not establish author FPS/base/end/trim conventions or a gold
  evaluation set. V4's dominant unreliable arm is right; V5's is left.
- **V9:** strongest next full-extraction candidate. All 29 assessed sample events pass the
  unchanged mask-aware policy and the inspected rear-hook stills show plausible hook motion.
  Review row 28 (out of order, 110-frame source span), 13 strict overlap pairs and 27 shared
  endpoints. Exact classes and boundaries still need sequential review.
- **V7:** promising pose samples, but 62 strict overlap pairs plus 16 shared-endpoint pairs
  materially limit clean source support. All rear-uppercut source rows are flagged under
  the conservative policy. Their diagnostic images show plausible uppercuts; do not silently
  remove the flags or drop that class from the problem. Resolve the annotation conventions
  and overlaps with a versioned human review before using a six-class held-out partition.
- **V8:** promising additional viewpoint, with turning/self-occlusion and only 21/25 sampled
  events retained. Lead hooks retain 2/5. No rear-hook annotations exist in this workbook.
  Keep partially reliable joints masked; do not lower thresholds to recover counts.
- **V10:** promising standing positive samples (30/30 retained), with displayed uppercut
  actions broadly matching sampled uppercut intervals. The video also includes kickboxing
  and plank exercises; review segment scope, source row 123's reversed interval and stance/
  lead-rear semantics before adding it to the first standing-boxing baseline. A “Plank +
  Uppercut” overlay does not itself make a sampled standing uppercut invalid. Unlabeled
  exercise footage remains unknown, not an automatic negative set.
- **V1/V2/V3:** hold out of the training pool pending alignment investigation. V1 sampled
  “jab/cross” intervals include on-screen slipping and a round title card. V2 row 91 falls
  in rest/instruction, and row 233's expected lead uppercut falls in sit-up jab/cross footage.
  V3 row 2 is a warm-up bend, row 446 is rest/instruction and close views lose landmarks.
  Many such frames have good pose confidence: confidence cannot repair wrong labels.
- **V6:** investigate class/phase alignment despite 20/20 sampled events passing pose policy.
  Only hook/uppercut labels exist; jabs/crosses must remain absent. Review source row 3's
  inconsistent numeric helpers, row 379's ordering and 214 strict overlap pairs. Explicit
  raw/24-FPS-time/30-FPS-time comparisons at rows 2/321/686 do not establish a replacement
  mapping. The 30-FPS late example maps to instruction; 24-FPS time shifts late anchors by
  roughly 1.8 seconds, enough to change punch phase. This is a hypothesis discrepancy, not
  a measured annotation error. Do not choose a mapping from pose retention.

All eight new sampled runs are diagnostic samples, not full training datasets. Complete
source and sampled counts, exclusions, class/joint statistics and recording-specific visual
links are in the ignored `reports/boxingvi/expansion/` artifacts. The actual workbooks contain
5,451 event rows across V1–V10; class counts come from those rows, not published totals.

## Manual evidence still needed

Inspect sequential video around early/middle/late events of every class, especially the
flagged examples above. Verify action identity, stance/lead-rear meaning and whether the
source bounds include onset, extension, recovery or adjacent punches. Check both source
index bases and end conventions. Stills are useful diagnostics but cannot certify timing.

For V1–V3, request the authors' exact annotated video versions, annotation FPS, origin/trim
information and any edit map. Existing helper rebases and URL playback timestamps are not
sufficient. The existing source-rate, 30-FPS and helper previews contradict simple global
hypotheses at some anchors; no automatic correction is justified. Preserve the originals
and store any later approved corrections in a separately versioned review sidecar.

V4/V5 also need a manually verified evaluation subset. V4's cut inventory is not certified;
V5 has a known cut at 10679 but not an exhaustive inventory. Independent IMAGE extraction
has no smoothing/tracking state across cuts. Later temporal features must not infer visual
continuity from continuous PTS or an empty cut list.

## Leakage-resistant partition strategy

Keep every recording, all its overlapping windows, augmentations and derived poses in one
partition. Do not randomly divide frames or punches. Before training, review near-duplicate
segments and any known shared boxer/session links. Keep linked groups together.

Local download metadata supports these conservative **publisher** groups:

| Guard group | Recordings |
| --- | --- |
| NateBowerFitness | V1, V6 |
| FightCamp | V2, V3, V4 |
| Precision Striking | V7, V8 |
| Self-Improvement Boxer | V5 |
| wearecult | V9 |
| Utah Lee | V10 |

Publisher groups are a leakage precaution, **not boxer IDs**. Boxer/session identities remain
null. Different publishers do not prove different people, and no boxer-independent claim
is justified. Verify identities/session provenance when possible without guessing from faces.

For a first engineering baseline, a fixed **V4 train / V5 development validation** experiment
is reasonable once a small positive evaluation subset is checked. It is a two-recording
prototype, without a final test claim. Both recordings have already informed data-policy
work and should not later be presented as untouched test data.

For a broader evaluation, provisionally reserve **V9 for test**, use **V7+V8 together for
validation** after endpoint/class support review, and use **V4+V5 for training**, adding V6
only after its alignment review. Keep V1–V3 quarantined until resolved. V10 can become another
independent source-family evaluation candidate after segment/label review. This is a proposed
strategy, not an assigned split: full poses and reviewed six-class support are still missing
for the held-out sources. Freeze the test manifest before classifier development; report
classes with absent support explicitly. An honest six-class three-way benchmark is not ready.

Fit scaling/statistics, class weights and any learned preprocessing only on the training
partition. Tune model hyperparameters on validation only. Do not choose a test mapping or
confidence policy using classifier accuracy. Report macro-F1, per-class support/confusion,
per-recording results and retention bias; a mask-only ablation is required before attributing
recognition gains to motion rather than view-dependent missingness.

## First baseline implementation, after review

1. Freeze immutable candidate/pose/label manifests and the existing `mask_aware70` policy.
   Use six provisional positive classes. Keep excluded source rows and unknown footage out
   of supervised class targets; no verified `no_punch` examples have been established.
2. Start with the eight core joints' image-aspect-corrected 2D coordinates, original scores
   and validity masks. Normalize using valid current-frame body anchors. Preserve unavailable
   features under masks; any neutral tensor placeholder is not an invented joint estimate
   and must not overwrite the pose archive. Avoid interpolation, future/full-clip statistics,
   direct row/video/absolute-time features and left-arm-specific fixes.
3. Implement a small mask-aware causal temporal convolution classifier in PyTorch, with
   deterministic seeds and a modest training budget. Exact timestamps remain available for
   temporal interpretation; do not assume 30 FPS. Omit cross-frame velocity features until
   scene boundaries are reviewed, or compute them only across verified continuous frames.
4. Run the two-recording engineering experiment and compare coordinate/mask/confidence
   ablations. Inspect failures, source-specific cues and class-dependent exclusions. No
   numerical accuracy target is promised by these data checks.
5. Expand only after the held-out source families have full extraction and reviewed labels.
   Level 2 additionally needs explicitly reviewed background/feints and onset/peak labels;
   unknown gaps are not valid negatives and source starts are not biomechanical onset truth.

See [EXTRACTION_RESUME.md](EXTRACTION_RESUME.md) for commands/recovery guarantees and
[BOXING_MOTION_AI_SPEC.md](BOXING_MOTION_AI_SPEC.md) for the authoritative architecture.
Licensing and redistribution remain unverified; generated data/media/reports stay out of Git.

## Completed full-video development support

| Class | V4 retained/eligible | V5 retained/eligible | Combined retained |
| --- | ---: | ---: | ---: |
| jab | 117/118 | 226/226 | 343 |
| cross | 39/39 | 138/138 | 177 |
| lead_hook | 140/144 | 68/69 | 208 |
| rear_hook | 73/74 | 39/39 | 112 |
| lead_uppercut | 71/75 | 39/39 | 110 |
| rear_uppercut | 87/104 | 73/73 | 160 |
| Total | 527/554 | 583/584 | 1110 |

V4 contains 35,363 extracted frames (19,735 all-body valid, 15,137 partially unreliable,
491 missing). Only one missing frame occurs within its eligible positive intervals; most
missing detections are outside those intervals, which remain unknown. These are pose-status
counts, not verified negative labels. Every full V4 frame and all 2,507 new sampled frames
match the cached exact source PTS inventory. Source hashes and the existing V5 poses remain
unchanged. Reuse was verified without invoking inference; 99 automated tests and Ruff pass.
