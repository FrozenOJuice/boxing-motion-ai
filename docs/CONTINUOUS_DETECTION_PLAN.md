# Level 1 continuous punch detection — implementation plan

**Current continuous dataset (2026-10-28):** [Continuous dataset v1](CONTINUOUS_DATASET_V1.md) contains the 12 accepted backgrounds and the **three** currently saved completed V4 event reviews (11 punches). The requested fourth completion is not persisted; intervals 4 and 5 remain excluded. Binary loader/smoke engineering is feasible, while six-class and continuous development/evaluation readiness remain blocked.

**Current dataset status (2026-10-28):** [Verified background v1](VERIFIED_BACKGROUND_V1.md) supplies 12 manually verified intervals (48.006 s). For this pilot, training is V4/V6 only, V5 is development-only, and V7/V8/V10 are evaluation-only; this supersedes broader training-source proposals below. Exhaustive event/evaluation labeling remains outstanding.

2026-10-26. Planning only; no detector, new labels or training implemented.

## Decision and genuine blockers

Proceed next with a small continuous-interval review pilot, then an engineering baseline. The existing full-video poses, exact-frame viewer and causal features are sufficient building blocks. **Verified negative footage and exhaustively reviewed continuous evaluation intervals are not yet available under an appropriate contract.** Those are genuine blockers to meaningful detection training/evaluation, not reasons to build another annotation platform.

Retain MediaPipe, the conservative 1,123-example provisional positive pool and every historical exclusion. Preserve the RTMPose comparison. The current pool contains 190 jabs, 97 crosses, 301 lead hooks, 108 rear hooks, 245 lead uppercuts and 182 rear uppercuts; it is not an exhaustive continuous timeline. V6:264/266 remain held, and the six confirmed disputes remain excluded. Neither a rejected punch nor a human note saying “no punch” automatically creates a verified background interval.

This completes the missing continuous-detection portion of Level 1 in the [specification](BOXING_MOTION_AI_SPEC.md). Causal processing is an engineering requirement; early-before-peak recognition and anticipation remain separate Level 2/3 goals. No early-recognition claim or performance target is imported into this pilot.

## Existing components and required extensions

| Existing component | Reuse | Small extension needed later |
|---|---|---|
| `vision/decoder.py`, `vision/contracts.py`, prepared `poses.npz` | Native decoded frame index, integer PTS/rational timebase, confidence, masks, scene/segment IDs | Bounded continuous-window manifest and hash validation; no new full extraction |
| `models.py`, `data/preparation.py` | Six punch types plus unknown; separate `BackgroundState.NO_PUNCH`; source provenance and unknown intervals | Separate versioned reviewed-interval/coverage contract; do not force background into punch `Event` |
| `data/annotation_review.py`, `services/review_workspace.py` | Manual/automated distinction, independent confirmations, immutable revisions, atomic writes, stale-write protection | Interval targets not tied to workbook rows, explicit absence/completeness attestation, new-event proposals |
| `api/media.py`, current React Review screen | Hash-bound exact-PTS frames, playback, timeline, skeleton toggle, Save/Next | Small “continuous intervals” queue using same viewer; interval boundaries and three outcomes |
| `training/data.py`, `velocity.py`, `augmentation.py` | Causal torso-normalized XY, exact-time gated velocities, unchanged masks, consistent mirror transform | Window loader with supervision and padding masks separate from joint validity |
| `training/models.py` | 32-channel TCN encoder, dilations 1/2/4, 15-frame receptive field | Per-frame activity/type outputs instead of complete-clip mean pooling |
| `training/multi_recording.py`, `data/curated_subset.py` | Curated selection, source-group guards, fixed seeds, provenance | Continuous-window sampling and fold manifests; retain positive-clip experiment code unchanged |

Current dashboard review targets are source-row cases. Its three punch confirmations cannot express “there are no punches anywhere in this interval.” Do not overload a confirmation, unknown punch, Hold or Exclude for that purpose. Existing manually verified source boundaries also do not establish a shared biomechanical onset definition or exhaustive annotation coverage.

## Actual interval inventory and starting queue

Read-only inventory from each `candidates-full-v1/unknown-intervals.json` and its full `poses.npz`:

| Recording | Decoded frames | Unknown frames | Unknown intervals | Internal ≥5 s gap candidates* |
|---|---:|---:|---:|---:|
| V4 | 35,363 | 31,081 | 448 | 15 |
| V5 | 15,519 | 10,451 | 430 | 6 |
| V6 | 46,705 | 36,485 | 440 | 31 |
| V7 | 28,062 | 25,743 | 106 | 33 |
| V8 | 24,081 | 21,742 | 157 | 23 |
| V10 | 26,917 | 25,111 | 123 | 10 |

*Interior gaps of at least five seconds whose centered four-second window lies in one known scene and extraction segment. Opening/closing gaps are omitted. Source interval coverage includes plausible excluded annotations; malformed bounds cannot establish coverage. These numbers are **not negative counts**, action prevalence or exhaustive labeling percentages. Unknown edits and unannotated punches may occur even in a numerically continuous gap.

These twelve approximately four-second windows are a bounded discovery queue, selected as centers of the two longest eligible gaps per recording. Frame ranges are zero-based **half-open**; displayed PTS below are rounded seconds only. Authoritative writes must use integer PTS/timebases from the arrays.

| Recording | Candidate A frames / source PTS | Candidate B frames / source PTS |
|---|---|---|
| V4 | [17121,17217), 714.088–718.092 s | [22883,22978), 954.412–958.374 s |
| V5 | [10451,10571), 348.715–352.719 s | [13484,13604), 449.916–453.920 s |
| V6 | [2883,2978), 120.245–124.207 s | [27921,28017), 1164.538–1168.542 s |
| V7 | [8644,8764), 288.421–292.425 s | [6838,6957), 228.161–232.132 s |
| V8 | [3349,3469), 111.745–115.749 s | [6396,6516), 213.413–217.417 s |
| V10 | [14413,14513), 576.520–580.520 s | [8222,8322), 328.880–332.880 s |

The inventory is saved locally in `reports/boxingvi/continuous-detection-plan-v1/inventory.json`. These windows have **not** been visually certified or assigned movement categories. Long gaps may be instruction, editing or missing punches; they are discovery candidates, not the final negative sampling distribution.

Use the queue to find pauses/ordinary movement, then deliberately add near-punch recovery/guard intervals and high-motion non-punch intervals. The previous pose audit's V6:264/266 guard-like intervals (approximately frames 20097–20107 and 20288–20297 shown there) are useful *review leads*, not approved negatives. Keep their existing holds unchanged. Defensive movement and feints have not been established in a verified inventory; specifically search for them, and report absence if none can be verified. Do not promise all categories exist.

Background categories are descriptive tags, not new classifier classes:

- Ordinary movement: steps, repositioning and arm movement without a punch.
- Guard/recovery: guard adjustments; distinguish a punch's return phase from subsequent guard motion using the boundary rule below.
- Defensive movement: slips, ducks, blocks or evasions without a simultaneous punch.
- Pause: visible boxer with little movement; exclude title cards, empty scenes and edits from in-domain negative training.
- Feint: an observed aborted/simulated attack with no completed punch under the review guideline. Intent is not inferable from a still frame; unresolved feint-versus-short-punch cases remain unknown.

Pose speed/missingness may prioritize diverse windows but never supplies a negative label. Include easy and moving negatives across viewpoints; selecting only static pauses would create a motion shortcut.

## Lightweight human verification

First review twelve discovery windows (~48 seconds of footage plus context), then cap a first data collection pass at **five 4–8 s negative candidates per recording**, one per category where available. This is a budget of up to 30 candidates, not an assertion of sufficient model data. Replace unusable candidates within that budget; report missing categories rather than relaxing evidence requirements.

Reuse Review: select interval → watch with one second of context each side → choose **No punches / Punch present / Not sure** → Save & Next. Context is shown but is not automatically certified. Allow shortening an interval to a clearly negative section. Optional category chips and notes suffice for negatives. Require explicit “I watched this whole interval; no punches occur here” and visibility/continuity confirmation. A sampled contact sheet or automatically completed playback is not an attestation.

“Punch present” can link a source event or propose a new event in a separate review layer; it never silently restores an excluded row or edits a workbook. “Not sure” retains unknown supervision. Preserve existing type, hand and boundary confirmations for punch reviews; a generic punch can have activity verified while its six-class identity remains unknown. Jab/cross still require verified hand semantics for a fully verified class.

Add a small versioned interval ledger using the existing file-based locking/revision pattern, separate from historical punch ledgers. Proposed fields: interval ID, recording/source-group ID, source/pose hashes, native [start,stop), endpoint PTS/timebases, known scene/segment, target kind, decision, movement tags, manual/automated origin, explicit review evidence, reviewed coverage, timestamp, prior revision hash and unresolved reasons. Unknown boxer/session IDs remain null. Save human input as `manual`; automated candidate selection cannot certify absence. Reject conflicts, stale hashes, out-of-bounds ranges, cuts, missing extraction and V9. Superseding a review invalidates dependent manifests rather than modifying them.

For evaluation, absence labels alone are insufficient: review **all action and background** in each selected continuous block. Add an explicit coverage-complete attestation, list every punch, and mark uncertain subintervals. No further manual review of the entire 1,123 positive pool is required for the pilot. The genuinely necessary additional work is negative certification plus exhaustive review of a small continuous benchmark.

## Event and timing semantics

Preserve source bounds and conventions unchanged. A new detection-reference version stores independently reviewed native-frame bounds and their evidence; never reinterpret source starts as verified onset. Proposed guideline to pilot before freezing:

- Onset: first visible committed departure of the punching arm into the punch trajectory, judged using surrounding footage; preparatory guard movement is excluded.
- End: first frame after the punch's active excursion/return finishes or becomes a distinct next action. Recovery can overlap another hand's punch; do not require a stationary guard. Store the first excluded frame as `stop_exclusive`.
- If an onset/end cannot be localized, retain an uncertainty range or mark boundary unknown. Do not invent a single exact timestamp. Freeze a clear subset for primary timing metrics; report uncertain examples separately.
- Peak is optional and not required for Level 1. It must never be inferred from maximum pose speed and called human truth.
- Separate same-hand consecutive punches when visually distinguishable. Preserve distinct event IDs and true overlaps; ambiguous mixtures remain unknown. A source overlap alone does not prove two punches, duplication or a correction.

Proposed prediction record: recording ID, event ID, estimated start/stop native indices and exact source times, predicted type/scores, first emission source time, finalization source time, observed-through frame, model/config versions and optional wall-clock processing latency. Backdating an estimated onset must not backdate when detection was actually emitted. Revisions to a live prediction retain the initial emission and subsequent history.

“No detection,” “no punch” and “unobservable pose” are distinct. Missing torso/arm observations retain masks; do not fill joints or declare background. Unknown annotations use an ignored supervision mask, not a seventh truth label. Explicit background may become a model target but remains a separate state in the event contract.

## Proposed small causal baseline

Implement later in a separate detection module; preserve the clip classifier and its artifacts. Reuse XY + validity-gated velocities, the 50% sequence-consistent mirror augmentation and existing causal TCN encoder. Geometry, visibility augmentation and artificial pre-clip context options remain disabled. Real preceding stream frames are required state, not a new augmentation.

A minimal factorized head predicts per-frame punch activity plus six conditional type logits. Train activity with masked binary loss on approved positive intervals and manually verified background; train type only on eligible labeled punch frames. No loss on unknown, held/excluded or conflicting overlap frames. Start with isolated/non-overlapping trusted windows; explicitly report this limitation instead of assigning one class to simultaneous actions. Separate overlapping-event evaluation as a stress set; it may require a later multi-event output design if common. Do not claim this pilot solves simultaneous punches.

Use native frames and exact PTS initially; retain the existing 15-frame receptive field and report its varying duration (~0.47–0.58 s across roughly 30–24 FPS). Velocities use actual elapsed time, so no nominal-FPS conversion is introduced. A time-resampled representation would be a separate future experiment.

Train bounded sequential windows with up to 14 preceding real frames for encoder warm-up; no loss on warm-up/padding. Compute initial velocity with a valid real predecessor where available. Reset state at cuts, extraction gaps and nonmonotonic timestamps. Warm-up may contain unknown labels but supplies input only. Never pad from future frames, restart at every punch boundary, normalize using an entire interval, or pass IDs/label metadata to the model. Keep masks and source provenance available for audit but metadata out of features.

Sample training-only by source group, then activity/class strata so long recordings and negatives cannot dominate. Begin with a documented roughly balanced positive/background supervised-frame budget, with existing group-aware positive sampling; no double class weighting. Real continuous evaluation preserves natural prevalence. Keep the unchanged pose validity policy; derive/report any window eligibility rules explicitly rather than pretending the clip-level mask-aware70 policy already defines windows. Long missing-pose runs trigger abstention/reset, with thresholds fixed on development and abstention duration reported.

Decode events online with development-fixed activity hysteresis, persistence and end-gap rules in **seconds**, not hidden nominal frame counts. Emit class confidence at the actual emission time; finalize boundaries after the end condition. Use no future smoothing. Do not automatically merge nearby punches: combinations need separate events, and same-class punches without a detectable valley are a known failure mode. Freeze decoder settings before held-out evaluation.

First run loader/causality and one-batch smoke checks, then one pilot seed (20261021) only after data readiness is approved. Include never-punch and simple motion-threshold sanity baselines, plus a mask-only control. Do not reuse six-class clip scores as calibrated activity probabilities. A modest detector training experiment requires a new explicit approval; none is run now.

## Recording-separated evaluation

Training sources: curated positives and newly verified negatives from V4/V6/V7/V8/V10. V5 remains development-only. Keep V7/V8 together, preserve the V1/V6 guard, and leave V9 completely inaccessible. No adjacent chunks from one recording cross folds. Unknown people/session identities prohibit boxer-independent claims.

Bounded evaluation pilot: two **30-second continuous blocks per recording**, one mixed action and one background-rich, reviewed exhaustively. This is six minutes total including V5, before replay/review overhead. Select using raw footage before detector predictions; avoid only selecting clean successes. Extend blocks for event context, score a declared interior, and mark edge-crossing/uncertain events ignored. If unresolved content prevents complete review, replace the block or report the remaining scored duration; never count unverified minutes as negative exposure. This small benchmark cannot precisely estimate rare false alarms; if no alarms occur in one minute, do not claim a reliable near-zero rate.

For each non-V9 source-group holdout, remove **all** that group's positive clips and background windows from training and checkpoint selection. Use V5 alone for selection/tuning; frozen fold evaluations are diagnostic generalization, not a newly independent test, because these sources have already informed development. The group-held-out benchmark includes both V7 and V8 when that group is held. V9 remains reserved until separately authorized, with the model/settings frozen first.

Freeze a deterministic one-to-one temporal event matcher before scoring: maximum-cardinality then maximum-IoU assignment, stable tie-breaks. Primary temporal IoU threshold .5, sensitivity .3 and onset-within-200-ms reported separately; these are proposed protocol choices, not established accuracy guarantees. Report class-agnostic matches and class-aware matches separately. Duplicate detections count as false positives; missed references count as false negatives; class errors appear in matched-event confusion and count as FP/FN in class-aware scores. Same-class overlapping events must each have distinct predictions to match.

Report:

- Event precision/recall/F1, per-class precision/recall, six-class macro and supported-class macro; absent classes explicitly identified.
- Unmatched event alarms per **fully reviewed scored minute**, and separately alarms per verified-background minute. Publish numerator, denominator, abstention time and unknown/ignored time; do not drop poor-pose minutes silently. Missed punches during abstention remain misses in the primary end-to-end metric.
- For matched events: signed first-emission minus reference-onset latency, median/p90 and fraction of negative delays; also first-correct-class emission latency. Backdated onset error and finalization delay are separate. Report misses alongside latency to avoid a fast-easy-event bias.
- Onset/end errors on timing-verified events, confusion matrices, errors by movement tag/viewpoint, and duplicate/merged-event failures. Boundary-uncertain examples are excluded only from precise timing metrics, with counts disclosed.
- End-to-end runtime separately from source-time latency: decoder + pose + features + model + event logic. Cached-pose throughput is not live pipeline performance.

No claims about pre-peak detection, landed punches, boxer identity or independent-test performance follow from this pilot.

## Milestones and acceptance gates

1. **Contract/viewer extension:** interval review, immutable manual evidence, continuity checks, exact timing, no V9; synthetic tests for revision history, scope, conflicts and unchanged original hashes. No training.
2. **Bounded review:** discovery queue, up to 30 negative candidates, six minutes of continuous reference blocks. Publish actual verified seconds, categories, class counts and unresolved areas by group. Require negatives and positive events in training/development, and at least one reviewable held-out group. Missing feints/defenses limit claims; do not invent them or demand an exhaustive new dataset.
3. **Manifest/window loader:** explicit positive/background/unknown masks, curated exclusions, source-group split checks, source/review hashes, complete reference coverage, no class metadata features. Test off-by-one endpoints, rational PTS, overlaps, changing revisions, cuts/gaps, invalid poses and repeated deterministic builds.
4. **Detector pilot only after approval:** future-perturbation causality test, batch-versus-stream equivalence, no chunk-boundary artifacts, masked-loss/padding tests, smoke run, frozen decoder/matching tests and one-seed group-held-out evaluation. Preserve every previous experiment.

The first blocker is missing verified absence/continuous coverage, not estimator replacement. The second is a consistent timing guideline for the small reference set. Limited true overlaps, missing negative categories and 2D pose failures should be measured as scope limitations before expanding implementation. Licensing remains unverified; no redistribution is proposed.

## Verification of this planning milestone

Only this document and ignored read-only inventory outputs were created. Verification: 442 Python tests passed (one existing Starlette/httpx deprecation warning), and Ruff passed; no frontend/backend behavior or contracts are changed. All 4,647 existing pose-audit protected hashes were rechecked and matched. The manual work above is proposed, not claimed complete.
