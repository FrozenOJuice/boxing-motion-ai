# First verified background dataset

2026-10-28. **All 12 Motion Pilot 2 intervals pass saved-review validation.** They provide **48.006 seconds / 1,151 native frames** of manually verified no-punch footage, suitable for a bounded detector data/loader pilot. They do not yet make a continuous detector experiment evaluation-ready. No detector training or additional candidate discovery occurred.

## Actual saved decisions

The ledger contains **13 immutable revisions for 12 interval IDs**. V4-motion-1 has an earlier full-frame inspection review and a later continuous-playback review; the latter is selected without deleting the former. The latest reviews span 2026-10-28 16:04:59–16:09:46 UTC.

For every latest decision:

- Origin is `manual`; decision is `no_punch`; status is `verified_background`.
- `watched_entire` and `visible_continuous` are explicitly true, and `unannotated_punches` is `no`.
- Versioned playback evidence is source/cache/interval-bound, uses permitted 1× playback and covers the complete required media range. Recomputed `viewing_validation` equals its stored summary, including `playback_complete=true`, `complete=true` and `completion_mode=continuous_playback`.
- Exactly **one** native frame is recorded as individually inspected; `exact_complete=false`. Complete playback is not presented as inspection of every frame. V4-motion-1 has four playback spans; the other eleven have one each. Their union, not merely an ended flag, establishes coverage.
- Original source annotations do not overlap the certified interval under the existing provisional mapping. Complete poses/native indices, exact endpoint PTS, source video/pose/preparation hashes and known scene/segment continuity are revalidated. Actual first/last included source frames decode at the registered PTS.
- Notes and optional movement tags are empty. No observed punch, ambiguity or missing-punch flag is recorded in these twelve decisions. This reports the saved human findings; it is not a new automated visual judgment.

The prior V4-motion-1 review retains its original 96-frame exact-inspection evidence and confirmations. The complete chain is validated, including the superseded revision. Latest review head: `1cc389740ef91efd0c7808723c473294417ab1767ad00326f5307681ede9661a`.

## Verified inventory

Native frame bounds are zero-based **[start, stop)**. Duration uses the exclusive endpoint's exact PTS, not a nominal FPS multiplication. Context before/after each interval is not included in the certification or duration.

| Interval | Native frames | Seconds | Discovery stratum (not human movement classification) |
|---|---|---:|---|
| V4-motion-1 | [33087,33183) | 4.004 | Quiet motion |
| V4-motion-2 | [1655,1751) | 4.004 | Quiet motion |
| V4-motion-3 | [17791,17887) | 4.004 | Lower-body motion |
| V4-motion-4 | [29707,29803) | 4.004 | Lower-body motion |
| V4-motion-5 | [432,528) | 4.004 | Moderate arm motion |
| V4-motion-6 | [33807,33903) | 4.004 | Mixed motion |
| V6-motion-1 | [44524,44620) | 4.004 | Quiet motion |
| V6-motion-2 | [960,1055) | 3.962 | Quiet motion |
| V6-motion-3 | [19589,19685) | 4.004 | Lower-body motion |
| V6-motion-4 | [35365,35461) | 4.004 | Lower-body motion |
| V6-motion-5 | [20884,20980) | 4.004 | Moderate arm motion |
| V6-motion-6 | [2278,2374) | 4.004 | Mixed motion |

V4 supplies **576 frames / 24.024 seconds**, V6 **575 frames / 23.982 seconds**. The discovery mix is 4 quiet, 4 lower-body, 2 moderate-arm and 2 mixed-motion intervals. These are pose-ranking strata. Guarding, footwork, defensive movements and feints were not separately human-tagged; do not claim verified counts for those categories. No extra human tagging is required to preserve the valid absence decisions.

## Versioned artifacts and reproduction

- Private dataset manifest: `reports/boxingvi/verified-background-v1/manifest.json` (`verified-background-1.0`). SHA256: `a96d2a87a9afe363f06f53d342caca5284bf08f9506230a2cb3c2514f9bd6bbb`.
- Git-tracked reference: `configs/verified_background_v1.json`, pinning the manifest and backup receipt hashes.
- Byte-for-byte copies of **all 18 interval-review revisions** (13 motion + 5 original), both pilot configs and both original candidate manifests: `reports/boxingvi/verified-background-v1/review-backup/`. Their 22-file SHA256 receipt is also pinned. This is durable review backup data, not disposable preview cache.
- Read-only exporter/checker: `boxing_motion_ai.data.verified_background`. It reuses `IntervalWorkspace`, the decision contracts, coverage validation, metadata store and exact frame service. It does not regenerate candidates, alter labels or train.

On a workspace where this export does not yet exist:

```sh
.venv/bin/python -m boxing_motion_ai.data.verified_background
```

To validate the existing export, current sources, review chain and backup:

```sh
.venv/bin/python -m boxing_motion_ai.data.verified_background --check
```

Creation refuses existing output/config rather than overwriting. The same live source/review snapshot produces the same manifest. Any new review revision, even another confirmation, makes the frozen version stale until explicitly versioned again. Latest holds or positive decisions cannot be silently included. Exact frame coordinates, endpoint integer PTS/timebases, source inputs/hashes, original candidate provenance, manual review timestamp/hash, explicit confirmations and raw playback evidence are retained. Original candidate manifests remain unchanged with their original `unknown` state; the new manifest adds a separate reviewed supervision layer.

Only listed intervals receive `no_punch` background supervision. All other unlabeled footage and uncertified context remain **unknown**, with no background loss. These records are absence intervals, not punch `Event` objects. Training remains disabled/not authorized in this export. Existing positive exclusions are not restored or relabeled.

## Diversity and remaining readiness gaps

This is a real improvement over the original gap-only pilot: 12/12 reviewed proposals were accepted as no-punch footage, versus five original V4 candidates that contained unmarked punches. This small, deliberately selected batch is not an unbiased estimate of future yield.

The set is adequate to exercise verified-negative loading, masking, continuity and provenance. It is **too small to establish robust negative diversity or generalization**: only 0.8001 minutes, two recording environments, selected for reliable landmarks, with unknown boxer/session identities. Quiet/low-activity and clean-pose selection can create motion/visibility shortcuts. Hard defensive movements, feints, rapid non-punch arm movements and poor-pose conditions are not demonstrated by the saved tags. Keep those limitations explicit; do not automatically mine or label more footage now.

The five original Punch present / unmarked-punch reviews remain intact as positive-event labeling leads, never negatives. Source workbook rows and the historical 1,123-example curated clip pool are not exhaustive activity timelines. Human-confirmed clip boundaries are useful references, but do not automatically establish a uniform detection onset/end definition or certify every inter-punch gap. Existing approved anchor reviews should be reused without redundant review; verify only genuine timing/coverage gaps needed by the detector.

**Current pilot split overrides historical classifier training membership:** only V4/V6 may supply training material; V5 is development-only; V7/V8 stay together and V10 remains evaluation-only. Retain the V1/V6 guard and null boxer/session identities. Do not import the whole historical multi-recording positive pool into this pilot, since it contains evaluation recordings. Its existing conservative V4/V6 subset is 520 + 295 = 815 provisional clips, preserving all holds; this is not 815 fully verified detection events. V9 remains untouched. No boxer-independent or independent-test claim is supported.

## Smallest practical next milestone

Prepare a **bounded exhaustive event-reference pilot**, not another negative-candidate batch or model experiment:

1. Add a small versioned event-list layer to the existing interval review workflow. Preserve original rows and separate activity, six-class identity/hand certainty, exact onset/end or uncertain ranges, and overlapping/mixed actions. A Punch present decision alone cannot provide event boundaries or prove all events were labeled.
2. Start with the five already reviewed V4 punch-present intervals: mark every observed punch, including previously unmarked events, using context and a frozen onset/end guideline. Reuse existing human-confirmed eligible anchors and inspect a small representative V6 positive subset only where the activity timing remains unverified. Keep V6:264/266 and all historical exclusions held; absence notes outside these twelve reviewed intervals do not generate negatives.
3. Exhaustively annotate the **six already prepared evaluation blocks**: two each in V7, V8 and V10, totaling **179.526 seconds**. Record all punches plus ambiguous/ignored ranges and explicit full coverage; the current ledger has no evaluation reviews. No new candidate generation is needed. Maintain V7/V8 grouping and evaluation-only use. Establish a short exhaustively reviewed V5 development sequence before selecting activity thresholds; development footage must not enter training or the held-out scoring pool.
4. Freeze the training-positive/background intersection and separate development/evaluation manifests. Only then seek approval for the causal activity/type loader smoke test and one-seed detector baseline from the existing [continuous-detection plan](CONTINUOUS_DETECTION_PLAN.md). Report event precision/recall, false positives per reviewed minute and true emission latency; do not use unknown exposure as negative time.

The genuine blocker is now **exhaustive positive-event/timing and continuous evaluation references**, not re-confirming these twelve negatives or reviewing every historical clip. The existing interval contract explicitly has `exhaustive_event_reference=false`; No punches verification cannot complete a mixed-action evaluation block.

## Verification and preservation

The full Python suite passes **486 tests**, including seven new export tests and the existing interval/continuous-coverage tests. The targeted interval/export regression run passes 35 tests. Ruff lint, formatting of changed Python files and diff whitespace checks pass. The existing Starlette/httpx deprecation warning remains. No frontend code changed.

The exporter was run on the actual saved reviews, then its independent `--check` path revalidated the frozen manifest, current source/review state and byte-for-byte backups. All **4,669 pre-change protected input/review hashes** match after validation; the inventory is in ignored `reports/boxingvi/verified-background-validation-v1/protected-inputs.json`. Original human review records, workbook/video files, curated datasets and historical results are unchanged. No V9 file was opened. Generated dataset manifests, backups, diagnostic receipts and media remain ignored; only code, tests, documentation and the small hash reference enter Git.
