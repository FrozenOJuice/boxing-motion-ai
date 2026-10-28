# Verified continuous dataset v1 — readiness

2026-10-28. **The saved state contains three completed V4 intervals, not four.** V4-bg-1–3 pass validation; V4-bg-4 and V4-bg-5 have no saved event revisions. A read-only check of the existing Chrome dashboard agrees: the first three are Completely labeled, while interval 4 shows zero punches and no active event draft. No browser navigation, save, certification or draft discard was performed for this investigation.

The versioned manifest freezes the verified subset: **11 human punch events, 12 verified background intervals, and 3 exhaustive positive-review intervals**. It explicitly records the requested count of four and a failed count-match readiness flag. Neither missing completion is fabricated. V4-bg-5 stays explicitly held, even if its review is changed later. V4-bg-4 is excluded until a genuine saved completion exists; no re-review is demanded merely to implement a binary smoke test.

## Saved events and timing

| Interval | Native [start, stop) | Type | Full event duration |
|---|---|---|---:|
| V4-bg-1 | [17121,17138) | lead uppercut | 0.709 s |
| V4-bg-1 | [17140,17164) | rear uppercut | 1.001 s |
| V4-bg-1 | [17164,17179) | lead hook | 0.625 s |
| V4-bg-2 | [22924,22940) | jab | 0.667 s |
| V4-bg-2 | [22941,22955) | rear uppercut | 0.584 s |
| V4-bg-2 | [22955,22971) | lead hook | 0.667 s |
| V4-bg-2 | [22971,22982) | rear uppercut | 0.459 s · partial overlap |
| V4-bg-3 | [11450,11460) | jab | 0.418 s |
| V4-bg-3 | [11460,11470) | rear uppercut | 0.417 s |
| V4-bg-3 | [11470,11476) | lead hook | 0.250 s |
| V4-bg-3 | [11476,11483) | rear uppercut | 0.292 s |

| Class | Events | Total full-event seconds |
|---|---:|---:|
| jab | 2 | 1.085 |
| cross | 0 | 0.000 |
| lead hook | 3 | 1.542 |
| rear hook | 0 | 0.000 |
| lead uppercut | 1 | 0.709 |
| rear uppercut | 5 | 2.753 |

The latest event sets have zero uncertain-type flags, zero overlapping pairs and no repeated event IDs. Consecutive intervals sharing an endpoint are preserved as consecutive, not reported as overlaps. No labels, boundaries or duplicates were automatically corrected. Full durations range from **0.250 to 1.001 seconds**, totaling **6.089 seconds**; they include the reviewer’s selected movement/return boundaries, not a newly inferred biomechanical phase. This validates saved evidence and exact source correspondence, not an independent visual re-adjudication of human judgments.

All 15 event-history revisions were replayed: 4 for interval 1, 5 for interval 2 and 6 for interval 3. Interval 3 includes a revised event; only its latest value enters the dataset, and its earlier value remains backed up. Each current completion is a manual `complete` operation with explicit `watched_entire=true` and `all_punches_labeled=true`. Combined exact-still/continuous-playback coverage is recomputed against the source-bound preview; it does not claim that every native frame was individually inspected. Every original upsert/change retains its boundary-inspection evidence. Source video hashes, original pose timelines, native indices, cuts/segments, rational PTS and actual decoded first/last included frames are revalidated.

One rear uppercut in V4-bg-2 is [22971,22982), while that reviewed interval stops at 22978. Its full bounds and 0.459-second duration are retained. The four frames beyond the exhaustive interval are explicitly outside the initial frame-supervision domain, not silently clipped out of the raw event. V4-bg-1’s first event also starts exactly at the review boundary. Preserve these edge conditions in future event/timing evaluation; surrounding context is not exhaustively labeled merely because a known punch extends into it.

## Supervised footage and unknown regions

| Material | Native frames | Seconds |
|---|---:|---:|
| 12 original verified no-punch intervals | 1,151 | 48.006 |
| 3 completed mixed/punch intervals | 287 | 11.970 |
| Total exhaustive footage | **1,438** | **59.976** |
| Punch-active frames inside exhaustive domains | 142 | 5.922 |
| Verified non-event frames inside completed punch intervals | 145 | 6.048 |
| All supervised background frames | 1,296 | 54.054 |

V4 supplies **35.994 seconds** (including all positives); V6 supplies **23.982 seconds**, all negative. The original 12 negatives remain six per recording. The additional 6.048 seconds are six non-event spans inside the three **explicitly completed human reviews**; their background supervision is supported by the all-punches-labeled attestation, not by a gap in the source workbook. They are not six independent new review candidates. Negative frames outnumber active frames by approximately 9.1:1. No background target is assigned outside these exhaustive domains. No unknown joint positions are filled; pose confidence, validity masks and timestamps remain in their existing hash-bound sources.

The frozen manifest separates raw full events from bounded `supervision_runs`, which carry exact endpoint PTS/timebases, activity targets, event IDs and conditional type-loss eligibility. Unknown types and simultaneous events disable single-class type loss while retaining verified activity. Masks for label supervision are separate from pose-validity masks. Overlaps are supported and reported rather than resolved by class priority. Conflicting background/event supervision blocks export.

## Versioning, backups and reproducibility

- Private dataset: `reports/boxingvi/continuous-dataset-v1/manifest.json`, schema `continuous-dataset-1.0`.
- Committed hash reference: `configs/continuous_dataset_v1.json`; manifest SHA256 `c7993318ee8f8a16e2cd89f04ad0acaf7ce95c0cad8aabad0c70079b683b2d59`.
- All 59 referenced review/source metadata files have byte-for-byte copies under the new dataset's `review-backup/`, including the event-history revisions and the accepted background dataset's prior backup history. These are durable human-review backups, not disposable preview cache. Videos/pose arrays remain in their original locations, referenced by existing hashes.
- `data/continuous_dataset.py` composes the existing background validator and event-ledger replay rather than implementing another review system. It refuses to replace an existing dataset/config and rejects source, review, backup or hash changes. The synthetic event fixture now includes its real source-group guard, matching production provenance.

```sh
# Only on a workspace where this new version does not exist:
.venv/bin/python -m boxing_motion_ai.data.continuous_dataset

# Revalidate the existing dataset, live reviews, sources and backup bytes:
.venv/bin/python -m boxing_motion_ai.data.continuous_dataset --check
```

A new review revision, including a subsequently completed fourth interval, invalidates this snapshot. Prepare an explicitly reviewed **new version**, rather than overwriting v1, automatically enabling the fifth interval or claiming the earlier snapshot contained four completions. Original ledgers, annotations and the accepted verified-background v1 are never rewritten. Current all-other-footage supervision is Unknown. The dataset has no development/evaluation examples and records `detector_training_authorized=false`.

## Readiness and genuine blockers

**Enough for data engineering and a binary training-only smoke test after approval; not enough for a validated six-class continuous-detection experiment.** No immediate additional manual annotation is necessary to implement/test a loader, causal processing or a small overfit check. Missing the fourth saved completion does not prevent that narrow work, and the rapid fifth interval may remain held indefinitely.

The limits are substantive:

- Cross and rear hook have **zero** events. Lead uppercut has one; the other classes have only 2–5. Four-class coverage is not adequate six-class coverage, and even the represented classes have too few examples for reliable performance estimates.
- All positive events are from V4. V6 contributes negatives only, creating a recording/activity shortcut. The entire supervised collection is about one minute, deliberately selected for reviewability; it does not establish natural punch prevalence or hard-negative diversity. Confidence/mask and recording-background shortcuts remain plausible.
- V5 has no exhaustively reviewed continuous development reference in this dataset. Its pre-segmented classifier validation pool cannot supply false-positive exposure or complete detection references.
- V7/V8/V10 have six prepared but unreviewed continuous evaluation blocks (179.526 seconds). Until all events/unknown ranges and full coverage are human-established, their unmarked footage cannot be scored as negative exposure. Keep V7/V8 together and V10 evaluation-only. Prior classifier work has already informed development using these sources: new detector holdouts can be recording-separated, but are not pristine independent-test evidence or boxer-independent evidence. Athlete/session identities remain unverified. V9 stays reserved and unopened.
- One partial event and one event touching the left review edge need an explicit future scoring-edge policy. Their saved labels/bounds are not automatically rejected or corrected. Raw interval completion establishes action inventory, not perfect biomechanical boundary consistency.

Do not random-split frames, adjacent windows or events from V4 into a supposed independent validation set. All current V4/V6 material is training-only; preserve the V1/V6 guard. Do not import the historical 1,123-example classifier pool, which includes recordings reserved for evaluation in this pilot.

If six-class positives are expanded later, reuse existing **eligible human-confirmed** references where sufficient instead of requiring blanket re-review. For example, V6:46 is a previously fully verified rear-hook anchor; it can be considered in a separately approved sparse-positive version, keeping surrounding footage Unknown. Previously confirmed cross anchors are in V8/V10, so they cannot fill this pilot's training deficit without leakage. A trustworthy V4 cross reference would require genuine targeted verification. No anchor, excluded dispute or provisional clip is imported into v1; V6:264/266 and all historical exclusions remain held.

## Smallest practical next implementation

After reviewing this dataset, implement the **window loader and a binary causal activity baseline**, without attempting to solve absent six-class targets first:

1. Reuse the existing torso-normalized XY + validity-gated exact-PTS velocities, original confidence/validity masks and approved mirror transform. Keep label-supervision masks separate. Use real predecessor/context input with up to 14 warm-up frames; no loss outside the certified domain or on warm-up/padding, and no interpolation across cuts, extraction gaps or nonmonotonic timestamps.
2. Reuse the 32-channel causal TCN encoder with dilations 1/2/4 and a per-frame **one-logit punch-activity head**, replacing clip pooling for this detector. Preserve the classifier and checkpoints. Types remain metadata for diagnostics; do not pretend to train or evaluate six-class recognition with absent classes. Streaming state must reset at discontinuities; test future-perturbation causality and batch/stream equivalence before fitting.
3. For a separately approved smoke fit only, use one fixed seed and a tiny fixed budget on this training pool. Balance supervised activity/background exposure during training and track negative-source contribution; do not let negative-only V6 dominate or feed recording IDs/metadata into features. Report training fit as an engineering check, with no model selection, threshold tuning or generalization claim from these same intervals. No need to label the rapid fifth interval first.
4. Before a meaningful detector comparison, establish a short **exhaustively reviewed V5 development sequence** with both activity and background, then exhaustively label the six already prepared V7/V8/V10 evaluation blocks. Freeze thresholds/event decoding on development before held-out scoring. Report event precision/recall, false positives per fully reviewed minute and actual first-emission latency; retain unknown/edge/abstention accounting. Missing cross/rear-hook training diversity becomes necessary for a six-class claim, not for the initial loader/smoke milestone.

No detector, loader, model head, training loop or new candidate generator is implemented in this milestone. The next action is approval of this bounded binary engineering milestone, not an automatic new training experiment or a demand to re-label the existing accepted negatives.

## Verification

The full Python suite passes **503 tests** (one existing Starlette/httpx deprecation warning). Seven new synthetic tests cover immutable export/check/backup, exact-time supervision, incomplete exclusion, the explicit fifth hold, overlap/unknown-type loss masking, split/conflict rejection and staleness after review changes. Existing event and background tests remain passing. Ruff, changed-Python formatting and diff whitespace checks pass. No frontend code changed.

The exporter and independent `--check` were run against actual saved human records and source video/pose hashes. All 4,709 pre-change protected source/review/artifact hashes are checked after preparation. No source labels, review history, source footage, pose tensors, model results or existing dataset versions are modified. No V9 file is opened. Generated manifests, raw backups and validation receipts remain Git-ignored; only code, tests, documentation and the small hash reference are committed.
