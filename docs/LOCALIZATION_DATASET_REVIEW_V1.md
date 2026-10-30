# Localization evidence and bounded review batch v1

Starting revision: `74d0bca`. **Queue preparation only; no new training or model promotion.**
The existing continuous dataset, activity detector, V2 analyzer, classifier, checkpoints and
source reviews are unchanged. The user approved minimal queue/registry enablement through the
existing editor, with no layout changes. This is the stopping point for human review.

## Audit: what genuinely exists

The accepted continuous v1 contributes **11 V4 events**, three exhaustive mixed intervals and
12 exhaustive negative intervals (six V4, six V6). Its 1,438 exhaustive native frames cover
59.976 seconds; 142 active frames and 1,296 background frames. All its positives are V4.
V4-bg-4 remains incomplete; V4-bg-5 remains explicitly held.

The live, hash-linked human ledger is version 19, SHA256
`f5d8ef1e437fa7f50beec40e81b9bb526b1a3422dd2d5b7d4f7426e30f90ce5c`.
Intersecting actual full human confirmations with the still-approved conservative source
selection identifies **four additional eligible punches**, not an expanded classifier pool:

| Human source reference | Type recorded by human | Native [start, stop) | Evidence receipt |
| --- | --- | --- | --- |
| V4:144 | Jab | [10404, 10411) | dashboard-reviews-v1/v00000011 |
| V4:343 | Lead hook | [18192, 18202) | dashboard-reviews-v1/v00000012 |
| V6:7 | Lead hook | [486, 497) | dashboard-reviews-v1/v00000016 |
| V6:46 | Rear hook | [8368, 8386) | dashboard-reviews-v1/v00000015 |

Each has a human accept revision, confirmed timing/type/hand, exact-frame evidence including
both included endpoints, no unresolved uncertainty, and membership in the preserved approved
selection. The exporter checks the original candidate digest, source inputs, native PTS and
immutable receipts. Inclusive reviewed endings become exclusive `end + 1`, never nominal-FPS
estimates. These add **two V4 and two V6 events**, yielding 15 unique training references.
They provide sparse positive activity and known event boundaries only. Their surrounding
footage is **Unknown**, not background or an exhaustive event inventory.

V4:497 and V6:613 have full human confirmations but remain outside the conservative approved
selection. They are reported separately and **not restored**. V6:264/266 and all historical
holds/exclusions remain held. Source workbook marks are never promoted to labels. There are
**zero boundary-verified V5 instance references** in the audited review ledger; provisional
assistant suggestions and classifier development examples do not change that.

The exact selected-list digest remains
`a758ac1c070560f82af0e3365ecef2f9da83fd01d99303d4c14e61341365ffb4`.
No V7/V8/V10 footage is decoded or selected here. No V9 path is opened.

## Prepared human review work

**Six intervals, 52.053 seconds of core footage:** five V6 training candidates totaling
40.041 seconds, and one V5 development sequence of 12.012 seconds. One second of approximate
native context on each side helps inspect boundary-crossing events; context is not exhaustive.
Times below are exact source-PTS seconds rendered as decimals for reading. Integer PTS/timebase
endpoints are retained in the manifest.

| Queue ID | Native core [start, stop) | Source seconds [start, stop) | Core suggestions | Purpose |
| --- | --- | --- | ---: | --- |
| V6-motion-101 | [450, 642) | [18.769, 26.777) | 3 | Hook/uppercut trajectories, guard and recovery; includes V6:7 |
| V6-motion-102 | [6400, 6592) | [266.933, 274.942) | 2 | Repeated lead-arm movement and recovery |
| V6-motion-103 | [8280, 8472) | [345.345, 353.353) | 3 | Rear-arm trajectories; includes V6:46 |
| V6-motion-104 | [25050, 25242) | [1044.794, 1052.802) | 2 | Uppercut trajectories and intervening movement |
| V6-motion-105 | [30490, 30682) | [1271.687, 1279.695) | 4 | Both-arm hook/uppercut combination |
| V5-motion-101 | [8100, 8460) | [270.270, 282.282) | 8 | Development-only jab/cross/hook combinations |

There are **22 source suggestions overlapping the cores**, including two already verified V6
anchors, plus four context-only suggestions. These are workload hints, not a verified count;
review must discover omissions, reject bad marks and inspect actual boundaries. No detector or
learned localization output selected these windows. Contexts are checked against native
scene/timestamp continuity, cuts and the preserved conservative source exclusions. The rapid,
excluded V6 source rows 391–396 were deliberately not queued.

Planning estimate: **20–40 minutes** for full viewing, about 44 suggested boundary inspections,
corrections, missed punches and uncertainty marking. This is an unmeasured estimate, not a
requirement to confirm 22 punches. Only 52 seconds of core viewing is mandatory; replay and
exact-frame work account for the rest. No blanket re-review of the 12 existing negatives or
11 existing V4 events is requested. Source suggestions cover varied trajectories; visibility,
footwork/defense diversity and actual punch counts must still be established by the human.

### How to review

Start the existing local application in the usual way, or from the repository root:

```sh
.venv/bin/python -c 'from pathlib import Path; import uvicorn; from boxing_motion_ai.api.app import create_app; uvicorn.run(create_app(Path.cwd()), host="127.0.0.1", port=8000)'
# In another terminal:
npm --prefix web run dev
```

Open `http://127.0.0.1:5174/`, then **Development → Review → Video intervals**.
Choose **Individual punches · V6 / V5** in the existing Review pilot selector.

1. Watch the complete core, including unmarked movement. Source annotations are provisional.
2. Add each actual punch; pause and use exact frames to set its first and last included frame.
   Saves retain native exclusive stops. Types default to Unknown: type/hand confirmation is
   unnecessary for class-agnostic localization. Do not guess a six-class label.
3. Mark occlusion, unclear movement or an inseparable combination as an **uncertain section**.
   Do not stretch one broad activity bout into a supposed individual punch.
4. Save events, then attest complete viewing and inventory before **Save complete & Next**.
   An empty completion additionally attests no punches and continuous boxer visibility.
   A seek, unchecked checkbox or partial playback cannot certify exhaustive review.
5. Include V6:7 and V6:46 in those cores' full inventories. Their prior references remain
   reusable independently; the editor does not manufacture new human revisions for them.
   Re-entered identical bounds are deduplicated by the dataset adapter. If careful new review
   contradicts an anchor, preserve the new revision and resolve that conflict explicitly
   before export; do not force agreement or silently replace the original evidence.

Events can extend into available context to preserve genuine full boundaries. A boundary at
or beyond the reviewed core is marked censored for core supervision. If the available context
cannot establish it, retain uncertainty. Mark invisible/ambiguous stretches even in a block
that otherwise has clearly visible punches. Reviewed uncertain sections remain Unknown.
The V5 queue item is prominently Development only. **It must never optimize weights.**

## Dataset adapter and immutable artifacts

This extends the existing interval-review registry, buffered viewer, exact-frame inspection,
coverage validation and append-only `IntervalEvents` transactions. No annotation platform or
new layout was built. A small `LocalizationReview` subclass adds uncertainty and explicit
empty-interval completion to a separate namespace; original event/activity routes are unchanged.
V5's legacy `array_slice_half_open` marks now appear as provisional suggestions in the editor.

- Queue pin: `configs/localization_review_v1.json`.
- Private immutable queue/audit: `reports/boxingvi/localization-review-v1/`.
- New manual ledger namespace: `reports/boxingvi/localization-event-reviews-v1/`.
- Evidence pin: `configs/localization_evidence_v1.json`.
- Current pending evidence: `reports/boxingvi/localization-evidence-v1-review-ready/manifest.json`.
- Earlier preparation snapshots remain preserved in their original separate directories.
- Original continuous v1 pin and SHA256 remain unchanged:
  `c7993318ee8f8a16e2cd89f04ad0acaf7ce95c0cad8aabad0c70079b683b2d59`.

`data/localization_evidence.py` is a review/supervision adapter over continuous v1, not a
replacement dataset framework or a new training entry point. Its manifest carries the original
intervals, preserved exclusions, four sparse anchors, pending/reviewed new intervals, canonical
exact-range event references, source groups, review heads and durable review-byte backups.
Pending intervals have no background supervision. Individually saved human punches may supply
sparse positive evidence before completion; only explicit full completion permits background
between events. Uncertainty overrides local activity targets, including overlapping event
regions; event-boundary eligibility is separate from activity and background supervision.

Each run has exact source PTS/timebase and native bounds. Core-edge censoring and known boundary
eligibility are explicit. Exhaustive boundary-absence domains exclude uncertain regions; the
existing localization loader's edge/support-band policy must still be applied when constructing
losses. Sparse anchors do not create negative boundary labels around themselves. Pose availability
is never used to establish a background target. Original feature arrays, masks and training code
are unchanged. Canonical identical bounds merge origins instead of double-counting imported
anchors; conflicting human background/positive evidence and changed anchor bounds block export.
Uncertain raw event references are not eligible boundary truth merely because they are saved.

Roles remain recording-separated: V4 training, V1/V6 one training source-group guard, V5
**development-only**, V7/V8 one evaluation group and V10 evaluation-only. There is no random
frame/event split and no adjacent-frame train/development leakage. V5 influenced prior classifier
development and is **not an independent final test**. Boxer/session independence is unverified.

```sh
# Revalidate current source/review hashes, queue, original dataset, evidence and backups:
.venv/bin/python -m boxing_motion_ai.data.localization_evidence --check

# Prepare/reuse the existing buffered clips, without extracting poses or saving reviews:
.venv/bin/python -m boxing_motion_ai.data.localization_evidence --previews

# After real review, freeze a NEW snapshot. Existing output paths are never overwritten:
.venv/bin/python -m boxing_motion_ai.data.localization_evidence --check \
  --snapshot reports/boxingvi/localization-evidence-after-human-review-v1
```

Preparation is exclusive creation; an existing queue cannot be regenerated over itself.
Live checks replay review history and validate source hashes/PTS. Frozen snapshots retain their
own review-byte backups. Later review edits are reflected only in a newly exported snapshot;
the current committed evidence pin remains the historical pending snapshot.

## Readiness and next fixed experiment

**The queue and adapter are ready. Training/development datasets are not ready for the next
controlled localization experiment.** All six new cores are currently unreviewed; no new
completion or negative label was fabricated. Four additional real anchors mitigate V6-negative-
only supervision, but only two V6 hook events cannot establish general transfer.

Once the bounded human work is available, the practical next experiment is:

1. Freeze a new evidence snapshot with complete V6 and V5 inventory/uncertainty masks; resolve
   anchor conflicts, verify both positive and background exposure per recording, and inspect
   boundary consistency. Keep unresolved regions excluded rather than demanding forced labels.
2. Adapt the existing native feature/supervision loader to this snapshot without altering the
   architecture. Add V5 feature generation strictly for development inference, preserving exact
   timing, observation masks and censor/support-band rules. Verify input redaction on all new
   windows: V6 has visible instructional HUD text; V5 has a changing corner number. Neither
   may enter RGB/motion features as a label or recording shortcut. No new features were generated
   or assumed safe on these windows in this milestone.
3. Reuse the same lightweight temporal model and three ablations (pose, RGB/motion, fused), seed
   20261029, 120 full-batch updates, CPU, AdamW .003/decay .0001, existing head weights and frozen
   offline boundary decoder. Same eligible V4/V6 data and budget across arms; no architecture,
   decoder or hyperparameter search. Train on approved V4/V6 evidence only. No V5 optimization,
   threshold selection, best-epoch selection or inference-derived labels.
4. Report all arms and frozen V2 on the same reviewed domains: one-to-one IoU ≥ .5 matches,
   misses, signed source-time boundary errors, incorrect splits/merges, background proposals per
   exhaustively reviewed minute, unresolved/unobservable time and runtime. Keep censored events
   and conditional errors visible. Publish V4/V6 fit separately from V5 development transfer;
   do not name a winner on these few examples or describe V5 as final validation.

Expanded V6 positives and within-recording negative/guard/recovery evidence directly address the
previous V4-positive/V6-negative shortcut. They do not remove athlete, camera, setting, HUD,
trajectory or label-style shortcuts. Broader recording/boxer diversity and separate exhaustive
held-out references would still be needed for credible reliability claims. No additional broad
annotation request or reserved evaluation access is part of this batch.

## Verification and remaining issues

The real exporter and independent `--check` run against the original human records and native
source hashes. All six buffered clips decode and independently round-trip their exact PTS;
no pose extraction repeats. Local Chrome opened every queue item, played each for ~1.6 seconds,
paused and advanced one exact source frame; there were no browser exceptions or annotation-save
requests. These are transport/queue checks, not human punch verification or full-clip inventory.
Screenshots at 1440×900 and 1280×800 were inspected and stay Git-ignored. The bundled browser
host was unavailable; a local headless Chrome/CDP fallback performed these checks.

Existing editor scrolling is still necessary, particularly at 1280×800; no layout redesign was
attempted. Context-only source marks can extend beyond the core, and annotations/HUD text can bias
human review; inspect the whole raw footage rather than confirming suggestions mechanically.
Uncertain sections use the same editor with class targets disabled; review completion does not
make those regions known. Contradictory anchor boundaries need explicit reconciliation before
snapshot export. V5 feature-cache support remains work for the subsequent approved experiment.

Automated verification covers exact boundaries/PTS, coverage/visibility attestations, uncertain
and incomplete supervision, censoring, deduplication/conflicts, V5 development roles, source-group
and reserved-source rejection, immutable snapshots/backups, idempotent saves, stale writes,
restart replay and original-review preservation. Frontend tests cover separate endpoints,
class-agnostic punch saves, uncertain ranges and explicit no-punch payloads. The full suite passes **708 Python tests** (one existing Starlette/httpx deprecation warning)
and **102 frontend tests**. Ruff, changed-Python formatting, Prettier, TypeScript, production
build and diff whitespace checks pass.

A preservation receipt checks **9,036 pre-existing files** spanning source data, reviews,
models and prior experiment/feature artifacts; no changed bytes are permitted. No source,
annotation, review, model or previous experiment is rewritten. Preview images and generated
manifests remain private and are not committed. No new training experiment has run.
