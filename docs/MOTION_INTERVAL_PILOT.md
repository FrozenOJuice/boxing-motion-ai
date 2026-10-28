# Motion-guided interval review — pilot 2

Prepared 2026-10-27. There are **12 unverified proposals**, six each from V4 and V6, totaling **48.006 seconds** inside the review bounds. Each has approximately one second of playback context on each side. All are unknown; no punch-absence label or human review was created. This is a second candidate-discovery pilot, not a detector or a negative dataset release.

## Open the new queue

Refresh http://127.0.0.1:8000, choose **Review → Video intervals**, then select **Review pilot → Motion pilot 2 · V4/V6**. Original pilot remains available in the same selector with its original 36 candidates and saved decisions. Changing pilots respects unsaved edits and cannot mix their histories.

Use the same buffered playback, speed controls, exact-frame inspection, provisional source-annotation overlays and **No punches / Punch present / Not sure** form. Statistical activity hints are shown as **Unverified**. They do not assert guarding, footwork, defensive movement or punch absence. When punches are visible, choose Punch present and indicate whether they were marked; do not force a negative outcome.

The twelve MP4 previews are pre-generated and cached. No source footage was modified. Inspection contact sheets and preview checks are private artifacts in `reports/boxingvi/interval-review-motion-v2/`; they are not substitutes for watching the entire interval.

## Selection method

The previous gap-center method found human-reported unmarked punches in all five reviewed V4 windows. Those five decisions remain `needs_event_labeling` for future event work. None was restored as a negative or relabeled.

This pass uses only the existing continuous MediaPipe poses for **V4 and V6**. It scans four-second windows at one-second spacing using source PTS/timebases, measures valid consecutive-joint displacement divided by actual elapsed time, and creates statistical review priorities. No classifier, learned activity model, pose imputation, new pose extraction or training is involved.

Geometry is aspect-corrected, hip-centered and torso-scaled. Arm activity uses elbows/wrists relative to the torso; feet activity uses ankles relative to the torso; hip translation supplies an image-plane body-motion proxy. Units are projected torso lengths per second. Body rotation, camera projection, torso-normalization noise and confident anatomical errors can affect these statistics. They cannot establish punch identity or absence.

### Exclusions and reliability

Every proposed window and its context must:

- Avoid source-annotation bounds with a one-second guard. All plausible raw annotation bounds are used, including held/excluded rows. For a reversed interval, the min/max envelope is used **only to exclude search space**; no event is corrected or approved.
- Avoid **every original-pilot interval**, including context plus a two-second guard. This excludes the five reviewed positive windows and also avoids recycling the unreviewed original candidates.
- Stay in one known scene and estimator segment, with complete native indices and increasing PTS; timing gaps over 100 ms and known cuts are rejected.
- Have valid shoulders/hips on every context frame, at least 90% validity for each elbow, wrist and ankle, and no continuous arm-invalid run longer than 150 ms. Invalid coordinates never contribute motion values.
- Have sufficient valid arm-frame pairs and no observed arm displacement speed above 12 torso lengths/second, a conservative artifact/fast-activity screening gate, not a biomechanical punch threshold.

These are **discovery gates**, not changes to stored masks, the 0.5 source confidence contract, mask-aware70 training policy or historical candidate eligibility. Validity does not prove anatomical correctness. The new gates reject all five old reviewed windows in a read-only motion diagnostic; this does not establish detector accuracy or imply that every reliable pose sequence is punch-free.

### Diversity ranking

For each recording, retain two quieter-motion, two lower-body-motion, one moderate-arm-motion and one mixed-motion proposal. Exclude the upper 30% of eligible arm-p90 activity, select quiet windows by lower combined arm/hip/foot activity, rank lower-body windows by greater foot/hip activity relative to arm activity below the recording median, and sample median-range arm and mixed movement for the remaining slots.

Prefer recording thirds that have fewer selections, then activity score, with deterministic frame-index tie breaks. Require at least 30 seconds between selected starts. This avoids six adjacent near-duplicate windows without weakening pose gates to fill an exact temporal quota. Quantiles and ranking use these training recordings only; no V5 or evaluation footage informs the ranking.

The proxies aim to offer possible quiet/guarding, repositioning and more challenging arm/body movement. Actual defense, feints and non-punch semantics remain for human review. There is no promise every category exists or that these windows will yield negatives. The moderate/mixed proposals may contain punches, which must remain positive/unknown outcomes.

| Recording | Windows passing source/continuity/pose gates | Source/original-queue exclusions | Context conflicts | Pose/spike exclusions |
|---|---:|---:|---:|---:|
| V4 | 304 | 613 | 24 | 526 |
| V6 | 197 | 1,324 | 51 | 368 |

These are counts of overlapping search windows, not independent examples or counts of negative footage. All 12 final proposals have 100% anchor coverage; minimum per-arm-joint coverage across selected context windows is approximately 98.6%. This supports inspectability, not a label-accuracy claim.

## Prepared intervals

Frames are native zero-based half-open bounds. Time is source-video time, not a YouTube offset. The labels in the final column describe motion statistics only.

| ID | Source start | Native frames | Duration | Review priority |
|---|---|---|---:|---|
| V4-motion-1 | 23:00.004 | [33087,33183) | 4.004 s | Less overall movement |
| V4-motion-2 | 01:09.027 | [1655,1751) | 4.004 s | Less overall movement |
| V4-motion-3 | 12:22.033 | [17791,17887) | 4.004 s | More lower-body movement |
| V4-motion-4 | 20:39.029 | [29707,29803) | 4.004 s | More lower-body movement |
| V4-motion-5 | 00:18.018 | [432,528) | 4.004 s | Moderate arm movement |
| V4-motion-6 | 23:30.034 | [33807,33903) | 4.004 s | Mixed movement |
| V6-motion-1 | 30:57.022 | [44524,44620) | 4.004 s | Less overall movement |
| V6-motion-2 | 00:40.040 | [960,1055) | 3.962 s | Less overall movement |
| V6-motion-3 | 13:37.024 | [19589,19685) | 4.004 s | More lower-body movement |
| V6-motion-4 | 24:35.015 | [35365,35461) | 4.004 s | More lower-body movement |
| V6-motion-5 | 14:31.037 | [20884,20980) | 4.004 s | Moderate arm movement |
| V6-motion-6 | 01:35.011 | [2278,2374) | 4.004 s | Mixed movement |

Each has zero overlapping source annotations under the existing provisional mapping. This is a conflict check, **not** evidence of no punches. Three exact frames per proposal were inspected for media availability and presentation sanity; no absence decision was inferred from those samples.

## Versioning and preservation

A small allowlisted pilot registry reuses the existing interval workspace. Default API behavior stays on the original pilot. `?pilot=motion-v2` selects the new manifest and independent append-only ledger; the same scope is retained in cached video URLs. Unknown pilot names and cross-pilot IDs fail rather than choosing another workspace.

- Original manifest/config and `reports/boxingvi/interval-reviews-v1/` are unchanged, including all five manual punch-present/unmarked decisions and their revision chain.
- `configs/interval_review_motion_v2.json` pins `reports/boxingvi/interval-review-motion-v2/manifest.json`; its SHA256 is `fa6ed95a96b246fb4d24fb1a2195bb5a9201ea042ad5d152503fd48f8866e18d`.
- New explicit saves will create revisions under `reports/boxingvi/interval-reviews-motion-v2/`. That ledger is currently empty. Back it up together with its manifest; it is durable review data, not disposable preview cache.
- The original five reviews remain bound to head `4ab1e93f3cf7205d41cff7299672a3efd0b168a4602cac772aced31c5742759f`. No existing decision is migrated, overwritten or inferred.
- V5 remains development-only. V7/V8 and V10 remain reserved for the detector pilot's evaluation. The V1/V6 guard and null boxer/session identities are retained. V9 is never accessed.

The deterministic preparation command is:

```sh
.venv/bin/python -m boxing_motion_ai.data.motion_candidates
```

It refuses to replace the existing private manifest and refuses a conflicting existing config pin. On the same input/review snapshot it can recreate a missing manifest that matches its pin. Further candidate rounds require another explicit version; do not edit the old manifest to swap candidates. Source data, analysis, contact sheets, MP4s and all review files remain Git-ignored.

## Strict coverage: friction assessment and recommendation

The gate is **unchanged**: every native interval frame must be recorded as inspected, plus explicit whole-interval and visibility attestations, before No punches can save. Buffered playback does not credit skips, seeks, hidden playback or buffering. Exact inspection remains available to fill missing frames. Punch present and Not sure do not require completing a negative certificate.

There are no genuine completed No punches reviews in the first five records, so they cannot quantify human effort for negative certification. The earlier actual-browser study left one frame uncredited at 1× in short clips despite zero reported dropped frames; conservative presentation callbacks can therefore create extra clicks that do not necessarily indicate inattentive review. Reopening also resets viewing coverage, and 0.25× replay increases viewing time substantially.

Recommended next usability experiment: a **one-click sequence through only the missing original frames**, retaining exact decode, visibility checks, displayed-frame evidence, a suitable dwell/context and explicit final attestation. It could automate navigation without inventing inspected ranges. Also show where coverage is missing and explain that a positive finding can be saved immediately. Measure genuine review time and rejection/yield by priority after this 12-case batch before changing coverage policy.

Do not replace the full-frame gate with a timer, `ended`, a percentage threshold or automatically selected confirmations. Persisting/resuming viewing evidence across sessions would need explicit source/bounds/revision binding and a reviewed policy; it is not implemented here. This milestone recommends improvements only and does not weaken or bypass verification.

## Verification

Synthetic tests cover mask-gated motion, elapsed timestamps, missing arms/feet, gaps and spikes, deterministic strata/separation, independent pilot heads/history, restart persistence, idempotent saves, cross-pilot/unsafe IDs and evaluation/development rejection. Frontend tests cover pilot switching, isolated save URLs/heads, scoped playback metadata and existing review/coverage behavior.

The full Python suite passes **469 tests** (one existing Starlette/httpx deprecation warning); the frontend passes **63 tests**. Ruff, TypeScript, Python/frontend formatting and production build pass. All twelve previews pass exact source/MP4 PTS round-trip verification. Headless Chrome verified switching between 12 new and 36 original entries, real V4/V6 buffered playback and exact next-frame inspection. The original five-review head remained identical; the new ledger remained empty. All **4,654 protected input/review hashes** matched after verification. No review POST is sent to the actual workspace, and no detector is trained.
