# First five interval-review findings

2026-10-27, America/Edmonton. Read-only assessment of the saved human decisions; no footage was automatically classified, no replacement candidates were prepared, and no training occurred.

## Verified saved decisions

The interval ledger contains five published revisions covering five different intervals, V4-bg-1 through V4-bg-5. All are `source: manual`, decision `punch_present`, and `unannotated_punches: yes`. Their current status is `needs_event_labeling`; `training_enabled` and `exhaustive_event_reference` remain false. There are no No punches or Not sure decisions in this first batch, no revisions superseding these decisions, and no newly certified background examples.

The existing workspace replay validates the immutable manifest binding, consecutive version numbers, parent-hash chain and decision semantics. Each detail query also validates the registered preparation/pose references and interval timing. The current source lookup returns **zero overlapping annotations for each of the five windows**, consistent with their selection from source-annotation gaps.

Source-video timestamps below are not YouTube offsets. Native intervals are zero-based and half-open, following the existing provisional mapping. The records were saved on October 27 at 21:56–21:57 MDT (their serialized UTC timestamps fall on October 28).

| Interval | Source-video time | Native frame interval | Recorded frame coverage | Human decision / unmarked punches |
|---|---|---|---:|---|
| V4-bg-1 | 11:54.088–11:58.092 | [17121,17217) | 95/96 | Punch present / Yes |
| V4-bg-2 | 15:54.412–15:58.374 | [22883,22978) | 78/95 | Punch present / Yes |
| V4-bg-3 | 07:55.100–07:59.104 | [11391,11487) | 71/96 | Punch present / Yes |
| V4-bg-4 | 14:38.086–14:42.090 | [21053,21149) | 55/96 | Punch present / Yes |
| V4-bg-5 | 13:40.278–13:44.282 | [19667,19763) | 95/96 | Punch present / Yes |

The five candidate windows total **19.978 seconds**. Their notes are empty and no movement tags were saved. Thus the records establish that the reviewer observed punches and reported unmarked punches in every window. They do **not** establish the number/type of punches, their onset/end frames, stance, or whether every other movement was reviewed. Those details must not be invented from the interval-level answers.

`watched_entire` and `visible_continuous` are false in these punch-present records. This is valid: identifying a punch does not require certifying the entire interval as punch-free. Complete frame coverage plus explicit whole-interval/visibility attestations remain mandatory only for a No punches certification. Partial coverage does not invalidate these positive observations and must not be upgraded into exhaustive reference labels.

The immediate training implication is clear: **all five stay excluded from background**. They may be retained as future event-labeling leads, but cannot automatically become new labeled training punches. No original workbook, source event, historical exclusion or human decision is changed. The five reviews do not need repeating to establish their unsuitability as negative examples.

## Does candidate selection need improvement?

**Yes, before using it as the main way to obtain background.** This first batch yielded 0/5 verified negatives and 5/5 reports of unmarked punches. All five are from V4, selected purposively rather than randomly; this is not an estimated dataset-wide missing-label rate and says nothing definitive about the unreviewed 25 candidates.

The preparation code in `data/interval_pilot.py` ranks internal unknown gaps by duration, takes a centered four-second window and checks continuity, separation and timing. It does not inspect visual activity or evidence of a pause. The containing V4 gaps were approximately 64.273, 63.063, 61.770, 21.480 and 18.978 seconds long. None of those gap lengths predicted a usable negative in this batch.

A long gap measures sparse source coverage. It cannot distinguish resting/guarding from unannotated punch demonstrations, combinations or other actions. These reviews support treating the source labels as insufficient for exhaustive continuous supervision. They do not establish that every missing mark is a publisher error or that the existing labeled punches are wrong; the source's intended coverage and provisional alignment remain separate questions. The selection safeguard worked—every candidate stayed unknown until review—but its ranking produced poor negative-review efficiency here.

## Recommended next step

Pause further **blind gap-center review** as the primary negative collection strategy. Preserve the existing queue and its decisions; do not delete or relabel the remaining candidates. Instead, approve a small **human-led interval-selection pass** before generating another queue:

1. Briefly survey footage in the prospective detector-training recordings V4 and V6, and identify a few visibly continuous stretches of pause, guard adjustment or ordinary movement. Start with at most three proposed 4–8-second windows per recording. This is a review budget, not a claim those stretches exist or a request to re-review the failed five.
2. Record the human-suggested native start/stop frames with context and source provenance. Check known cuts, extraction gaps, pose visibility and existing source marks. Overlapping source annotations are a conflict to retain for review, not something to erase. Selecting a window still leaves it unknown.
3. Apply the existing buffered playback and strict No punches workflow to the entire proposed window. A human spotting a quiet stretch during a skim is only a proposal; all-frame coverage and explicit absence/visibility attestations are still required to certify it. Preserve Punch present and Not sure outcomes without forcing a negative.
4. Measure **verified negative seconds per reviewed candidate/minute**, rejection reasons and movement tags. Expand only if this small pass improves yield. Then deliberately add verified moving backgrounds—guard changes, footwork or defensive movement—and feints only where actually distinguishable, so an eventual detector cannot succeed merely by separating movement from stillness.

A minimal future dashboard “Suggest interval”/start-end selector would let the reviewer nominate these windows without manual JSON editing. It should create a separately versioned candidate set, preserve the original manifest and keep all prior reviews bound to their original hashes. The current ledger is pinned to one immutable manifest, so simply replacing or appending entries in that file would invalidate review history. This capability is a proposed next implementation step, not implemented in this assessment. No large annotation platform is needed.

If automated proposal ranking is later approved, use existing reliable-pose motion summaries only as a *review-ordering aid*, with both lower- and moderate-motion strata and some non-ranked controls. Missing/occluded landmarks must never look like zero motion. Pose confidence and low speed cannot certify no punch, and a poorly tracked punching arm can appear quiet. Keep humans as the negative-label authority. Do not use the current six-class positive-clip classifier as a no-punch detector.

V5 remains development-only. V7/V8 remain together and, with V10, reserved for this detector pilot's evaluation; keep those observations out of training-source selection/tuning. Preserve the V1/V6 guard and unknown boxer/session identities. V9 remains untouched. The existing 1,123-example curated positive pool and all prior classifier experiments remain historical/provisional assets; no detector dataset is released by this assessment.

## Audit trail and checks

Ledger head: `4ab1e93f3cf7205d41cff7299672a3efd0b168a4602cac772aced31c5742759f`.
Pinned pilot manifest: `0a3d424bfaccc67c157bf4d42abfdd7bfba283cea6b0a54ca222dce628b3ff8d`.
The first five record hashes and a compact factual summary are retained in ignored `reports/boxingvi/first-five-interval-review-20261027/`. Original records are read-only and rehashed after the checks. No dashboard POST is issued.

Verification: **20 relevant tests passed** (`pytest tests/test_interval_review.py tests/test_review_clips.py -q`); **Ruff passed** (`ruff check .`), and `git diff --check` passed. All five human-record hashes and all 4,647 previously protected file hashes matched. One existing Starlette/httpx deprecation warning remains. Only documentation is changed; no new selection logic, candidates, labels or model behavior is introduced.
