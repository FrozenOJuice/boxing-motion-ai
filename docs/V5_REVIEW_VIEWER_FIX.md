# V5 interval review viewer repair

Starting revision: `2af74f9`. Frontend viewer/mapping correction only; no queue recreation,
annotation changes, dataset export, model training or layout change.

## Preservation before edits

The actual append-only localization review ledgers were replayed before editing code. All
five V6 intervals were complete; V5-motion-101 was incomplete with **zero saved events and
zero review revisions**. The five saved V6 inventories contain 5, 4, 0, 8 and 1 entries
respectively. This is their saved human state, not a new interpretation of those labels.

All 29 existing ledger/lock files were copied and SHA256-inventoried under the private,
Git-ignored `reports/boxingvi/v5-viewer-fix-backup-20261030T045946Z/`. Heads, completion flags
and event payloads were also recorded separately. No human completion, edit, removal or
save request was made by browser verification. The after-check compares both bytes and
replayed state and rejects additional review files. Browser-only viewing coverage and
unsaved drafts are not persisted event revisions; the user's original browser was not
reloaded or controlled by the test scripts.

## Observed failure and causes

Before changes, isolated local Chrome played the **entire real V5 preview (14.047 seconds)**
and then exercised backwards seeking and alternating arrows. Initially those worked, but
later the requested frame became 8099 while the visible image, frame counter and slider
remained at 8100. Subsequent arrows repeatedly derived their destination from that retained
8100 frame, even when another request was pending. The private reproduction records requested
indices, visible image identities, media clock and coverage separately.

The concrete frontend defects were:

- **Requested and displayed indices were conflated in navigation.** Image loading is
  asynchronous, but slider value, arrow destinations and disabled states followed the last
  decoded pixels instead of the latest explicit request. A pending request could therefore
  make dragging snap backwards and repeated steps repeat the same destination.
- **Pending/error UI could be hidden.** The entire ExactFrame overlay had zero opacity until
  exact readiness; that also hid its loading/error/retry interface. There was no deadline for
  a request or image decode that stayed pending. Retaining old pixels is useful, but it must
  not silently conceal an outstanding request.
- **A retained cached image did not necessarily emit another load event.** Returning to that
  frame while a different target was pending needed to re-publish its verified identity;
  otherwise parent readiness could remain stale. Late decodes also needed invalidation on
  timeout, not only on a changed target.
- **Playback events could conflict with explicit navigation.** Pause handling always saved
  the media clock as a resume point, including an explicit manual seek. The `ended` flag also
  overrode a newly chosen earlier frame and restarted playback at the preview beginning.
  Delayed presentation callbacks wrote `shown` even when playback was no longer active.

The baseline also observed pending frame requests; their transport delay is not evidence of
bad source PTS or corrupt pose data. Direct reads of the affected V5 frames succeeded and
returned their correct native identity/PTS. This change fixes the demonstrated state-handling
and recovery defects; it does not claim a measured root cause for every intermittent backend
or browser transport delay. No cache was deleted to obtain a successful result.

## Changes

`BufferedIntervalViewer` uses an explicit navigation target while exact inspection is pending.
The counter and image continue to describe confirmed pixels; the slider's accessible text
states both target and displayed frame during loading. Repeated steps advance the requested
native index by one. Boundary marking requires matching requested, displayed and decoded
identities. Exact loading/error UI remains visible, and a failed frame can be retried or
replaced by another target without resetting video or review state.

`ExactFrame` bounds the whole outstanding target load to 15 seconds, retains old pixels with
an explicit message, invalidates late decode generations on failure, and re-publishes a
retained image when matching current metadata arrives. Requests, HTTP failures, image failures
and decode stalls cannot silently enable inspection or credit unseen frames.

Only active, nonseeking playback callbacks update the displayed playback position. Manual
seek intent is established before pausing; its later pause event cannot overwrite the target
with an old resume clock. End playback selects the real final source frame. A manual earlier
selection after `ended` plays from that selection; ordinary native pause/resume still retains
its fractional media clock. No frame/time resampling or coverage policy was changed.

## V5 suggestions: separate schema mismatch

All eight returned suggestions use legacy **`v5-candidate-1.0`**, which stores the prepared
native range in `array_slice_half_open`. Backend interval selection already understood that
field, but `annotationSegments` read only `source_interval_half_open`. Thus all eight rows
were returned by the API but disabled by the frontend.

The renderer now accepts the existing slice **only for V5 with that specific legacy schema**
and only when the newer mapping field is absent. Explicitly null/invalid newer mappings,
unknown schemas, reversed ranges and missing mappings stay unavailable. No raw spreadsheet
number, helper formula or guessed offset is used; source objects are not mutated.

Authority is the original preparation contract in `docs/DATASET_PREPARATION.md`,
`configs/v5_recording.yaml` (`frame_offset: 0`, provisional inclusive ending), and the
hash-verified prepared rows/contiguous native frame array. All eight rows 386–393 were checked:
the prepared slices equal `[candidate_start_index, candidate_end_index + 1)`. Their exact
native PTS/timebases were checked independently. For example, row 386 is [8140,8146),
PTS [271605,271805) at timebase 1/1000. **No −24 helper subtraction was applied.**

This establishes how to display the already-prepared provisional references, not that the
author's annotation convention or those punch boundaries/classes have been independently
verified. The eight suggestions remain Original dataset · Provisional and require human review.

## Verification and resuming

Local Chrome tests use the real V5 interval and real source-frame API, in isolated temporary
browser profiles. They cover:

- Full playback through the preview end, followed by seeks/steps at source frames 8070,
  8100, 8138, 8300, 8459 and 8490 (context and core boundaries included).
- Previous/next agreement between decoded image, frame counter and timeline; mouse timeline
  clicks/drags; backwards seeking while playing; resuming a chosen middle frame after `ended`.
- Rapid steps while an earlier real frame request is deliberately delayed; a failed transport
  followed by visible retry; an indefinitely held request followed by timeout and navigation
  to another real frame. Fault injection changes transport only, never pixels or model results.
- A separate nearly complete playback at **99%**, then a backwards seek to **8138** and repeated
  steps 8137→8136→8135→8136→8137→8138. Coverage remains 99%, incomplete, throughout those steps.
- Seeking before playback does not credit the intervening footage. Boundary controls become
  available only after confirmed inspection. All eight provisional suggestions are enabled.

No browser exception or review-save POST occurred. Private reproduction logs, final browser
records and screenshots are under `reports/boxingvi/v5-viewer-*`; no visual artifact is committed.
Focused tests cover delayed/stale metadata and decodes, timeout/retry, retained-image returns,
end replay, pause/callback races, uncertainty/coverage and the legacy mapping's fail-closed rules.

The five V6 completions and V5's saved state are checked byte-for-byte and by ledger replay
after verification. Models, source annotations, dataset manifests and Python services were not
changed. Resume **Development → Review → Video intervals → Individual punches · V6 / V5 →
V5 interval**. If an old page still has the old controls, reload after retaining any unsaved
draft notes. A reload requires rewatching the short V5 core to re-establish browser-only viewing
coverage; it does not erase saved human reviews. Do not interpret a displayed 99% as complete
or fabricate the final missing viewing evidence.

Final checks: **111 frontend tests**, **78 relevant Python tests** (one existing Starlette/httpx
deprecation warning), TypeScript, Ruff, Prettier, production build and diff whitespace checks
pass. Final Chrome verification covers 30 recorded states, including the exact 99%/8138 case.
All 29 backed-up review files and all six replayed states remain identical; all five V6
completions remain complete, and V5 still has no saved event or completion. A development-server
hot reload can reset browser-only viewing coverage or an unsaved draft; persisted review work
is unaffected. No cache, source file, review revision or historical artifact was deleted.
