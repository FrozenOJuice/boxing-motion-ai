# Phase 1 — workflow integrity and workspace structure

Implemented only Phase 1 of [the approved plan](../docs/UI_UX_REDESIGN_PLAN.md).
Production code and the plan itself are unchanged. Phases 2–4 are not implemented.

## What works

- Videos routes to `/videos`; opening a video routes to `/videos/:videoId`.
  Browser Back/Forward works. Returning to Videos restores library scroll and tile
  focus; entering Analyze focuses the video heading.
- Each video keeps its own playhead, explicit boundary draft, selected result and
  inspector tab. App memory preserves navigation, and prototype-only sessionStorage
  restores these on refresh. Local video blobs still resolve through the existing
  prototype IndexedDB store. Missing video URLs show a recovery screen.
- Set start and Set end never invent the other boundary. Incomplete, reversed,
  one-frame and overlong selections cannot be classified. Dragging a range explicitly
  establishes both boundaries. The indexed sample uses actual source timestamps,
  at least two frames, and the existing Python gate of at most three elapsed seconds
  between first and last included frames. No pose-quality assessment is simulated.
- A source-frame seek disables boundary marking until its indexed image is loaded.
  Handle clamping seeks to the boundary actually applied. Bounds changes clear the
  current prediction while preserving saved records; seeking alone retains it.
- Natural playback completion retains the last source still. Play then restarts at
  zero. Replay selection always uses its own bounds and stops on the final included
  frame before the exclusive end boundary. Late playback promises cannot undo Pause.
- The right inspector has keyboard-accessible Selection and Saved punches tabs.
  Range and classification action remain outside the scrollable tab body. Saved rows
  have readable time/source labels and no separate 74–88px scroll window. Opening a
  saved row selects its range, seeks to its start and returns to Selection.
- Repeated classification of the same range reopens the existing result rather than
  appending duplicates. Existing recorded sample outputs remain recorded outputs;
  new example selections remain examples through navigation and refresh. Existing
  legacy selections are retained, including those that no longer pass the range gate.
  Such invalid ranges cannot be rerun until corrected.
- The measured timeline/control stack replaces the fixed 118px sizing reservation.
  The viewer keeps its native aspect ratio; timeline, transport and primary action
  remain visible at the three requested desktop sizes. The dark theme is retained.

## What is simulated

New classification is a labeled 450ms example transition using the existing sample
scores, not a model call. A known range can reopen an existing saved output. The two
recorded sample outputs come from the preserved reference artifacts. Neither case
claims a new model run or verified punch occurrence. Model scores remain uncalibrated.
No backend API, pose extraction, training, research review or dataset write runs here.

## Browser verification

Native Safari was used for actual interactions and screenshots. `review.html` supplies
real iframe viewports of 1440×900, 1280×800 and 1024×768 CSS pixels, scaled only for
presentation. Direct workspace URLs were used separately to test native browser history
and refresh. The existing prototype dev server was reused and left running.

Before changing code, both reported bugs were reproduced:

1. Set start alone created a one-frame `0.000–0.041` range and enabled Classify.
2. Playback ended at `5.963 / 6.004`; pressing Play stayed at the final frame.

After the fix, Safari verified:

- Start-only and end-only drafts show the missing boundary and a disabled Classify.
- Setting both boundaries on the same frame still shows the two-frame validation error.
  Stepping and explicitly setting the end then permits a valid two-frame range.
- Natural completion stops at the last indexed still; pressing Play starts from zero
  and progresses again (observed at 0.250 seconds).
- The saved 1.084–1.459 selection replays to 1.417, before its exclusive boundary.
- Back to Videos, native browser Back/Forward, reopen and refresh preserve state.
  Sample and local-video drafts/results remain isolated.
- Inspector arrow keys switch tabs and move focus. Timeline dragging creates a new
  candidate; handle dragging clamps correctly and clears stale output.
- New example classification increments Saved once; reclassifying keeps the same count.
- At 1024×768, more than five saved rows fit with the current range and Classify visible.
  There is one scrollable inspector body, rather than two stacked small lists.

| Viewport | Evidence stored locally (ignored)                                                                                                                 |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1440×900 | `artifacts/phase1/screenshots/videos-1440x900.png`, `selection-1440x900.png`, `saved-1440x900.png`                                                |
| 1280×800 | `artifacts/phase1/screenshots/example-1280x800.png`, `saved-1280x800.png`                                                                         |
| 1024×768 | `artifacts/phase1/screenshots/videos-1024x768.png`, `saved-1024x768.png`, `range-1024x768.png`, `example-1024x768.png`, `incomplete-1024x768.png` |

Captures are real UI pixels; only empty native-capture padding is trimmed. Raw copies
remain locally. No screenshots, media or generated verification artifacts are committed.

Automated verification: **31 frontend tests**, TypeScript, formatting and production
build pass. Tests cover explicit/incomplete drafts, frame-count and elapsed-time rules,
playback completion/replay, late callbacks/promises, history/focus/scroll restoration,
refresh and per-video isolation, missing URLs, saved-result provenance, deduplication,
cancelled classification, tab keyboard controls and absence of production API calls.

## Review steps

1. Run `npm --prefix prototype run dev`; open `http://127.0.0.1:5186/videos`.
2. Open Shadowboxing sample. Clear an existing draft if restored. Press Set start:
   the end must remain “Not set” and Classify must stay disabled.
3. Set the end on that same frame: the two-frame error must remain. Step forward and
   explicitly set the end again, or drag a larger range; Classify becomes available.
4. Classify and inspect the example label. Switch to Saved punches, reopen a recorded
   output, and compare its distinct source label. Repeating a range must not add a row.
5. Play to the end, then press Play again. Replay a selected range separately.
6. Use Back to Videos, reopen, native Back/Forward and refresh. Check draft, time,
   result and tab restoration. Open a different local video to confirm isolation.
7. Open `http://127.0.0.1:5186/review.html` and review all three target viewport buttons.

## Remaining limits (later phases)

- Exact source stepping/frame-count validation is restricted to the indexed sample.
  Arbitrary uploads remain browser-time based; this prototype has no local pose or
  inference service. Their three-second gate is time based, not a claimed source-frame
  validation or punch-quality check.
- Session restoration across refresh depends on browser session storage; app memory
  still preserves navigation if storage is blocked. Local videos depend on the existing
  browser storage and are not recoverable in another browser or after site-data cleanup.
- Existing upload preparation/error/cleanup limitations, Safari black posters and the
  demo's library presentation remain Phase 2 work. No new upload lifecycle is added.
- Global typography, broader target sizing, 200% zoom/reflow, VoiceOver, portrait-media
  and cross-browser validation remain Phase 3 work. Model details may scroll; the range
  and primary action remain visible.
- Production integration, Python/ML validation and Development migration remain Phase 4
  and need separate authorization. No source-group or V9 protection is changed.
