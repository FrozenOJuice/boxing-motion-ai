# Review experience redesign

The Review screen now follows **Select punch → Watch → Decide → Save & Next**.
The versioned curation API now supports [three independent confirmations](INDEPENDENT_ANNOTATION_VERIFICATION.md).
Original data, source groups, exclusions and V9 protections are unchanged. Dataset, Experiments and Models retain their existing
views; the shared navigation and page heading have been simplified.

## Try it locally

The existing local backend serves the latest frontend build. Refresh
http://127.0.0.1:8000. To launch from a stopped session, use the commands in
[DASHBOARD_V02.md](DASHBOARD_V02.md#launch). No new dependencies are required.

1. Open **Review**. Use the queue filters for **Disputed punches** (six) or
   **Proposed anchors** (twelve), or choose a recording/search for a punch.
2. For reproducible checks, search **V4:497**, a dispute, or **V4:144**, an anchor.
   The original punch label is beside the playback controls. Search **V6:613** and **V6:7**
   for another dispute/anchor pair. Queue punch numbers are positions in the filtered
   list; source-row identifiers are in Advanced details.
3. Play/pause, step a frame, or scrub the timeline. Hover/focus the timeline markers
   for start/end frame labels; clicking a marker jumps there. The highlighted range
   marks the punch within surrounding footage.
   Toggle **Skeleton** if useful. **Playback options** contains direct frame entry
   and context length. Space toggles playback; arrows step when focus is outside
   form controls. Playback waits for every decoded image, pauses on manual seeking,
   and stops at the window end or a retrieval error. It is not realtime playback.
4. Choose **Yes**, **Change it**, or **Not sure**. No decision is preselected.
   Corrections appear only after Change it, including frame inputs and “Use frame
   as start/end” buttons. Yes explicitly resets any draft correction to the original
   mapped label and boundaries, even when an older review proposed a correction.
5. Select **Punch type is correct**,
   **Lead/rear hand is correct** and/or **Start/end timing is correct** independently.
   Save any verified subset with **Save & Next**. Timing requires viewing both exact
   boundary frames; a new type/hand check requires viewing the punch. Three checks
   constitute full verification; fewer remain partially verified and held. Notes are
   optional for unchanged annotations. Corrections retain their brief explanation.
   Prior scoped human checks may be restored with their original reviewer attribution;
   editing the label or interval clears them. **Not sure** keeps its selectable reasons.
6. Open **Advanced details** for predictions, automated observations, annotation
   provenance, exact timing, landmark confidence, source quality and full review
   history. Nothing in these sections is automatically treated as a human decision.

Inspecting the UI alone does not save a review. Only attest and save a real case if
you have personally reviewed it. Revised decisions append to history and never
rewrite source annotations. Corrections still require separate preparation before
training; this redesign does not perform that step.

## Display categories and exact decision mapping

Queue categories are presentation-only, not new backend dispositions:

| Category | Meaning |
| --- | --- |
| To review | No definitive human decision and no current attention flag |
| Reviewed | Latest human decision accepts or excludes the punch; this does **not** mean every reviewed punch is accepted |
| Partially verified | A human has verified some aspects, while full acceptance remains held |
| Needs attention | Dispute, unresolved hold/ambiguous/mixed boundary, automated exclusion, or unavailable review metadata |

Human holds remain Needs attention. The exact disposition is available in Advanced
details, including whether a Reviewed punch was accepted or excluded. A model or
assistant observation cannot produce a Reviewed category by itself.

| Primary choice | Explicit outcome | Existing API action |
| --- | --- | --- |
| Yes | Verify selected aspects of the original annotation | `verify` |
| Change it | Verify selected aspects of the proposed label/boundaries | `verify` |
| Not sure | Punch/arm/timing unclear, or multiple punches | `ambiguous` |
| Not sure | Hard to see, needs another review, or Other | `hold` |
| Not sure + More options | Explicitly exclude from future candidates | `exclude` |

Not sure requires a reason; its documented mapping records an unresolved state.
Exclusion is a separate explicit option, never selected automatically. Every
save retains its manual/automated source, explicit statements/optional notes, server timestamp, source bindings, exact frame evidence,
attestation, validation and revision history through the unchanged backend.

## Verification

- Full Python suite: **369 passed**, no skips; Ruff passed. One existing upstream
  Starlette/httpx TestClient deprecation warning remains.
- Frontend: **17 tests passed**; TypeScript, formatting and production build passed.
  Tests cover disclosure, explicit decision mapping, original/corrected label handling,
  boundary checks, attestation reset, stale-write draft retention, Save & Next,
  display-category semantics and exact-frame playback waiting/pause/end behavior.
- Real metadata: all six disputes and twelve anchors returned available case and edit
  state. Exact start/end frames were retrieved for V4:497, V4:144, V6:613 and V6:7.
  The review-ledger hash was unchanged; no real review writes were performed.
- Browser visual inspection was blocked by computer-use restrictions on the current
  URL. No browser window was opened, and browser inspection was stopped. Responsive
  layout and appearance still need human visual acceptance in the existing local tab;
  API/DOM/build checks do not replace that visual check.

The backend and prepared data are unchanged. Generated previews and production build
outputs remain ignored; only frontend source, tests and documentation are committed.

## Stable playback and compact layout refinement

The viewer now retains its displayed frame while a hidden replacement image loads
and completes `HTMLImageElement.decode()`. The same decoded image element is promoted
into view; pixels, skeleton, displayed frame number, timeline position and exact timing
metadata update together before paint. Requested navigation remains separate from the
displayed frame. Aborted metadata requests and superseded image-decode generations
cannot replace a newer selection. Boundary edits remain disabled while the requested
frame is not displayed.

The stage has a fixed 16:9 shape sized from the available viewport, independent of
loading, error messages or source-frame dimensions. Footage and overlays share a
letterboxed plane preserving the source aspect ratio. Error/loading notices overlay
the stage rather than replacing it. A failed request pauses playback and states which
previous frame remains visible; Retry fetches the requested frame again. No approximate
frame is substituted. Before the first successful frame, the same reserved stage stays
empty with an actionable error.

Only the displayed image and its pending replacement are retained by the component.
The existing server preview cache and browser image handling remain in use. Speculative
adjacent-frame requests were deliberately omitted: they can trigger expensive exact
video decoding and contend with active navigation. No dependency or unbounded image
cache was added.

Redundant section page titles are removed. Reduced vertical spacing and a
viewport-dependent footage-height cap make room for the viewer, controls and three
decisions on common laptop/desktop sizes. Correction forms and advanced details can
still extend below the fold. Small screens keep the responsive stacked layout and
may require scrolling.

Verification for this refinement: 22 frontend tests cover delayed image decoding,
out-of-order metadata/decode completion, retained frame/overlay/timestamp correspondence,
load and decode errors, retries, playback stopping, initial loading, and resize events
at 1440×900, 1366×768, 1280×720 and 390×844. Resize checks verify retained DOM/state;
jsdom does not perform physical layout or paint. TypeScript, formatting, production
build, Ruff and the 369-test Python suite pass (the existing TestClient deprecation
warning remains). Browser visual inspection remains unverified because computer-use
inspection of the current browser URL was restricted; no browser was opened.

To visually verify, refresh the local dashboard, select a punch, rapidly step in
both directions and scrub while frames load. The old footage should remain visible
until replaced; its label and skeleton must agree. Resize the window through the
sizes above and check that the primary decision buttons fit on desktop. Under slow
network conditions, verify loading does not change the stage size. If a frame fails,
check the explicit previous-frame message and Retry. These checks do not require
saving any annotation review.


## Manual reviews without names

The dashboard has no reviewer-name field, profile, account or authentication step.
It no longer reads or writes the old browser name preference. Select the independently
verified aspects and explicitly Save or Save & Next. The server records `source: manual`
and a timezone-aware timestamp; the frontend cannot choose or spoof this source field.
No name is generated or substituted. Existing named reviews and receipts remain intact,
and their legacy names are still visible in Advanced details/history.

Automated observations are displayed separately as automated and cannot acquire
verification merely by loading the dashboard. Old records without a source field use
their original human/assistant actor for display classification, without rewriting the
record or changing its canonical hash. New automated records use `source: automated`;
they cannot carry manual confirmations. The retained internal `actor_type` is an actor
category for compatibility, not an account or personal identifier.

[Independent verification](INDEPENDENT_ANNOTATION_VERIFICATION.md) is unchanged: partial
checks remain held, full six-class labels require both type and lead/rear hand checks,
and timing requires exact boundary evidence. New manual checks carry their originating
revision IDs. Previously named origins keep their names in historical records; unnamed
origins are simply manual reviews. No assertion is copied from automated observations.

Notes are optional for unchanged annotations. Meaningful corrections retain their brief
reason, and uncertainty offers selectable reasons with an optional note. Existing source
fingerprints, revision links, stale-write detection, original exclusions and V9 protections
remain in force. No real review is saved during automated verification.

Current checks: **427 Python tests**, **37 frontend tests**, Ruff, TypeScript, formatting
and production build pass. New tests cover anonymous saves, server-assigned manual source,
automated-source rejection, timestamp/verification preservation, historical named receipts
and hashes remaining unchanged, name-free UI and ignoring old browser name preferences.
One existing upstream Starlette/httpx TestClient deprecation warning remains.
