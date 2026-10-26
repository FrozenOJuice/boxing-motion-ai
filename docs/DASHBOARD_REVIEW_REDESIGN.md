# Review experience redesign

The Review screen now follows **Select punch → Watch → Decide → Save & Next**.
The versioned v0.2 curation API, original data, source groups, exclusions and V9
protections are unchanged. Dataset, Experiments and Models retain their existing
views; the shared navigation and page heading have been simplified.

## Try it locally

The existing local backend serves the latest frontend build. Refresh
http://127.0.0.1:8000. To launch from a stopped session, use the commands in
[DASHBOARD_V02.md](DASHBOARD_V02.md#launch). No new dependencies are required.

1. Open **Review**. Use the queue filters for **Disputed punches** (six) or
   **Proposed anchors** (twelve), or choose a recording/search for a punch.
2. For reproducible checks, search **V4:497**, a dispute, or **V4:144**, an anchor.
   The original punch label is above the large viewer. Search **V6:613** and **V6:7**
   for another dispute/anchor pair. Queue punch numbers are positions in the filtered
   list; source-row identifiers are in Advanced details.
3. Play/pause, step a frame, or scrub the timeline. Start/end buttons mark the
   annotated interval; the highlighted range includes surrounding footage.
   Toggle **Skeleton** if useful. **Playback options** contains direct frame entry
   and context length. Space toggles playback; arrows step when focus is outside
   form controls. Playback waits for every decoded image, pauses on manual seeking,
   and stops at the window end or a retrieval error. It is not realtime playback.
4. Choose **Yes**, **Change it**, or **Not sure**. No decision is preselected.
   Corrections appear only after Change it, including frame inputs and “Use frame
   as start/end” buttons. Yes explicitly resets any draft correction to the original
   mapped label and boundaries, even when an older review proposed a correction.
5. Enter your name and evidence. Confirmation also requires lead/rear evidence and
   viewing both boundary frames. Explicitly attest your review, then **Save & Next**.
   Save without advancing is available once a decision is selected. Editing a field
   clears attestation. Save failures retain the draft; navigation warns before
   discarding unsaved work.
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
| Needs attention | Dispute, unresolved hold/ambiguous/mixed boundary, automated exclusion, or unavailable review metadata |

Human holds remain Needs attention. The exact disposition is available in Advanced
details, including whether a Reviewed punch was accepted or excluded. A model or
assistant observation cannot produce a Reviewed category by itself.

| Primary choice | Explicit outcome | Existing API action |
| --- | --- | --- |
| Yes | Confirm original label and mapped boundaries | `confirm` |
| Change it | Confirm the explicitly proposed label/boundaries | `confirm` |
| Not sure | Can't tell what the punch is | `ambiguous` |
| Not sure | Leave it for a later review | `hold` |
| Not sure | Don't use this punch | `exclude` |

Not sure requires selecting one of its three outcomes; it never silently chooses
an internal state. Ambiguous and Hold still require unresolved uncertainty. Every
save retains reviewer notes, server timestamp, source bindings, exact frame evidence,
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
