# Label missing punches in the five V4 intervals

2026-10-28. This bounded workflow adds separate human event annotations to V4-bg-1 through V4-bg-5, the five original intervals already reviewed as Punch present. It does not alter their original reviews, BoxingVI workbooks, existing punch curation, or the 12 verified Motion Pilot 2 backgrounds. No detector training or candidate discovery is included.

## Use the dashboard

The usual launch is unchanged:

```sh
npm --prefix web run build
.venv/bin/python -m boxing_motion_ai.api
```

Open http://127.0.0.1:8000 → **Review → Video intervals**. In **Review pilot**, choose **Label missing punches · V4**. Only the five approved intervals are listed.

1. Watch the interval and its context with normal buffered playback. Slow motion, pause, previous/next frame, timeline navigation and paused skeleton overlays remain available.
2. Click **Add punch**. No type or boundaries are preselected. Pause and use exact-frame stepping to find the first included punch frame, then **Set start here**. Find the last included punch frame and **Set end here**. The controls are enabled only for a loaded exact still, not an approximate moving-video position.
3. Choose Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut or Rear Uppercut. If identity or lead/rear semantics are unclear, choose **Type uncertain**. This stores `unknown`, without guessing a six-class label or stance.
4. Click **Save punch**. It stays on this interval, preserves viewing coverage and lets you add the next punch. Separate events may be consecutive or overlap; neither case is automatically merged or corrected.
5. Added punches appear as clickable human-label segments above the original source-annotation timeline. Each overlap has its own clickable lane. Click a human segment to inspect its exact start and edit it, or use its **Edit punch** button. **Remove punch** removes it from the current event list; all earlier revisions remain in **Labeling history & details**.
6. After watching the whole reviewed interval and saving every punch, explicitly check **I watched the entire interval and labeled every punch.** Then **Save complete & Next** saves the completion and advances only on success. On the final interval it stays there with saved completion feedback. Reopening shows the saved status; any later event add/change/removal resets it to In progress and requires a fresh completion save.

Completion is never inferred from playback ending, selecting a class, or saving one event. The whole-interval checkbox starts unchecked on each visit. Seeking leaves unseen ranges uncovered, interval switching resets viewing evidence, and hidden/offscreen playback does not earn coverage. Existing versioned continuous-viewing validation is reused; individual still inspections remain separate. Uncertain event types remain unknown even if the action inventory is complete. An interval with an existing positive review cannot be certified complete with zero events; leave it in progress and reconsider its original review if needed rather than inventing a punch.

Use surrounding footage to identify the first visible committed departure into the punch trajectory, excluding preparatory guard adjustments. End at the last included frame before the punch's active excursion/return finishes or becomes a distinct next action, following the [detection plan](CONTINUOUS_DETECTION_PLAN.md). A new action need not wait for a stationary guard. Do not force a boundary you cannot establish: leave the event draft or interval incomplete. This milestone supports unknown types, not fabricated precision for ambiguous timing.

An event may extend into available context if it genuinely overlaps the reviewed interval. Such an event is marked **Partial overlap**; its full original-source bounds are preserved. Context alone is not certified by interval completion, and an event entirely outside the reviewed interval is rejected. Missing context or uncertain edge timing remains unresolved; do not clip the punch and call that a verified onset/end.

## Engineering and preservation

`services/interval_events.py` reuses the existing strict model, `PunchType`, native timeline/continuity validation, source-bound coverage validator, exact decoder, safe metadata store and ledger locking. The existing workbook `Event` contract requires workbook-row provenance, so it is deliberately not populated with fake source rows for newly observed events. The new `HumanPunch` contract instead stores a stable UUID, native zero-based [start,stop), punch type and explicit type uncertainty.

UI End means **last included frame**. The API stores **stop = End + 1**, with original integer PTS/timebases for both start and exclusive stop, exact rational timing, scene/segment and edge-overlap status. No nominal-FPS conversion or guessed source offsets are used. Boundaries must stay inside available context, overlap the interval and pass continuity checks. New or changed boundary positions require exact inspected-frame evidence; unchanged previously inspected boundaries can be reused for type-only edits. Source-video hashes and actual endpoint frames are revalidated before writes.

New records live under ignored `reports/boxingvi/interval-events-v1/V4-bg-N/v00000001/review.json` and subsequent immutable revisions. Each records `source=manual`, timestamp, request ID, previous revision hash, full interval/source provenance, the original positive-review hash, operation, current event snapshot and completion state. Completion records retain full viewing evidence and explicit confirmations. Removal is a tombstone operation in the history, not deletion of source data. The shared atomic revision publisher fsyncs the new bundle and renames it into place while the existing safe ledger lock is held; older files are never overwritten.

GET `/api/intervals/{id}/events` and POST of the same route expose current state/history and validated upsert/remove/complete operations. GET `/api/interval-event-status` supplies queue progress. Routes are restricted to the five fixed original V4 IDs and require their latest original review to remain Punch present. They share existing localhost/host/origin/JSON-write guards. A changed source review or provenance invalidates the event-history binding for explicit reconciliation; it is not silently migrated. Duplicate requests are idempotent; reused IDs with different payloads and stale concurrent writes fail. Failed saves retain drafts and do not advance.

All newly stored event decisions are manual actions, never automated judgments. The original source marks remain separately labeled provisional. Unknown footage remains unknown: neither an empty event list, removed event nor a completion action automatically generates background examples. `background_inferred=false` and `training_enabled=false` are preserved. Future preparation must explicitly version any use of these labels; the classifier's historical datasets and the accepted background manifest remain unchanged.

V5 remains development-only; V7/V8/V10 remain evaluation-only for this pilot, V7/V8 remain grouped and the V1/V6 guard is retained. V9 and arbitrary filesystem targets are inaccessible. No new identity or authentication fields, model code, data downloads or dependencies were introduced.

## Verification

Backend tests use tiny synthetic videos and temporary review ledgers. They cover add/edit/remove, restart persistence, exact timestamps, uncertain types, overlapping/consecutive events, old-history preservation, idempotency/stale writes, rejected uninspected or invalid boundaries, explicit full-coverage completion and invalidation after edits, empty-positive completion rejection, scope and request-origin guards. Frontend tests cover draft validation, end-to-exclusive conversion, save/edit/remove/reload, uncertainty, retry/duplicate prevention, completion gating/advancement and overlapping clickable human timeline segments separate from source marks.

Actual Chrome testing used the real V4 queue and footage: five-only navigation, smooth complete playback, exact start-marker jump, one-frame stepping, boundary controls, uncertain-type selection, cancel and interval switch. Browser testing did not save any event or fabricate human confirmations. Screenshot and preservation inventory are under ignored `reports/boxingvi/interval-event-labeling-validation-v1/`. Persistence and completion writes were exercised only with synthetic fixtures.

Back up `reports/boxingvi/interval-events-v1/` together with the original interval manifest and review ledgers. These are durable human annotations, not disposable cached previews. The original 12-background manifest still validates against its unchanged review head.

Final checks: **496 Python tests** and **76 frontend tests** pass; Ruff lint, changed-Python formatting, frontend formatting, TypeScript and production build pass. One existing Starlette/httpx deprecation warning remains. All **4,694 protected input/review/export/backup hashes** match, and the actual event ledger contains zero human revisions from testing. Actual save/revise/remove/completion/persistence tests were isolated in synthetic temporary workspaces. The original annotation datasets, 18 prior interval-review revisions, 12 verified backgrounds and historical experiments remain unchanged; V9 was not accessed.
