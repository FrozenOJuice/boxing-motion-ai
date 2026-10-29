# Boxing Motion AI — UI/UX assessment and redesign plan

**Review date:** 29 October 2026

**Assessed source baseline:** `9639ed2`

**Status:** Assessment and proposed design only; requires approval before implementation.

## 1. Recommendation

Build a focused local media application around **Videos → open video → select one punch → classify → review the prediction in the same workspace**. Development is a separate secondary area. Do not add a Dashboard, a top-level Analyze destination, or a Results destination.

The production application has useful real functionality but still presents a developer-oriented upload form and research workbench. The prototype establishes a much better structure and properly proportioned viewer, but it is not ready to migrate. Its remaining problems are incomplete interaction states, difficult access to saved punches, very small text, and an upload/library experience that does not yet represent the production lifecycle.

**Recommended next implementation milestone:** finish the prototype's navigation, playback, selection and inspector behavior described in Phase 1. Keep production unchanged until that working experience passes visual review. Reuse production's analysis contracts later; do not replace them with prototype simulation or browser-only validation.

### Priority findings

| Priority                   | Finding and evidence                                                                                                                                                                                        | Required outcome                                                                                                                                 |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| P0 — task blocker          | Production `.has-video .analysis-stage` caps height at `clamp(180px, 30vh, 340px)`. At 1024×768, results/Classify fall below the initial viewport because the analysis grid stacks.                         | Fit the uncropped viewer to available width **and height**; keep timeline, transport and Classify visible together.                              |
| P0 — interaction integrity | In the prototype, pressing Set start at frame 1 immediately creates `0.000–0.041` and enables Classify. Clicking it produces example output without setting an end.                                         | Represent an incomplete selection explicitly. Require both boundaries and the existing pipeline's frame/range constraints before classification. |
| P0 — playback              | In Safari, full playback ended at source frame 144, `5.963 / 6.004`. Pressing Play again left it on the last frame rather than restarting.                                                                  | Explicit ended/replay behavior: Play at the end starts the video from the beginning; Replay selection always uses its bounds.                    |
| P1 — result access         | The prototype's saved list is limited to 88px, or 74px on shorter screens. Only about one or two punches are visible. The result body scrolls independently above it.                                       | Give saved punches a usable local inspector view; avoid two tiny stacked scroll areas.                                                           |
| P1 — continuity            | Back to Videos and reopening the same video resets the unsaved range and playhead. Neither UI has URL-based workspace navigation.                                                                           | Preserve the draft and playhead per video during navigation; support browser Back/Forward and refresh recovery.                                  |
| P1 — readability           | Prototype score labels, timestamps and metadata use 10–11px; the simulation disclosure is 9px. More compact does not mean more usable.                                                                      | Use readable 12–14px supporting text and 14px normal control labels; reclaim space by removing redundant structure.                              |
| P1 — trust/lifecycle       | Prototype local uploads do not perform pose extraction or inference. The library lacks per-video cleanup, status and real recent-open ordering. Production has real jobs but hides videos in a select menu. | One real video library with actual lifecycle states, clear limits, safe cleanup and no invented progress or results.                             |
| P2 — consistency           | Production Review, Analyze and Overview use different layout rules; the two applications have separate control/style systems.                                                                               | One boxer-facing component system and scoped styles; retain internal review behavior behind a separate shell.                                    |

P0 means it blocks a dependable core workflow, not that the task justifies changing models or research data.

## 2. What was actually inspected

### Browser method and coverage

Both running applications were inspected in **native Safari**, including screenshots and accessibility-tree observations. A temporary review page outside the repository embedded the actual applications at independent **1440×900, 1280×800 and 1024×768 CSS viewports**. Display scaling fitted these into the host window; it did not change their CSS viewport dimensions. Source inspection confirmed text sizes and layout rules, so scaled screenshots were not mistaken for literal font-size measurements.

| Application / state                | 1440×900                                 | 1280×800                                     | 1024×768                                           |
| ---------------------------------- | ---------------------------------------- | -------------------------------------------- | -------------------------------------------------- |
| Prototype Videos                   | Visually inspected                       | Visually inspected                           | Visually inspected                                 |
| Prototype Analyze before selection | Visually inspected                       | Visually inspected                           | Visually inspected                                 |
| Prototype Analyze with prediction  | Example output inspected                 | Example output inspected, including playback | Recorded sample output inspected, including replay |
| Production Analyze                 | Ready video / empty prediction inspected | Recorded result inspected                    | Recorded result layout inspected                   |

Additional browser checks:

- Prototype Set start → one-frame selection → example classification; full playback and attempted restart; saved selection opening and bounded replay; Back to Videos and reopening.
- Native prototype file selection rejected existing over-30-second test footage with “Choose a video no longer than 30 seconds.” The library remained usable.
- Prototype Development, disclosure and keyboard-help dialogs inspected at compact desktop size. Escape closed Development and restored focus to its trigger.
- Production Review/Punches, Review/Video intervals, Dataset, Experiments and Models visually inspected at 1440×900. Existing saved analysis results were opened without running new inference.
- Review source frames loaded in this session. A universal “broken video review” failure was **not** reproduced. There are still layout, scrolling and interaction inconsistencies; do not equate them with a demonstrated failure of every viewer.

**Limits:** Production development screens were not exhaustively tested at all three sizes. Slow processing, unavailable poses, interrupted jobs, browser storage exhaustion, portrait footage, 200% zoom, VoiceOver and other browsers require additional validation. Their recommendations below combine source inspection with explicit future acceptance tests, not invented observations. Fresh successful uploads and a new live production classification were not run during this assessment.

Audit captures and the temporary viewport tool are under `/private/tmp/boxing-ui-audit/`, outside Git. Screenshots are evidence of rendered UI, not proposed mockups. No screenshots or generated visual artifacts belong in this commit.

### Supporting checks and source evidence

- Prototype: **14 tests pass**; TypeScript passes.
- Production frontend: **92 tests pass**; TypeScript passes.
- These establish a useful baseline, not UX approval: the one-frame selection and end-of-video behavior survived the current tests.
- No production upload/classify/remove request or review-save action was performed. No model training, model evaluation or accuracy measurement was run.

Key implementation references:

- [Production shell](../web/src/App.tsx), [Analyze](../web/src/Analyze.tsx), [production styles](../web/src/styles.css).
- [Review viewer](../web/src/Viewer.tsx), [buffered interval viewer](../web/src/BufferedIntervalViewer.tsx), [review workbench](../web/src/ReviewWorkbench.tsx), [interval review](../web/src/IntervalReview.tsx), [Overview tools](../web/src/Overview.tsx).
- [Prototype application](../prototype/src/App.tsx), [player](../prototype/src/Player.tsx), [timeline](../prototype/src/Timeline.tsx), [storage](../prototype/src/storage.ts), [styles](../prototype/src/styles.css).
- [Existing assisted-recognition contract](ASSISTED_PUNCH_RECOGNITION.md), [local jobs](../src/boxing_motion_ai/analysis/jobs.py).

## 3. Current design assessment

### Application structure and navigation

**Production:** Analyze is the default. A fixed sidebar exposes Analyze plus four Development tools. Development is labeled, but those destinations still take much of the app's navigational space and attention. Upload, the local-video dropdown, removal, analysis, explanatory copy and saved selections share one page. There is no recognizable Videos home.

**Prototype:** Videos as home, a video-opened workspace and a labeled Back action are correct. Development is visually secondary and truthfully represented as an informational dialog. However, the Videos tab has no active parent indication inside the workspace, and internal screen state does not behave like normal URL navigation. A click removes the focused library item without explicitly focusing the workspace heading; classification also left focus on the document in the observed session.

**Decision:** Keep the Videos-first hierarchy. Replace state-only navigation with a small route model and deliberate focus restoration. Do not bring the production sidebar into the boxer experience.

### Layout and video proportions

**Production:** A slogan, upload panel and video dropdown consume the space above the player even after a video is ready. At 1440×900 the ready footage is constrained to roughly 270px high by the 30vh rule, with unused horizontal stage space. At 1280×800 lower selection controls/history need page scrolling. At 1024×768 the results panel is below the fold. The layout prioritizes explanatory form sections over the video task.

**Prototype:** The width/height fit and `object-fit: contain` preserve the sample's native 16:9 aspect ratio. The footage is substantially larger and the timeline/transport remain visible at every tested size. Keep this principle. The `clientHeight - 118` reservation is nevertheless a hard-coded assumption about controls, text wrapping and zoom. At 1024×768 the video is width-limited and blank space remains underneath; stretching it vertically would recreate the original error. The tall bordered empty inspector exaggerates this unused space.

**Decision:** Measure the actual control stack, fit the viewer within the remaining space and let the empty inspector be content-led. Use empty space when native proportions require it; do not fill it with dashboard cards or crop the footage.

### Controls, timeline and selection

There is already only one principal seek track in each boxer workflow; preserve that. Production's plain native range does not show the selected interval. Its separate boundary buttons are precise but disconnected from the video position. The prototype adds a visible range and handles, but a drag on the same filmstrip both seeks and creates/replaces a selection. That behavior is not obvious to a new boxer. “In” and “Out” plus bracket symbols are familiar to editors but less clear than “Set start” and “Set end.”

The prototype's one-boundary shortcut silently invents the other boundary. Both handles can nearly overlap for short selections. A 12px handle is too difficult to acquire reliably. It has slider semantics and keyboard adjustment, but the announced min/max do not describe the other boundary's constraint. The ruler floors to whole seconds; finer scales need explicit subdivision, not repeated indistinguishable labels.

Production and prototype also describe the same recorded range differently: production shows **0.334s between boundary frames**, while the prototype displays **0.375s** for the exclusive interval. Both follow their stated definitions, but the mismatch is confusing. Adopt one canonical interval contract and one user-facing duration calculation.

### Prediction and saved punches

Keeping prediction beside the video is correct. Both implementations distinguish uncalibrated scores from verified confidence, and the prototype labels examples separately from recorded model output. Keep these distinctions.

The current right panel spends several regions on range, replay, Classify, repeated simulation wording, prediction, scores, model details and a very short saved list. At 1280×800 and 1024×768, the score explanation/model details require scrolling while the saved list has another scrollbar. The generic “Classified punches” heading also groups real saved outputs with simulated selections. There is no clear draft-versus-saved lifecycle, stable punch naming or “next punch” action. Repeated classification can append duplicate prototype rows.

**Decision:** Keep one inspector, with local **Selection** and **Saved punches (n)** tabs. These are views inside Analyze, not primary application destinations. Use one scrollable body per tab; retain range and primary action outside it. Clicking a saved row selects its range, seeks to its start and opens Selection. Source type remains explicit. Updating bounds clears the current output; it does not silently overwrite the old saved result.

### Videos and upload

The prototype's ordinary grid and single Upload action are good. One video stays a normal-size tile instead of expanding across the page. A largely empty collection is acceptable; manufactured filler would make it worse.

The remaining weaknesses are concrete:

- The sample looks like an owned library video. A small metadata line does not establish a strong empty-library/demo distinction.
- “Recent” is not backed by reopening behavior: `openedAt` is assigned on import, not updated when a video reopens. Production's list is sorted by random directory IDs rather than user recency.
- New users cannot see preparation state, last activity or safe cleanup on a video tile. Prototype cleanup requires clearing browser site data; production removal is a prominent button beside the video selector and has no confirmation in its handler.
- Browser poster capture can return black; the file icon fallback is honest but makes a collection hard to scan. Browser-clock local footage also lacks the sample's real filmstrip and exact stepping.
- File limits differ: production supports MKV and validates/transcodes through Python, while the prototype uses browser decoding and does not accept MKV. The prototype's 150 MB text actually guards 150 MiB.
- `recentVideos()` and sample fetch start without a separate library-loading state, so “No videos yet” can appear before loading finishes. The sample can mask the genuine zero-upload state.

Preserve the backend's existing **four saved-video limit**, processing limits and cleanup scope. A scalable grid is not authorization to increase capacity. Test 20-item layout fixtures for overflow, but do not claim the real application already stores 20 uploads.

### Loading, errors and accessibility

The over-duration error was understandable, preserved the library and could be dismissed. Native dialog focus/Escape behavior worked. The prototype has a skip link, named icon buttons and visible focus rings; its muted colors are generally sufficiently distinct. This is not a reason to replace the dark direction.

Gaps include the missing library-loading state, generic opening status without a production processing lifecycle, no unavailable-result prototype state, weak discoverability of disabled exact-frame controls, tiny supporting text and handle targets, navigation focus loss, and untested zoom/screen-reader behavior. Some production errors prescribe removing/reuploading the video even when a frame retry or playback reload would be a safer recovery. Do not treat data deletion as the normal response to a transient playback failure.

### Development screens

| Screen             | Current issue                                                                                                                                                     | Change / remove                                                                                                                                                   | Retain                                                                                                                                 |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Review — punches   | Global navigation plus queue plus viewer compete for width. Decision/save controls approach or cross the bottom of the viewport. Styling differs from Analyze.    | Put it in a separate Development shell; provide a stable decision/save region and shared focus/control primitives later. Do not rebuild the workflow now.         | Queue filters, exact source frames, annotation context, uncertainty, dirty-state protection, save/next behavior and review provenance. |
| Review — intervals | Additional viewing, annotation and event-labeling states make it substantially more complex than punch recognition. Actions and evidence sit below the viewer.    | Make viewing status and save eligibility legible; avoid duplicating those controls in boxer Analyze. Treat specific playback bugs as scoped fixes.                | Buffered media, inspected-frame/continuous-viewing evidence, annotation distinctions and review gates.                                 |
| Dataset            | Valid research tables and metrics dominate if treated as a user dashboard. Source-group terms assume specialist knowledge.                                        | Remove it from boxer navigation; keep a clear page heading, table overflow and expandable provenance within Development.                                          | Existing counts, roles, source separation, missing-data distinctions and restrictions.                                                 |
| Experiments        | Large experiment cards, technical path-like table labels and an “Inspect results ↗” treatment can suggest external navigation even when it switches a local view. | Prefer a compact experiment selector/list and explicit local selection states if this tool is later refined. No broad overhaul is required for the boxer release. | Original result scope, negative studies, historical/reference status and metadata.                                                     |
| Models             | Development metrics and confusion matrices could be misread as expected accuracy on an uploaded video.                                                            | Keep them in Development; expose only concise model identity/limitations in Analyze. Never place these accuracy figures beside a new user's prediction.           | Run selection, class metrics, confusion matrix, checkpoint provenance and evaluation-scope caveats.                                    |

## 4. Recommended product structure

```text
Videos                         /videos             (default; / redirects here)
  Upload / preparation         dialog + video tile status
  Analyze a video              /videos/:videoId
    Current selection          shared player + timeline + inspector
    Saved punches              local inspector tab, same workspace
Development                    /development/*      (secondary entry)
  Review / Dataset / Experiments / Models
Help                           contextual dialog, not another main destination
```

Browser Back/Forward follows these routes. Back to Videos restores library scroll/focus. Reopening a video restores its session draft and playhead. Refresh reopens an available video/result or shows a recoverable missing-video state. A saved selection may have a stable route/query identifier; it must still render in the same workspace.

The boxer sees **videos and punch predictions**, not datasets, pilots, checkpoints, seed values or experiment scores. The six-class model remains assisted recognition: users select a movement they believe contains one punch. No automatic punch discovery, No punch class, verified labels, statistics or coaching are added.

## 5. Detailed Videos and upload specification

### Layout

- Use a centered content container up to 1280px, 24px desktop gutters (16px at 1024). Header: “Videos” on the left, one primary “Upload video” on the right.
- Under the header, one concise guidance line: “One boxer · Continuous shadowboxing · Up to 30 seconds.” Put additional format/size limits in the upload dialog, backed by the actual adapter contract.
- Use a grid of approximately 240–300px-wide tiles; 4 columns at a wide library, 3 at 1024 when labels remain readable. Never stretch a single tile across the container.
- A tile has a contained thumbnail, duration overlay, filename/title, actual preparation state and saved-punch count when nonzero. Use 14px title and 12px metadata. Show the full filename on focus/hover or in details.
- Do not show dimensions, checkpoint details, fake dates or analytics on every tile. Dimensions belong in video information. Show recency only when a real last-opened timestamp exists.
- Keep “Try a sample” as a clearly separate secondary text action in prototype/empty-library guidance. It opens the reference video; it does not insert pretend user uploads.

### States and interaction

| State                      | Required behavior                                                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Initial load               | Stable library placeholder or short status; no “empty” claim until loading completes. Preserve existing content during refresh.                                                                                                       |
| No uploads                 | Compact “No videos yet” guidance beneath the existing header action. No second upload button, huge drop card or fake recent videos.                                                                                                   |
| One / several videos       | Same tile size and metadata hierarchy. Open by click, Enter or Space. Library scroll is independent of workspace state.                                                                                                               |
| File selected              | Validate obvious size/type errors; show filename and actual opening/upload state. Native picker and file drop use the same handler.                                                                                                   |
| Queued / preparing         | Tile names the actual stage. Use determinate progress only when counts exist; otherwise use an indeterminate indicator. No fabricated percentage/ETA.                                                                                 |
| Ready                      | Tile opens Analyze. Production must use the existing prepared preview, source-frame inventory and cached poses, not browser-only equivalents.                                                                                         |
| Failed                     | Tile remains visible with a concise reason and supported recovery. Never display a failed job as an empty library.                                                                                                                    |
| Storage / capacity failure | Explain actual capacity/disk/storage limitation. Keep already available videos accessible. Offer scoped cleanup, not browser-wide data clearing.                                                                                      |
| Remove video               | Put a secondary menu action on that video. Confirm that original upload, generated media and its analysis results will be removed. Use the existing scoped API; never touch research data. Disable while the backend forbids removal. |

Use existing prepared source-frame PNG endpoints for reliable production posters where possible. Do not recapture an unreliable browser frame or add a second decoding pipeline just to style the library. If a required metadata field is unavailable, omit it or explicitly propose the small API addition separately; do not fabricate it. Rename, search and collection organization can wait until the small real library warrants them.

## 6. Detailed Analyze specification

### Layout and sizing

Use a compact 56px global header and approximately 52px workspace bar: **Back to Videos**, video title, optional video-information/help action. Remove the production slogan, repeated upload panel and video dropdown from the open workspace.

The main region contains the viewer stack and right inspector. Suggested starting dimensions, to be verified rather than hard-coded as universal breakpoints:

| CSS viewport | Gutters / gap | Inspector | Approximate 16:9 viewer width × height |
| ------------ | ------------- | --------- | -------------------------------------- |
| 1440×900     | 24px / 20px   | 304px     | 1068 × 601px                           |
| 1280×800     | 20px / 16px   | 288px     | 936 × 527px                            |
| 1024×768     | 16px / 16px   | 272px     | 704 × 396px                            |

These are design targets, not measurements of the current app. Cap the total workspace near 1600px. For every clip, fit `width = min(availableWidth, availableHeight × nativeAspectRatio)` after measuring the actual timeline/transport stack. Derive height from the same ratio. Do not assign a competing minimum height or fixed 16:9 ratio to portrait footage. Preserve all pixels with contain fitting.

At the three target sizes and 100% zoom, normal video/selection/Classify use must not require page scrolling. Output/model details may scroll within the inspector. At narrower effective widths or 200% zoom, switch to a single-column/reflow layout with a reachable Classify action; allow document scrolling instead of shrinking text or clipping controls.

### Viewer and transport

- Viewer remains the largest object. Timeline sits directly below it with a small 8px gap and shares its width. No second native-control scrubber or decorative progress bar.
- One transport row: Play/Pause, previous/next frame, current/total time, playback speed, fullscreen; selection actions use the remaining space consistently. Keep all hit areas at least 36px high in ordinary desktop use.
- Use explicit “Set start” and “Set end” labels, with I/O in tooltips or optional keyboard hints. Secondary frame buttons keep recognizable frame-step icons and accessible names.
- Preserve the source-frame/preview mapping. While playing, follow presented media time; on exact stepping, show the requested source frame before treating it as ready. Do not infer frames using nominal FPS.
- Handle ready, buffering, seeking, ended, unavailable-frame and playback-error states explicitly. An error/status overlay must not change stage dimensions.
- Play after end restarts at zero. Replay selection starts at its first included frame and stops before the exclusive boundary. Replay is not a second playback system.
- Offer frame retry or media reload for recoverable failures. Keep the current bounds/result while retrying. Fullscreen failure should be a nonblocking message; test fullscreen outside an iframe that may restrict it.

### One timeline and a simple selection contract

Adopt one visible rule and teach it once: **click to seek; drag across a punch to select; drag either edge to refine**. Show this concise hint before the first complete selection, close to the timeline. Set start/end offers a precise, keyboard-accessible alternative, without changing modes.

- Setting only one boundary creates an incomplete draft, not a one-frame “ready” punch. Use “Set the end of the punch” or “Set the start” near the disabled action.
- Distinguish playhead, selected range and saved-selection markers by shape/labels as well as color. Do not add another full annotation lane for six colors.
- Give boundary handles 24px or larger hit regions around narrow visible marks. Resolve overlapping targets deterministically. Keyboard arrows move one actual source boundary; Home/End respect the other edge.
- Use source-time positions for filmstrip, ruler, handles and playback. Add millisecond/subsecond ticks at useful scales; do not repeat floored labels. Optional timeline zoom should magnify the **same** track only if 30-second clips prove too imprecise to trim; it is not required to start.
- Use one internal half-open frame interval `[start, stopExclusive)` matching the existing API. Adapt the inclusive end-frame control once. Display duration using source timestamps through the stop boundary, including the final included frame's interval.
- Require at least the existing minimum frame count, legal bounds and the existing maximum selection span. Do not invent a new minimum punch duration or pose threshold from this UX review. Tell the user why a selection is invalid before sending a request; the backend remains authoritative.
- A complete range is a user-designated candidate punch, never proof that a punch happened. Changing its bounds invalidates the displayed prediction. Seeking/stepping alone does not erase it.

### Inspector state design

Use local **Selection** and **Saved punches (n)** tabs in the right inspector. Selection is the default. Keep its selected range and primary action outside the scrollable body. This replaces the current tiny list pinned beneath another scroll region, not the integrated workspace itself.

| State                      | Inspector content / action                                                                                                                                                                                                   |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No selection               | “Select one punch” and one instruction. No scores, empty model/pose sections, disabled result card or repeated generic headings. Saved tab remains available if real selections exist.                                       |
| Incomplete / invalid draft | Actual boundary values and one actionable reason. Classify stays disabled. Do not imply pose quality was checked.                                                                                                            |
| Complete valid range       | Start, end, duration, Replay selection, primary Classify punch. No fabricated prediction or readiness score.                                                                                                                 |
| Classifying                | Stable layout, actual in-progress state, duplicate submission blocked and status announced. Keep focus; late responses cannot attach to a different video/range.                                                             |
| Available result           | Punch type, six consistently ordered score rows, short “Uncalibrated scores” explanation and restrained experimental label. Model identity/details are expandable. No percent-confidence gauge, accuracy metric or coaching. |
| Unavailable                | “Prediction unavailable” plus the actual pipeline reason and specific recovery, such as adjusting the range or choosing clearer footage. Show no punch label or score bars. Keep replay and the selected range.              |
| Request failure            | Retry for the same candidate, without discarding it or showing old scores as current. Distinguish a transport failure from insufficient pose quality.                                                                        |

Saved rows use time range, predicted type or “Unavailable,” and source status where necessary. A selected row has an explicit selected state. Opening it synchronizes player, range and output. A new candidate clears the current result but preserves previous saved records. Reclassifying identical video/range/model reuses or updates that result rather than appending indistinguishable duplicates. Do not add selection deletion or renaming that the real data contract cannot support yet.

In prototype mode, keep a visible global prototype identifier and **Example output / No model ran for this selection** beside simulated output. Never persist a simulation as a real result or upgrade it on reload. Recorded sample output retains its distinct provenance. In the eventual connected app, pose coverage and processing time belong under details, separately labeled from recognition accuracy.

## 7. Visual system and interaction rules

Use one restrained neutral-dark system, not a different theme for each screen. Define tokens once and use shared primitives rather than local CSS patches.

| Element                         | Proposed rule                                                                                                                                                                                                  |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Canvas / panel / raised surface | `#121416` / `#1B1E22` / `#23272C`; avoid a card around every subsection.                                                                                                                                       |
| Primary / secondary text        | `#E8EBEF` / `#AEB7C4`; secondary text still needs readable size.                                                                                                                                               |
| Action / selection accent       | `#C0E583` with `#19220E` text on filled actions. Reserve it for the current selection and primary action; no per-punch decorative colors.                                                                      |
| Borders                         | `#343A42` for nonessential separators; stronger `#66717F` where a control boundary must be distinguishable.                                                                                                    |
| Typography                      | System sans; 24/30px page title, 18/24px prediction, 14/20px body/controls, 12/16px metadata. Tabular numerals for time and scores. Avoid 9–10px essential copy and all-caps/letter-spaced technical eyebrows. |
| Spacing                         | 4px base scale: 4, 8, 12, 16, 24, 32. Use 16px panel padding, 8px related-control gaps and 24px between independent sections. Separate by hierarchy or a divider, not extra nested cards.                      |
| Shape                           | 6px control radius, 8px major-surface radius, 1px borders. Shadows only for dialogs/menus; no decorative gradients.                                                                                            |
| Controls                        | 36px ordinary desktop height, 40px primary action where space allows, 44px for coarse pointers. Visible hover, active, selected, disabled, busy and focus states. Use one icon set at 18–20px.                 |
| Focus                           | 2px high-contrast outline with offset. Hover is never the only way to reveal a required action. Tooltips open on focus too.                                                                                    |
| Status                          | Pair color with text/icon. Errors explain what happened and what can be done; success does not become a permanent badge collection.                                                                            |

Calculated static token-pair contrast: primary text/canvas **15.44:1**, secondary text/panel **8.26:1**, filled-action text/accent **11.58:1**, stronger control border/panel **3.37:1**. These are candidate-palette calculations, not a certification of rendered UI. Verify all actual states, transparency, overlays and focus combinations.

Accessibility requirements:

- Keyboard can upload/open a video, play/pause, seek, adjust both bounds, classify and reopen a result without pointer-only gestures. Modal focus is trapped and restored; route changes focus the main heading; returning focuses the originating tile.
- Limit global shortcuts to the active workspace. Never steal keys from fields, menus, dialogs or assistive controls. Announce selection values and constraints; do not announce playback time continuously to screen readers.
- Use semantic tabs and selected state for inspector views. Named meter values describe model scores, not confidence. Expose busy, error and unavailable states without depending on color.
- Target WCAG 2.2 AA: text contrast, control contrast, keyboard access, focus visibility, target sizing and reflow. Test with VoiceOver and zoom; source-level ARIA attributes alone do not establish conformance.
- Respect reduced motion. No animated score counting or unnecessary page transitions.

## 8. Frontend simplification and reuse

### Keep the proven contracts; replace presentation deliberately

| Reuse / isolate                                                                                                                                                                                            | Simplify / replace                                                                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Existing upload validation, local job processing, preview and exact-frame endpoints, cached MediaPipe poses, deterministic feature/model inference, unavailable-result reasons and prediction persistence. | Production `Analyze.tsx` mixes job loading/upload/removal, media state, selection, result history and presentation. Split those responsibilities without changing inference.        |
| Prototype's native-aspect fit principle, one timeline, buffered player, source-frame helpers, shared Button/IconButton/Dialog primitives and explicit provenance labels.                                   | Prototype `App.tsx` owns navigation, library, import, dialogs, selection and result UI in one large module. Extract a small set of feature components; do not migrate it wholesale. |
| Review dirty-state guards, exact-frame evidence, continuous-viewing/inspection tracking, uncertainty and immutable/source-group restrictions.                                                              | Do not merge ordinary Analyze playback time with review viewing evidence. Share low-level timing math or presentational controls only after their different contracts are tested.   |
| Read-only development results and their original evaluation scope.                                                                                                                                         | Scope Development styles; avoid copying its metadata cards, global selectors or metric components into boxer pages.                                                                 |

Recommended boundaries: `BoxerShell`, `VideosPage`, `VideoTile`, `UploadStatus`, `AnalyzeWorkspace`, `VideoPlayer/Transport`, `SelectionTimeline`, `PunchInspector` and `SavedPunchList`. Keep a small route owner and typed video/selection state; a new generic dashboard framework or large UI dependency is unnecessary.

Use one authoritative playback/selection controller. Represent incomplete, valid, classifying, available, unavailable and failed states explicitly instead of independent booleans that permit contradictory UI. Keep stable video/selection IDs and guard against stale responses.

Put data access behind a narrow adapter: production jobs/results versus prototype browser storage/example output. Share presentation contracts where appropriate, **not storage namespaces or inference semantics**. Revoke browser object URLs when media is released. Library recency and UI preferences must not be written into research records.

Production already avoids fetching research metadata while Analyze is active; preserve that separation. Development currently requests reviews, inventory and experiments together for any development destination. Route-local reads and a small shared cache can reduce unnecessary work later, without altering data semantics.

Consolidate styles by tokens and component ownership. The production stylesheet has multiple appended review/Analyze overrides, including the final 30vh cap. Remove obsolete rules within an approved feature migration, not by another global override at the end. Do not delete old development viewers merely because their appearance overlaps: their frame/evidence behavior differs.

## 9. Prioritized implementation roadmap and approval gates

Each phase must deliver a working, reviewable increment. This document authorizes none of them by itself.

### Phase 1 — prototype workflow integrity and workspace structure

**Deliver:** Videos-first routed workflow, correct playback completion, explicit selection draft state, and the inspector's Selection/Saved punches views. Keep simulated inference visibly simulated.

**Acceptance:**

- Opening, Back to Videos, browser Back/Forward and reopening preserve the appropriate video, playhead, draft and focus. Refresh has an explicit recovery path.
- Setting only one boundary never enables Classify. Existing valid/invalid frame/range rules are represented and explained; no new ML threshold is invented.
- Click seeks; drag selects; handles and keyboard edits agree on actual timestamps. Changing bounds clears current scores and preserves older saved records.
- Play after the natural end restarts; selection replay ends on the final included frame. Repeated play/pause/seek/step does not desynchronize the playhead.
- Saved punches are one local tab away, with enough space for at least five readable rows at 1024×768. Opening a row synchronizes range, frame and source-labeled output.
- At all three requested sizes, player, transport, timeline and primary action are visible together. No second timeline, Results page, hero block or duplicate upload action appears.
- Targeted regression tests cover the two browser-found failures and navigation/result races; real browser verification is required in addition to tests.

**Gate:** Approve the revised prototype flow before expanding library or production work.

### Phase 2 — library, preparation and recovery states

**Deliver:** A dependable local-video entry experience, with truthful loading/empty/ready/failed states and supported cleanup. Use prototype adapters/fixtures where production is not connected, explicitly marked.

**Acceptance:**

- Zero, one and four real-capacity items use the same grid. A 20-item test fixture verifies scrolling/overflow only; it does not change production capacity.
- Loading cannot flash “No videos yet.” The demo is distinct from personal uploads. Metadata and recent ordering are backed by actual values.
- File picker/drop share validation. Unsupported/corrupt, over-duration, over-size, storage-full and unavailable-storage cases retain existing videos and explain recovery.
- A processing tile uses only real stage/count information or explicitly labeled test simulation. Successful completion opens the correct video once, without reprocessing on subsequent selection.
- Safe per-video removal is scoped and confirmed; processing/busy restrictions remain enforced. No research record or saved review can be removed through this flow.

**Gate:** Approve the first-use and recovery experience with actual local footage.

### Phase 3 — visual system, accessibility and desktop density

**Deliver:** Apply the specified tokens and controls, replace tiny text and narrow targets, tune layout using available space and measured content, and finish output/error states.

**Acceptance:**

- Capture Videos, empty/incomplete/valid selection, available/unavailable prediction, Saved punches, upload/loading/error and help states at 1440×900, 1280×800 and 1024×768. Label test fixtures; never invent real AI results.
- Native footage aspect ratio differs by no more than 1% from its rendered image ratio, with no crop/stretch. Test landscape, portrait and rotated metadata examples without changing the media pipeline.
- Primary action and selected range remain visible as score/model details scroll. One inspector body scrolls at a time; saved entries are not confined to a 74–88px viewport.
- All essential copy is at least 12px at 100% zoom. Normal controls use the shared sizes; boundary targets meet at least 24×24px with larger pointer-friendly targets where possible.
- Keyboard-only and VoiceOver flows complete the main task; focus survives navigation, classification and dialog closure. 200% zoom/reflow does not clip actions.
- Text/control/focus contrast is measured in rendered states. Reduced-motion preferences work. Tooltips are accessible and disabled controls explain limitations nearby.
- Tests, TypeScript, formatter and build pass; screenshots require human visual approval, not just pixel-difference thresholds.

**Gate:** Approve the complete isolated experience. Passing automated checks alone is insufficient.

### Phase 4 — separately authorized production integration

**Deliver only after explicit approval:** Connect the approved boxer UI through a production adapter to the existing Python workflow, while placing the current internal tools behind Development. This is not an automatic continuation of prototype approval.

**Acceptance:**

- A new single-boxer video works without a workbook; exact frames, prepared preview and poses come from the existing pipeline. Reopening or classifying another range does not rerun pose extraction.
- Identical input remains deterministic; compatible reference clips reproduce stored predictions through the adapter. Frame indices/timestamps, exclusive end bounds and unavailable reasons remain unchanged across the UI boundary.
- No production simulation path can return example scores. Available, unavailable and request failure are distinct. Six-class scores remain uncalibrated and no punch-occurrence claim is added.
- Processing time, pose coverage and recognition accuracy remain separate concepts. No upload produces a claimed accuracy statistic without labeled evaluation.
- Existing local upload limits and scoped cleanup remain intact. Any missing thumbnail/metadata capability is a separately identified, bounded API task, not a silent backend redesign.
- Reviews, experiments, datasets, checkpoints, source groups and V9 protections remain unchanged. Development save gates, dirty-state handling, viewing evidence and existing tests continue to pass.
- Run relevant Python/frontend tests, Ruff, TypeScript, formatting and production build, plus real-browser smoke tests of both boxer and Development workflows. Review data/provenance preservation before rollout.

Defer search, extensive naming/organization, batch tools, automatic suggestions/detection, statistics, coaching, new training and architecture changes. They do not solve the current core-task usability problems.

## 10. Final visual and functional sign-off

In addition to the phase checks, recruit at least five first-time users for a short task-based review. These are proposed acceptance targets, not results already measured:

1. Find Upload and open an available video without guidance.
2. Select a complete punch, replay it, classify and explain what the result does—and does not—mean.
3. Find a previous prediction and return to the video library without losing it.
4. Recover from an invalid upload and an unavailable prediction.

Target at least four of five completing each core task without facilitator rescue, and all participants understanding that the AI has not verified that a punch occurred or supplied a probability of correctness. Record task completion, selection mistakes, recovery success and time to a usable result. Do not mix these usability measures or processing duration with recognition accuracy.

Approval should settle the Videos-first structure, selection rules, inspector tabs and density targets. Implementation begins only with a subsequent instruction. This assessment makes no production code, prototype migration, model, dataset, experiment or review-decision change.
