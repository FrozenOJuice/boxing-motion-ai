# Videos-first workspace review

This refinement is ready for visual review. It remains an isolated prototype, with no
production migration.

## Structure and proportions

- **Videos is home and the first navigation item.** Analyze is entered by opening a
  video. It has no independent top-level destination. A labeled Back to Videos action
  returns to the library; Development stays secondary.
- The player fits the footage's native aspect ratio into both the available height and
  width. A ResizeObserver recalculates it when the workspace changes size, reserving
  space for the timeline and controls. The stage no longer stretches into a shallow
  rectangle. Footage uses `object-fit: contain`, never fill cropping.
- Removing the separate tools row and below-player saved-punch strip gives the video
  substantially more room. The single timeline sits immediately below the player,
  followed by one transport row: playback/frame controls, time, speed/fullscreen, In/Out.
- Workspace/header width is capped at 1600 pixels; the library is capped at 1320 pixels.
  Width and height media queries reduce panel width and saved-list height where needed.
- The right panel shows instructions before selection; range, replay and Classify after
  selection; prediction and scores only after classification. Range and Classify stay
  fixed above the scrollable output. Previously classified punches remain in a compact
  independently scrollable list. Empty technical sections and placeholder scores are gone.
- The conventional thumbnail grid has **one primary Upload video action**. It handles
  an empty library without a second upload control and keeps one item at normal grid
  size rather than stretching it across the page. Larger collections scroll normally.

## Actual browser verification

Native Safari ran the real app through `review.html`, whose iframe provides independent
CSS viewports. Display scaling fits the review into the host window without changing
its layout viewport. All requested sizes were inspected visually after the final edits.

| Viewport | Capture | Observed |
| --- | --- | --- |
| 1440×900 | [Videos](artifacts/proportions/screenshots/videos-1440x900.png) | Compact thumbnail grid, single Upload action, secondary Development |
| 1440×900 | [Before selection](artifacts/proportions/screenshots/analyze-empty-1440x900.png) | Large native-proportion video, simple panel instructions |
| 1440×900 | [Selection](artifacts/proportions/screenshots/analyze-selection-1440x900.png) | Selected range and primary action, no empty prediction sections |
| 1440×900 | [After classification](artifacts/proportions/screenshots/analyze-classified-1440x900.png) | In-place prediction, source label, six scores, saved punches |
| 1280×800 | [Workspace](artifacts/proportions/screenshots/analyze-classified-1280x800.png) | Uncropped 16:9 footage; timeline, transport and Classify visible together |
| 1024×768 | [Workspace](artifacts/proportions/screenshots/analyze-classified-1024x768.png) | Native proportions retained; all primary controls visible without page scrolling |
| 1024×768 | [Videos](artifacts/proportions/screenshots/videos-1024x768.png) | Normal grid and metadata at compact desktop width |

Observed interactions: open video from the library, drag a source-snapped range, classify
without navigation, reopen recorded sample output, step source frame 38 → 39, replay a
saved interval, and return via Back to Videos. Replay stopped at source frame 35
(1.417 seconds), before the exclusive 1.459-second boundary.

Screenshots are real browser pixels. Only empty white padding from the native Safari
capture is trimmed; raw captures remain alongside the exports. Media and screenshots
remain ignored locally rather than redistributed in Git.

Automated verification: **14 tests**, TypeScript, Prettier and build. Tests explicitly
cover Videos-first navigation, a single upload action, empty/one-item/20-upload library
states, conditional panel content, saved output, simulation labels, stale-output clearing,
source timestamps, handle constraints and replay boundary races. Library state fixtures
are test-only; no fabricated library records are inserted into the application.

## Prototype limits for review

- Exact source frames remain available only for the indexed sample. Arbitrary local
  uploads use time seeking, with a clear precision note and disabled exact-frame buttons.
- Safari can produce black upload thumbnails and an initially black local preview.
  Library thumbnails use a video-file fallback; production should use pipeline thumbnails.
- Model details and score explanations may require output-panel scrolling on short
  desktops. Range, replay, Classify and the saved-punch list remain visible.
- Unsaved selection/playhead state resets on leaving the workspace. Saved selections
  reopen correctly. Naming, search, deletion, duplicate handling, mobile and dedicated
  screen-reader/device review remain follow-ups.

New classifications are clearly labeled **Prototype example / No model ran for this
selection**. The two sample selections preserve actual recorded model outputs. All
scores remain labeled uncalibrated. No new model accuracy claim is made.

## Scope

Only `prototype/` changes. Production source, Python services, models, datasets, reviews,
experiments and V9 are untouched. No production API, training or inference runs. No push,
deployment or migration is performed. The existing local preview server is reused and
left running because this refinement did not start it.
