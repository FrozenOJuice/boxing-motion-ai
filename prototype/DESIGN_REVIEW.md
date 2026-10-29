# Unified workspace refinement — October 29, 2026

The prototype is ready for **design review and approval**, not production migration.
All changes remain inside `prototype/`.

## What changed

- Analyze is the default and the normal workflow stays in one workspace. Results is no
  longer a primary page or navigation destination.
- A large video stage, familiar transport, secondary source-frame steps and one
  synchronized selection timeline occupy the main workspace.
- The right panel holds selection times, replay, prediction, six score bars, selection
  quality and collapsed model information. Its classification action stays visible
  while panel details scroll.
- Saved punches reopen their range and output beside the same player. Changing bounds
  clears the previous prediction so scores never remain attached to a changed range.
- Videos is a compact thumbnail library with an obvious upload action and concise
  local-media metadata. Clicking a video opens Analyze.
- Development remains a quiet secondary entry, represented by an informational dialog.
  No production tool is connected.
- Neutral dark surfaces, restrained green actions/selections, shared controls and
  consistent typography replace the earlier light layout. No remote fonts are needed.

New classification is deliberately simulated and labeled **Prototype example / No model
ran for this selection**. The two reference selections retain their real recorded model
outputs. Scores are labeled uncalibrated; neither example output nor saved model output
is a verified accuracy claim. Selection quality reports real range/frame availability;
live pose coverage is explicitly not evaluated.

## Browser verification

Reviewed in native Safari using the real app embedded by `review.html`. The iframe has
actual independent CSS viewports; only its display is scaled to fit the host browser.

| State | Viewport | Capture |
| --- | --- | --- |
| Videos | 1440×900 | [Library](artifacts/refinement/screenshots/videos-1440x900.png) |
| Analyze, no selection | 1440×900 | [Workspace](artifacts/refinement/screenshots/analyze-1440x900.png) |
| Analyze, recorded sample output | 1440×900 | [Saved prediction](artifacts/refinement/screenshots/analyze-saved-1440x900.png) |
| Analyze, new classification | 1440×900 | [Explicit example output](artifacts/refinement/screenshots/analyze-classified-1440x900.png) |
| Analyze, recorded output | 1280×800 | [Intermediate desktop](artifacts/refinement/screenshots/analyze-saved-1280x800.png) |
| Analyze, new classification | 1024×768 | [Compact desktop](artifacts/refinement/screenshots/analyze-classified-1024x768.png) |
| Videos | 1024×768 | [Compact library](artifacts/refinement/screenshots/videos-1024x768.png) |

Captures are actual browser pixels and remain ignored locally because the reference
footage is not distributed through Git.

Observed: native file upload and local decoding, reopening browser-stored media,
playback, source-frame stepping (frame 27 → 28), pointer-drag selection, bounded replay,
saved-output reopening, example classification without navigation, stale-output clearing,
and responsive panel/transport layouts. A replay race found in the browser was fixed:
late compositor callbacks now remain within the exclusive end boundary. The recorded
26–35 selection replays to source frame 35 at 1.417 seconds, before its 1.459-second
exclusive boundary.

Automated verification: 11 tests, TypeScript, Prettier and production build. Tests cover
source timestamps, non-crossing keyboard handles, late replay callbacks, provenance,
simulation disclosure, disabled/busy classification, stale-output clearing, separate
navigation, dialog focus restoration and absence of production API calls.

## Remaining UX issues

1. Safari can produce black upload posters, including after seeking before capture.
   The library now substitutes a recognizable video-file icon for unreadable/black
   thumbnails. The initial local-video frame can also be black until playback starts.
   Production should use pipeline-generated thumbnails and a verified preview.
2. Exact source-frame stepping is available for the indexed reference sample only.
   Arbitrary uploads use browser time seeking and a neutral timeline track rather than
   fake filmstrip frames. Production needs the real frame inventory for every upload.
3. On shorter desktops, quality/model details need panel scrolling after a prediction.
   The primary action remains fixed. This is an intentional density tradeoff to review.
4. Leaving Analyze for Videos resets the unsaved range/playhead; saved selections reopen
   correctly. Restoring the active workspace across navigation remains a follow-up.
5. Large libraries need naming, search and removal controls. Duplicate uploads currently
   appear separately. Mobile, screen-reader, Windows/Firefox and portrait-media review
   remain outside this desktop prototype pass.

## Preservation

Production files are unchanged. A fresh before/after comparison verifies the 4,787
protected dataset, review, model and experiment paths against this refinement's starting
snapshot. V9 source material is not opened, decoded or hashed. Media, screenshots and
verification receipts stay ignored. No model execution, production API request, push,
deployment or production migration is performed.
