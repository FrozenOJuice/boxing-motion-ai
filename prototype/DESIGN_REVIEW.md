# Design review and browser verification

## Direction

**Upload** is a focused library: one upload target, real recent-video rows, and an optional
sample. There is no dashboard or internal navigation mixed into this starting point.

**Analyze** uses one compact header, a large video, a shared transport and one filmstrip
track. The playhead, range handles, time labels and replay all use the same clock.
The primary classification action sits next to the selection rather than competing
with a prediction sidebar. Controls have visible hover, disabled and keyboard-focus states.

**Results** separates the saved-selection list from the selected result. A replayable
selection precedes the prediction and all six uncalibrated scores. Source labels
separate real saved sample outputs from simulated new results. Model information is
collapsed by default. There are no decorative statistics or inferred boxing advice.

The design is new: no production layout, navigation component, viewer component or
stylesheet is imported. Development tools are outside the main flow and represented
only by an explicit informational entry in the prototype footer.

## Browser review

The first pass was inspected directly in Chrome. The final responsive pass used native
Safari with `review.html`, creating actual independent CSS viewports. Native browser
controls were used because the browser connector was unavailable. The review toolbar's
scale only fits each viewport into the host window; media queries evaluate the stated
viewport dimensions.

| Viewport | Upload | Analyze | Results |
| --- | --- | --- | --- |
| 1440×900 | [Screenshot](artifacts/screenshots/upload-1440x900.png) | [Screenshot](artifacts/screenshots/analyze-1440x900.png) | [Screenshot](artifacts/screenshots/results-1440x900.png) |
| 1280×800 | [Screenshot](artifacts/screenshots/upload-1280x800.png) | [Screenshot](artifacts/screenshots/analyze-1280x800.png) | [Screenshot](artifacts/screenshots/results-1280x800.png) |
| 1024×768 | [Screenshot](artifacts/screenshots/upload-1024x768.png) | [Screenshot](artifacts/screenshots/analyze-1024x768.png) | [Screenshot](artifacts/screenshots/results-1024x768.png) |

Additional captures: [simulated result labeling](artifacts/screenshots/results-preview-1440x900.png)
and [keyboard-help dialog](artifacts/screenshots/keyboard-dialog.png). Captures are real
browser pixels. Empty padding from the native Safari capture was trimmed for the local
exports; the app UI was not altered. Raw captures remain beside the exports in `raw/`.
Screenshots include local third-party sample footage and remain ignored rather than
being redistributed through Git.

Observed interactions:

- Native file selection, local decoding and playback of a six-second video.
- Actual recent-video persistence through a full browser refresh.
- Sample playhead seeking and pointer-drag selection on the single timeline.
- Individual source-frame stepping and I/O bounds from the keyboard.
- Bounded replay of a selected range.
- Enabled/disabled classification states and the explicitly simulated result transition.
- Real saved sample prediction versus clearly labeled prototype preview.
- Native modal keyboard focus and Escape dismissal.
- Selection metadata remaining on one line at 1024 pixels after a layout refinement.

Automated verification: **10 interaction/contract tests**, TypeScript, Prettier and the
production build pass. DOM tests support interaction checks; the viewport observations
and screenshots above came from the real browser.

The workspace was simplified after the first visual pass: duplicate vertical header
space was removed, navigation now resets scrolling, range adjustment snaps to actual
sample timestamps, and the narrow-desktop Results preview was shortened so all six
scores fit more comfortably. The token styles were consolidated after iteration.

## Remaining UX issues and review questions

1. **Safari thumbnail capture:** a newly imported clip can produce a black first-frame
   thumbnail while playback is correct. This remains a prototype issue to resolve before
   production integration; it does not represent failed ML processing.
2. **Arbitrary-upload frame precision:** browser-clock seeking is available, but exact
   stepping is intentionally restricted to the indexed sample. Production integration
   must supply the real frame inventory and corresponding stills for every upload.
3. **Results scrolling:** model details and the score explanation can require scrolling
   on shorter desktops, especially when expanded. The primary prediction and replay
   remain above them; a future review should decide whether a more compact detail
   treatment is preferable.
4. **Long libraries/selections:** search, naming, deletion, pagination and bulk actions
   are not designed here. The prototype currently targets a small local collection.
5. **Further access/device review:** desktop mouse/keyboard and focus states were checked;
   screen-reader testing, Windows/Firefox, portrait footage and touch behavior still need
   dedicated review. No formal accessibility conformance is claimed.

These are design/prototype limits, not changes or new promises about model accuracy.
New classification always stays simulated. No automatic detection or production migration
is included or authorized by this prototype.

## Preservation

Only `prototype/` is added. Existing tracked production files are unchanged. All 4,787
previously protected dataset, review, model and experiment hashes are checked against
the pre-prototype snapshot. V9 source material is not opened, decoded or hashed.
Private receipts are under `artifacts/`. No production API was called and no review
was created. No push or deployment is performed.
