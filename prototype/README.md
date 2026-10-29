# Boxing Motion AI — isolated design prototype

A new React + TypeScript interaction design for Upload, Analyze and Results. This is
**not a production frontend migration**. No production components, styles, services,
review workflows or API routes are imported or changed.

## Run

```sh
cd prototype
npm ci
npm run prepare:demo
npm run dev
```

Open **http://127.0.0.1:5186/**. For viewport review, open
**http://127.0.0.1:5186/review.html**. The separate review page embeds the actual app
in an iframe with a real 1440×900, 1280×800, 1024×768 or 1920×1080 CSS viewport.
It scales the display to fit the browser window; it does not change the app's layout
viewport or fabricate screenshots.

`prepare:demo` copies only the existing local Milestone 1 preview and indexed PNGs
into ignored `public/demo/`. It verifies the preview hash before copying. An optional
argument selects the original upload directory:

```sh
npm run prepare:demo -- /absolute/path/to/the/reference/upload
```

The current checkout already has that local sample. No media, weights or datasets are
downloaded. Without it, local uploads still work and the sample action is unavailable.
The sample footage, generated build, screenshots and verification receipts stay ignored
by Git. Review screenshots are available locally under `artifacts/screenshots/`.

## What is real, and what is simulated

| Interaction | Behavior |
| --- | --- |
| Choose/drop footage | Actual browser metadata/codec checks and local video playback. No server upload. |
| Recent videos | Actual video blobs and thumbnails in a prototype-only IndexedDB database. Reopen works after refresh. |
| Sample playback | Actual video; source frame timestamps and indexed PNGs from the existing local upload. |
| Timeline | One source-time track for playhead and selection. Drag a range, drag handles, or use I/O. End boundaries are exclusive; replay stops on the final included frame. |
| Frame buttons | Exact indexed sample frames. Disabled for arbitrary uploads, whose exact frame inventory is unavailable to this prototype. |
| Speed / fullscreen / replay | Actual media controls, subject to browser support. |
| New classification | A deliberately simulated 450 ms UI transition; **no model request**. |
| Sample results | Two actual saved model outputs for the matching sample windows, labeled “Saved sample output.” |
| New result | Clearly marked “EXAMPLE RESULT” and “This selection was not classified.” Scores are borrowed from the sample to review the layout, not generated or asserted for the chosen movement. |
| Saved prototype selections | Actual browser-local metadata under a separate prototype storage key. They always reopen as previews, never verified predictions. |
| Model information | Read-only descriptions of the pinned reference represented by the saved sample output. |
| Development tools | Informational dialog. No connection to the existing development application. |

The sample scores in `src/demo-data.json` come from existing local outputs
`prediction-26-35.json` and `prediction-84-101.json`, with their checkpoint hash and
source timestamp mapping preserved. These are neither ground truth nor an independent
accuracy evaluation. No model, pose extraction, training or review write runs here.

## Design system

- Quiet light surfaces, dark ink and one green action/selection accent.
- Shared typography, spacing, border, radius and color tokens in `src/styles.css`.
- Shared Button, IconButton, Dialog, Icon and Brand primitives in `src/ui.tsx`.
- A shared media controller and video stage for Analyze and Results.
- A single timeline component; no second scrubber or progress bar competing with it.
- Flat lists, separators and plain model details rather than dashboard cards or metrics.
- Persistent prototype identification, specific simulation labels and no confidence claims.

Keyboard support: Space plays/pauses when the workspace is focused; arrow keys step
sample frames; I/O set bounds; F requests fullscreen. Tab reaches controls, handles
have slider semantics and arrow-key adjustment, dialogs use native modal focus behavior
and Escape dismissal. Focus outlines and reduced-motion preferences are supported.
Normal text colors were checked for at least 4.5:1 contrast on the canvas; this is not
a formal accessibility certification.

## Checks

```sh
npm test
npm run typecheck
npm run format:check
npm run build
```

The tests cover source-time/VFR selection, non-crossing keyboard handles, time formatting,
preview provenance, storage restoring only previews, explicit simulation labels, model
information and absence of calls to the production API. Browser visual verification and
remaining issues are recorded in [DESIGN_REVIEW.md](DESIGN_REVIEW.md).

Prototype storage is scoped to origin/port 5186 and names prefixed with
`boxing-motion-design-prototype`. It never reads production browser preferences or
review records. Browser site-data cleanup removes these local prototype uploads.
