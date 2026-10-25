# Boxing Motion AI Dashboard v0.1

Local read-only development dashboard, implemented 2026-10-25. It builds on the
[P1 services](DASHBOARD_SERVICES.md). No training, inference, annotation editing,
database, authentication or cloud deployment is introduced.

## Launch

From the repository root, install the locked Python dashboard extra and build the web app:

```sh
uv sync --locked --extra dashboard
npm --prefix web ci
npm --prefix web run build
uv run --locked --extra dashboard python -m boxing_motion_ai.api
```

Open **http://127.0.0.1:8000**. The launcher binds only to loopback and serves the
compiled `web/dist` frontend. Ctrl-C stops it. The existing `boxing-motion-ai` CLI
is unchanged. To retain all research/test dependencies in the environment, include
`--extra pose --extra train` in the sync/run commands as well.

For frontend development, keep that backend running, then in another terminal:

```sh
npm --prefix web run dev
```

Open **http://127.0.0.1:5174**. Vite proxies `/api` to port 8000; port 5174 avoids the
other local project's port 5173. Both servers use strict fixed ports and loopback.
`npm --prefix web run preview` uses port 4174 with the same API proxy. No CORS-wide
access or arbitrary host override is configured. Node 22.12+ is required (Node 22
in CI; local verification used Node 26). Python remains 3.11.

The dashboard requires the registered local metadata, approved prepared pose files
and original MKV sources already obtained for the project. It never downloads
media or models. Missing/stale artifacts appear as errors, not fabricated examples.
Runtime frame decoding uses the FFmpeg libraries in PyAV; a separate FFmpeg CLI
installation is not required for these routes.

## What is available

- **Review Queue:** all 83 existing ledger cases, including the six priority disputes
  and twelve targeted anchors; filter by recording, state, or source row/class text.
  The original 79-case diagnostic review plus four precautionary neighbors remain
  distinct from human-approved annotations. Predictions are unavailable for neighbors
  without diagnostic evidence; they are never invented.
- **Recording viewer:** actual decoded source frame, integer frame stepping, arrow-key
  shortcuts, annotation navigation, start/middle/end and ±10/30/60 contextual frames.
  The optional skeleton displays only valid landmarks and edges. Missing coordinates
  serialize as null, not invented positions. Confidence fields and masks are preserved.
- **Details:** source class, provisional canonical label, raw/mapped boundaries, end
  assumption, source uncertainty, latest review attribution, existing frozen prediction
  and class probability with its fold/source provenance. Probability is not a calibrated
  measure of annotation correctness.
- **Dataset:** six registered recordings, existing mask-aware class counts, separate
  training/development roles, source-group guards and human review progress. The
  existing 1,156-example training selection is distinct from the 1,125-example curated
  proposal. V5 retains its 583 development examples. Missing statistics withhold totals.
- **Experiments:** preserved velocity, multi-recording and RGB study summaries and
  stored fold/seed/model metrics. Historical negative results remain visible.
- **Models:** twelve existing reference/control metric artifacts across three seeds,
  train/development scores, per-class precision/recall/support and confusion matrices.
  No checkpoint is loaded or model executed.

The 83-case ledger is the initial UI navigation scope, not every annotation in the
dataset. The paginated API annotation lookup remains available for later navigation
expansion. Current recording registry remains V4/V5/V6/V7/V8/V10; V9 is rejected
before source access. Unknown boxer/session identities stay unknown.

## Exact-frame correspondence

`api/media.py` is separate from metadata-only queries. A frame request:

1. Resolves a registered recording through the existing curation configuration and
   pinned preparation metadata. Callers provide only recording IDs/native indices.
2. Opens fixed source paths with the shared no-symlink regular-file reader. Verifies
   the prepared NPZ and original video SHA256 against existing pins. Source verification
   is reused only while device/inode/size/mtime/ctime signatures remain unchanged.
3. Loads the existing native-index/PTS/timebase arrays with `allow_pickle=False`,
   verifies contiguous indices and strictly increasing rational timestamps, and preserves
   masks/coordinates/confidences. At most two recording timelines stay in memory.
4. Uses PyAV/FFmpeg to seek to a preceding keyframe and decode forward. It **requires
   exact rational PTS equality** with the requested indexed frame. Nominal FPS,
   browser playback times, V5 helper offsets and nearest-frame substitution are not used.
5. Returns a resized JPEG (maximum 1280 pixels) plus the exact source index, integer
   PTS string, rational timebase/time strings, scene/segment IDs and unmodified pose
   values. Resizing is display-only; timestamps/frame ordering are unchanged.

Pose overlays use existing `vision.contracts` joint names/edges and existing masks.
They are not newly estimated, smoothed or interpolated. Review can step across a
scene boundary; scene/segment IDs and known cut markers remain visible. No pose
processing crosses it. Context is explicitly labeled before/after the punch and
never assigned `no_punch`. Invalid/reversed/unresolved intervals are withheld from
the viewer without altering the original source row.

Generated JPEGs and receipts live only in **`.dashboard-cache/`**, ignored by Git.
Cache keys include source video/pose hashes, native index, exact timestamp and renderer
version. JPEG receipts are checked on reads; stale/corrupt sources or previews fail
visibly. Writes are atomic, and symlinked cache directories/files are refused. Cache
misses are serialized to one decoder with two decode threads and a bounded decode
window. First access may take seconds while large source files are hashed. Cache
size is not automatically pruned in v0.1; original reports/previews are never removed.

## API and safety boundary

Run `python -m boxing_motion_ai.api --help` for the trusted `--root` application
option. There is deliberately no host option. The app rejects nonlocal Host/Origin
and cross-site browser requests. This is a local development boundary, not an
authenticated service suitable for network deployment.

Read endpoints: `/api/health`, `/api/inventory`, `/api/statistics`, `/api/reviews`,
`/api/curation`, `/api/recordings/{id}/annotations`, `/api/cases/{id}/{row}`,
`/api/experiments`, `/api/experiments/{id}`, and
`/api/experiments/{id}/metrics/{metric_id}`.

Frame endpoints: `/api/recordings/{id}/frames/{index}` (metadata) and its `/image`
suffix (JPEG with `X-Native-Frame` and `X-Exact-Timestamp` headers).

Existing query envelopes distinguish missing (404), invalid (422), stale (409) and
incomplete metadata (200 with explicit status/individual errors). No write endpoints
exist. Only `web/dist` is statically served; raw data, reports, checkpoints and cache
folders are not mounted. Preserved paths returned as provenance are not asset URLs.
No training module is imported to list metadata or decode review frames.

## Verification

```sh
uv run --locked --extra pose --extra train --extra dashboard pytest --require-no-skips
uv run --locked --extra dashboard ruff check .
npm --prefix web test
npm --prefix web run typecheck
npm --prefix web run build
npm --prefix web run format:check
```

Backend tests generate a tiny synthetic variable-timing MKV and compare its decoded
PTS/pixels to API responses. They cover invalid IDs/V9, missing media, changed-source
hashes, exact-PTS failures, cache reuse/corruption, symlinks, immutable sources, masks,
HTTP status mapping, origin/host guards and rejection of write methods. Frontend
component tests use explicitly synthetic fixtures only in test files; the running
application always uses project APIs. Tests cover labels/probabilities/uncertainties,
frame stepping, filtering, failed-image states, valid-only overlays and boundary
conventions. CI has base Python, full Python with dashboard extras, and frontend jobs.

Real-data verification:

| Case | Native frame | Exact time | Evidence |
|---|---:|---|---|
| V4 row 497 | 21695 | `452431/500` seconds | API decode matches the preserved diagnostic frame/PTS pair. |
| V4 row 493 | 21662 | `451743/500` seconds | Browser viewer displays source frame and 32/33 valid joints. |
| V5 row 262 | 5377 | `179413/1000` seconds | API decode matches the preserved diagnostic frame/PTS pair. |
| V5 row 262, next frame | 5378 | `89723/500` seconds | Browser stepping displays the next native frame and matching timing. |

These checks establish display correspondence, not the truth of punch labels or
boundaries. The curation/human-review blockers remain unchanged. Original annotations,
source media, pose arrays, experiment artifacts and the reserved V9 set are preserved.

## Deliberately incomplete

No review decisions can be edited or saved. No full-speed video playback, audio,
new pose extraction, model inference/training, new experiment launch, dashboard job
system or Level 2 pipeline is implemented. Full annotation browsing, write concurrency,
and any authenticated/network deployment are separate milestones.

Final checks: **352 Python tests passed**, zero skips with full extras; base install
**160 passed, 69 expected optional skips**; **8 frontend
component tests passed**; Ruff, TypeScript, production build, formatting and CLI help
passed. An upstream Starlette warning notes that its httpx TestClient adapter is
being deprecated; the locked tests pass without suppressing it. No remote CI run
is claimed. Native-browser checks also confirmed dataset totals, review progress,
model metrics, V4/V5 display, frame stepping and overlay toggling.

Implementation references: [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
and [Vite setup](https://vite.dev/guide/). Dependencies are pinned in `uv.lock` and
`web/package-lock.json`; these references are not runtime network dependencies.
