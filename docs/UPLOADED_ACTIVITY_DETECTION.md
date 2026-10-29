# Continuous uploaded-video activity scanning

Implemented locally on 2026-10-29. This extends the production Analyze upload workflow;
the separate frontend prototype is unchanged.

The result is **experimental possible punching activity**, using the existing training-only
binary smoke model. An activity bout may contain several punches. These are model predictions,
not verified human annotations, punch counts, automatic six-class recognition, calibrated
confidence or demonstrated detection accuracy. No activity output is a background certificate;
unobservable time is reported separately.

## Run and use

Use Python 3.11, Node >=22.12, FFmpeg/ffprobe and the existing local weights:

```sh
uv sync --locked --extra dashboard --extra pose --extra train
npm --prefix web ci
npm --prefix web run build
uv run --locked --extra dashboard --extra pose --extra train python -m boxing_motion_ai.api
```

Open http://127.0.0.1:8000, or use `--port 8001` if necessary. Restart an older backend.
Run one server per checkout. No external inference API or cloud service is used.

1. In Analyze, upload a continuous, single-boxer shadowboxing recording (existing limits:
   30 seconds, 150 MiB, 60 FPS, 1800 frames). Wait for preparation and pose extraction.
2. Choose **Scan for activity**, immediately below the video timeline. Neither manual punch
   boundary is required. The player remains usable while cached poses are scanned.
3. Click a returned time/frame range to inspect its exact first source frame. Play uses the
   existing buffered preview. A range does not populate manual punch boundaries or invoke
   the six-class classifier. Boundary interruptions/censoring are identified.
4. Expand **Unobservable intervals — not background** when present. Observable-frame counts
   and scan wall time measure processing/pose coverage, not recognition accuracy.
5. Existing manual start/end selection, classification, replay and saved predictions still work
   independently. Results are scoped to the chosen upload and reload when reopening that video.

Existing local-video removal deletes only that upload's source, poses, preview and generated
classification/activity artifacts. It remains blocked during active work. Everything lives in
ignored `.analysis-local/<opaque-upload-id>/`; no upload or generated result belongs in Git.
Existing four-video storage limits still apply. There is no broad dashboard/navigation change.

## Frozen inputs and source restrictions

`configs/uploaded_activity_v1.json` explicitly pins:

- `reports/boxingvi/detector-smoke-v1-verified/checkpoint.pt`, SHA-256
  `b24f527e6f50a883a1c2bd6ae8c054f029aefba07ce2f23ee8d2acbe1655fd63`.
- Existing binary TCN architecture, 65-channel `continuous-xy-velocity-1.0` features,
  V4-only smoke training configuration, seed 20261021, epoch 30, held V4-bg-4/5 intervals,
  and training dataset fingerprint. Checkpoint metadata and strict state-dict loading are checked.
- Original torso normalization, validity masks, backward velocity computation, model code,
  pose contract, decoder code/config and activity evaluation protocol by hash.
- Existing MediaPipe full pose task SHA and joint order; cached IMAGE-mode poses must use
  the existing 0.5 visibility/presence threshold and match the uploaded source and pose hashes.

The feature/model/decoder implementations and thresholds are unchanged. The new adapter reads
prepared upload artifacts only, without a BoxingVI workbook, registered recording lookup,
review ledger or training dataset access. Historical `ReviewedWindows` restrictions remain intact.
No V9 media was opened; no reserved development/evaluation recording was used to fit or tune
anything. Upload predictions do not acquire dataset roles or enter human-review records.

## Timing, streaming and grouping

Preparation already rotates footage consistently, validates decoding and monotonic native PTS,
and creates an independently checked buffered preview with the same source-relative clock.
The scan checks every cached pose index/PTS/timebase against `video.json`, including exact rational
source timestamps, player-relative times and the known exclusive coverage endpoint.

The original source encoder uses twice the median adjacent timestamp spacing as fixed
full-prepared-source clock metadata. This is retained, rather than estimating an FPS from pose
motion. This is **prepared-video streaming**, not an online camera-clock estimator: pose features
are per-frame, velocities use only the preceding valid observation, and model/decoder decisions
use no future pose or probability inputs. Rechunking the same prepared recording preserves state.

Model history and derivatives reset at missing/unreliable torso poses, pose recovery, scene/segment
changes, declared cuts and timestamp gaps. The effective maximum gap is the smaller of the
original source-clock rule and the frozen decoder's 1/10 second. No inference occurs on
unobservable frames. Partial joint/arm failures use the original feature masks; observability is
**the existing reliable torso/pose-status gate**, not a guarantee that a complete punch is visible.
There is no automatic scene-cut or multi-person detector; continuous single-boxer footage remains
a user requirement. Declared discontinuities are honored when present.

The frozen decoder keeps activation 0.65, deactivation 0.35, activation duration 2/25 second,
low termination duration 2/25 second and maximum adjacent gap 1/10 second. Its legacy `reviewed`
input gate is set internally to allow the prepared upload stream; this does **not** assert human
review. Exported upload records explicitly say `human_reviewed: false`.

The new causal grouping wrapper implements the already frozen activity protocol: merge events
whose exact inter-event pause is **strictly shorter than 1/4 second**, within a contiguous observed
segment only. Preserve the first onset and actual first emission, and the last event end. Await
the quarter-second silence, including any qualifying pending activation, or censor at interruption/
EOF. Never bridge missing observations, timestamp gaps or resets. Consecutive punches may form one
bout. Neither splitting nor threshold tuning is added.

End time is the first excluded source boundary. When it is unknowable across a gap, its exact
value stays `null`; the UI displays the last included observed timestamp with a boundary warning,
without extrapolating a duration. Gaps and long unobserved tails have explicit open-time-support
annotations rather than fabricated frame indices. Source-time first emission is distinct from
backdated estimated onset and from operational wall time. The synchronous finished scan reveals
the full record, including its causal activation/finalization trace; it is not live browser telemetry.

## Record, API and cache

- `POST /api/analyze/videos/{id}/scan`: existing local-origin/JSON protection; returns 202 while
  processing or 200 with a reusable completed result. The same active request is idempotent.
- `GET /api/analyze/videos/{id}/activity`: not_started, processing, ready with result, or failed
  with a retry message. Upload IDs are opaque, validated and storage-scoped.
- `activity-<input-fingerprint>.json`: `uploaded-activity-detection-1.0` deterministic record.
  Includes grouped bouts, raw frozen-decoder events, immutable activation/finalization snapshots,
  exact source points and exclusive stops, actual first emissions, censoring/interruptions,
  unobservable intervals, resets, per-observation uncalibrated model scores, feature hash,
  checkpoint/dataset/pose/feature/decoder/grouping provenance and batch/stream checks.
  Frozen decoder fields named `confidence` remain its raw, explicitly uncalibrated summaries;
  they are not accuracy or correctness probabilities.
- `activity-state.json`: operational request ID, state, artifact SHA and scan wall seconds.
  Wall time is deliberately excluded from the reproducible inference record. Pose extraction
  time remains in the existing separate `pose.json`.

The fingerprint binds the upload ID, source bytes, poses, preparation/pose metadata, frozen pin,
new inference/grouping implementation and Python/Torch/NumPy/platform runtime identity. CPU
inference uses two threads and deterministic Torch algorithms. Cache reads revalidate bindings,
result hash, schema and video identity. Changed inputs/weights/code, stale or cross-video results,
worker failures/timeouts, corrupted state and interrupted jobs fail closed and can be retried;
missing/changed pinned weights have no substitute. Duplicate clicks do not queue duplicate scans.
Scans run in an isolated bounded subprocess on the existing local executor without extracting
poses again. Existing manual classification is blocked during scans and vice versa. A retry can
replace an owned failed generated result, never a research artifact. This is a single-server local
cache, not distributed job coordination or a portable cross-hardware bitwise reproducibility claim.

## Demonstrated verification

On this local macOS/CPU environment:

- A fresh browser upload of the existing permitted six-second V6-derived diagnostic MP4
  (`source_sha256=e33fe92864e112fc319acb5b534438b2374d84ec74fe2b67cdac0826bc220002`)
  prepared 144 native frames without annotations or a workbook. All 144 met the existing
  observability gate. The real pinned detector scanned the full recording and emitted:

  | Zero-based source range | Source-relative seconds | Actual first emission |
  | --- | --- | --- |
  | [32, 41) | 1.334409–1.709712 | 1.417810 s |
  | [105, 119) | 4.378530–4.962334 | 4.461931 s |

  This is an integration smoke demonstration on previously used diagnostic footage, **not an
  independent or annotated accuracy evaluation**. No correctness claim is made about either range.
- Scan wall time was approximately 2.64 seconds, separate from 12.39 seconds of pose extraction.
  Streaming/chunk sizes 13 and 64 gave identical records; independent recomputation was byte
  identical. Maximum streaming-versus-full-segment model difference was 2.39e-7; decoder batch
  and stream events/updates matched exactly. Repeated POST returned the cached artifact without
  changing its mtime or the cached pose bytes/mtime.
- A genuine one-second no-person upload ran through real MediaPipe: 0/25 observable frames,
  no activity bouts, and an explicit [0, 1 second) unobservable interval. It was not called background.
- Safari end to end: native file upload, preparation status, scan without selection, processing
  state, returned ranges, clicks to displayed frames 33 and 106, buffered playback to EOF,
  frame stepping, independent manual boundaries/classification, reopening cached results,
  switching videos without carrying ranges, and the unobservable result. The manual classifier
  returned its real six scores for the separately selected [105,119) range; no automatic type
  identification was performed. An input-change failure was displayed and successfully retried.
- Synthetic tests cover no activity, one/multiple bouts, strict quarter-second boundaries,
  merged consecutive events, actual emission timing, immutable emitted prefixes, exact nonzero
  source origins, chunking and future-motion invariance, all-missing poses, gaps/cuts/scenes/
  segments, mismatched clocks, pin corruption/missing checkpoint, stale/cross-video caching,
  duplicate requests, worker failures/retry, busy guards, interrupted/corrupt state and disk
  failure lock release. Existing manual classifier/media/API tests remain passing.

Verification receipts and browser captures are local-only under ignored
`reports/boxingvi/upload-activity-verification-v1/`. Existing research artifacts were compared
against a pre-change 4,797-file hash inventory excluding V9; none changed. No screenshots,
generated inference output, checkpoint or dataset are committed.

Checks: **622 Python tests passed**, Ruff check and changed-file Ruff formatting passed;
**95 frontend tests passed**, TypeScript, Prettier and production build passed.
Frontend tests pass with one worker; an earlier parallel run under concurrent test/build load
hit two existing review-screen timing timeouts, and the complete serial run passed. Python
reports the existing FastAPI/Starlette httpx deprecation warning.

Still unvalidated: detection precision/recall, false alarms, missed activity, boundary accuracy,
latency against human references, transfer to new boxers/cameras/lighting, combinations beyond
this smoke clip, multi-person/sparring/bag footage and general camera discontinuities. No held-out
pilot inference, scoring, new training, threshold tuning, statistics or coaching was performed.
