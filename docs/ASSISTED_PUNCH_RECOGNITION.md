# Milestone 1: Assisted Punch Recognition

Later addition: [continuous uploaded-video activity scanning](UPLOADED_ACTIVITY_DETECTION.md)
now runs the separate binary smoke model without manual selection. The six-class workflow
described here remains assisted; the original milestone scope below is historical.

Analyze is the default local application. Upload a short continuous shadowboxing video,
select one punch, and run an existing six-class model. Review, Dataset, Experiments and
Models remain under Development, with their existing workflows and preservation guards.
No automatic detection, suggestions, statistics, coaching, training or architecture change
is included.

## Run locally

From the repository root, with Python 3.11, Node >=22.12 and FFmpeg/ffprobe installed:

```sh
uv sync --locked --extra dashboard --extra pose --extra train
npm --prefix web ci
npm --prefix web run build
uv run --locked --extra dashboard --extra pose --extra train python -m boxing_motion_ai.api
```

Open http://127.0.0.1:8000. Restart an older running backend before using the new upload
routes. If port 8000 is occupied, add `--port 8001` and open http://127.0.0.1:8001.
Use one analysis server per checkout. This is a loopback-only local application, not a
public upload service. Frontend development still uses port 5174 and the port-8000 backend.

The existing checkpoint, its provenance, and `weights/pose_landmarker_full.task` must
be present locally. They remain ignored and are neither downloaded nor committed by
this feature. Missing or changed artifacts produce an error rather than placeholder
predictions. No cloud service or external API is used for analysis.

## Use Analyze

1. Upload MP4, MOV, M4V, MKV or WebM footage: one boxer, continuous take, visible arms
   and hips. Limits: 30 seconds, 150 MiB, 60 FPS, 1800 decoded frames, dimensions
   32–4096 pixels per side. Filename extension and decoded container/stream are checked.
2. Wait for decoding and pose extraction. Status shows the current stage and frame
   progress. The result survives browser refreshes and normal server restarts.
3. Use Play for smooth buffered playback. Pause, seek with the Video timeline, or use
   the previous/next source-frame buttons for exact inspection. Set punch start and
   end on displayed frames. The UI numbers frames from 1; the end frame is included.
4. Choose one complete punch spanning at most three seconds between boundary frames.
   Classify selected punch runs the pinned model. Replay selection plays only that
   range and returns to its exact last frame.
5. Inspect all six uncalibrated scores. Select another punch without repeating pose
   extraction. Saved selections can be reopened from the Selections list.

Scores describe competition among six known punch types, not the probability that a
prediction is correct. The classifier has no No punch class and cannot verify that the
selection contains a punch. Single-person framing, continuous footage and punch selection
are user requirements; no automated person-count, cut, stance or punch-occurrence check
is claimed. Predictions are experimental, particularly for new boxers and views, rear
hooks and uppercuts. No accuracy estimate is generated from an unlabeled upload.

## Local files and cleanup

All upload-specific files are under ignored `.analysis-local/<random-id>/`:

- Original `source.video` bytes and source SHA256.
- Silent H.264 buffered preview and its hash.
- Lossless PNGs of resized, oriented decoded frames, indexed in presentation order.
- Exact source PTS/timebases, frame-to-preview mapping and media metadata.
- One cached `poses.npz` plus pose-model, joint-order, validity and source provenance.
- Per-selection prediction JSON, processing status/progress and diagnostic worker log.

Remove local video deletes that upload and all its derived files/results. It cannot
remove research recordings, reviews or models. Removal is blocked while processing or
classification uses the upload. There is no automatic expiry. At most four saved videos
and two pending uploads are allowed; processing uses one background worker. Decoded PNGs
have a 768 MiB per-video budget. A failed job remains visible for explicit removal.
An interrupted upload is removed; interrupted processing is shown as failed after restart
and must be removed/reuploaded. No misleading resume or completed state is invented.

For a complete reset, stop the analysis server and remove only `.analysis-local/`.
Never use this cleanup instruction on `reports/`, `data/`, `weights/` or review ledgers.
The generated frames can use substantially more disk space than the compressed input.

## Model and input contract

`configs/assisted_level1_v1.json` pins the accepted original reference:

- `level1-velocity-v1/seed-20261021/xy_velocity/tcn/best.pt`, epoch 32.
- Checkpoint SHA256: `bd314c2f608a0f73b5b13bca912510a513e7491c6062a8c73446404f146bdbfa`.
- Original V4 training, V5 development selection; no new selection, ensemble or promotion.
- Feature schema `torso-core-2d-velocity-1.0`, 65 channels, unchanged CausalTCN and pooling.
- Fixed class order: jab, cross, lead_hook, rear_hook, lead_uppercut, rear_uppercut.

The loader verifies checkpoint/provenance hashes, saved feature/configuration metadata,
state shapes, seed/epoch and unchanged source-code hashes for the reused feature,
quality, architecture and pose contracts. MediaPipe weights are separately pinned.
Inference is CPU/eval mode with deterministic Torch operations and two threads.

`analysis/worker.py` calls the existing MediaPipe backend in IMAGE mode at threshold .5
and the existing pose-record validity contract, without a workbook. Each video is
extracted once. `analysis/inference.py` reuses `training/data.features`,
`training/velocity.joint_velocities`, the existing mask-aware70 policy and `CausalTCN`.
Clip velocities start invalid at the selected first frame, matching training. Exact
elapsed source time is retained; masks/coordinates are never filled or promoted.

Selections require at least two frames, the existing quality gate, reliable torso
normalization, and no declared scene/segment or excessive timestamp gap. Failed gates
return unavailable without scores. The frame-index range is checked before accessing
cached arrays. Source and cached-pose provenance are checked separately from model scores.

Media preparation and MediaPipe run in separate bounded subprocesses to avoid the
existing macOS FFmpeg-library conflict. Workers have timeouts. Decoding validates every
frame and strictly increasing PTS; dimensions cannot change mid-video. Quarter-turn
rotation metadata is applied before both pose extraction and preview creation; reflected
or arbitrary-angle display transforms are rejected. The preview is decoded independently
to verify exact source-relative PTS correspondence and final duration. No nominal-FPS
extrapolation or frame interpolation supplies missing timestamps. Unsupported timing
fails with an error. Pixel dimensions are reduced for display/inference, not presented
as original-resolution imagery. Original uploaded bytes remain intact.

The isolated upload adapter does not relax historical PoseClips/ReviewedWindows gates.
Uploads and predictions never enter annotation ledgers or acquire training/evaluation
roles. V9 remains reserved; V1/V6 and V7/V8 restrictions and all historical holds remain.

## Verification

Read-only compatible-reference regression:

```sh
.venv/bin/python -m boxing_motion_ai.analysis.verify_reference \
  --output reports/boxingvi/assisted-reference-replay-new.json
```

Choose a fresh output filename. This reads existing V5 prepared poses and original
predictions, not V9 or new training material. All **583** reference predictions match
by class. Maximum softmax-score difference is **4.1723e-7** against a **1e-6** tolerance
(single-selection versus original batched inference); repeated inference is exactly equal.
This is adapter compatibility evidence, not independent generalization evidence.

Actual Chrome verification used the production build and a new six-second upload cut
from existing V6 shadowboxing footage. It required no workbook: 144 frames processed,
100% torso/one-arm mask coverage, about 7.3 seconds for poses and 13.9 seconds total on
this machine. These timings and coverage are not recognition accuracy. Native Chrome
upload, progress, full buffered playback, timeline seeking, individual-frame stepping,
boundary selection, trained-model scores, saved selections after refresh, and bounded
replay were exercised. A later selection reused unchanged pose bytes/mtime; repeated
requests returned identical results. No human review or verification was saved.

Live invalid-file and >30-second uploads produce useful errors. A real MediaPipe pass
on a no-person clip produces zero pose coverage and an unavailable classification,
without invented scores. Those three temporary verification uploads were removed through
the scoped cleanup API. Rotation was also compared to FFmpeg autorotation on an asymmetric
90-degree metadata fixture, with identical oriented pixels. Synthetic tests cover all
four quarter-turns, VFR/nonzero PTS, decode/frame/storage limits, quality gates, model pin
changes, source bounds, concurrent requests, local request protections and cleanup.

Private receipts are under ignored `reports/boxingvi/milestone1-verification/`. This
includes reference replays, rotation, real inference/failure checks and a preservation
snapshot. The existing 4,787 protected inputs are verified byte-for-byte unchanged.
No V9 file was opened, decoded or hashed. Real-data checks require the local artifacts;
CI tests use generated media and synthetic weights instead of private data.

On this macOS installation, real MediaPipe execution needs a normal local process;
the terminal sandbox blocks its graphics-context initialization even with CPU inference.
The live-server checks were run outside that sandbox. The application isolates a worker
failure and reports it rather than allowing it to crash the server.

**604 Python tests and 92 frontend tests pass.** Ruff, changed-Python formatting, TypeScript, frontend
formatting and production build pass. Repository-wide Ruff formatting still reports the
pre-existing quote-style difference in untouched `src/boxing_motion_ai/cli.py`; it was
left unchanged. The existing Starlette/httpx deprecation warning remains.
