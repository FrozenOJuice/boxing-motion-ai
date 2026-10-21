# Resumable pose extraction and multi-recording validation

The CLI uses explicit recording configurations for V1–V10. Every mapping remains provisional;
configuring raw decoded indices does not certify author FPS, indexing, trimming or labels.
The inspected layouts include A/C/E events (V1/V9/V10), blank headers (V2/V5), literal class
capitalization differences and duplicated side columns. V6's 119 counter-only rows are
preserved as auxiliary rows, not counted as annotations. Formatting to Excel's last row is
ignored when locating actual values/formulas. Helpers and mirrored columns are checked, never
silently applied or corrected. Missing source classes remain absent.

## Run/restart V4 full extraction

```sh
uv run boxing-motion-ai extract-poses \
  --video data/raw/boxingvi/RGB_videos/V4.mkv \
  --workbook data/raw/boxingvi/Annotation_files/V4.xlsx \
  --recording configs/v4_recording.yaml --config configs/v4_pose_full.yaml \
  --model weights/pose_landmarker_full.task \
  --output data/processed/boxingvi/V4/pose-full-image-v1 --resume

uv run boxing-motion-ai prepare-recording \
  --pose-run data/processed/boxingvi/V4/pose-full-image-v1 \
  --workbook data/raw/boxingvi/Annotation_files/V4.xlsx \
  --output data/processed/boxingvi/V4/candidates-full-v1

uv run boxing-motion-ai assess-pose-quality \
  --dataset data/processed/boxingvi/V4/candidates-full-v1 \
  --output reports/boxingvi/expansion/quality/V4
```

Use a new output directory for preparation/quality regeneration; these protect existing
outputs. Original media, annotations, extracted V5 poses and prior review artifacts stay intact.

## Run/restart the remaining samples

```sh
uv run boxing-motion-ai validate-pose-samples \
  --root data/raw/boxingvi --processed data/processed/boxingvi \
  --output reports/boxingvi/expansion --model weights/pose_landmarker_full.task \
  --recording configs/v7_recording.yaml --recording configs/v8_recording.yaml \
  --recording configs/v9_recording.yaml --recording configs/v6_recording.yaml \
  --recording configs/v1_recording.yaml --recording configs/v2_recording.yaml \
  --recording configs/v3_recording.yaml --recording configs/v10_recording.yaml
```

This reuses hash-matched completed extractions/preparations/reports and resumes interrupted
IMAGE journals. A failed source does not prevent the other sources from being inspected;
`batch-state.json` lists failures and the command returns a non-success status. A native
estimator crash is isolated to its recording's subprocess. Logs are under `jobs/`.

The sampler selects five ordinal positions per available class, independent of pose quality.
Where an entire source class is annotation-flagged, selected flagged rows are diagnostic only:
flags are preserved and those rows do not become clean candidates. The report separates
absent source classes, flagged-only classes, unextracted rows, missing poses and policy failures.
No gap is assigned `no_punch`.

Create readable class sheets from existing overlays, without decoding the video again:

```sh
uv run boxing-motion-ai review-pose-sample \
  --pose-run data/processed/boxingvi/V7/pose-sample-expansion-v1 \
  --plan reports/boxingvi/expansion/plans/V7 \
  --output reports/boxingvi/expansion/visual-review/V7
```

`diagnose-alignment --help` compares raw zero/one-based and declared-rate hypotheses against
prior exact ffprobe timestamps only when original input hashes match. It separates strict
interval overlaps from shared endpoints. Neither a helper rebase nor a YouTube `t=` parameter
is accepted as a video offset. PTS continuity does not establish visual scene continuity.

`summarize-datasets --help` combines prepared manifests, unchanged-policy reports and explicit
visual-review decisions. It reports full and sampled support separately and refuses stale
input hashes or changed policy thresholds. No sample retention is extrapolated to a video.

## Recovery contract and limits

- Pose results are newline-delimited records in `poses.jsonl.partial`. Every 100 results are
  flushed/fsynced; the journal is also flushed on normal exit or a handled exception.
- A restart validates complete records, exact timing, masks, recording identity, input hashes,
  settings, dependency/Python versions and pose-producing code. An unterminated crash tail is
  saved separately before truncation. Corrupt committed records cause a refusal.
- The decoder replays the prefix from frame zero to verify indices and PTS. Completed **pose
  inference is not repeated**. This trades restart decoding cost for reliable frame identity.
- IMAGE mode has no hidden temporal estimator state. Interrupted VIDEO-mode tracking cannot
  be resumed exactly and is refused. Full runs and legacy sample runs retain their decoder.
- New samples still sequentially decode/check every source frame through the last requested
  window. They skip resizing/RGB transfer outside selected windows. Selected pixels and VFR
  timestamps are tested for equality against the original decoder.
- An OS advisory lock prevents two writers to a journal. Locks release on process death.
  The present lock implementation targets this macOS/Linux environment (`fcntl`).
- Completion renames the journal and records its SHA-256. `--resume` is a checked no-op for an
  unchanged completed run. Batch reuse can retain completed immutable outputs across unrelated
  code edits; interrupted inference requires the original pose-producing implementation.
  A code/config/model mismatch is an explicit refusal, never an implicit mixed-version append.
- Keep the run's code version, lockfile and configuration to resume it later. Restoring the
  pinned version or choosing a new output directory is required after incompatible changes.

All generated data/reports/media remain ignored. No new datasets or weights are downloaded,
and no classifier is trained by these commands. Source licensing remains unverified.
