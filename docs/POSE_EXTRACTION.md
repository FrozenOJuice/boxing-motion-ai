# V5 pose extraction — initial Phase 1 slice

The user authorized V5 raw annotation frame numbers as a **provisional** mapping. This enables
pose-quality investigation, not model training or a claim that the dataset gate is verified.
The pipeline follows the [specification](PROJECT_SPEC.md): sequential decoder → single-boxer
pretrained pose estimator → versioned pose records → private visual overlays. There are no
features, punch classifiers, training jobs or inferred background labels in this slice.

## Reproduce the quality sample

```sh
uv sync --locked --extra pose
uv run --extra pose boxing-motion-ai extract-poses --help
```

Provide the original local `V5.mkv`, `V5.xlsx` and Google's **Pose Landmarker Full** task bundle.
Model source: [MediaPipe model overview](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models).
The locally tested version is the version-1 float16 bundle:
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`.
Store it under `weights/pose_landmarker_full.task`; the CLI never downloads anything. Keep weights
and all third-party review images private while usage/redistribution rights remain unverified.

```sh
uv run --extra pose boxing-motion-ai extract-poses \
  --video data/raw/boxingvi/RGB_videos/V5.mkv \
  --workbook data/raw/boxingvi/Annotation_files/V5.xlsx \
  --model weights/pose_landmarker_full.task \
  --config configs/v5_pose_review.yaml \
  --output data/processed/boxingvi/V5/pose-review-reproduce
```

Choose a new output directory for each run. Config keys are validated; misspellings fail.
Explicit CLI options override YAML. The checked-in YAML selects short windows covering the
18 alignment-review source rows, all six literal punch names, flagged rows and the transition
between decoded frames 10678/10679. These windows are **not training clips** or verified events.
For a full-video extraction, omit `--config` and `--range`; full extraction is a separate run.
Repeated `--range START:STOP` options select zero-based, half-open *extraction* ranges. This says
nothing about the annotation authors' end convention. No frames inside a requested range are
skipped. Decoder numbering always begins at source frame zero, including when earlier frames
are outside the requested ranges. Decode remains sequential, so late sample windows incur
full-prefix decode cost.

On this macOS host, MediaPipe's CPU delegate still needs access to graphics initialization.
The restricted sandbox caused a native abort; running the same task with normal OS access
succeeded. A CPU delegate does not guarantee this package can initialize in a headless sandbox.
PyAV decoding runs in a local subprocess to isolate its FFmpeg libraries from the different
FFmpeg libraries imported by MediaPipe/OpenCV. This resolves duplicate native-class warnings;
it is not a network service. `MPLCONFIGDIR` can point to a writable cache if required.

## Outputs and pose contract

- `manifest.json`: status/scope, source and model SHA-256, code hashes, Git revision/dirty flag,
  dependency/platform versions, resolved settings, schema and pose quality counts. Pinned
  dependencies and hashes support reproduction; cross-platform floating-point identity is
  not promised. A sample run does not claim a known complete source frame count.
- `poses.jsonl`: one record per requested frame, in source presentation order. Each stores
  exact integer `pts` and rational `time_base`, zero-based decoded `frame_index`, convenience
  seconds, estimator milliseconds, source/inference dimensions and range `segment_id`.
- `annotations.provisional.json`: all nonempty V5 rows, original A/B/C and D/E formulas,
  questionable-row flags and explicit unknown author FPS/base/end/trim conventions. Whole-number
  Excel floats (e.g. `239.0`) retain their raw values and additionally get integer candidate
  indices using the existing audit validator; fractional cells are rejected, not rounded. No raw
  value is repaired, canonical class mapping adopted, or unlabeled interval called `no_punch`.
- `overlays/*.jpg` and `skeleton-contact-sheet.jpg`: downscaled source pixels with skeletons,
  exact frame indices/PTS, pose quality and **unverified** raw-index candidate labels. The
  display candidate uses inclusive endpoints only to surface review candidates conservatively.

`pose-1.0` fixes the 33 MediaPipe landmark names/order in `vision/contracts.py`. Coordinates
are raw normalized image x/y and MediaPipe's hip-relative z in image-x scale units. They are
not calibrated 3D measurements. No body normalization, mirroring or world coordinates are
added. Left/right means estimator anatomy; stance, lead/rear hand and boxer identity remain
unknown. `num_poses=1` is a single-person assumption, not verified identity tracking.

Each landmark retains visibility and presence scores. A valid-mask entry requires finite
coordinates/scores, x/y inside the image and both scores at least the configured threshold
(and at most one). Finite out-of-image or low-confidence values remain in the raw record but
are masked invalid. Nonfinite/missing values serialize as null. A missing detection produces
33 null points and 33 false mask entries. No interpolation, zero-pose substitution or previous
pose carry is performed. Backend errors fail the run rather than becoming missing detections.

`ok` requires all 22 body landmarks (indices 11–32) valid. `unreliable` means a pose was returned
but at least one body landmark failed this strict check; valid joints remain available under
their individual masks. `missing` means no pose was returned. These statuses do not certify
anatomical correctness. Confident wrist swaps and occlusion hallucinations still need review.

MediaPipe uses synchronous VIDEO mode and current/past tracking. Range gaps reset the estimator
and its relative timestamp origin; sample-window results can differ from full-video tracking.
Integer estimator milliseconds are rounded from actual relative PTS. Original timestamps remain
unchanged. Nonmonotonic source PTS or milliseconds collisions fail rather than being silently
retimed. No future interpolation or additional temporal smoothing is applied.

Outputs are marked `complete` only after successful extraction, backend close and report writing.
Failures retain a `failed` manifest and `.partial` records for diagnosis, without a successful
`poses.jsonl`. A native abort/power interruption can leave `running` plus partial output; that
is not a completed run. Existing outputs are never overwritten by the extraction command.

## Required visual review before further work

Inspect skeleton attachment at shoulders, elbows, wrists, hips, knees and feet across the six
classes. Check fast extensions, crossed/occluded arms, anatomical side consistency, and whether
high-confidence points follow the correct body part. Review the cut at 10678/10679 for tracking
lag, and rows 272/323/442/536 independently of pose quality. Annotation correspondence is still
provisional; a good skeleton does not verify a punch class or its boundary. Author FPS,
zero/one indexing, end convention, trim offsets and manually reviewed onset/peak remain unresolved.

The generated run report under `reports/boxingvi/` records local measured results and links to
private media. Raw/processed data, model weights and reports remain Git-ignored. No images or
third-party annotations are committed with the implementation.

## Tests

```sh
uv run pytest
uv run ruff check .
uv run --extra pose pytest
```

Core synthetic tests need neither footage nor weights. Optional integration tests generate tiny
synthetic variable-rate media to check the real decoder, and synthetic pixels to check overlays.
They skip when the pose extra is absent. Tests cover raw-row preservation/flags, exact rational
timing, masks, range state resets, reproducibility with a deterministic fake backend, output
preservation, failed/truncated runs, millisecond collisions, config validation and CLI diagnostics.

Sources: [MediaPipe Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python),
[PyAV frame timing API](https://pyav.basswood.io/docs/stable/api/frame.html),
[original BoxingVI release](https://github.com/Bikudebug/BoxingVI).

## Full extraction and preparation — 2026-10-20

The accepted prototype is extended by [full extraction and candidate preparation](DATASET_PREPARATION.md).
Use `configs/v5_pose_full.yaml` for the full run: independent IMAGE inference, confirmed cut
metadata and explicitly selected review frames. New runs use pose schema 1.1 and default to
IMAGE mode; the earlier VIDEO sample remains reproducible through its explicit review config.
Preparation accepts only complete IMAGE runs, preventing temporal pose state across unlisted cuts.
