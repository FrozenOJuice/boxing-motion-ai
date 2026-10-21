# Per-recording pose validation

The same extraction, preparation and fixed-mask quality functions serve inspected recordings
V1–V10. All ten local workbook layouts now have individually inspected configurations.
Timing configurations remain provisional; accepting a video ID is not evidence of a verified
frame mapping. See [EXTRACTION_RESUME.md](EXTRACTION_RESUME.md) for the expansion workflow.

`RecordingConfig` records the literal workbook header, helper formulas and class vocabulary,
an explicit integer frame offset, a provisional endpoint assumption, source URL and separate
boxer/session identities. Configured columns hold source start/end/class; the parser rejects a
changed header or width and flags inconsistent helper formulas. Capitalization aliases must
be explicit (V4 `Lead hook`); no stance or left/right-to-lead/rear inference occurs.

Annotation columns are explicit per recording: A/B/C for several workbooks, A/C/E for
V1/V9/V10. Headers, helper formulas, mirrored values and auxiliary-only rows are validated.

The current mapping supports integer source-coordinate offsets to decoded frame indices.
It does not infer annotation FPS, interpret URL `t=` as an offset, resample to nominal FPS or
apply helper formulas. A recording needing rate/time remapping must first establish the
coordinate system from evidence and extend the mapping explicitly. Do not force it into
an integer-offset configuration. Index base and end conventions remain unverified.

`recording_id` is the source V-number. `boxer_id` and `session_id` remain null without evidence.
Distinct recordings do not establish distinct boxers or sessions. Splits are unassigned;
identity uncertainty must be resolved before claiming boxer-disjoint evaluation.

## Reproduce the V4 sample

Run with the existing local footage, workbooks and model; no command downloads them.
Use new output directories when reproducing, as existing outputs are protected.

```sh
uv run boxing-motion-ai plan-pose-sample \
  --workbook data/raw/boxingvi/Annotation_files/V4.xlsx \
  --recording configs/v4_recording.yaml \
  --output reports/boxingvi/v4-sample-plan

uv run boxing-motion-ai extract-poses \
  --video data/raw/boxingvi/RGB_videos/V4.mkv \
  --workbook data/raw/boxingvi/Annotation_files/V4.xlsx \
  --model weights/pose_landmarker_full.task \
  --recording configs/v4_recording.yaml \
  --config reports/boxingvi/v4-sample-plan/pose-config.json \
  --output data/processed/boxingvi/V4/pose-sample-image-v1

uv run boxing-motion-ai prepare-recording \
  --pose-run data/processed/boxingvi/V4/pose-sample-image-v1 \
  --workbook data/raw/boxingvi/Annotation_files/V4.xlsx \
  --allow-sample --output data/processed/boxingvi/V4/candidates-sample-v1

uv run boxing-motion-ai assess-pose-quality \
  --dataset data/processed/boxingvi/V4/candidates-sample-v1 \
  --video data/raw/boxingvi/RGB_videos/V4.mkv \
  --output reports/boxingvi/v4-mask-aware

uv run boxing-motion-ai compare-pose-quality \
  --dataset data/processed/boxingvi/V4/candidates-sample-v1 \
  --dataset data/processed/boxingvi/V5/candidates-full-v1 \
  --output reports/boxingvi/v4-v5-comparison
```

The sampler chooses five equally spaced ordinal positions in each source class among
structurally eligible annotations, before reading pose quality. Coverage depends on where
that class occurs in the recording; it is not uniform random footage. Full original rows,
including excluded rows, remain in preparation output.

Sample NPZ arrays store only extracted frames, with original `frame_index`, integer PTS and
rational time base. `source_interval_half_open` holds decoded coordinates;
`array_slice_half_open` holds compact array positions. These differ for sparse samples.
`quality_assessable` requires every frame of the candidate to be extracted. Unextracted
footage is unassessed, never a missing detection or background negative. The source frame
count stays null for partial runs. Unknown-interval inventories cover only the decoded
prefix in a partial run; they do not claim full-video coverage.

Full runs use the same functions without `--allow-sample`. `configs/v4_pose_full.yaml` is
prepared for a future full run; no full V4 extraction is implied by that file. V5's confirmed
cut at 10679 is not inherited. IMAGE inference has no cross-frame state. Any later temporal
features must respect independently verified scene boundaries; an empty cut list does not
certify continuity.

The approved `mask_aware70` policy is unchanged: original masks at 0.5, all four anchors valid
on at least 95% of event frames, at least one complete arm on at least 80%, at least 70% of
core joint entries valid, and at most 10% entirely missing frames. Scores, coordinates and
masks are preserved; no interpolation or confidence lowering occurs. Retention is not proof
of accurate poses or labels. See [MASK_AWARE_POSES.md](MASK_AWARE_POSES.md).

Reports and images remain under ignored `reports/` and `data/processed/`. Source files are
read only, checksummed and never corrected in place. Licensing/redistribution is unverified.
