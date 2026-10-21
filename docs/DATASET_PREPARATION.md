# V5 full poses and provisional punch candidates

This Phase 1 slice prepares data, not a trained classifier. It retains every source annotation
and extracted frame. It does not approve source annotation accuracy, licensing or held-out splits.

## Reproduction

Use Python 3.11, the locked `pose` extra and the existing local pretrained task bundle:

```sh
uv sync --locked --extra pose
uv run --extra pose boxing-motion-ai extract-poses \
  --video data/raw/boxingvi/RGB_videos/V5.mkv \
  --workbook data/raw/boxingvi/Annotation_files/V5.xlsx \
  --model weights/pose_landmarker_full.task \
  --config configs/v5_pose_full.yaml \
  --output data/processed/boxingvi/V5/pose-full-image-v1
uv run --extra pose boxing-motion-ai prepare-v5 \
  --pose-run data/processed/boxingvi/V5/pose-full-image-v1 \
  --workbook data/raw/boxingvi/Annotation_files/V5.xlsx \
  --output data/processed/boxingvi/V5/candidates-full-v1
uv run pytest
uv run ruff check .
```

Use new output directory names when reproducing; existing artifacts are preserved. No command
fetches footage or weights. On the tested macOS machine, MediaPipe needs normal graphics access
even with the CPU delegate; see [runtime details](POSE_EXTRACTION.md). No new dependencies were
needed beyond the existing pose extra.

## Cuts and temporal state

Full extraction defaults to **IMAGE mode**, independently detecting pose on each frame. There
is no inter-frame pose tracking or temporal smoothing. No interpolation is implemented. This
prevents earlier frames from influencing a pose after any edit, including unidentified edits.
It trades some tracking speed/stability for a conservative extraction baseline.

The confirmed transition between decoded frames 10678/10679 is recorded as `scene_cuts: [10679]`
(first post-cut frame). The estimator is also explicitly recreated before frame 10679, and
`scene_id`, `segment_id` and `scene_cut_before` mark the boundary. Original PTS and decoded indices
remain unchanged. `pose-1.1` adds those fields to the original landmark contract. The list of
confirmed cuts is **not exhaustive**. A regular PTS clock does not prove visual continuity.

Preparation requires complete IMAGE-mode records, verifies cut metadata/resets, and excludes
intervals spanning a confirmed cut or entering its two-frame guard on either side. Guard size
is a documented conservative review policy, not an inferred source annotation offset. No later
smoothing/interpolation may disregard these boundaries. Unknown edits remain a visual-review
limitation for sequence content, even though independent inference has no cross-cut pose state.

The earlier VIDEO-mode prototype remains available for comparison via `v5_pose_review.yaml`.
It is not the input to this preparation pipeline. Full-run pose quality can differ from tracked
sample windows; missing/reliable status is measured again on the full run.

## Candidate and class policy

Raw A/B map provisionally to zero-based decoded frame identifiers with **no −24 subtraction**.
Preparation selects A through B inclusive (`[A, B+1)` array slices) solely as an explicit working
assumption. Author FPS/base/end convention/trim offset remain unverified and null. Raw numeric
cells, including integral floats, source class text and helper formulas are retained unchanged.
No onset/peak, identity, stance or lead/rear-hand evidence is fabricated.

Only exact source names map to provisional expected classes:

| Literal source name | Provisional class |
| --- | --- |
| Jab | jab |
| Cross | cross |
| Lead Hook | lead_hook |
| Rear Hook | rear_hook |
| Lead Uppercut | lead_uppercut |
| Rear Uppercut | rear_uppercut |

These are expected source labels, not independently verified techniques. The literal `Jab `
trailing-space variant is **unmapped and excluded**, pending an explicit reviewed alias policy.
This pipeline is separate from the verified canonical Event/audit mapping: it never supplies
an unverified convention to `audit --mapping`.

Every source row appears in `candidates.jsonl`, including exclusions. Annotation-eligible rows
must have a mapped exact class, plausible in-bounds interval, no extra source diagnostic flags,
no known-questionable row ID, and no known cut/guard intersection. Default excluded row IDs are
272/323/442/536. Reversed/equal/out-of-order bounds, duplicates, suspect overlaps, malformed helper
formulas, whitespace variants and unrecognized classes remain excluded. Conservative overlap
screening uses inclusive candidate endpoints and may exclude rows that would merely touch under
an exclusive author convention. No error is repaired and no corrected replacement is inferred.

`pose_clean_candidate` additionally requires all 22 body joints valid on **every event frame**.
Rows with missing or unreliable poses retain masks and detailed counts and can be inspected as
annotation-eligible but not pose-clean. This is a strict quality screen, not keypoint accuracy
or label truth. No confidence threshold was tuned using classification performance.

## Artifacts and loading

- `preparation.json` / `.md`: class balance, annotation exclusions, eligible rows with unreliable
  or missing pose data, strict clean counts, assumptions, hashes and split limitations.
- `poses.npz`: one shared sequence, no duplicate clip storage. `coordinates[T,33,3]` and
  `confidence[T,33,2]` use float64; invalid finite raw values are retained, absent values are NaN.
  `valid_mask[T,33]` is authoritative for later feature construction. Coordinates are normalized
  image x/y and hip-relative z, not calibrated world measurements. Confidence order is
  visibility/presence; joint order and quality status codes are in `preparation.json`.
  Integer arrays preserve frame index, PTS, rational timebase numerator/denominator, scene and
  estimator segment. `pose_status`: 0=all body joints valid, 1=unreliable, 2=missing.
- `candidates.jsonl`: source row/provenance, exclusions, quality and `array_slice_half_open`.
  Read the selected coordinate/confidence/mask/time arrays using this slice. No temporal
  resampling, padding, feature normalization or zero-coordinate filling is performed.
- `unknown-intervals.json`: complement of plausible source annotation intervals, including
  excluded rows where their bounds are usable. Those intervals remain **unknown**, never
  `no_punch`. Inside annotation bounds, labels are still provisional; coverage is not exhaustive
  ground truth. Reversed intervals are not swapped to claim coverage.

Preparation checks full contiguous records, exact increasing rational timestamps, mask/status
consistency, source workbook hash, and cut metadata before creating output. Failed writes are
marked failed. Source pose JSONL, workbook and media are read only. All generated data/media and
model weights remain Git-ignored. Only small configs, code, synthetic tests and docs belong in Git.

## Training readiness

Inspect `classes_without_pose_clean_candidates` and `training_readiness` before proceeding.
If a class has zero strict-clean examples, a clean six-class baseline is blocked; no class is
automatically removed and no threshold is silently relaxed. The report also counts examples
with all six arm joints valid, to distinguish finger/foot problems from uncertain arm positions.

These are V5-only development candidates. They can support an explicitly provisional engineering
baseline after review, but **cannot establish independent Level 1 performance**. Every row has
`split_group=V5` and `split=null`. No frames/clips are randomly distributed across splits. Other
recordings and verified boxer/session grouping are needed for independent validation/test data.
Annotation alignment, boundary definitions, technique accuracy and rights remain unresolved.

Tests cover full-frame preservation, masks, explicit API mode, estimator reset before a cut,
cross-cut/guard exclusions, exact class handling, unknown intervals, deterministic candidate
serialization, input hash mismatches and rejection of corrupt/truncated pose streams.

MediaPipe mode/API reference: [official Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python).

## Partial-pose follow-up

[Mask-aware assessment](MASK_AWARE_POSES.md) compares the strict screen against partial-joint
coverage policies. Its proposed development index is separate from `pose_clean_candidate`:
it never relabels an unreliable point as valid or changes the original preparation report.
Retained partial sequences still require masked features and independent shortcut evaluation.

## Multi-recording expansion

The V5 prototype and original policy remain documented above. Current per-recording layouts,
sparse/full preparation and restart guarantees are in [EXTRACTION_RESUME.md](EXTRACTION_RESUME.md).
See [BASELINE_READINESS.md](BASELINE_READINESS.md) for recording-specific alignment gates and
the proposed leakage-resistant baseline path. No model training was performed.
