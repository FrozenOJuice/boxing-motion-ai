# V6 full provisional dataset preparation

**Complete: 46,705 source frames and 302 mask-aware provisional punch examples.**
All original exclusions and the approved 15 holds are preserved. No source bounds,
labels, overlaps, helpers or suspected duplicates were corrected. No new model was
trained. V4-only mirror augmentation remains the reference, V5 development-only,
and V9 inputs were not opened. See [the approved readiness review](V6_READINESS.md)
and [the representation assessment](POSE_REPRESENTATION_ASSESSMENT.md).

## Actual yield

| Class | Source annotations | Annotation exclusions | Pose-policy rejections | Retained |
|---|---:|---:|---:|---:|
| Jab | 0 | 0 | 0 | 0 |
| Cross | 0 | 0 | 0 | 0 |
| Lead hook | 125 | 61 | 0 | 64 |
| Rear hook | 95 | 63 | 0 | 32 |
| Lead uppercut | 259 | 134 | 0 | 125 |
| Rear uppercut | 206 | 125 | 0 | 81 |
| **Total** | **685** | **383** | **0** | **302** |

The earlier ceiling is now measured retention under mask-aware70, not an assumed
80% scenario. It remains **provisional label quality**, not independently verified
ground truth. V6 supplies four labeled classes, not a standalone six-class dataset.
Visible but unlabeled straight punches are not assigned jab/cross automatically.

The 383 annotation exclusions are:

- 367 overlap-flagged source rows, including shared endpoints and the near-duplicate
  rear-hook intervals at 101/102. None restored or automatically trimmed.
- Row 3's contradictory numeric E/F helpers, retained as an unresolved source anomaly.
- Approved holds 321, 598, 620 and 675–686: 15 distinct additional rows, comprising
  eight lead and seven rear uppercuts. Pose reliability cannot clear their class/phase concerns.

Row 379 also has an out-of-row-order flag, already included in the overlap count.
Reason counts are nonexclusive; 367 + 1 + 15 = 383 unique exclusions. The structured
report lists every excluded source row, raw cells, flags and reasons. No further
annotation-eligible event fails the unchanged pose policy.

## Extraction and preservation

New artifacts, all ignored by Git:

- `data/processed/boxingvi/V6/pose-full-image-v1/`: journal, manifest, original
  annotation snapshot, 24 skeleton overlays and contact sheet.
- `data/processed/boxingvi/V6/candidates-full-v1/`: complete raw pose arrays,
  original coordinates/confidences/masks, candidates and unknown-interval inventory.
- `reports/boxingvi/v6-full-v1/quality/V6/`: per-joint diagnostics, unchanged
  policy index and representative quality-review imagery.
- `reports/boxingvi/v6-full-v1/readiness/`: structured/readable final reports,
  all exclusions, per-class comparisons and descriptive geometry diagnostics.
- `reports/boxingvi/v6-full-v1/preservation-and-sample-consistency.json`: integrity
  checks for 2,971 original/reference/sample/code files and sample/full consistency.

Every decoded index 0–46704 and its exact integer PTS/rational time base matches
the existing source inventory. Source PTS run from 0 to 1947.946 s. No frames were
dropped or resampled. IMAGE inference at width 960 and confidence 0.5 processes
frames independently: no tracking state, smoothing, interpolation or carried joints.
Raw normalized image XYZ and both confidence scores remain stored even when invalid;
absent detections remain null/NaN with false masks.

The earlier 346-frame V6 sample matches the new full extraction **exactly** for
coordinates, confidence, masks, status and exact PTS. Original video/workbook and
accepted sample/reference artifacts retain their hashes. Native graphics access
was needed to initialize the established CPU MediaPipe pipeline; the initial
sandbox failure was recovered through the existing resume mechanism.

`configs/v6_recording_provisional_v1.yaml` differs from the preserved original
recording config only by the approved holds. Raw offset zero and inclusive-end
mapping remain provisional. Author FPS/base/end/trim and lead/rear semantics remain
unverified; no global helper or FPS correction was adopted. A complete cut inventory
is not certified. Independent IMAGE inference prevents cross-cut estimator state,
but does not prove that every positive interval is visually continuous.

## Pose quality

Across the complete source: 31,121 `ok` frames (66.63%), 15,444 `unreliable` (33.07%)
and 140 missing (0.30%). Missing detections occupy five runs; the longest is frames
0–81. The report stores exact boundaries/times for the longest runs. These are
whole-recording statistics, including unknown footage, not punch accuracy.

There are **no missing detections inside the 302 retained intervals**. Only 61
examples pass strict all-body validity: 26 lead hooks, one rear hook, 25 lead
uppercuts and nine rear uppercuts. The other **241** contain unreliable body joints
but satisfy the unchanged mask-aware policy: anchors in 95% of frames, one complete
reliable arm in 80%, 70% valid core entries and at most 10% missing detections, with
the original 0.5 masks. No threshold was lowered and no missing coordinate invented.

Frame-weighted invalid elbow percentages on annotation-eligible intervals before
pose filtering (left / right; V4/V5 use preserved full-data quality reports):

| Class | V4 | V5 | V6 |
|---|---:|---:|---:|
| Lead hook | 0.7 / 34.8 | 2.1 / 4.2 | 1.5 / 6.8 |
| Rear hook | 0.0 / 99.1 | 76.9 / 0.0 | 4.9 / 5.3 |
| Lead uppercut | 0.4 / 58.9 | 1.1 / 0.0 | 4.9 / 21.7 |
| Rear uppercut | 0.0 / 96.1 | 77.1 / 0.3 | 11.6 / 5.7 |

V6 wrist invalidity is 4.2%/4.1% for rear hooks, 3.8%/20.1% for lead uppercuts and
10.4%/4.4% for rear uppercuts. Eligible-interval elbow failures are entirely saved
visibility scores below 0.5, without low presence, nonfinite or out-of-image elbow
entries. This establishes the threshold mechanism, not physical occlusion or
correct landmark placement. Glove overlap/body turns and occasional unreliable
lower-body estimates remain relevant. Reviewed new overlays span all four classes;
high confidence is not independent geometric validation.

Targeted low-coverage examples are row 72 / frame 9725 (rear hook), row 151 / frame
15348 (lead uppercut) and row 285 / frames 23013–23022 (rear uppercut). They show
reliable visible-arm chains alongside low-visibility, uncertain estimates of the
other arm. In row 285, the rising/far arm is unreliable while the guard-side chain
remains usable. **One reliable arm does not guarantee observation of the punching
arm.** These are diagnostic review flags, not automatic new exclusions or verified
anatomical/stance assignments. No confidence score certifies the unobserved joints.

V6 has substantially better bilateral rear-class visibility than V4/V5. Its lead
uppercuts still have notable right-arm missingness. Descriptive maximum joint-mask
class MI is 0.158 bits (right elbow); class-associated masks remain possible shortcuts.
Scripted dependent clips do not support independent statistical significance or
held-out accuracy claims, and no mask classifier was trained here.

## Diversity and representation implications

V6's frontal/oblique views and turns provide wider projected shoulder spans than
V4's weak rear-class views: median shoulder-width/torso is 0.597 for rear hooks and
0.592 for rear uppercuts, versus V4 0.102/0.154 and V5 0.234/0.319. These are projected
geometry proxies, not calibrated orientation or stance. Unknown boxer/session IDs
stay null. Keep V1/V6 together under `natebowerfitness-unresolved`; it is a conservative
source-group guard, not verified identity. Carry the readiness provenance with this
dataset; the existing training gates remain unchanged and do not admit V6 yet.

Median first-to-last sampled frame spans of retained provisional intervals:

| Class | V4 seconds | V5 seconds | V6 seconds |
|---|---:|---:|---:|
| Lead hook | 0.292 | 0.334 | 0.647 |
| Rear hook | 0.250 | 0.300 | 0.792 |
| Lead uppercut | 0.250 | 0.234 | 0.626 |
| Rear uppercut | 0.250 | 0.267 | 0.584 |

V6 intervals are substantially longer even though V4/V6 share approximately the
same decoded cadence. Execution, recovery inclusion and annotation conventions
may all contribute; these are not verified biomechanical punch durations. Length
and phase cues could undermine transfer. Do not retime/trim them automatically or
warp every clip to a fixed percentage of completion.

The most promising next isolated representation test is **existing normalized XY
plus validity-gated causal displacement per exact elapsed second**, with matched
mask controls. Angle-only inputs would inherit V4/V5's opposite hidden-arm patterns;
2D orientation normalization cannot recover out-of-plane geometry. The separate
[technical assessment](POSE_REPRESENTATION_ASSESSMENT.md) documents benefits, limitations,
the unchanged reference, and a controlled experiment proposal. No representation,
resampler, architecture, augmentation or training implementation changed in this task.

## Reproduction and validation

Use the existing local source/model files; no download is part of these commands:

```sh
uv run boxing-motion-ai extract-poses \
  --video data/raw/boxingvi/RGB_videos/V6.mkv \
  --workbook data/raw/boxingvi/Annotation_files/V6.xlsx \
  --recording configs/v6_recording_provisional_v1.yaml \
  --config configs/v6_pose_full.yaml --model weights/pose_landmarker_full.task \
  --output data/processed/boxingvi/V6/pose-full-image-v1 --resume
uv run boxing-motion-ai prepare-recording \
  --pose-run data/processed/boxingvi/V6/pose-full-image-v1 \
  --workbook data/raw/boxingvi/Annotation_files/V6.xlsx \
  --output data/processed/boxingvi/V6/candidates-full-v1
uv run boxing-motion-ai assess-pose-quality \
  --dataset data/processed/boxingvi/V6/candidates-full-v1 \
  --output reports/boxingvi/v6-full-v1/quality/V6
uv run python -m boxing_motion_ai.data.v6_preparation \
  --output reports/boxingvi/v6-full-v1/readiness
uv run pytest -q
uv run ruff check .
```

Extraction resumes/reuses a matching validated run. Preparation/quality/readiness
require fresh destinations; do not overwrite the accepted v1 artifacts. The final
report uses frozen v1 paths, so a newly versioned preparation needs its own explicit
provenance paths. The optional quality renderer can add exact-PTS source sheets to
an existing quality report without inference.

**191 tests passed; Ruff and `git diff --check` passed.** Synthetic tests cover
exclusion preservation, exact PTS/masks, missing runs, diagnostic validity/continuity,
report writing and rejection of changed inputs/policy. Generated data/media/reports,
environments and weights stay ignored and untracked. Original dataset provenance:
[BoxingVI](https://github.com/Bikudebug/BoxingVI),
[V6 metadata source](https://www.youtube.com/watch?v=cQW-h8yGbRc).
Licensing/redistribution remains unverified. Stop for review before any feature or
training experiment.
