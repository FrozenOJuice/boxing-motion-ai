# V5 partial-pose policy and model-input recommendation

This is a read-only quality investigation, not model training or a change to source annotations,
pose coordinates/confidence, masks, timestamps or splits. It builds on [dataset preparation](DATASET_PREPARATION.md).

## Reproduce

```sh
uv run --extra pose boxing-motion-ai assess-pose-quality \
  --dataset data/processed/boxingvi/V5/candidates-full-v1 \
  --output reports/boxingvi/v5-mask-aware-reproduce \
  --video data/raw/boxingvi/RGB_videos/V5.mkv
uv run pytest
uv run ruff check .
```

Omit `--video` for numerical diagnostics only. Optional visual review decodes the source once,
without pose re-inference, and verifies each selected frame's exact PTS/timebase. No new dependencies
are added. Output must be a new directory outside the prepared dataset/raw data. The command
verifies the prepared NPZ hash, preserves all source files and records input/code hashes.

Outputs: `quality.json`, `joint-failure-matrix.csv`, `policies.md`, `policy-index.json` and optional
`visual-review/`. Candidate IDs refer to the original `candidates.jsonl` and `poses.npz`, which
are identified by hashes and never copied, rewritten or imputed. Existing annotation exclusions
remain in force under every policy; no unreviewed gap becomes `no_punch`.

## Definitions and scope

Use annotation-eligible candidates only; source class/boundary assumptions remain provisional.
Per-joint diagnostics report both frame-weighted failure fractions and candidate-balanced mean
failure fractions. Failure reasons are **mechanical mask conditions**, potentially overlapping:
nonfinite coordinates/scores, outside-image coordinates, low visibility/presence or invalid score
range. These do not by themselves identify physical occlusion, motion blur or pose-estimator error.
Confidence scores are not calibrated keypoint accuracy measurements.

All retention comparisons use the **unchanged original 0.5 validity masks**. Post-hoc 0.3/0.5/0.7
sensitivity changes diagnostic masks only; it does not lower detection thresholds, recover missing
poses, change the saved dataset, or establish low-confidence estimates as correct.

The compared policies are fixed and class-independent; none is tuned using trained recognition
performance. They are working curation proposals, not accuracy-validated thresholds:

| Policy | Retention requirement |
| --- | --- |
| strict_body | All 22 body joints valid on every event frame |
| strict_arms | Both shoulders, elbows and wrists valid on every event frame |
| core90 | >=95% anchor frames, >=80% frames with one complete arm, >=90% valid core entries |
| mask_aware80 | Same anchor/arm checks; >=80% valid core entries |
| mask_aware70 (recommended for development) | Same anchor/arm checks; >=70% valid core entries |
| anchors_only_diagnostic | >=95% anchor frames; no arm check; intentionally permissive comparator |

Coverage policies also require no more than 10% completely missing pose frames. An anchor frame
has both shoulders and both hips valid. Core joints are shoulders, elbows, wrists and hips (8).
A complete arm has its shoulder, elbow and wrist valid; either arm can satisfy admission, but
input channels keep fixed anatomical left/right ordering. This is never label-conditioned arm
selection, stance inference, class merging or discarding the less visible arm's valid observations.
Finger/face/foot estimates remain stored, although this proposed first model does not require them.
A retained example may still contain invalid joints; its invalid coordinates remain masked out.

Offline coverage filters use complete candidate intervals for **curation only**. Aggregate coverage,
source row, absolute frame/time, event duration and known end frame must not become causal model
features. They can encode labels or future information.

## Mask/class association without training

The code computes categorical mutual information between fixed coverage bins and source classes,
plus a predeclared low-left-elbow-coverage contingency table. It fits no classifier or decision
boundary and reports no prediction accuracy. Seeded label permutations compare the maximum MI
across all body joints; circular clip-order shifts are a separate descriptive sensitivity check.
Neither establishes independent-session significance: V5 is one dependent, scripted recording.
High-dimensional mask/class association can arise from real action geometry, camera viewpoint,
script order, estimator bias or their interaction. This analysis cannot isolate those causes.

Missingness can be informative even when raw mask/confidence channels are omitted: placeholders,
feature gating and zeroed channels can still reveal it. Do not claim a coordinate-only model
eliminates this risk. More recordings/viewpoints and group-held-out tests are essential.

## Recommended first temporal-model handling (proposal only)

1. Start with fixed-order 2D core-joint coordinates and masks. Keep raw z, all confidence scores
   and the full 33-joint record for provenance; monocular z is not calibrated depth. Do not
   silently promote low-visibility coordinates or map anatomical side to lead/rear from a label.
2. Normalize each frame using **currently valid** hip and shoulder anchors, accounting for image
   aspect ratio. If the root/scale is invalid or degenerate, mark affected features invalid for
   that timestep. Do not use a future frame or full-clip mean as a fallback.
3. Preserve every timestep and exact delta-time. Derived velocity/angles require all component
   joints valid; velocity needs consecutive source frames, positive exact delta-time, and the
   same scene/segment. Reset temporal state at known cuts; never interpolate or smooth across
   gaps/cuts. The edit inventory is incomplete; independent IMAGE poses avoid cross-cut pose
   tracking, but future sequence continuity still requires review.
4. Use masked operations. A future dense tensor may encode an invalid feature with an **explicit
   neutral placeholder plus a separate mask**, solely as a model-input representation. It is
   not a joint estimate and must never overwrite the pose dataset. Never feed NaNs to ordinary
   dense layers or let invalid raw coordinates influence normalization/velocities. No tensor
   encoding, padding, reconstruction or forward filling is implemented in this investigation.
5. Begin with mask-gated features; use confidence for reliability/gating. Treat direct confidence
   input channels as an explicit later ablation, since they may amplify the same shortcut.
   Keep padding masks distinct from missing-joint masks. A causal TCN must consume only current
   and past frames, not complete-event quality summaries or future padding/end information.
6. After training is authorized, compare mask-only, coordinate/mask-aware and confidence-channel
   ablations on **independent source groups**, macro-F1/per-class metrics and severe-occlusion
   subsets. Class-independent joint masking augmentation is a possible train-only control, not
   evidence of solved bias and not a reason to invent joint positions. No such model/augmentation
   was trained or implemented here.

A mask-aware six-class **development selection** can be prepared without claiming trustworthy
six-class recognition performance. Retention is not evidence of accurate hidden-arm trajectories.
Annotation alignment, identity/session grouping, independent evaluation and rights remain unresolved.

Sources: [MediaPipe landmark score definitions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/components/containers/NormalizedLandmark),
[Pose Landmarker overview](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
