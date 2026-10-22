# Level 1 validity-gated velocity study

**Velocities meaningfully improve aggregate V4→V5 recognition in all three seeds,
especially rear uppercuts, but do not solve balanced six-class recognition.** Mean
development macro-F1 rises **0.4912→0.6081**, accuracy **71.87%→78.50%**. Rear hooks
do not improve on average, lead uppercuts improve in only one seed, and jab recall
falls in all three. This experiment was subsequently accepted as the provisional
XY+velocity+mirror reference; earlier XY results remain preserved. The
[frozen transfer follow-up](LEVEL1_VELOCITY_TRANSFER.md) finds inconsistent external
benefits. Neither study is a held-out test or boxer-independent generalization claim.

## Representation and continuity contract

The optional `torso-core-2d-velocity-1.0` representation wraps the existing loader:

- Channels 0–40 are the unchanged `torso-core-2d-1.0` XY/confidence/mask features.
  Channels 41–64 contain eight anatomical core joints × `[vx, vy, velocity_valid]`.
- Velocity is `(normalized_xy[t] - normalized_xy[t-1]) / exact_elapsed_seconds`.
  Integer PTS and rational time bases are subtracted as `Fraction` values before
  converting the interval to floating point. Large absolute timestamps do not
  lose the small difference through float cancellation.
- Both endpoint joints and torso normalizations must be reliable. Missing poses,
  skipped indices, changed scene/segment IDs and timestamp discontinuities invalidate
  the pair. Non-increasing timestamps are rejected as contract errors.
- The frozen continuity rule permits at most twice the source's median positive
  adjacent-frame PTS interval: **0.084 s in V4, 0.066 s in V5**. This is derivative
  validity metadata, not a changed dataset-quality threshold or annotation FPS
  assumption. Source timing metadata is fixed before clip construction; derivatives
  use no future positions or whole-clip coordinate statistics. This is prerecorded
  Level 1, not a validated streaming cadence-estimation algorithm.
- The first annotated frame has invalid velocity: no pre-clip context is added.
  Unusable derivatives are zero sentinels plus false masks. Joint coordinates are
  never filled; there is no clipping, smoothing, resampling or new exclusion.
- Mirroring negates X and vx and exchanges anatomical left/right joint channels,
  confidence and both mask types. Y/vy stay unchanged. Lead/rear class labels stay
  unchanged; no stance is assumed. Mirror decisions remain training-only at 50%.

Valid derivative coverage, excluding the first-frame non-opportunities, is **86.19%**
of within-clip joint pairs in V4 and **89.35%** in V5. PTS, source frames, raw pose
arrays, quality indices, annotations and original uncertainty remain immutable.

## Matched experiment

V4 training has the same **527** selected positives; V5 development has the same
**583**. V6 and all other recordings are excluded from training. V9 inputs are not
opened. The original mask-aware70 policy, class weights and exclusions are unchanged.

Each representation uses seeds **20261021/20261022/20261023**, 40 epochs, batch size
32, AdamW learning rate 0.001, weight decay 0.001, dropout 0.1 and two CPU threads.
V5 macro-F1 selects the best epoch, with earlier ties retained. Visibility augmentation
and context are disabled; the new strict configuration schema does not admit either
intervention or a different recording split.

The TCN source is byte-identical to the preserved reference: three causal blocks,
32 hidden channels, receptive field 15 frames, original pooling and six outputs.
Only the input projection widens 41→65, increasing **11,046→11,814 parameters**.
Shared initialization is copied exactly; added columns start at zero, preserving
initial predictions and global RNG state. Data order, mirror decisions and shared
initialization match across arms. Original XY normalization is unchanged, and all
three XY full/mask controls reproduce their preserved mirror-reference metrics
**exactly**. Native sampling remains unchanged, so the physical receptive field
still differs between V4 and V5.

Both arms include mask-only controls: eight original masks versus eight original
masks plus eight derivative-validity masks. The latter has no coordinates,
confidence or vector values; its input projection widens 8→16 (9,990→10,246 parameters).
This checks added validity cues, although no control provides a causal decomposition
of what the full network learns.

## Results

Selected full-feature checkpoints, evaluated without augmentation:

| Seed | XY epoch | XY macro-F1 | Velocity epoch | Velocity macro-F1 | Paired gain | XY accuracy | Velocity accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20261021 | 31 | 0.5111 | 32 | 0.6597 | +0.1486 | 75.13% | 80.10% |
| 20261022 | 31 | 0.4887 | 26 | 0.5987 | +0.1100 | 72.04% | 79.07% |
| 20261023 | 28 | 0.4737 | 40 | 0.5658 | +0.0921 | 68.44% | 76.33% |
| **Mean** | | **0.4912** | | **0.6081** | **+0.1169** | **71.87%** | **78.50%** |

Macro-F1 sample SD is 0.0188 for XY and 0.0477 for velocity; paired-gain SD is 0.0289.
These are three optimizations on the same recordings, not independent boxer/session
replications or a significance test. V5 has been repeatedly used for model selection.

| Class | XY mean recall | Velocity mean recall | Change, percentage points |
|---|---:|---:|---:|
| Jab | 91.45% | 86.28% | −5.16 |
| Cross | 96.38% | 96.86% | +0.48 |
| Lead hook | 80.88% | 87.25% | +6.37 |
| Rear hook | 21.37% | 21.37% | 0.00 |
| Lead uppercut | 0.00% | 10.26% | +10.26 |
| Rear uppercut | 21.92% | 78.54% | +56.62 |

- **Rear uppercuts:** recall improves in every seed, from 46.6/15.1/4.1% to
  65.8/87.7/82.2%. This is the most consistent class-level gain.
- **Rear hooks:** 5.1/20.5/38.5% becomes 30.8/20.5/12.8%. The mean is unchanged;
  the third seed regresses substantially. Across the three repeated V5 evaluations,
  64.1% of rear hooks become rear uppercuts. Improved uppercut recall partly shifts
  the competition between these classes rather than resolving both.
- **Lead uppercuts:** 0/0/0% becomes 30.8/0/0% (12 of 39 correct in the first seed).
  Overall 81.2% become lead hooks; the mean gain hides two complete failures.
- **Lead hooks:** improve in two seeds and regress in one. **Jabs** regress in all
  three, increasingly becoming lead hooks. Crosses remain near their previous ceiling.

Mask-only macro-F1 changes **0.3546→0.3492**; every seed decreases slightly
(−0.0035/−0.0016/−0.0110). Full-feature paired gains exceed mask-control gains in
all seeds. Added derivative-validity cues alone do not explain the observed gain
in this comparison. This does not prove robust motion reasoning or rule out
class-correlated pose-estimator artifacts and remaining boundary/length shortcuts.

### Overfitting

At selected checkpoints, mean train/development F1 is **0.7957/0.4912** for XY and
**0.9118/0.6081** for velocity: the gap remains approximately 0.304 in both arms.
At epoch 40 it is **0.8044/0.4248** versus **0.9261/0.5722**, gaps 0.3796 and 0.3539.
Velocities make V4 easier to fit and improve V5, but substantial domain overfitting
persists. Two velocity seeds deteriorate after their best epochs; the third selects
the final epoch. Accuracy alone would hide the weak uppercut/hook classes.

## Noise and extreme derivatives

Apparent speeds in normalized image-plane torso units/second, valid pairs only:

| Recording | Median | 90th percentile | 99th percentile | Maximum |
|---|---:|---:|---:|---:|
| V4 | 1.115 | 4.286 | 9.657 | 66.987 |
| V5 | 1.094 | 5.924 | 13.600 | 39.952 |

Exact-PTS sheets are under `reports/boxingvi/level1-velocity-review-v1/`:

- **V4 row 551, frames 24468–24470:** the estimated left wrist jumps near the feet
  for one frame, then returns near the glove. Visibility 0.635/presence 0.998 passes
  the unchanged mask. The consecutive spikes are 66.49/66.99 torso units/s over
  genuine 41/42 ms intervals. Torso scale also changes 0.1392→0.1743 image-height
  units on the return. Source imagery supports estimator mislocalization, not a
  real hand trajectory or a timestamp-subtraction bug.
- **V4 row 225, frame 13893:** a similar near-foot wrist estimate produces 58.68
  torso units/s despite high reported visibility. Masks cannot certify geometry.
- **V5 row 389, frames 8231–8232:** the maximum coincides with a rapidly extending
  arm; these images do not establish that every high-tail derivative is erroneous.

Thus noisy, nominally valid poses can generate unrealistic apparent trajectories.
No speed cutoff, winsorization, smoothing, source correction or new quality rule
was applied; the evaluated models saw these values. These are not measured physical
glove speeds in metres/second. The study does not isolate how much outlier noise
helps or hurts learned predictions, so do not attribute every gain to true motion.

## Artifacts, reproduction and validation

Implementation: `training/velocity.py` (optional wrapper/contract), extended mirror
handling, experiment schema 1.5, `training/velocity_study.py`, exact-PTS reviewer and
synthetic tests. Existing XY feature code, TCN architecture, source data and previous
configuration files were preserved. The prior reference remains available unchanged.

```sh
uv run python -m boxing_motion_ai.training.velocity_study \
  --config configs/level1_velocity_study.yaml \
  --output reports/boxingvi/level1-velocity-smoke-repeat --smoke
uv run python -m boxing_motion_ai.training.velocity_study \
  --config configs/level1_velocity_study.yaml \
  --output reports/boxingvi/level1-velocity-repeat
uv run python -m boxing_motion_ai.training.velocity_review \
  --experiment reports/boxingvi/level1-velocity-repeat \
  --output reports/boxingvi/level1-velocity-review-repeat
uv run pytest -q
uv run ruff check .
```

Use fresh output directories. `reports/boxingvi/level1-velocity-v1/` contains summary,
all per-class precision/recall/confusions, predictions, augmentation audit, curves,
best/final checkpoints, velocity distributions, configurations, source snapshots,
initialization hashes and preserved-input manifest. `checkpoint-verification.json`
records fresh best-checkpoint reloads reproducing all 24 train/development prediction
sets. Velocity checkpoints require their recorded 65/16-channel input interface;
legacy XY-only diagnostic utilities should not be applied blindly to them.

**207 tests passed; Ruff and `git diff --check` passed.** Tests cover exact elapsed
time, invalid joints/anchors, missing poses, scene/segment/index/timestamp gaps,
large-PTS precision, reflection/derivative consistency, involution, causal features,
shared initialization/RNG and rejected split/intervention changes. Real-data smoke
and full runs pass selection, shuffle, mirror, reference-reproduction and provenance
checks. 3,021 preserved artifact files are unchanged; V4/V5 and accepted V6 dataset
hashes still match. V6's exclusions remain preserved and it was not used in training.
All generated data, reports, figures and weights remain Git-ignored; no download
was performed. Unknown footage stays unknown, and identities/boundaries remain
provisional. Stop here for review before any additional intervention.
