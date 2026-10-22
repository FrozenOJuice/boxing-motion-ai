# Relative-arm geometry: investigation and controlled Level 1 study

**A useful partial result, not a replacement reference.** Additional validity-gated
arm geometry improves mean macro-F1 on V5/V6/V7/V8, but regresses on V10 in all three
seeds. Lead-uppercut recall improves in every seed on every evaluated recording;
rear hooks and rear uppercuts remain unrecognized outside V5. Keep the accepted
**V4-only XY + validity-gated velocities + 50% mirror** provisional reference unchanged.

## Representation investigation and decision to proceed

The [accepted velocity-transfer study](LEVEL1_VELOCITY_TRANSFER.md) improved only V7
on average, regressed on V8/V10 in all three seeds, and had zero external rear-hook/
rear-uppercut recall. We reviewed its preserved metrics/confusions and investigated
geometry on the same prepared selections: **1,739 eligible positives** across V4, V5,
V6, V7, V8 and V10. No annotation or quality-policy selection changed.

The important coverage distinction is **partial versus bilateral observation**.
V4 anatomical-right rear-class geometry is almost absent, but at least one arm's
geometry is available on **at least 97.98% of frames in every represented class**.
This supports a limited additive, explicitly masked comparison; it does not support
claims of complete bilateral arm observation or recovery of hidden punch trajectories.
A bilateral requirement would have rejected the experiment unnecessarily: it was
not part of the existing mask-aware policy. Neither class labels nor stance identify
which anatomical arm is punching in these statistics.

### Candidates and masks

The optional version `torso-arm-geometry-1.0` adds six feature groups per anatomical
side. All use the unchanged aspect-corrected, hip-centered, torso-scaled positions.

| Candidate | Definition | Required reliable landmarks | Limitation |
|---|---|---|---|
| Shoulder-relative wrist XY | Wrist minus same-side shoulder | Shoulder, wrist, original normalization | Still camera-axis/projected coordinates |
| Upper-arm vector | Elbow minus shoulder | Shoulder, elbow, normalization | Yaw, foreshortening and erroneous elbows |
| Forearm vector | Wrist minus elbow | Elbow, wrist, normalization | Hidden elbows/wrists and short projected bones |
| Elbow cosine | Cosine between shoulder−elbow and wrist−elbow | All three joints and nondegenerate segments | Projected 2D angle, not measured anatomical 3D flexion |
| Upper arm relative to torso | Dot and signed cross with unit hip-midpoint→shoulder-midpoint axis | Arm endpoints plus all torso anchors | Removes in-plane rotation, not out-of-plane rotation |
| Forearm relative to torso | Same torso-axis orientation for forearm | Arm endpoints plus all torso anchors | Same projection and forearm-visibility limitations |

Each group has its own validity bit. Missing values are zero sentinels, never invented
landmarks. A fixed numerical guard of **0.001 torso units** masks near-zero projected
segments for angle/unit-vector calculations; it does not change source masks, confidence
thresholds, coordinates, or example retention. No interpolation, smoothing, stance
canonicalization, Z/world-coordinate reconstruction or velocity change is introduced.

The original 65 XY/velocity channels remain byte-identical as a tensor prefix.
Each side contributes 11 geometric values plus six validity bits: **99 full channels**.
The mask-only control extends the original 16 bits with 12 geometry-validity bits:
**28 mask channels**, with no geometric values, coordinates or confidence.

Features are frame-local; original velocities remain exact-PTS causal backward
calculations. Mirroring exchanges anatomical sides, negates vector X and signed
cross components, and preserves dot products, elbow cosine and class labels.
Confidence and all original/new masks swap consistently. Mirroring twice restores
the full input, and recomputing geometry from mirrored base features agrees with
mirroring computed geometry. No original stance is assumed.

### Coverage across recordings

These are frame-weighted fractions within eligible annotated clips, not whole-video
coverage or independently verified anatomical accuracy. L/R means anatomical channels,
not lead/rear. In these selections, upper/forearm torso-orientation coverage equals
its corresponding vector coverage; forearm and elbow coverage also coincide.

| Recording | Clips | Wrist-relative L/R | Upper-arm/vector-axis L/R | Forearm/elbow/forearm-axis L/R |
|---|---:|---:|---:|---:|
| V4 | 527 | 99.19% / 52.63% | 99.90% / 48.29% | 99.16% / 46.39% |
| V5 | 583 | 66.27% / 98.97% | 63.61% / 98.63% | 63.11% / 98.57% |
| V6 | 302 | 94.90% / 88.96% | 94.14% / 87.61% | 93.56% / 87.51% |
| V7 | 66 | 90.97% / 93.40% | 84.72% / 90.62% | 83.91% / 89.70% |
| V8 | 120 | 92.90% / 84.42% | 89.41% / 81.53% | 88.49% / 81.53% |
| V10 | 141 | 94.47% / 98.12% | 92.18% / 95.59% | 91.53% / 95.36% |

The major limitation remains V4: among **73 rear-hook / 87 rear-uppercut** clips,
right elbow angles appear at all in only **1 / 7 clips**, on **0.19% / 1.85%** of class
frames. Right shoulder-relative wrist coverage is only **1.50% / 3.02%**. None of those
clips has either right feature available on at least half its frames. Opposite-arm
coverage is high, so aggregate “either arm” availability hides this gap. The half-frame
statistic is descriptive only; no new selection threshold is applied.

Lead uppercuts have V4 right-angle coverage **36.57%**, with majority coverage in
22/71 clips; V5 has left/right coverage **98.89%/100%**. V5 rear-hook/rear-uppercut
left coverage is **22.86%/21.75%**, versus approximately full right coverage. Mirroring
redistributes measured sides and missingness, without creating a new camera view
or recovering the unobserved arm. New conjunction masks can expose class-related
visibility patterns, motivating the matched control.

V6 lacks source jab/cross annotations; V8 lacks source rear hooks. V7's 20 source
rear-uppercut annotations remain excluded, rather than absent from its workbook;
its cross/rear-hook selections contain only one example each. V10 has just four lead
uppercuts. All exclusions, V6 holds (321, 598, 620, 675–686), V10 row 123 and other
review flags remain intact. Unlabeled footage remains unknown.

### Perspective and stability

A shoulder-width-based facing axis is poorly supported: V4 rear-hook/rear-uppercut
projected shoulder-width/torso medians are **0.101/0.166**, versus **0.250/0.314** in V5
and **0.582/0.580** in V6. We use the existing torso axis rather than infer facing/stance
from narrow projected shoulders. This still cannot undo camera yaw or perspective.

Available V4 right forearms have p05 projected length **0.079 torso units**; V5 left
forearms have p05 **0.089**. There are 28/63 observations respectively below a descriptive
0.05-unit probe. Very short projected segments make orientation sensitive to jitter.
The probe never changes validity. V5 left projected elbow changes exceed a descriptive
30° per-frame probe on **894/2,522** valid continuous pairs (35.4%). Such changes can
reflect genuine fast motion, projection or estimator errors; they are not automatically
mislabeled as bad poses. Diagnostics never span missing detections, cuts, skipped indices,
segment changes or existing timestamp-gap boundaries. Complete distributions and
class/side coverage are retained in the readiness JSON.

## Controlled experiment

- **V4 train: 527; V5 development: 583.** Three matched seeds: 20261021/20261022/20261023.
- Preserved frozen XY+velocity reference models are reused, not trained again or fine-tuned.
  New models train from the same base initialization for **40 epochs**, batch 32,
  AdamW learning rate 0.001, weight decay 0.001, dropout 0.1, two CPU threads.
- Same causal TCN source, three blocks, hidden size 32, receptive field 15 frames and
  original pooling. Only the necessary input projection widens **65→99**, increasing
  full parameters **11,814→12,902**; control **16→28**, **10,246→10,630**.
- Original shared weights and zero-initialized added columns preserve initial reference
  predictions. All three full/control runs match the preserved reference's shared
  initialization hashes, shuffle order, mirror decisions and training/development rows.
- Mirror probability remains **50% during training only**. Visibility/context remain
  disabled. Native sampling, quality thresholds, labels and class weights are unchanged.
- A two-epoch/two-examples-per-class smoke test passes before the full experiment.
  Best V5 macro-F1 selects each epoch with the original earliest-tie rule. All selections
  finish before V6/V7/V8/V10 evaluation; no external tuning/reselection/ensembling occurs.
- All **30** frozen reference prediction sets (V5 plus four diagnostic recordings,
  three seeds, full/control) reproduce **exactly**, including probabilities/source rows.
  New/base inputs retain identical source rows and unchanged original feature prefixes.
- V7/V8 remain one unresolved source group; V1/V6 grouping precautions persist.
  Boxer/session identities remain unknown. V9 inputs are never opened. These external
  recordings are reused diagnostics, not unbiased independent boxer test sets.

## Full-feature results

Means across three seeds. Six-class F1 uses the original fixed taxonomy; supported
F1 separately averages classes with eligible ground truth. Predictions into unsupported
classes remain errors; outputs are not restricted or renormalized.

| Recording | Accuracy reference → geometry | Six-class F1 reference → geometry | Supported F1 reference → geometry | Supported-F1 paired changes in seed order |
|---|---:|---:|---:|---|
| V5 | 78.50% → 76.10% | 0.6081 → 0.6568 | 0.6081 → 0.6568 | +0.0147, +0.0749, +0.0566 |
| V6 | 11.81% → 27.70% | 0.0674 → 0.1245 | 0.1010 → 0.1867 | +0.0647, +0.1309, +0.0614 |
| V7 | 51.01% → 68.18% | 0.3650 → 0.4212 | 0.4380 → 0.5055 | +0.0149, +0.2028, -0.0152 |
| V8 | 34.72% → 38.33% | 0.2389 → 0.2655 | 0.2867 → 0.3186 | +0.0174, +0.0484, +0.0298 |
| V10 | 18.20% → 14.66% | 0.1249 → 0.0846 | 0.1249 → 0.0846 | -0.0425, -0.0728, -0.0056 |

V5, V6 and V8 improve F1 in all seeds; V7 improves in two of three. V10 regresses in
all three. Seeds reuse the same recordings and dependent clips; no significance or
boxer-independent generalization is claimed. No external aggregate is used to pick
a preferred seed or change the configuration.

### Per-class mean recall

Cells are reference → geometry (%). `—` indicates no eligible ground truth, not zero
recall. V7 excluded rear uppercuts are distinguished in the source-support report.

| Class | V5 | V6 | V7 | V8 | V10 |
|---|---:|---:|---:|---:|---:|
| jab | 86.3% → 76.1% | — | 91.7% → 95.8% | 56.2% → 60.4% | 4.1% → 2.0% |
| cross | 96.9% → 89.1% | — | 100.0% → 100.0% | 46.0% → 33.3% | 0.9% → 0.0% |
| lead_hook | 87.3% → 83.8% | 42.2% → 71.4% | 85.0% → 81.7% | 35.8% → 42.0% | 88.0% → 65.3% |
| rear_hook | 21.4% → 21.4% | 0.0% → 0.0% | 0.0% → 0.0% | — | 0.0% → 0.0% |
| lead_uppercut | 10.3% → 70.1% | 6.9% → 30.4% | 23.1% → 55.6% | 17.5% → 31.7% | 33.3% → 83.3% |
| rear_uppercut | 78.5% → 76.7% | 0.0% → 0.0% | — | 0.0% → 0.0% | 0.0% → 0.0% |

**Lead uppercuts improve in every matched seed on all five evaluated recordings.**
V5 per-seed recall changes from 30.8/0/0% to 76.9/76.9/56.4%; V6 from 17.6/2.4/0.8%
to 40.0/39.2/12.0%. This is the strongest reproducible benefit. V10's 83.3% mean is
based on only four examples and does not establish reliable generalization there.

**Rear hooks do not improve on average in V5 and remain 0% externally.** Rear uppercuts
remain 0% wherever externally supported, and V5 recall falls slightly overall with
opposite changes across seeds. Geometry does not solve the bilateral observation or
lead/rear-semantic uncertainty. V5's balanced F1 gain trades away jab/cross/lead-hook
recall and reduces overall accuracy. V10 loses lead-hook recall from 88.0% to 65.3%,
with near-zero jab/cross recognition despite its lead-uppercut gain.

All per-seed precision/recall, confusion matrices and prediction probabilities are in
[the structured report](../reports/boxingvi/level1-limb-study-v1/summary.json).
[The readable report](../reports/boxingvi/level1-limb-study-v1/summary.md) includes full
and mask-only mean 6×6 matrices for each recording with explicit true/predicted axis
order. These are mean counts over identical clips, not additional independent examples.

## Mask-only controls and overfitting

| Recording | Mask accuracy reference → geometry | Mask supported F1 reference → geometry | Full geometry supported F1 |
|---|---:|---:|---:|
| V5 | 55.69% → 54.43% | 0.3492 → 0.3475 | 0.6568 |
| V6 | 20.31% → 21.08% | 0.1110 → 0.1164 | 0.1867 |
| V7 | 36.36% → 44.44% | 0.2699 → 0.2950 | 0.5055 |
| V8 | 41.94% → 41.67% | 0.2815 → 0.2744 | 0.3186 |
| V10 | 30.50% → 28.13% | 0.1368 → 0.1278 | 0.0846 |

New masks alone do not explain the full gains: mask-only F1 is essentially unchanged
on V5, only slightly higher on V6, and lower in all three V8 seeds. V7's control also
improves (two seeds), so extra missingness cues may contribute there. Full features
substantially outperform mask controls on V5/V6/V7, but V8 accuracy still trails its
mask-only control and V10 full metrics remain worse than mask-only. Separate control
models are not a causal decomposition of which cues the full network uses; this does
not prove stance-independent or motion-specific learning.

Selected full-feature epochs and F1:

| Seed | Reference / geometry epoch | Reference V4 / V5 | Geometry V4 / V5 | Geometry final V4 / V5 |
|---|---:|---:|---:|---:|
| 20261021 | 32 / 10 | 0.9086 / 0.6597 | 0.6993 / 0.6745 | 0.9499 / 0.5669 |
| 20261022 | 26 / 18 | 0.9052 / 0.5987 | 0.7894 / 0.6736 | 0.9313 / 0.5187 |
| 20261023 | 40 / 40 | 0.9215 / 0.5658 | 0.9347 / 0.6224 | 0.9347 / 0.6224 |

All models still train for the same 40-epoch budget. Seeds 1/2 peak on V5 earlier
(epochs 10/18), then continue fitting V4 while development F1 falls. Final V4 F1 is
0.93–0.95 versus V5 0.52–0.62: overfitting persists. Smaller gaps at selected early
epochs are development-selected evidence, not an independent generalization result.
Training curves, checkpoints, original annotations/poses and previous results are
preserved. No architecture, regularization or augmentation adjustment followed these results.

## Recommendation and limits

Keep geometry optional and the accepted XY+velocity+mirror reference unchanged pending
review. Geometry is worth retaining as a promising lead-uppercut intervention, with
consistent V6/V8 F1 gains, but **it is not a consistent improvement across environments**
and does not fix rear punches. Do not automatically promote it or expand training
based on V5 alone.

The strongest remaining bottleneck is representative, observed rear-arm geometry in
the training source, together with provisional annotation/lead-rear conventions.
Before another modeling change, prioritize a separately approved, leakage-guarded
data-diversity comparison using the prepared V6 source; its bilateral coverage is
much better. Preserve all existing V6 holds and the V1/V6 group guard. Do not relax
masks, infer stance, fill joints or treat unknown footage as no-punch to improve yield.
No such expanded training was performed here.

Projected 2D elbow angles remove in-plane rotation/translation mathematically; they
are not invariant to out-of-plane yaw, depth or occlusion. Torso directions remain
projection dependent. Source confidence is not ground truth anatomical correctness.
Current results do not certify labels, punching-side visibility, boxer identity,
full-video event detection or early recognition. V9 remains reserved and untouched.

## Reproduction and validation

```sh
uv run python -m boxing_motion_ai.training.limb_investigation \
  --config configs/level1_limb_investigation.yaml \
  --output reports/boxingvi/level1-limb-readiness-repeat
uv run python -m boxing_motion_ai.training.limb_study \
  --config configs/level1_limb_study.yaml \
  --output reports/boxingvi/level1-limb-smoke-repeat --smoke
uv run python -m boxing_motion_ai.training.limb_study \
  --config configs/level1_limb_study.yaml \
  --output reports/boxingvi/level1-limb-study-repeat
uv run pytest
uv run ruff check .
```

Use fresh ignored destinations. The study configuration freezes the actual recorded
coverage report, accepted reference and training budget; the readiness repeat is for
inspection and does not silently replace its input. Ordinary experiment configurations
can optionally select `level1-experiment-1.6` / `xy_velocity_limb`; V4/V5 split gates
and disabled visibility/context remain strict. Nothing changes the reference registry.

Readiness JSON retains all feature-group coverages by class/side, source rows,
source/preparation/policy hashes, length/angle distributions and source-group guards.
The experiment retains 60 paired prediction files, all confusion matrices, selected/
final training metrics, 12 selected checkpoint hashes, augmentation audits and complete
per-run source/lock/config snapshots. No dataset, report, checkpoint, generated media
or virtual environment belongs in the Git commit. Previous artifacts and original
inputs are hash-verified unchanged.

Validation: **239 tests passed** (`uv run pytest -q`). `uv run ruff check .` and `git diff --check` passed. Independently recomputed all 60 prediction-file
confusion matrices, accuracy and F1; verified 12 checkpoint hashes and **3,598** preserved
prior files unchanged. All 30 frozen reference prediction sets reproduced exactly.
New synthetic tests cover masks, degenerate bones, known angles/vectors, in-plane
rotation, physical mirroring/involution, causality, continuity, unchanged base inputs,
zero-column initialization/RNG preservation, strict split/config gates and report writing.
