# V4 → V5 recognition failure diagnosis

2026-10-21. Read-only analysis of `reports/boxingvi/level1-initial-v1/`. No model was trained,
no architecture, feature pipeline, quality threshold, original data or existing experiment was
changed. V9 contents were not opened. All analyses use the original 527 V4 / 583 V5 retained
positive clips, with their provisional labels and raw-index/inclusive-end assumptions.

The best-supported problem is a **viewpoint/visibility domain shift amplified by mask and
clip-length shortcuts**. Translation/scale normalization works, but does not normalize image
reflection, body rotation, anatomical channel assignment or which arm is occluded. More training
on the same V4 representation made transfer worse. There is no evidence of a simple slicing or
class-index bug that would explain the collapse.

## Artifacts and reproduction

New artifacts are under `reports/boxingvi/level1-diagnosis-v2/` (ignored by Git):

- `diagnosis.json`: distributions, frozen-model interventions, implementation checks and hashes.
- `diagnosis.md`: comparison table and links to all **16** visual examples.
- `distributions.png`, `class_visibility.png`: recording/class comparisons.
- `review/`: normalized-pose strips and **80 exact-PTS decoded source frames** in annotated sheets.
- `visual-cases.json`: source rows, original annotation intervals and reviewed frame indices.
- `interpretation.md`, `review-first.md`: findings and the six-sheet priority review.

Reproduce from the repository root with a fresh output directory:

```sh
uv sync --locked --extra pose --extra train
uv run --extra pose --extra train python -m boxing_motion_ai.training.diagnostics --experiment reports/boxingvi/level1-initial-v1 --output reports/boxingvi/level1-diagnosis-repeat --media-root data/raw/boxingvi/RGB_videos
uv run --extra pose --extra train pytest
uv run ruff check .
```

Omit `--media-root` for a pose-only diagnosis. The tool never optimizes weights. Counterfactual
input changes operate on copies for frozen inference only; none is an implemented improvement.
It verifies prepared/source-video hashes, reproduces original predictions, and hashes existing
checkpoints, predictions, configurations, curves and reports before/after. Existing artifacts
were unchanged. Seeks accelerate visual decoding; acceptance requires **exact stored PTS/time
base**, with decoded indices obtained from the matching prepared inventory, never seek positions
or FPS multiplication. Context outside an annotation remains unknown.

## Which failures are transfer failures?

| Source class | V4 recall, best epoch 4 | V5 recall, best epoch 4 | V4 recall, final epoch 40 | V5 recall, final epoch 40 |
|---|---:|---:|---:|---:|
| cross | 97.4% | 0.0% | 100.0% | 0.0% |
| rear_hook | 98.6% | 0.0% | 63.0% | 0.0% |
| rear_uppercut | 0.0% | 0.0% | 96.6% | 1.4% |

Cross/rear-hook failures are clear recording transfer failures even at the selected checkpoint.
The selected checkpoint **has not learned rear uppercuts on V4 either**; V5 macro-F1 chooses an
early epoch before that class is learned. Later V4 rear-uppercut learning still fails to transfer.
Simply training longer or selecting the final checkpoint does not solve the problem.

At epoch 4, 101/138 V5 crosses become lead_uppercut and 36 become lead_hook. Rear hooks split
28/39 lead_uppercut, 11/39 lead_hook. Rear uppercuts split 48/73 lead_uppercut, 25/73 lead_hook.
This is systematic source-conditioned confusion, with zero recall across those classes.

The failure is present before and after the known V5 cut at frame 10679. Scene 0 has 429 clips,
macro-F1 0.1781; scene 1 has 154, macro-F1 0.1876. All three failed classes have zero recall in
**both** scenes at the selected checkpoint. A single transition-local tracking failure cannot
explain the recording-wide result; the scene inventory is still not certified exhaustive.

## Viewpoint, pose geometry and visibility

Source sheets show V4 bag work directed predominantly toward **screen left**, while the sampled
V5 actions travel toward **screen right** in a different camera view. These are observations of
projected movement, not verified stance/handedness identities. Neither boxer/session identity nor
a recording-wide orthodox/southpaw mapping has been established. Pose-estimator anatomical
left/right assignments are estimates, not independently verified body-part identities.

Valid-only median wrist X relative to hips is generally negative in V4 and positive in V5.
For crosses, median-of-clip-median left/right wrist X is −0.857/−0.671 torso units in V4,
versus +0.432/+0.705 in V5. Conditional statistics for an often-hidden wrist describe only its
visible frames. The class-conditioned shoulder-width/torso ratio also differs: rear_hook
0.102 versus 0.234, rear_uppercut 0.154 versus 0.319. Rotation/foreshortening changes remain after
hip-centering and scalar normalization. Aligning a coordinate axis to a very narrow projected
shoulder span would itself risk instability.

Mean **per-clip valid-frame fraction**, selected clips only:

| Class | V4 L elbow / R elbow | V5 L elbow / R elbow | V4 L wrist / R wrist | V5 L wrist / R wrist |
|---|---|---|---|---|
| cross | 1.000 / 0.318 | 0.191 / 1.000 | 1.000 / 0.487 | 0.266 / 1.000 |
| rear_hook | 1.000 / 0.009 | 0.217 / 1.000 | 0.988 / 0.014 | 0.289 / 1.000 |
| rear_uppercut | 1.000 / 0.022 | 0.225 / 0.997 | 0.981 / 0.030 | 0.272 / 1.000 |

Both recordings retain all torso anchors on all selected frames. The characteristic hidden side
reverses. Frame-weighted below-threshold visibility explains roughly 68/99/98% of V4 right-elbow
frames for cross/rear_hook/rear_uppercut, versus 81/77/77% of V5 left-elbow frames. Presence never
falls below 0.5 for these elbow entries. The estimator frequently reports presence while regarding
the joint as occluded. Confidence thresholds expose a real view-dependent reliability issue;
this does not demonstrate that lowering the threshold would recover correct coordinates.
Confident but incorrectly placed or side-assigned joints remain possible; sheets cannot certify
all such cases.

Frozen TCN input interventions, with original source labels held fixed:

| Counterfactual input | V5 macro-F1, best | V5 macro-F1, final |
|---|---:|---:|
| Original | 0.1831 | 0.0763 |
| Reflect normalized X only | 0.0171 | 0.0071 |
| Exchange left/right joint channels only | 0.0113 | 0.1331 |
| Reflect X **and** exchange joint channels, masks and confidence | 0.4320 | 0.3782 |

At the best checkpoint the combined counterfactual raises cross recall to 81.9%, but rear-hook
recall is only 10.3% and rear-uppercut recall remains zero. It is incomplete and is **not a verified
V5 mapping or valid improved-model result**. It is a post-hoc, potentially out-of-distribution
sensitivity test on development data. The same transformation hurts V4 macro-F1 (0.5288→0.0717).
A blanket left/right swap is unsupported. The evidence supports testing training-side view
robustness, rather than adopting a V5-specific correction.

## Mask, length, temporal-order and padding shortcuts

The mask-only TCN learns V4 well (final macro-F1 0.6375), so masks/length/temporal boundaries carry
substantial training-label information without movement coordinates. Descriptive class MI of
quartile-binned wrist validity is 0.742 bits for V4 right wrist and 0.863 bits for V5 left wrist.
Exact frame count has descriptive MI 0.385/0.351 bits. These plug-in estimates use different
cardinalities and dependent clips; they are not causal effects, significance tests or directly
comparable feature-importance scores.

The frozen best mask-only TCN's V5 macro-F1 is 0.1742. Replacing **every real token with the same
all-valid mask**, preserving only each clip's length, still gives 0.1713 and changes just 5.8% of
predictions. On V4 the same intervention drops 0.5006→0.1002. The model's V5 collapse can therefore
be almost reproduced by a length/boundary-only input, while its V4 predictions rely strongly on
actual mask structure. This is a diagnostic counterfactual, not a trained length-only baseline.

With identical all-one tokens and no extra right padding, changing real length alone makes the
best mask-only TCN predict cross at lengths 3–4, jab at 5–10, and lead_hook at 11–32. Logits change
by up to 6.20. The static mask baseline is exactly invariant in this test. Convolutional left
padding and input/conv biases create a position-from-clip-start signal that remains after masked
mean pooling. This is an architectural **boundary shortcut**, rather than a bug admitting padded
batch frames into pooling.

Right-padding invariance holds to <1e-6 logit error; changing future inputs causes exactly zero
change in past encoded outputs. The static model is order-invariant as expected. Shuffling real
frames changes best full-TCN V4 macro-F1 only 0.5288→0.5215, and V5 0.1831→0.1868. Reversing time
gives 0.4897/0.1838. Temporal features do affect individual predictions (V5 shuffle changes 16.6%),
but correct temporal order provides little demonstrated cross-recording benefit. At the final
checkpoint V4 shuffled F1 remains 0.8395 versus 0.8680 original. Much of the solution is compatible
with posture, coverage and boundary cues rather than transferable punch dynamics.

Invalid-coordinate gating is necessary and correct. However, zero sentinels, explicit masks and
confidence channels all reveal missingness. The static coordinate mean/std uses all real frames,
so an intermittently missing joint's coordinate mean is also shrunk toward zero. Removing the
explicit mask channel alone would leave missingness encoded in coordinates/confidence; it would
also hide useful reliability information. There is no justification for trusting invalid joints.

## Class balance, duration and native frame rates

| Class | V4 count | V5 count | V4 median frames / center-span seconds | V5 median frames / center-span seconds |
|---|---:|---:|---|---|
| jab | 117 | 226 | 8 / 0.292 | 7 / 0.200 |
| cross | 39 | 138 | 6 / 0.209 | 8 / 0.233 |
| lead_hook | 140 | 68 | 8 / 0.292 | 11 / 0.334 |
| rear_hook | 73 | 39 | 7 / 0.250 | 10 / 0.300 |
| lead_uppercut | 71 | 39 | 7 / 0.250 | 8 / 0.234 |
| rear_uppercut | 87 | 73 | 7 / 0.250 | 9 / 0.267 |

Seconds are exact-PTS-derived **first-to-last sampled frame spans**, not verified biomechanical
punch durations or annotation-FPS conventions. Quantized PTS differences are handled using whole
source spans for effective rates, rather than taking the inverse of rounded 33/42ms steps.

Source effective decoded rates are 23.9760 and 29.9700 FPS. A 15-sample receptive field spans
**14 intervals**, about 0.5839 seconds in V4 versus 0.4671 in V5 (20% shorter on V5). The model
receives no delta-time channels; equal displacements over differing physical times are processed
at differing effective speeds. **99.43%** of V4 clips and **95.88%** of V5 clips have fewer than
15 samples. Mean per-clip fraction of encoded frames with a full 15-sample history is only
0.146%/0.639%. Nearly all pooled states are affected by clip-start initialization.

Duration/class relations also change: e.g. jab is shorter in V5, rear hooks longer. A uniform
frame-rate factor does not explain every difference. Correcting the model's sampling clock would
not establish the source authors' annotation frame rate, offsets or boundary conventions.

V4's most common class is lead_hook (26.6%); V5's is jab (38.8%). Cross grows from 7.4% to 23.7%.
Inverse-frequency training weights were correctly computed only from V4. Reweighting does not
supply missing viewpoints or recover the often-unobserved punching arm. A single source and seed
provide little evidence about generalization beyond this development pair.

## Implementation audit and remaining uncertainty

Verified on all selected examples:

- Model/feature source hashes match the frozen experiment snapshot; labels map to the saved
  six-class ordering and inspected literal class map with **zero mismatches**.
- Raw start/end equal the selected decoded interval start/inclusive end under the explicitly
  provisional convention, with **zero slicing mismatches**. Source arrays retain exact indices,
  PTS and rational time bases; no timestamp enters the classifier as metadata.
- Hip midpoint error is zero; normalized torso-length error ≤1.2e-7. All invalid joint feature
  entries are exactly zero. No invalid raw coordinate is silently used or filled.
- Normalization is frame-local and aspect-correct for both verified 3840×2160 sources. Median
  within-clip scale CV is 1.90%/1.76%; there is no recording-wide exploding-scale symptom.
- Saved checkpoint probabilities reproduce within 1.8e-7, with identical predicted classes.
  Masks are original binary core masks in the ablation. Models are in evaluation mode.
- Known scene/segment boundaries are not crossed by selected clips; no smoothing or interpolation
  occurs. Complete-clip pooling is Level 1 and does not establish online/early recognition.

No confirmed implementation defect was found in those checks. Sentinel-pooled statistics,
non-invariant geometry and sequence-start cues are consequential design choices. Unknown
stance, possible pose side-assignment errors, unreviewed action classes/onsets/endpoints,
frame-index base and author FPS remain unresolved. Representative sheets show motion near the
provisional intervals, including after the cut; they do not certify the full label alignment.
No automatic relabeling, offset, threshold change or class removal is supported.

## Ranked causes and next controlled improvements

1. **Viewpoint/body-orientation and hidden-arm reversal — strongest evidence.** Signed wrist
   geometry, class-conditioned shoulder projection, opposite-side masks, source sheets and the
   frozen reflection-plus-swap sensitivity all agree. Unknown stance/pose-side assignment may
   contribute, but is not a verified cause on its own.
2. **Mask/clip-length/causal-boundary shortcuts — directly demonstrated.** The model learns
   source-specific missingness; V5 predictions can largely be reproduced without missingness or
   motion, using clip length and boundary responses.
3. **Single-source overfitting and limited class/view coverage — directly observed.** Final V4
   macro-F1 0.868 versus V5 0.076, and 39 training crosses, constrain what this run can establish.
   Early V5 epoch selection also leaves rear uppercuts unlearned at the selected checkpoint.
4. **Different physical sampling clocks and action/annotation durations — measured contributor,
   effect size not isolated.** Short clips and weak order sensitivity suggest fixing FPS alone
   is unlikely to address the dominant failure.
5. **Residual alignment/label errors — still possible, not established as the dominant cause.**
   The failure spans both known scenes; available visual samples do not support a global offset.

Recommended **three single-factor comparisons**, after review/approval; none implemented:

1. **Training-only physical mirror augmentation.** Compare unchanged baseline against V4-only
   augmented training using consistent X reflection and anatomical left/right permutation of
   coordinates, confidence and masks. Verify the relative lead/rear taxonomy on the six priority
   sheets first. Keep V5 unchanged, and retain the original checkpoint comparison. No V5-specific
   canonicalization or blind anatomical relabeling. Keep architecture, class weights, policy,
   candidates, epoch budget and seed fixed. Primary outcomes: V5 macro-F1 and the three failed
   class recalls, with mask-only controls and early/final gaps.
2. **Class-independent structured visibility augmentation.** A separate comparison against the
   original baseline: predeclare label-independent elbow/wrist masking during V4 training, with
   consistent coordinate/confidence gating and artificial-mask provenance. Preserve reliable
   torso anchors, original records and all six classes. No interpolation or threshold lowering.
   Test whether mask-only discrimination and view-specific error asymmetry shrink without
   sacrificing useful valid-joint information. For the static baseline, valid-count coordinate
   moments are a subsequent isolated feature comparison, not bundled into this intervention.
3. **Remove the artificial clip-start reset cue with real causal pose context.** Independently
   supply up to 14 preceding real frames, bounded by known scene/segment starts; pool/classify
   only the original annotation interval. Keep those context frames explicitly unknown. Preserve
   absolute indices/PTS and gate unreliable detections. Keep native FPS unchanged initially to
   isolate this boundary intervention. Add context/mask-only controls because scripted neighboring
   punches can introduce a different shortcut. A common-clock, timestamp-based sampling experiment
   can follow separately; do not combine it with mirror or masking when attributing gains.

Run these as small, fixed-budget development comparisons before considering broader training
recordings. V9 remains reserved. Broader, permitted, verified training views will ultimately be
needed to establish whether a learned model transfers beyond this pair. There is no proposal to
increase model size, train longer on unchanged V4, or start early recognition on this evidence.

## Minimal manual review

Start with just these **three paired comparisons (six sheets)**. Each has five frames: context,
start, middle, end and context, plus a separate normalized-pose strip. Check projected punch
direction, whether anatomical left/right overlays follow the correct visible arm, and whether the
provisional interval contains the stated action. No new full-video annotation pass is requested.

| Pair | V4 source row / raw interval | V5 source row / raw interval |
|---|---|---|
| cross | 53 / 7837–7842 | 273 / 5469–5476 |
| rear_hook | 254 / 14472–14480 | 370 / 7645–7652 |
| rear_uppercut | 317 / 16093–16097 | 338 / 6860–6868 |

V4 row 317 is deliberately a **best-checkpoint error** (rear_hook prediction); there is no correct
rear-uppercut example at that checkpoint. Optional seventh sheet: V5 row 503, frames 11673–11680,
shows the same cross→lead_uppercut failure after the known cut. Correct V5 jab/lead-hook/
lead-uppercut examples and other errors are in the full 16-case index. Red source-overlay points
are invalid estimates, not trusted landmarks or fabricated joints. Context labels remain unknown.

## Verification

Full automated suite: **113 passed**, including five new synthetic diagnostic tests. Ruff and
`git diff --check`: **passed**. Tests cover non-mutating input perturbations, constant-token
length/boundary controls, padding and causal-prefix checks, order-invariant predictions, and
exact-PTS seek/forward decoding with rejection of a mismatched timestamp. All **38** hashed
original experiment artifacts remained unchanged; all 16 source/pose sheet pairs were generated.
Generated diagnostics/media and existing trained weights remain Git-ignored. Model architecture,
training features, thresholds, source data and V9 were not changed. No optimizer was run.
