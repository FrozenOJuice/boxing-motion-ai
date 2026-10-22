# Experiment 2: mirror + temporary visibility augmentation

2026-10-21. The policy and comparisons below were fixed before full training. Only temporary
visibility augmentation is added to the accepted mirror pipeline. No architecture, feature
representation, source quality policy, data split or context-window change is part of this study.

## Lead-uppercut inspection before implementation

Read the mirror study and examined its three full-feature mirrored checkpoints, all V5
lead-uppercut predictions, source/canonical label maps and six representative source/pose strips.
New read-only evidence is under `reports/boxingvi/level1-visibility-preflight-v1/`:
`preflight.json`, `visual-cases.json`, and `review/` (30 exact-PTS source frames plus pose strips).
The selected source rows are V4 **66, 344, 559** (correct at seed 20261021's selected epoch 31)
and V5 **187, 196, 570** (lead_hook errors at that checkpoint). They span different parts of the
recordings. Labels and inclusive raw-index boundaries remain provisional.

| Mirrored seed | Epoch | V4 lead-uppercut recall | V5 recall | V5 counts predicted jab / lead_hook |
|---|---:|---:|---:|---|
| 20261021 | 31 | 88.7% | 0.0% | 5 / 34 |
| 20261022 | 31 | 95.8% | 0.0% | 33 / 6 |
| 20261023 | 28 | 78.9% | 0.0% | 29 / 10 |

Verified mapping: source `Lead Uppercut` maps to canonical `lead_uppercut` and model index 4 in
both datasets. All 527/583 selected source-to-canonical labels match their recorded maps; no
class-index mismatch was found. V4 supplies 71 lead-uppercut training clips with positive
inverse-frequency weight. The class is learned on V4 and not accidentally absent from training.
Previous checks establish raw-index slicing, correct torso normalization, mask gating and
consistent physical mirror semantics. V5 row 187 was recognized as lead_uppercut by the initial
unaugmented model but becomes lead_hook under the mirrored model, with unchanged source data.
This is not evidence of a newly introduced dataset-wide label permutation.

A strong remaining **class-conditioned visibility mismatch** survives mirroring. Mean per-clip
fraction of frames with both elbows valid:

| Source class | V4 | V5 |
|---|---:|---:|
| jab | 86.3% | 96.4% |
| lead_hook | 60.9% | 96.5% |
| lead_uppercut | 38.7% | 98.9% |
| rear_hook | 0.9% | 21.7% |
| rear_uppercut | 2.2% | 22.2% |

Mirroring changes which side is hidden, not the number of visible joints. The learned rule can
associate missingness/pose coverage with classes and prefer jab/lead_hook for well-observed V5
uppercuts. This is a suspected contributor, not a proven sole cause. Median observed wrist
vertical ranges for lead-uppercut clips are similar (0.225 versus 0.232 torso units), conditional
on valid observations across either wrist; they do not identify the actual punching arm.

The reviewed source intervals show compact upward/bent-arm motions in different camera views
and styles. These sparse images do not independently certify every class, punching hand or
biomechanical boundary. Pose-side assignment mistakes, annotation ambiguity and lost visual
information remain possible. No confirmed label/implementation defect or evidence supporting
automatic relabeling was found. No original annotations are corrected. Visibility augmentation
can weaken a shortcut but **cannot create the originally missing reliable training landmarks**.

## Fixed augmentation policy

`temporary-forearm-occlusion-1.0`, applied after the unchanged 50% whole-clip mirror augmentation:

- Independently select each V4 optimization clip with probability **0.5**, regardless of label,
  existing validity, source time or mirror decision.
- Choose left/right with equal probability. Choose wrist-only or elbow-and-wrist with equal
  probability, corresponding to a joint occlusion or a co-occluded forearm.
- Choose one contiguous temporal span uniformly between **20% and 50%** of the real clip's length,
  round up, minimum one frame, cap at length−1. Choose a valid start uniformly. This models a
  temporary occlusion, never a full-clip erasure or masking outside the sequence.
- Set all five model feature channels of affected joints to zero: coordinates, confidence and
  validity. Set corresponding mask-only channels to zero with the same event. No invalid entry
  ever becomes valid, no position is filled, and no confidence threshold is lowered.
- Protect shoulders, hips and the torso-normalization-valid bit. Keep labels, source arrays,
  timestamps, indices, sequence lengths, padding and original quality selections unchanged.
- If a sampled joint is already invalid, keep the no-op event in the audit. Do not resample based
  on visibility, because that would make the event distribution class-dependent.
- Evaluate original V4 and V5 clips, without either augmentation. Do not reapply source retention
  thresholds to synthetic occlusions or drop training examples/classes.

An independent Torch generator (seed + 2000003) isolates visibility choices from mirroring
(seed + 1000003), initialization, dropout and shuffling. Full/mask-only controls share identical
augmentation schedules. Per-event row/span/joint and newly hidden usable/source-entry counts are
logged. RNG states are saved in checkpoints. Version-1.0/1.1 experiment configs keep visibility
augmentation disabled; version 1.2 explicitly requires probability 0 or 0.5.

## Matched experiment

Three fixed seeds: **20261021, 20261022, 20261023**. Each has two conditions and two inputs:

1. Mirror-only full TCN and mask-only TCN (rerun controls).
2. Mirror + visibility full TCN and mask-only TCN.

This is **12 runs**, each with the same 40 epochs, 527 V4 training clips, 583 V5 development clips,
model, 41/8-channel features, optimizer, class weights and remaining hyperparameters as the
accepted mirror study. No training-data expansion, pre-clip context, resampling or model-size
change. V9 remains reserved and unopened. Each mirror-only control must reproduce its previous
metrics exactly. Initial weights, source hashes/rows/policy, training order and mirror schedules
must match in each pair; full/mask-only visibility events must match. Previous mirror-study files
are hashed before and after. All original files and previous experiments remain read-only.

Primary outcome: V5 macro-F1 at the independently selected best V5 epoch (earlier ties), with
accuracy, all class recalls, confusion, final-epoch behavior and mask-only comparisons. Inspect
lead_uppercut, rear_hook and rear_uppercut explicitly; accuracy increases do not establish
balanced six-class improvement. V5 is repeatedly selected development, not an unbiased test.
Three seeds on the same recording pair do not support boxer-independent or significance claims.

## Reproduction and outputs

Run from the repository root, using fresh output directories:

```sh
uv sync --locked --extra pose --extra train
uv run --extra pose --extra train boxing-motion-ai train-visibility-study --help
uv run --extra pose --extra train boxing-motion-ai train-visibility-study --config configs/level1_visibility_study.yaml --output reports/boxingvi/level1-visibility-smoke-repeat --smoke
uv run --extra pose --extra train boxing-motion-ai train-visibility-study --config configs/level1_visibility_study.yaml --output reports/boxingvi/level1-visibility-repeat
uv run --extra pose --extra train pytest
uv run ruff check .
```

`--smoke` uses the first seed, two clips/class and two epochs in all four combinations; its
metrics are diagnostic only. Full outputs: `reports/boxingvi/level1-visibility-v1/`, containing
summary JSON/Markdown, per-class recall/confusion plots, matched configurations, per-run
provenance/source snapshots, predictions, per-class metrics/confusions, loss/F1 curves,
augmentation logs and best/final checkpoints. Data, weights and generated artifacts remain
Git-ignored. No visibility hyperparameter tuning is authorized within this initial comparison.

## Measured results

Completed the smoke comparison and all **12** full runs. Every mirror-only control reproduced its corresponding previous seed/model metrics **exactly**. Paired initial weights, shuffle orders, source data/policy selections and mirror schedules matched. Full/mask-only visibility events matched. No policy, probability, seed or budget was changed after observing results.

Best-checkpoint V5 metrics, **mean ± sample standard deviation** over the same three seeds. The same 583 development clips are reused; this is not independent held-out evaluation.

| Input | Condition | Accuracy | Macro-F1 |
|---|---|---:|---:|
| tcn | mirror_only | 71.87% ± 3.35% | 0.4912 ± 0.0188 |
| tcn | mirror_visibility | 70.61% ± 1.24% | 0.4879 ± 0.0173 |
| tcn_mask_only | mirror_only | 56.89% ± 2.53% | 0.3546 ± 0.0187 |
| tcn_mask_only | mirror_visibility | 53.92% ± 0.99% | 0.3103 ± 0.0309 |

Full-feature paired macro-F1 changes were **−0.0322, +0.0192, +0.0032**; mean **−0.0033** (sample SD 0.0263). This is no consistent improvement with this fixed policy. The small mean change does not support a significance claim. Selected epochs changed from 31/31/28 (mirror-only) to 23/28/28 (mirror+visibility).

### Six-class coverage

| Source class | Full mirror recall | Full mirror+visibility recall | Mask mirror recall | Mask mirror+visibility recall |
|---|---:|---:|---:|---:|
| jab | 91.4% | 90.7% | 92.3% | 92.9% |
| cross | 96.4% | 94.2% | 54.3% | 48.3% |
| lead_hook | 80.9% | 83.3% | 38.7% | 24.5% |
| rear_hook | 21.4% | 25.6% | 17.1% | 22.2% |
| lead_uppercut | 0.0% | 0.0% | 0.9% | 0.0% |
| rear_uppercut | 21.9% | 13.7% | 20.1% | 16.9% |

- **Lead uppercut remains a complete failure:** 0/39 correct in every selected full model under either condition. With visibility augmentation, V4 lead-uppercut recalls are 91.5%, 83.1%, 71.8%, while V5 stays zero. V5 errors shift toward lead_hook: across seeds 76.9% lead_hook and 23.1% jab, compared with 42.7%/57.3% under mirror-only. The class head is active: 48.9% of V5 rear uppercuts are predicted as lead_uppercut under the new condition. This is not an absent output class; learned class/view/pose distinctions remain wrong.
- **Rear hook gains are limited and inconsistent:** mean recall 21.4%→25.6%, with paired seed values 5.1→7.7%, 20.5→48.7%, 38.5→20.5%. Mean confusion is 25.6% lead_uppercut, 23.9% rear_uppercut, 12.8% cross, 12.0% lead_hook and 25.6% correct. One seed loses substantial recall despite the mean increase.
- **Rear uppercut worsens:** mean recall 21.9%→13.7%, with paired seed values 46.6→21.9%, 15.1→11.0%, 4.1→8.2%. Confusion with lead_uppercut rises from 27.9% to 48.9%. This is a consequential class tradeoff, not progress toward reliable rear-punch recognition.
- **Other classes:** jab falls slightly (91.4→90.7%), cross falls (96.4→94.2%), while lead_hook rises modestly (80.9→83.3%). Worst-class recall stays zero in all full-model runs. Mean macro recall falls from about 52.0% to 51.3%.

**The policy does not improve balanced six-class recognition, and does not even increase mean overall accuracy in this comparison.** A small rear-hook mean gain cannot compensate for continued lead-uppercut failure and lower rear-uppercut recall.

### Shortcut controls

Mask-only macro-F1 declines in all seeds: paired changes −0.0537/−0.0303/−0.0490; mean **−0.0443** (sample SD 0.0124), from 0.3546 to 0.3103. Full-feature performance remains higher than mask-only in every seed. The mean margin grows from 0.1366 to 0.1776, mainly because the mask control gets worse while full-model performance is essentially flat. This is consistent with weakened visibility cues or harder mask-only learning, but does not establish reduced shortcut reliance inside the full model or better temporal movement understanding. Masks retain original class-correlated missingness, clip length and causal start-boundary signals; adding occlusions cannot remove all of these.

### Overfitting and realized augmentation

At epoch 40, mean full-feature training/development macro-F1 changes from **0.8044/0.4248** (mirror-only) to **0.7868/0.4039** (mirror+visibility). The mean gap is essentially unchanged: 0.3796→0.3829. All new final development scores are below their selected best scores. The intervention reduces training fit a little without improving final development fit. Mask-only final training/development means fall from 0.5719/0.3229 to 0.4871/0.2833.

Per full model, 10,599 / 10,740 / 10,523 of 21,080 presentations received a scheduled visibility event (50.28% / 50.95% / 49.92%). Approximately 18–19% of scheduled events hid no additional usable entry because those selected joints were already invalid. Those events were not resampled. Newly hidden usable joint-frame entries were 38,506 / 38,895 / 38,365, about 3% of all core joint-frame slots presented during training. The identical schedules and counts apply to the matching mask-only controls. Actual loss of valid entries naturally depends on original visibility, while event selection is label- and visibility-independent.

### Interpretation and stopping decision

**Retain mirror-only as the current reference. Keep visibility augmentation optional and do not promote this tested policy to the default.** It has not rescued the three difficult classes or improved balanced recognition. This result does not rule out all occlusion policies, but no rate/span tuning or additional training was performed within this study.

The preflight visibility mismatch remains relevant: artificial occlusion can only remove information; it cannot supply the reliably visible training poses, camera/stance coverage or semantics absent from the original recording. Missing stance/lead-hand context, 2D viewpoint ambiguity, pose assignment errors and annotation uncertainty remain possible contributors. No specific label swap, new offset or annotation correction is verified. Preserve the prediction/source-row artifacts for targeted future review rather than beginning a broad reannotation pass. No pre-clip context, architecture change or data expansion was implemented. Stop for review.

### Verification

**124 tests passed; Ruff and `git diff --check` passed.** New tests verify that masks only decrease, hidden coordinates/confidence/validity are jointly gated to zero, torso anchors and the normalization flag are protected, real sequence boundaries/padding/timestamps and labels remain unchanged, private RNG can be restored without changing mirror choices, schedules are independent of labels and existing missingness (including no-ops), disabled mode consumes no RNG, probabilities are validated, and evaluation uses unaugmented source clips. Source data/policy hashes and previous mirror-study files remain unchanged. Weights, generated media and reports remain Git-ignored. V9 was not opened.
