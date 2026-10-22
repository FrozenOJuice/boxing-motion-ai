# Initial Level 1 experiment

Protocol frozen before the initial 40-epoch run, 2026-10-21. This is an authorized initial
six-class positive-clip recognition experiment. It does not complete the Level 1 product gate
(full-video event detection and independent test evaluation are still outstanding).

## Data and split

- V4: **527 training clips**; V5: **583 development clips**. V9 is reserved for future evaluation;
  its files are never opened by this experiment. No clip or frame from V5 enters optimization.
- All six classes are retained. No excluded annotation is restored, no original file is edited,
  and unlabeled footage is omitted as unknown. No `no_punch` examples exist in these experiments.
- Source masks, approved `mask_aware70` thresholds, joint order, input hashes and row selections
  are checked before training. Selection is recomputed from the original arrays. Source rows,
  uncertainties and excluded rows are recorded. Quality policy version is
  `recording-mask-policy-index-1.1`; legacy `v5-preparation-1.0` is explicitly supported alongside
  `recording-preparation-1.1`.
- Identities are recording V4/V5, with **boxer/session unknown**. This is recording-separated
  development, with no claim of boxer-independent performance. Source annotation classes,
  boundaries, indexing, trimming and scene inventory remain provisional.

| Class | V4 train | V5 development |
|---|---:|---:|
| jab | 117 | 226 |
| cross | 39 | 138 |
| lead_hook | 140 | 68 |
| rear_hook | 73 | 39 |
| lead_uppercut | 71 | 39 |
| rear_uppercut | 87 | 73 |

## Features and models

Feature schema `torso-core-2d-1.0`: shoulders, elbows, wrists and hips, fixed MediaPipe ordering.
Each frame independently uses aspect-correct image XY, centered at the hip midpoint and divided
by the shoulder-midpoint to hip-midpoint distance. All four torso anchors must be valid and the
scale must exceed 1e-6; otherwise that frame's normalized joint features are invalid. No motion
smoothing, temporal interpolation, lookahead, stance mirroring or imputation occurs. Invalid
coordinates/confidences become gated zero feature sentinels with explicit validity channels;
original raw coordinates/confidences, masks and NaNs remain in the read-only dataset loader.
The estimator's relative z is omitted from this first baseline.

The main input has 41 channels: XY, two confidence scores and a usable bit per core joint, plus a
torso-normalization-valid bit. The mask-only input has only the eight **original binary masks**,
with no coordinates, confidence, normalization geometry or timestamps. It cannot read source row,
recording identity or absolute time. Sequence length still influences coverage and temporal edge
patterns, which remain potential shortcuts.

Integer decoded indices, integer PTS and rational time bases remain available in each loader
item. `exact_times()` exposes exact rational seconds; floating seconds are also provided. No
FPS conversion, resampling or duration-derived index calculation is used. Padding is masked out
of pooling/loss, and clip labels are never applied to unknown background frames. Clips crossing
known scenes or extraction segments fail validation. Uncatalogued cuts remain a source limitation.

- **Static:** order-invariant per-channel mean and population standard deviation, then a linear
  six-class head. No temporal order information. Mask channels explicitly distinguish invalid
  coordinate sentinels in these pooled features.
- **TCN:** input projection to 32 channels; three residual, left-padded kernel-3 convolutions
  with dilations 1, 2, 4; channel-only LayerNorm, ReLU, dropout 0.1; masked clip-mean pooling and
  six-class head. Temporal encoding has a 15-frame receptive field. Pooling uses the complete
  annotated clip: this is **Level 1**, not an early-recognition evaluation. Native frame rates
  differ (V4 approximately 23.976 vs V5 approximately 29.970), so the temporal field's physical
  duration differs. No unsupported annotation-FPS equivalence is inferred.

Both models also run with mask-only input under the same split, seed and epoch budget.
The static model is deliberately simple; temporal model superiority is not assumed.

## Reproduction and artifacts

Run from the repository root, using fresh output directories:

```sh
uv sync --locked --extra pose --extra train
uv run --extra pose --extra train pytest
uv run ruff check .
uv run --extra pose --extra train boxing-motion-ai train-level1 --config configs/level1_smoke.yaml --output reports/boxingvi/level1-smoke-repeat
uv run --extra pose --extra train boxing-motion-ai train-level1 --config configs/level1_initial.yaml --output reports/boxingvi/level1-initial-repeat
```

The smoke configuration uses two evenly spaced examples per class per split for two epochs;
its metrics are diagnostics only. Initial training is 40 epochs, batch 32, AdamW learning rate
0.001, weight decay 0.001, gradient norm cap 1, seed 20261021, deterministic CPU algorithms,
two threads and zero loader workers. Class-weighted cross-entropy uses inverse V4 class
frequencies only; there is no resampling or validation-derived weighting. No hyperparameter
search or post-result retuning was performed. Best epoch is selected by V5 macro-F1, ties earliest.
The final epoch is also retained to expose later overfitting. There is no independent test result.

Each output contains configuration, environment and dataset provenance, package source snapshot,
lockfile, source hashes and selected rows. Every run stores `curves.csv`, `curves.png`,
`confusion.png`, `metrics.json`, source-row predictions and `best.pt`/`last.pt` checkpoints
(model, optimizer, Torch/loader RNG, seed, configuration and provenance digest). CPU seeds are
set for Python, NumPy and Torch. Bitwise reproducibility across different platforms or PyTorch
versions is not promised; see [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
The TCN explicitly pads only the past, rather than relying on symmetric convolution padding;
see [Conv1d documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html).

All generated artifacts, data, environments and checkpoints are ignored by Git. Original raw
media and annotation files are not opened for training; cached prepared inputs are verified
against their recorded hashes. Inputs are rechecked at the end to detect concurrent changes.

## Measured initial results

Full run: `reports/boxingvi/level1-initial-v1/`. Completed smoke: `reports/boxingvi/level1-smoke-v2/`. A repeat of the earlier smoke produced identical metrics for all four runs.

Best checkpoints, selected independently by V5 macro-F1:

| Run | Epoch | V4 accuracy | V4 macro-F1 | V5 accuracy | V5 macro-F1 |
|---|---:|---:|---:|---:|---:|
| static | 6 | 47.25% | 0.4200 | 37.74% | 0.1326 |
| tcn | 4 | 55.22% | 0.5288 | 30.53% | 0.1831 |
| static_mask_only | 3 | 23.91% | 0.1200 | 27.27% | 0.1748 |
| tcn_mask_only | 23 | 53.13% | 0.5006 | 41.51% | 0.1742 |

The training-majority class is **lead_hook** (140/527 V4 clips). Predicting that class on V5 gives 11.66% accuracy and macro-F1 0.0348. V5 instead has jab as its dominant class (226/583, 38.77%); that is a descriptive validation prior, not a trained reference.

Selected TCN per-class development metrics:

| Source class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| jab | 0.8138 | 0.5221 | 0.6361 | 226 |
| cross | 0.0000 | 0.0000 | 0.0000 | 138 |
| lead_hook | 0.2126 | 0.5441 | 0.3058 | 68 |
| rear_hook | 0.0000 | 0.0000 | 0.0000 | 39 |
| lead_uppercut | 0.0902 | 0.5897 | 0.1565 | 39 |
| rear_uppercut | 0.0000 | 0.0000 | 0.0000 | 73 |

TCN confusion matrix (rows = source label; columns = prediction), ordered jab, cross, lead_hook, rear_hook, lead_uppercut, rear_uppercut:

```text
118   7  51   2  48   0
  1   0  36   0 101   0
 24   0  37   0   7   0
  0   0  11   0  28   0
  2   0  14   0  23   0
  0   0  25   0  48   0
```

### Findings and stopping decision

- **Severe overfitting and recording transfer failure:** the full-feature TCN ends at V4 macro-F1 0.8680 versus V5 0.0763. Its development loss rises from 2.2338 at the selected epoch 4 to 4.9150 at epoch 40, while training loss falls from 0.8742 to 0.2920. The selected checkpoint is substantially earlier than the end. This is not a usable six-class recognizer.
- **Class collapse:** at the selected TCN checkpoint, cross, rear_hook and rear_uppercut have zero recall. Of 138 crosses, 101 become lead_uppercut and 36 become lead_hook. Of 73 rear uppercuts, 48 become lead_uppercut and 25 become lead_hook. The static model mostly predicts jab and lead_hook; its 37.74% accuracy conceals macro-F1 0.1326 and zero recall for four classes. Accuracy alone is misleading.
- **Shortcut evidence:** without coordinates or confidence, the mask-only TCN reaches V4 training macro-F1 0.6375 by epoch 40. Its selected V5 macro-F1 0.1742 is close to the full-feature TCN's 0.1831; static mask-only scores 0.1748 versus full static 0.1326. These results provide no convincing evidence of transferable movement recognition. Mask-only TCN accuracy 41.51% is dominated by jab and lead_hook, with four classes at zero recall. It is not evidence of broad six-class success. Mask patterns, clip lengths and temporal padding edges are not disentangled by this ablation; missingness alone is not proven causal.
- **Likely contributors, not established causes:** previous audits found opposite-side arm occlusion in V4 versus V5. View/stance differences, provisional annotation alignment and native-FPS differences may also matter. This experiment does not establish which factor causes each error and does not justify swapping left/right labels or applying offsets.
- **No suspiciously high development performance:** the concerning high numbers are in-sample training scores and an accuracy value inflated by class collapse. V5 has been used for epoch selection, so these development metrics are optimistic rather than an independent evaluation. One seed and dependent clips do not support significance claims.
- **Stop for review:** retain the baseline artifacts. Before any model improvement, review cross/rear-hook/rear-uppercut V5 errors using recorded source rows, revisit paired V4/V5 view/handedness and alignment evidence, and agree on controlled mask/clip-length experiments. Do not access V9 for diagnosis or model selection. Do not move to early recognition or full-video deployment on this evidence.

### Verification

Automated tests: **108 passed** (including nine new synthetic training/loader/model tests). Ruff: **passed**. CLI help works. Tests cover exact rational timing, stale hashes, fixed splits, policy changes, known scene boundaries, source exclusions, invalid-coordinate gating, frame-local normalization, padding invariance, temporal causality, finite gradients, metrics and checkpoint/report writing. All prepared input hashes matched again after experiments. Git tracks no dataset files, reports, environments or checkpoints.

## Follow-up diagnosis

The [frozen-checkpoint failure diagnosis](LEVEL1_FAILURE_DIAGNOSIS.md) compares V4/V5 viewpoint,
visibility, duration and boundary shortcuts, reproduces saved predictions, and provides 16 source
video/pose examples. It recommends three controlled comparisons for review; no further training
or model/feature changes were performed during that diagnosis.

## Controlled mirror experiment

[Experiment 1](LEVEL1_MIRROR_STUDY.md) adds optional training-only physical reflection and compares
three matched seeds for original/mirrored full-feature and mask-only TCNs. Initial experiment
artifacts remain unchanged; the mirror study uses fresh ignored output directories.

## Controlled visibility experiment

[Experiment 2](LEVEL1_VISIBILITY_STUDY.md) first inspects lead-uppercut failures, then compares
mirror-only against mirror plus temporary, class-independent wrist/forearm masking. It retains
the three seeds, existing architecture/data/policy and mask-only controls, without pre-clip context.

## Controlled pre-clip context experiment

[Experiment 3](LEVEL1_CONTEXT_STUDY.md) compares mirror-only against real causal context from the
same full-video poses, with exact timestamps and pooling only on original punch frames. It keeps
visibility augmentation disabled, the original eligible examples, architecture and three seeds.

## Recording-group expansion

The [V7/V8 expansion](LEVEL1_DATA_EXPANSION.md) keeps mirror-only as reference, adds new
training sources behind a frozen provisional alignment/data gate, and keeps V7/V8 in one
unresolved source group. V5 remains development; V9 is not opened. The comparison retains the
TCN/features/policy and three seeds, with matched mask-only controls. Visibility augmentation
and pre-clip context remain disabled.
