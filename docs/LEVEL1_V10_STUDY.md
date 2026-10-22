# V4-only versus V4+V10 mirror-only Level 1 study

Adding V10 improves **rear-uppercut recall in all three seeds**, but does **not**
improve balanced six-class recognition. Mean macro-F1 falls from **0.4912 to
0.4848**, accuracy falls from **71.87% to 68.15%**, and lead-hook recall regresses
in every seed. Keep **V4-only mirror augmentation as the reference**. Retain this
expanded experiment and V10 preparation for comparison rather than promoting it.

## Frozen comparison and data integrity

The unchanged `configs/level1_v10_proposed.yaml` is the starting proposal. Its
accepted paths/counts/readiness hashes are checked and translated into a runnable
study saved with each run. The separate `v10-training-gate-1.0` freezes V4/V10
training, V5 development, source groups, dataset/policy/pose hashes and all retained
V10 row identities. Historical V7/V8 gates and their source-group dictionary retain
their original contract. V7/V8 are not loaded for this experiment; V9 inputs are
not opened. Reading archived experiment artifacts for preservation is not using
their recordings as model inputs.

| Class | V4 train | V10 additions | Expanded train | V5 development |
|---|---:|---:|---:|---:|
| jab | 117 | 49 | 166 | 226 |
| cross | 39 | 36 | 75 | 138 |
| lead_hook | 140 | 25 | 165 | 68 |
| rear_hook | 73 | 6 | 79 | 39 |
| lead_uppercut | 71 | 4 | 75 | 39 |
| rear_uppercut | 87 | 21 | 108 | 73 |
| total | 527 | 141 | 668 | 583 |

V10 rows **9, 10, 123 and 130–137** remain excluded. No annotation, frame bound,
coordinate, confidence, mask or source timing is changed. Labels remain provisional;
unknown footage is omitted, never assigned `no_punch`. Recording IDs remain separate
from conservative source groups and unknown boxer/session IDs. V10 is
`utah-lee-unresolved`, V4 `fightcamp-unresolved`, V5
`self-improvement-boxer-unresolved`. These do not establish boxer independence.

The frozen architecture, `torso-core-2d-1.0` features, normalization, timing,
collation, causal 15-frame receptive field, punch pooling and `mask_aware70` policy
are unchanged. The runner checks feature/timing function ASTs and the model file
against the preserved mirror study. Mirror probability remains 0.5, with consistent
coordinate/confidence/mask left/right reflection and unchanged relative class labels.
Visibility augmentation and pre-clip context remain disabled.

Each arm trains full-feature and mask-only TCNs at seeds **20261021, 20261022,
20261023**: 12 models, 40 epochs each, batch 32, hidden 32, dropout 0.1, AdamW
learning rate/weight decay 0.001 and two CPU threads. Class weights use the same
inverse training-frequency formula. V5 selects the best macro-F1 epoch with earlier
ties retained; its 583 examples never enter optimization. Paired initial model hashes
match, and full/mask shuffle and mirror logs match within every seed/arm.

Adding data changes the class mixture, inverse-frequency weights and number of
optimizer steps: **680 per V4-only model versus 840 per expanded model**. The epoch
budget is unchanged. This experiment does not isolate those effects from viewpoint,
execution style, native frame-rate differences or annotation uncertainty.

The real-data smoke test passed first: one seed, two epochs, at most two examples
per available class per recording. Smoke scores are plumbing checks, not performance
evidence. All complete V4-only controls reproduce the preserved mirror-reference
selected epochs, training/development metrics and final-epoch metrics **exactly**.

## Development results

Values below are means over the three matched seeds; ± is sample standard deviation
over seeds, not an uncertainty interval across independent recordings.

| Input | Training | V5 macro-F1 | V5 accuracy | Final train F1 | Final V5 F1 |
|---|---|---:|---:|---:|---:|
| Full poses | V4 | 0.4912 ± 0.0188 | 71.87% | 0.8044 | 0.4248 |
| Full poses | V4+V10 | 0.4848 ± 0.0209 | 68.15% | 0.8654 | 0.4453 |
| Masks only | V4 | 0.3546 ± 0.0187 | 56.89% | 0.5719 | 0.3229 |
| Masks only | V4+V10 | 0.3157 ± 0.0115 | 50.14% | 0.4883 | 0.2740 |

| Seed | V4 full F1 | Expanded full F1 | Paired change | Selected expanded epoch | Expanded rear-uppercut recall |
|---|---:|---:|---:|---:|---:|
| 20261021 | 0.5111 | 0.4894 | −0.0217 | 40 | 52.05% |
| 20261022 | 0.4887 | 0.4620 | −0.0267 | 40 | 54.79% |
| 20261023 | 0.4737 | 0.5029 | +0.0293 | 26 | 45.21% |

Only one seed improves overall macro-F1. The paired mean change is **−0.0064**,
sample SD **0.0310**; this is not a reliable overall improvement. Accuracy changes
are −8.40, −3.60 and +0.86 percentage points respectively. No test of statistical
significance across unseen recordings is supported by three seeds on one V5 video.

Mean per-class development precision/recall, full-feature TCN:

| Class | V4 precision | Expanded precision | V4 recall | Expanded recall | Recall change (pp) |
|---|---:|---:|---:|---:|---:|
| jab | 84.54% | 95.12% | 91.45% | 85.55% | −5.90 |
| cross | 86.61% | 63.83% | 96.38% | 94.93% | −1.45 |
| lead_hook | 52.94% | 38.27% | 80.88% | 35.78% | −45.10 |
| rear_hook | 36.75% | 35.97% | 21.37% | 18.80% | −2.56 |
| lead_uppercut | 0.00% | 12.17% | 0.00% | 11.11% | +11.11 |
| rear_uppercut | 55.88% | 59.38% | 21.92% | 50.68% | +28.77 |

Rear-uppercut recalls improve from **46.58/15.07/4.11%** to
**52.05/54.79/45.21%**. The direction is consistent, but the paired gains vary
substantially (+5.48/+39.73/+41.10 pp). Precision improves on average, though not
in every seed. This is a useful targeted V5 result, not a balanced model improvement.

Rear-hook recalls are **10.26/15.38/30.77%** after expansion, versus
5.13/20.51/38.46%; two seeds worsen. Lead-uppercut recalls are
**30.77/2.56/0.00%**: the mean gain is driven mainly by one seed, and the class
remains effectively unrecognized in another. Four new examples in one V10 sequence
do not establish robust lead-uppercut diversity.

Jab recall worsens in **all three seeds**, even though its precision improves.
Lead-hook recalls fall from 86.76/83.82/72.06% to **30.88/25.00/51.47%**.
Cross recall remains high, but its precision loses 22.78 pp due to false positives.

## Confusion, shortcuts and overfitting

Every model retains its full precision/recall/F1/support metrics, probabilities,
confusion matrix, curves and best/final checkpoints. The comparison also contains
`confusion-comparison.png`, a visually checked four-panel mean-count matrix.
Each mean matrix still represents 583 V5 examples, not 1,749 independent tests.

The expanded full model predicts, on average per seed:

- **33.7 of 68 lead hooks as crosses**, versus 0.7 before expansion. Correct lead
  hooks fall from 55.0 to 24.3.
- **22.3 of 39 rear hooks as rear uppercuts**, versus 12.7 before expansion.
- **37.0 of 73 rear uppercuts correctly**, versus 16.0. Remaining rear uppercuts
  include 14.0 called lead uppercuts and 8.7 called rear hooks.
- **4.3 of 39 lead uppercuts correctly**; 17.0 become lead hooks and 15.0 crosses.

These errors expose changed class boundaries and a rear-uppercut prediction bias;
they do not authorize relabeling any source annotations.

The expanded mask-only model's mean rear-uppercut recall reaches **48.40%**, close
to the full model's 50.68%. Its rear-uppercut precision is only **28.93%**, versus
59.38% with full features. Masks/length/boundaries can therefore reproduce much of
the recall gain, but full features are substantially more selective. Do not claim
all improvement is movement learning, or that masks alone explain every prediction.
Mask-only inputs retain sequence length and padding/boundary cues, so this control
cannot distinguish those shortcuts from class-dependent missing joints.

Mask-only aggregate F1 decreases in all seeds (paired mean −0.0389). The full-feature
gain minus mask-only gain is +0.0325 on average, but the full-feature aggregate gain
itself is negative. This difference is descriptive, not a causal movement estimate
or evidence that shortcut reliance has been resolved.

Overfitting persists. The mean final train/development F1 gap increases from
**0.3796 to 0.4201**. Different training populations make training F1 imperfectly
comparable; V5 is the common development population. The third expanded seed peaks
at epoch 26 (F1 0.5029), then falls to **0.3846** while train F1 reaches **0.8697**.
The other two seeds select epoch 40, the existing budget boundary; no additional
epochs or selection changes were tried after seeing these results.

V5 has been used repeatedly for development and epoch selection. These are optimistic
development results, not an unbiased test, broad generalization or boxer-independent
success. V9 remains reserved and untouched.

## Artifacts, reproduction and disposition

Local ignored artifacts:

- `reports/boxingvi/level1-v10-data-v1/training-gate.json`: frozen V4/V10 gate.
- `reports/boxingvi/level1-v10-smoke-v1/`: real-data smoke study.
- `reports/boxingvi/level1-v10-v1/summary.md`, `summary.json`, `runs.json`:
  all seed/arm/control results and reference checks.
- `comparison.md`, `comparison.json`, `confusion-comparison.png`: class precision,
  recall, paired gains, mean confusion and overfitting summaries.
- Each seed/arm/model directory: predictions, metrics, curves, configuration,
  source snapshot, augmentation logs and best/final checkpoints.

For a new reproduction, use fresh ignored paths and the same accepted proposal:

```sh
uv run python -m boxing_motion_ai.training.v10_study \
  --proposal configs/level1_v10_proposed.yaml --freeze-gate \
  --output reports/boxingvi/level1-v10-data-repeat/training-gate.json
uv run python -m boxing_motion_ai.training.v10_study \
  --gate reports/boxingvi/level1-v10-data-repeat/training-gate.json \
  --output reports/boxingvi/level1-v10-smoke-repeat --smoke
uv run python -m boxing_motion_ai.training.v10_study \
  --gate reports/boxingvi/level1-v10-data-repeat/training-gate.json \
  --output reports/boxingvi/level1-v10-repeat
uv run python -m boxing_motion_ai.training.v10_results \
  --directory reports/boxingvi/level1-v10-repeat
uv run pytest
uv run ruff check .
```

V10 provides useful targeted rear-uppercut variation, with a substantial tradeoff
in lead-hook recognition and continued shortcut risk. Do **not** replace the
V4-only reference with this mixture. Preserve both results for review before any
further data-selection, loss, architecture, augmentation or training experiment.
No additional intervention is implemented here.

Validation: `uv run pytest -q` **176 passed**; `uv run ruff check .` and
`git diff --check` passed. Synthetic checks cover V4/V10 loading and training,
reserved/mixed split rejection before input reads, unchanged-policy gate rejection,
recording-qualified predictions and paired-seed/mean-confusion calculations.
The final verification checks all 12 models' training/development identities,
candidate counts, exclusion preservation and exact reference reproduction.
All **2,326** prior experiment files match their pre-study hashes; frozen source,
pose, preparation, policy and readiness hashes also match. Generated reports,
datasets, weights and the local environment remain ignored and untracked.
