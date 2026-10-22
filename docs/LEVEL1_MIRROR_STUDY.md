# Experiment 1: training-only horizontal mirroring

2026-10-21. Protocol fixed before the full study. This implements only the accepted mirror
comparison after [the failure diagnosis](LEVEL1_FAILURE_DIAGNOSIS.md). Visibility augmentation,
pre-clip context, feature redesign and architecture changes are outside this experiment.

## Protocol and reflection semantics

V4 remains the sole training recording (527 approved clips), V5 development (583), and V9
reserved and unopened. All six provisional source classes and `mask_aware70` selections remain
unchanged. Original media, annotations, extracted poses, policy indices and previous experiments
are read-only. No unknown/background frame receives a punch or `no_punch` label.

`horizontal-core-reflection-1.0` works on copies of existing `torso-core-2d-1.0` features:

- Reflect hip-relative normalized horizontal coordinates as **X → −X**. This is equivalent
  to reflecting source image X → 1−X before the unchanged torso normalization.
- Exchange anatomical left/right shoulders, elbows, wrists and hips, including **all** five
  channels per joint: X, Y, visibility, presence and usable mask. The torso-valid bit is invariant.
- Mask-only input exchanges the same eight anatomical validity channels. Missing coordinates
  remain gated zeros with masks; there is no imputation or threshold adjustment.
- Make one independent Bernoulli(0.5) decision per whole clip each training presentation, shared
  by all its frames and both full/mask-only feature views. Time order, length, indices and PTS
  are unchanged. Evaluation of V4 and V5 always uses original unaugmented clips.
- Preserve **all punch class labels**. Physical reflection changes anatomical handedness of the
  imagined boxer but preserves which hand is leading/rearward. A jab remains a jab, cross a cross,
  and lead/rear hook/uppercut remain the same relative classes. No original stance is inferred.
  This is training augmentation, not an assertion that V5 should be reflected or relabeled.

The comparison has seeds **20261021, 20261022, 20261023**, each with unaugmented and mirrored
training for both full-feature TCN and mask-only TCN: **12 training runs**. The non-temporal
baseline is not rerun because this comparison isolates the accepted TCN augmentation experiment.
Every run retains the existing architecture, native-frame features, optimizer/loss, batch size,
class weighting, dropout, learning rate, decay, 40-epoch budget and quality policy. Configurations
inherit the frozen initial experiment and are checked for equality. Model/feature source hashes
must match the initial snapshot. The unaugmented original-seed metrics must reproduce exactly.

Within each model/seed pair, initial weights and epoch-by-epoch training order match. A private
augmentation generator (seed + 1000003) leaves global initialization/dropout and shuffle RNG
streams unchanged. Full/mask-only controls use identical augmentation decisions and shuffle
orders. The realized number of mirrored clips fluctuates around 50%; the Bernoulli probability
is fixed. Checkpoints retain augmentation RNG state, and per-epoch row logs/digests permit audit.
Each epoch still has the same number of optimizer steps and clips; augmentation does not double
or filter the dataset. Training metrics also use original clips, making overfitting comparisons
consistent across conditions.

## Evaluation and reproduction

Primary measure: V5 macro-F1 at the independently selected best V5 epoch, with earlier ties.
Report accuracy, all six recalls, precision/F1/support and confusion matrices; inspect final
training/development scores and curves for overfitting. Compare full-feature gains with mask-only
gains and each class, rather than treating accuracy improvement as six-class success. Three
seeds on the same recordings do not establish boxer-independent generalization or provide three
independent test sets. V5 remains repeatedly used development data, not a held-out test.

Run from the repository root with fresh output paths:

```sh
uv sync --locked --extra pose --extra train
uv run --extra pose --extra train boxing-motion-ai train-mirror-study --help
uv run --extra pose --extra train boxing-motion-ai train-mirror-study --config configs/level1_mirror_study.yaml --output reports/boxingvi/level1-mirror-smoke-repeat --smoke
uv run --extra pose --extra train boxing-motion-ai train-mirror-study --config configs/level1_mirror_study.yaml --output reports/boxingvi/level1-mirror-repeat
uv run --extra pose --extra train pytest
uv run ruff check .
```

The smoke run uses the first seed, two clips/class/split and two epochs for both conditions and
both inputs. Its metrics are diagnostics only. Main outputs are under
`reports/boxingvi/level1-mirror-v1/`: study/derived configurations, `summary.json`/`summary.md`,
recall/comparison plots, seed/arm source snapshots and provenance, predictions, curves,
confusions, best/final checkpoints, initial-model digests and per-epoch augmentation logs.
All generated artifacts remain Git-ignored. Existing version-1.0 configs still disable mirroring;
version-1.1 configs explicitly require probability 0 or 0.5.

## Measured results

Completed the smoke study and all **12** full runs. The original-seed unaugmented TCN and mask-only metrics reproduced the initial experiment **exactly**. All paired initial-weight digests, shuffle orders and full/mask-only augmentation decisions matched. Realized mirror rates by seed were 49.72%, 50.60%, 49.27% (21,080 presentations per model), consistent with fixed Bernoulli probability 0.5.

V5 results at independently selected best epochs, **mean ± sample standard deviation** across 20261021/22/23. These seeds reuse the same V5 clips; they are not independent data groups.

| Input | Mirroring | Accuracy | Macro-F1 |
|---|---|---:|---:|
| tcn | unaugmented | 24.07% ± 9.74% | 0.1497 ± 0.0360 |
| tcn | mirrored | 71.87% ± 3.35% | 0.4912 ± 0.0188 |
| tcn_mask_only | unaugmented | 41.74% ± 0.26% | 0.1840 ± 0.0099 |
| tcn_mask_only | mirrored | 56.89% ± 2.53% | 0.3546 ± 0.0187 |

Full-feature paired macro-F1 gains: **+0.3281, +0.3342, +0.3622** (mean +0.3415, sample SD 0.0182). Full-feature selected epochs were 4/39/5 without augmentation and 31/31/28 with it.

### Does it improve every punch class?

**No.** Mean recall improves for five classes, but improvements are not consistent for every class in every seed. Lead uppercuts regress to zero recall at all selected augmented checkpoints. Rear-punch coverage remains inadequate.

| Source class | Full original recall | Full mirrored recall | Mask original recall | Mask mirrored recall |
|---|---:|---:|---:|---:|
| jab | 39.7% | 91.4% | 89.2% | 92.3% |
| cross | 0.0% | 96.4% | 1.0% | 54.3% |
| lead_hook | 45.1% | 80.9% | 59.3% | 38.7% |
| rear_hook | 0.0% | 21.4% | 0.0% | 17.1% |
| lead_uppercut | 19.7% | 0.0% | 0.0% | 0.9% |
| rear_uppercut | 16.9% | 21.9% | 0.0% | 20.1% |

- **Jab/cross/lead hook:** full-model recall improves in every seed. Cross rises from zero in all controls to 95.7–97.1%; lead-hook recall reaches 72.1–86.8%; jab reaches 87.6–93.4%. These are real improvements against the fixed provisional development labels, beyond an accuracy-only change.
- **Rear hook:** recall rises from zero to 5.1%, 20.5%, 38.5%. It is still poor and strongly seed-dependent. Mean confusion is 32.5% rear_uppercut, 22.2% cross, 13.7% lead_uppercut, 10.3% lead_hook, and only 21.4% correct.
- **Rear uppercut:** mean recall rises slightly, 16.9%→21.9%, but individual paired values are 0→46.6%, 50.7→15.1%, 0→4.1%. Thus one seed worsens substantially and two augmented seeds remain weak. Mean confusion remains spread across lead_hook (29.7%), lead_uppercut (27.9%), rear_hook (14.6%) and the correct class (21.9%).
- **Lead uppercut:** all 39 V5 examples are misclassified in each selected augmented model. Across seeds they become jab (57.3%) or lead_hook (42.7%). Original mean recall 19.7% came entirely from the first seed; the other two original controls were already zero. This is a clear remaining failure, not six-class completion. The study does not determine whether pose ambiguity, viewpoint coverage, annotation uncertainty or class competition explains it.

### Movement information versus missingness patterns

Mask-only mirroring improves macro-F1 from **0.1840 to 0.3546**, with paired gains +0.1630/+0.1908/+0.1581. Exchanging side-specific missingness patterns therefore contributes to improved transfer even without coordinates or confidence. This control retains native clip length and causal boundary cues, so its gains are not attributable solely to masks.

Full features outperform mirrored mask-only in **all three seeds**, by macro-F1 0.1740/0.1144/0.1216 (mean 0.1366). The paired full-feature augmentation gain exceeds the mask-only gain by 0.1651/0.1434/0.2041 (mean 0.1709). This is evidence of useful additional pose/confidence information, especially for cross and lead_hook. It is not a causal percentage decomposition or proof that the TCN learned temporal movement dynamics. Jab recall is already similarly high using masks alone, and rear-hook/rear-uppercut full-model mean recall barely exceeds their mask-only counterparts. Shortcuts remain important.

### Overfitting

At final epoch 40, full-feature mean training/development macro-F1 changes from **0.8923/0.0685** without mirroring to **0.8044/0.4248** with mirroring. The mean gap shrinks from 0.8239 to 0.3796, but remains large. Mirrored final development F1 is below the selected best mean 0.4912 in every seed (final scores 0.4482/0.3913/0.4348). Mirroring substantially reduces recording-specific overfitting without resolving it. Per-run loss/F1 curves and best/final checkpoints are preserved.

### Conclusion and stopping point

**Keep mirroring as a promising provisional training option, not a complete six-class solution.** It improves mean macro-F1 as well as accuracy and robustly rescues crosses, but fails lead uppercuts and leaves rear hooks/uppercuts unreliable. The result supports viewpoint/visibility shift as an important contributor to the initial failure. It does not verify V5 stance, justify V5-specific reflection, or establish boxer-independent generalization.

No probability, seed, architecture, threshold, candidate selection, feature representation or training-budget tuning was performed after seeing these results. Visibility augmentation and real pre-clip context remain unimplemented. Stop for review before another experiment.

### Verification

**119 tests passed; Ruff and `git diff --check` passed.** New tests verify exact double-reflection restoration, equivalence with raw-image reflection before normalization, consistent confidence/mask swaps, retained invalid gating/padding, unchanged six-class labels and timing, isolated augmentation RNG, disabled mode, rejected non-50% probabilities and unaugmented evaluation. Existing data/quality hashes and all prior experiment files remain unchanged. Weights and generated reports remain ignored by Git. V9 was not opened.
