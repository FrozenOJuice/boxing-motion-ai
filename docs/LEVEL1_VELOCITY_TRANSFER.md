# Frozen velocity transfer diagnostics

XY + validity-gated velocities + 50% mirror augmentation is now the **provisional
Level 1 reference**, by acceptance of the V4→V5 experiment. The reference registry
is `configs/level1_reference.yaml`; earlier XY checkpoints and every prior study remain
preserved. This registry identifies the three original per-seed V5-selected checkpoints,
not an ensemble or a newly trained model.

**The improvement does not transfer consistently beyond V5.** Only V7 improves on
average; V6 is mixed across seeds and V8/V10 regress in all three. Both representations
have zero rear-hook and rear-uppercut recall on every external recording where those
classes have eligible examples. Recommend investigating a representation with less
viewpoint dependence before launching V6-expanded training. This is a recommendation
for review, not an implemented change or a rollback of the accepted reference.

## Frozen comparison

- Original V4 training (527 examples), V5 development selection (583), seeds
  **20261021, 20261022, 20261023**. Load the accepted velocity-study `best.pt` checkpoints
  for XY/XY+velocity, full/mask-only: **12 frozen checkpoints**, **48 external evaluations**.
- Before transfer, all 12 frozen V5 prediction sets reproduce with **zero probability
  difference and identical predicted classes**. Architecture source hashes, normalization
  AST and velocity computation/continuity-policy AST match frozen implementations.
- Same eligible source rows, labels, exact PTS/frame indices, mask-aware70 policy and
  preparation hashes for each pair. No resampling, filtering, correction, model selection,
  optimization, augmentation at evaluation, velocity clipping or threshold changes.
  Original V4 class weights are retained for the unchanged evaluation loss procedure;
  reported accuracy/F1/confusions are the original unweighted clip metrics.
- All existing V6 holds remain, including 321, 598, 620 and 675–686. V7 excluded rear
  uppercuts and V10 row 123/other exclusions remain excluded. Unlabeled footage remains
  unknown. All source mappings and unknown boxer/session IDs remain provisional.
- V7/V8 remain one unresolved `precision-striking-unresolved` group. V6 retains the
  `natebowerfitness-unresolved` V1/V6 guard; V10 is `utah-lee-unresolved`. These are source
  precautions, not verified distinct boxer identities. V9 inputs are never opened.
- These recordings already informed previous diagnostic/preparation studies (and
  V7/V8/V10 expansion studies). They are **reused diagnostic environments, not untouched
  test sets**. Seeds measure optimization variation, not independent boxer generalization.

## Metrics and eligible coverage

Means below are across matched seeds, not pooled independent repeats. Six-class F1
uses the unchanged six-class definition, including zero F1 for unsupported classes.
Supported F1 separately averages only classes with eligible ground truth. Predictions
into unsupported classes remain errors; outputs are never restricted or renormalized.

| Recording | Eligible | Accuracy XY → velocity | Six-class F1 XY → velocity | Supported F1 XY → velocity | Paired supported-F1 changes, seed order |
|---|---:|---:|---:|---:|---|
| V6 | 302 | 21.30% → 11.81% | 0.0781 → 0.0674 | 0.1171 → 0.1010 | +0.0593, -0.0910, -0.0163 |
| V7 | 66 | 46.46% → 51.01% | 0.3073 → 0.3650 | 0.3688 → 0.4380 | +0.1706, -0.1182, +0.1553 |
| V8 | 120 | 46.67% → 34.72% | 0.2806 → 0.2389 | 0.3368 → 0.2867 | -0.0361, -0.1111, -0.0031 |
| V10 | 141 | 30.73% → 18.20% | 0.2310 → 0.1249 | 0.2310 → 0.1249 | -0.0624, -0.1447, -0.1113 |

| Class | V6 source / eligible | V7 source / eligible | V8 source / eligible | V10 source / eligible |
|---|---:|---:|---:|---:|
| jab | 0 / 0 | 29 / 8 | 25 / 16 | 49 / 49 |
| cross | 0 / 0 | 30 / 1 | 32 / 21 | 36 / 36 |
| lead_hook | 125 / 64 | 60 / 20 | 88 / 54 | 26 / 25 |
| rear_hook | 95 / 32 | 5 / 1 | 0 / 0 | 6 / 6 |
| lead_uppercut | 259 / 125 | 50 / 36 | 29 / 21 | 10 / 4 |
| rear_uppercut | 206 / 81 | 20 / 0 | 25 / 8 | 25 / 21 |

V6 has no jab/cross source annotations. V8 has no rear-hook source annotations.
V7 has 20 rear-uppercut source annotations, but none retained; this class is **excluded,
not absent from the original workbook**. V7 cross and rear hook each have only one
eligible example. V10 lead uppercuts have only four, so those recall estimates are fragile.

## Per-class mean recall

Each cell is XY → velocity (%); `—` means no eligible ground truth, not zero recall.

| Class | V6 | V7 | V8 | V10 |
|---|---:|---:|---:|---:|
| jab | — | 100.0% → 91.7% | 60.4% → 56.2% | 43.5% → 4.1% |
| cross | — | 100.0% → 100.0% | 55.6% → 46.0% | 0.0% → 0.9% |
| lead_hook | 91.1% → 42.2% | 80.0% → 85.0% | 61.1% → 35.8% | 74.7% → 88.0% |
| rear_hook | 0.0% → 0.0% | 0.0% → 0.0% | — | 0.0% → 0.0% |
| lead_uppercut | 4.8% → 6.9% | 15.7% → 23.1% | 7.9% → 17.5% | 83.3% → 33.3% |
| rear_uppercut | 0.0% → 0.0% | — | 0.0% → 0.0% | 0.0% → 0.0% |

Confusions are available for **every recording × seed × representation × control**
in the local [structured report](../reports/boxingvi/level1-velocity-transfer-v1/summary.json).
The [readable generated report](../reports/boxingvi/level1-velocity-transfer-v1/summary.md)
contains the mean full-feature 6×6 matrices with explicit axis order. Per-seed predictions,
probabilities, source rows and error associations are retained alongside them.

The main errors are systematic: V6 XY predicts lead hook on 78.0% of clips; velocity
shifts 46.4% of predictions to jab despite no jab source labels. V8 velocity also shifts
40.3% to jab and loses lead-hook recall. V10 velocity predicts lead hook on 78.0% of
clips, losing jab and lead-uppercut recognition. V7 lead-uppercut confusions shift away
from cross, but mostly toward jab/lead hook, with only a small average recall gain.
There is no external replication of the accepted V5 rear-uppercut gain.

## Frozen mask-only controls

The original XY control has eight joint-validity inputs; the velocity control adds
eight derivative-validity inputs. Both use their original trained checkpoints, no
coordinates/confidence/vector values, and exactly the same evaluation clips.

| Recording | Mask accuracy XY → velocity | Mask supported F1 XY → velocity | Full velocity supported F1 |
|---|---:|---:|---:|
| V6 | 20.53% → 20.31% | 0.1095 → 0.1110 | 0.1010 |
| V7 | 37.37% → 36.36% | 0.2775 → 0.2699 | 0.4380 |
| V8 | 42.78% → 41.94% | 0.2753 → 0.2815 | 0.2867 |
| V10 | 29.79% → 30.50% | 0.1336 → 0.1368 | 0.1249 |

Mask controls barely change between representations. Full velocity is worse than its
mask-only control on V6/V10 and almost tied on V8; V7 has a clear full-feature advantage.
Missingness/sequence-boundary cues remain viable correlates. These separate models do
not establish a causal fraction of shortcut reliance inside the full network.

## Velocity tails and mistakes

The tail threshold is the **already accepted V4 training valid-pair p99,
9.65745449 torso lengths/s**, fixed before external evaluation. A predeclared diagnostic
severe threshold of 50 torso lengths/s is also recorded; no external valid pair exceeds
it. Neither threshold changes inputs, quality selection or evaluation. A tail value is
not automatically a pose error: this is normalized 2D apparent speed, not physical speed.
Changing torso projection, normalization jitter and genuine arm motion can all contribute.

| Recording | Tail pairs / valid pairs (%) | Clips with any tail | Maximum speed (torso/s) | Median frames / exact-PTS center span (s) |
|---|---:|---:|---:|---:|
| V6 | 157 / 37009 (0.42%) | 103 / 302 | 28.44 | 16 / 0.6260 |
| V7 | 76 / 6013 (1.26%) | 29 / 66 | 22.20 | 11.5 / 0.3505 |
| V8 | 124 / 10393 (1.19%) | 58 / 120 | 32.27 | 12 / 0.3670 |
| V10 | 115 / 12049 (0.95%) | 62 / 141 | 20.15 | 12 / 0.4400 |

By class, cells give **tail-pair percentage / maximum torso lengths/s**; `—` means
no eligible examples. Counts, clip maxima/median/p99 and exact-frame largest pairs are
retained in each recording's diagnostics JSON.

| Class | V6 | V7 | V8 | V10 |
|---|---:|---:|---:|---:|
| jab | — | 0.00% / 7.65 | 1.19% / 12.42 | 0.31% / 17.91 |
| cross | — | 4.29% / 12.24 | 1.75% / 14.67 | 0.25% / 11.77 |
| lead_hook | 0.54% / 20.41 | 1.58% / 21.60 | 1.03% / 32.27 | 2.50% / 20.15 |
| rear_hook | 0.94% / 28.44 | 0.00% / 5.15 | — | 2.10% / 13.43 |
| lead_uppercut | 0.36% / 21.54 | 1.32% / 22.20 | 1.37% / 12.52 | 1.88% / 11.60 |
| rear_uppercut | 0.16% / 18.70 | — | 0.83% / 11.43 | 0.85% / 12.03 |

Mean velocity-model error rates in tail-containing versus other clips are V6
**85.8% / 89.4%**, V7 **59.8% / 40.5%**, V8 **67.8% / 62.9%**, V10 **63.4% / 96.2%**.
There is no uniform association between tails and errors. Class conditioning matters:
V10 tails occur in 23/25 lead hooks (a recognized class), but only 9/49 jabs (a failed
class). V6 lead hooks have similar tail/non-tail error rates, 57.1%/58.3%; rear hooks
and rear uppercuts fail in both groups. V8 jab tail clips fare worse (66.7% vs 30.0%
error), while cross/lead-uppercut patterns go the other way. Small, dependent strata
cannot establish causation; failure is not explained solely by extreme derivatives.
The new external maxima are below the earlier V4 false-valid spike of 66.99 torso/s.
No new visual adjudication of individual tail events is claimed.

## Recording, duration and missingness shortcuts

V6 center spans are roughly twice or more the typical V4 0.21–0.29 s class medians,
with 247/302 clips at least 15 frames; most V7/V8/V10 clips are shorter than 15. V4/V6
native cadence is approximately 24 Hz, V7/V8 approximately 30 Hz, V10 approximately
25 Hz. Exact PTS drives velocities, but the TCN still has a **15-frame** receptive field,
so its physical history differs by recording. Within longer clips the start-padding
pattern occupies a smaller fraction of pooled frames. These are unchanged, plausible
transfer issues; this study does not manipulate padding or fit a duration-only classifier.
First-frame derivative masks remain false and carry the same boundary cue as before.

Fixed-bin descriptive class association (bits) is:

| Recording | Frame-count bins MI | PTS-span bins MI | Largest single-joint coverage-bin MI |
|---|---:|---:|---:|
| V6 | 0.079 | 0.153 | 0.158 |
| V7 | 0.150 | 0.087 | 0.219 |
| V8 | 0.177 | 0.057 | 0.189 |
| V10 | 0.169 | 0.077 | 0.141 |

Bins are predeclared (frames 8/15; seconds 0.3/0.6/1.0; joint coverage quartiles).
These finite-sample plug-in associations are descriptive, not bias-corrected significance
or predictive accuracy. All-original-joints-valid clips have velocity error rates
85.8%, 16.7%, 57.1%, 87.6% on V6/V7/V8/V10, versus 90.5%, 65.2%, 70.9%, 76.1% with
some invalid core joints. Thus visibility associates strongly with errors in V7/V8,
but cannot explain the V10 collapse. Class composition confounds these pooled rates;
class-conditioned counts are included in the per-seed association files.

Recording identity also predicts label availability and class balance (V6 only hooks/
uppercuts; V7 no eligible rear uppercuts). A pooled recording-wide score would obscure
that shift. V7 and V8 differ even within their guarded shared source group, arguing
against interpreting one V7 average gain as robust environmental transfer. Unknown
boxer/session identities prevent boxer-independent claims. Clean-room tests of shortcut
causes would require new controlled experiments; none were silently performed here.

## Recommendation

1. Keep the accepted XY+velocity+mirror reference frozen, with the earlier XY reference
   available. It is a V5 development improvement, not a general transfer solution.
2. Before V6-expanded training, propose a controlled **validity-gated, local joint-angle/
   relative-limb representation** comparison, aimed at reducing camera projection and
   normalized-XY dependence. Keep exact-PTS motion and explicit masks as controls;
   2D angles are still viewpoint dependent and unreliable when projected segments are
   short or occluded. Specify those gates before a future run; implement nothing now.
3. Keep V6 as an eligible provisional expansion source, with its present holds and
   V1/V6 group guard. A subsequent V4 vs V4+V6 study should be a separate frozen-config
   experiment, evaluated by class/group with V5 development and V9 still reserved.
   The present failures motivate diversity but do not demonstrate that adding V6 will
   improve another recording. Do not use these diagnostics to select epochs or tune
   velocity thresholds.

## Reproduction, preservation and checks

```sh
uv run python -m boxing_motion_ai.training.velocity_transfer \
  --config configs/level1_velocity_transfer.yaml \
  --output reports/boxingvi/level1-velocity-transfer-repeat
uv run pytest
uv run ruff check .
```

Use a fresh ignored output directory and the existing accepted checkpoints/prepared
inputs. This command evaluates only; it has no optimizer or training branch. Default
training recording gates remain unchanged; external loading requires an explicit
separate diagnostic gate and still rejects V9 before reading any files.
The registry is declarative, not an executable training configuration.

`reports/boxingvi/level1-velocity-transfer-v1/` contains JSON/Markdown summaries,
48 source-row prediction files, all association files, selected-checkpoint hashes,
input/exclusion provenance and V5 reproduction results. Earlier experiment/input
manifest hashes were verified before and after the run. Datasets, reports, source
media, virtual environments and checkpoints remain ignored; no previous result,
annotation, pose, video or label was overwritten. No training, tuning, architecture,
feature, masking or quality-policy change was performed.

Validation: **219 tests passed** (`uv run pytest -q`, 23.62 s);
`uv run ruff check .` and `git diff --check` passed. Independently recomputed all
48 prediction-file confusion matrices/accuracy/F1; verified 12 checkpoint hashes and
**3,476** preserved prior artifact/input hashes unchanged. Git ignore checks passed for
raw/processed data, generated reports, virtual environments and checkpoint files.
