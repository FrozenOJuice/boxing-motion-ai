# Multi-recording Level 1 recognition and grouped generalization

**Multi-recording training substantially improves V5, but transferable six-class
recognition is not established.** The XY+velocity model reaches mean V5 macro-F1
**0.7968**, versus the preserved V4-only reference's **0.6081**. The four group-held-out
pilot folds score only **0.1637–0.4230** six-class F1 for A; rear-hook recall collapses
on almost every unseen group. Full models generally beat mask/duration controls,
showing transferable pose content in some environments, alongside strong recording
and visibility dependence. This does not prove temporal movement understanding.

Keep **V4-only XY+velocity+mirror** as the accepted provisional reference pending review.
The multi-recording models are candidate experiments, not automatically promoted.
Geometry stays optional. No new feature engineering, architecture or RGB pipeline
was introduced.

## Pool, source labels and uncertainties

All existing approved provisional examples are used; **1,156 training candidates**
from V4/V6/V7/V8/V10, **583 V5 development examples**. Selected source rows, pose files,
source policy hashes and pose-journal hashes match the accepted readiness inputs.
No annotation, media, coordinate, confidence or mask is modified. Existing exclusions,
including V6 321/598/620/675–686 and V10 123/transition holds, remain unchanged.

| Recording | Jab | Cross | Lead hook | Rear hook | Lead uppercut | Rear uppercut | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4 | 117 | 39 | 140 | 73 | 71 | 87 | 527 |
| V6 | 0 | 0 | 64 | 32 | 125 | 81 | 302 |
| V7 | 8 | 1 | 20 | 1 | 36 | 0 | 66 |
| V8 | 16 | 21 | 54 | 0 | 21 | 8 | 120 |
| V10 | 49 | 36 | 25 | 6 | 4 | 21 | 141 |
| V5 (development) | 226 | 138 | 68 | 39 | 39 | 73 | 583 |

Training totals are **190/97/303/112/257/197** in the class order above. Raw class
fractions range from 8.4% cross to 26.2% lead hook; V4 supplies 45.6% of raw candidates.
V6 has no source jab/cross annotations. V8 has no source rear hooks. V7's 20 source
rear uppercuts are all excluded; its one retained rear hook and one cross are fragile
cells. The V7/V8 group has all six labels jointly, but only one rear-hook example.

The literal source labels map consistently to the same six provisional classes,
allowing case/edge whitespace differences. Known exercise-transition/phase and
lead/rear concerns are documented in [V6 readiness](V6_READINESS.md),
[V6 preparation](V6_PREPARATION.md), [V10 preparation](V10_PREPARATION.md) and the
[V7/V8 review](LEVEL1_DATA_EXPANSION.md). No new verified label contradiction blocked
training. This is **literal mapping consistency, not biomechanical certification**:
authors' frame bases/end conventions, some interval phases and boxer stance/lead-rear
semantics remain unverified. No stance is inferred from anatomical channels and no
suspect annotation is restored or relabeled. Unlabeled footage remains unknown.

Source groups are guarded by origin, not assumed verified athlete identity:

- V4: `fightcamp-unresolved`.
- V6: `natebowerfitness-unresolved`, with **V1/V6** held together if V1 is ever added.
  V1 is absent from this experiment and rejected by its training whitelist.
- V7/V8: one `precision-striking-unresolved` group, never separated across folds.
- V10: `utah-lee-unresolved`.
- V5: development-only `self-improvement-boxer-unresolved`.

Boxer/session IDs remain explicitly unknown. V9 inputs are never opened. Licensing
and redistribution status are not newly verified; all source data stay local/ignored.

### Duration differences

Native frames and exact PTS are retained. V4 per-class median center spans are
0.209–0.292 s, versus V6's 0.584–0.792 s, V7 0.334–0.467 s, V8 0.300–0.500 s and
V10 0.360–0.560 s. These are first-to-last decoded frame spans, not independently
verified biomechanical punch durations. V6 median rear-hook length is 20 frames,
versus seven in V4. Native sampling also differs (roughly 24/30/25 Hz by recording),
so the unchanged 15-frame TCN receptive field covers different physical durations.
Nothing is resampled, retimed, trimmed or corrected to equalize these distributions.

## Training-only sampling

A frozen hierarchy assigns each training example probability
`(1/6) × sqrt(N_class,group)/sum_groups sqrt(N_class,group)`
`× sqrt(N_class,recording)/sum_recordings_in_group sqrt(N_class,recording)`
`× 1/N_class,recording`.

Classes receive equal expected mass; square-root source allocation reduces large
sources while avoiding the full equal-group weight on singleton cells. Sampling is
with replacement, with **N_training_pool draws per epoch**. All probabilities/counts
are recomputed from each fold's training examples only. No V5/held-out frequencies
enter this sampler. Unit cross-entropy class weights replace inverse-frequency loss
weights to avoid balancing classes twice; this is part of the documented sampling
change, not an optimizer or architecture intervention.

| Recording | Raw share | Expected sampled share |
|---|---:|---:|
| V4 | 45.59% | 38.90% |
| V6 | 26.12% | 21.55% |
| V7 | 5.71% | 7.33% |
| V8 | 10.38% | 12.26% |
| V10 | 12.20% | 19.96% |

Every class has expected mass 1/6. V7's singleton rear hook is still expected about
**10.9 draws/epoch**, and its cross about **9.6**: replacement does not create new
execution diversity. This risk is logged, not hidden or removed after seeing results.
Equal class mass cannot remove structural source/label associations when cells are
absent; it does not create V6 straight punches or diverse V7 rear hooks.
Sampler histories contain every selected index and mirror decision. Both representations
and controls use identical draws/augmentation decisions for matched seeds.

The larger pool also increases optimizer steps: 37 batches/epoch versus 17 for the
old 527-example V4 reference. Thus the V4-only comparison bundles source diversity,
sampling/loss balancing and update exposure; it is not a data-only causal ablation.
A versus B is matched within every new run.

## Fixed experiment and fold protocol

**A:** existing XY + validity-gated velocities + 50% training-only mirror.
**B:** A plus the previously tested optional relative-arm geometry. Feature, velocity,
mirror and model source bodies match the accepted study; only input widths differ.

Same causal TCN, hidden 32, three blocks, 15-frame receptive field, original pooling,
dropout 0.1; **40 epochs**, batch 32, AdamW LR 0.001/weight decay 0.001, gradient clip 1,
two CPU threads. Full channels 65/99 and mask-only 16/28 are unchanged from prior
studies. Shared initialization and zero added columns are preserved. Visibility and
pre-clip context stay disabled; quality remains mask-aware70 with original thresholds.

Main experiment: all five training recordings, seeds **20261021/20261022/20261023**,
V5 selection/development. Four leave-one-source-group-out folds use **20261021 only**:

| Held-out group | Training recordings | Training count | Selection |
|---|---|---:|---|
| V4 | V6, V7, V8, V10 | 629 | V5 only |
| V6 | V4, V7, V8, V10 | 854 | V5 only |
| V7/V8 | V4, V6, V10 | 970 | V5 only |
| V10 | V4, V6, V7, V8 | 1015 | V5 only |

Each fold starts from scratch; main-model weights are never reused as fold initialization.
Held-out examples are not passed to the training/selection function. Every fold retains
all six training classes. V5 remains development-only, and is never a LOGO training or
held-out group. Best V5 macro-F1 selects the epoch with the original earliest-tie rule.
All selections in a case finish before held-out model evaluation. No fold results
change later fold settings, thresholds, architecture, augmentation or checkpoint choice.

There are **28 trained models**: 12 main models (three seeds × A/B × full/mask-only)
and 16 pilot models (four folds × A/B × full/mask-only). A two-epoch, two-candidates-per-
class-per-recording smoke test covers every case. An initial smoke caught a batch-mask
interface error; it was corrected and covered by synthetic training tests before full
training. Two successful repeated smoke runs match all 20 model states, audits, metrics
and V5 prediction sets exactly.

## V5 main comparison

All six classes are present, so supported-class F1 equals six-class macro-F1.

| Seed | A F1 | B F1 | A accuracy | B accuracy | A/B selected epoch |
|---|---:|---:|---:|---:|---:|
| 20261021 | 0.7532 | 0.7818 | 85.42% | 85.59% | 31/22 |
| 20261022 | 0.8376 | 0.7639 | 88.68% | 84.73% | 16/34 |
| 20261023 | 0.7996 | 0.7431 | 88.34% | 85.59% | 28/34 |
| **Mean** | **0.7968** | **0.7629** | **87.48%** | **85.31%** | |

The preserved V4-only reference has F1 **0.6081**, accuracy **78.50%**. V4-only optional
geometry had F1 **0.6568**, accuracy **76.10%**. Both new representations improve V5 F1
in all three matched seeds over their corresponding V4-only results. A outperforms B
on average; B beats A in only one seed. No reference or seed is automatically promoted.

Relative to V4-only A, V5 mean rear-hook recall rises **21.4→42.7%**, lead uppercut
**10.3→77.8%**, rear uppercut **78.5→92.7%**. Lead-hook recall falls **87.3→78.9%**.
These are substantial development gains, not independent test/generalization claims.

## Group-held-out pilot

One seed per fold, no error bars suggesting multiple independent trainings. V7/V8
are pooled **once** for group-level metrics; they are also reported separately below.
Six-class F1 keeps all six labels; supported F1 averages only classes with eligible
ground truth. Predictions into absent classes remain errors, without output restriction.

| Held-out group | A six F1 | B six F1 | A supported F1 | B supported F1 | A accuracy | B accuracy |
|---|---:|---:|---:|---:|---:|---:|
| V4 | 0.1637 | 0.1397 | 0.1637 | 0.1397 | 19.54% | 22.96% |
| V6 | 0.2824 | 0.3199 | 0.4236 | 0.4798 | 48.01% | 50.00% |
| V7/V8 | 0.4230 | 0.4839 | 0.4230 | 0.4839 | 46.77% | 59.68% |
| V10 | 0.2683 | 0.2195 | 0.2683 | 0.2195 | 34.04% | 30.50% |

Equal-group mean six-class F1 is **0.2843 for A / 0.2908 for B**. This is a descriptive
mean across four heterogeneous source groups, not a statistically established B win.
B helps V6/V7-V8 but hurts V4/V10. V6 has only four ground-truth classes, so its six-
class F1 is lower than supported F1; the full label space is retained for every prediction.
Neither representation demonstrates reliable six-class source transfer.

### Per-recording precision and recall

Cells are **A → B**, with main V5 averaged over three seeds and other columns from
the corresponding held-out pilot. `—` denotes unsupported ground truth. Existing
metric JSON keeps precision-zero conventions and false-positive counts for absent
classes; an absent class's recall is not claimed to be 0%. All counts remain provisional.

Mean per-class recall:

| Class | V5 main | V4 held out | V6 held out | V7 held out | V8 held out | V10 held out |
|---|---:|---:|---:|---:|---:|---:|
| jab | 91.9% → 87.5% | 9.4% → 1.7% | — | 37.5% → 87.5% | 50.0% → 62.5% | 26.5% → 28.6% |
| cross | 97.1% → 97.8% | 43.6% → 5.1% | — | 0.0% → 0.0% | 52.4% → 61.9% | 8.3% → 2.8% |
| lead_hook | 78.9% → 88.2% | 23.6% → 55.7% | 90.6% → 85.9% | 20.0% → 60.0% | 44.4% → 53.7% | 92.0% → 88.0% |
| rear_hook | 42.7% → 29.9% | 0.0% → 0.0% | 6.2% → 15.6% | 0.0% → 0.0% | — | 0.0% → 0.0% |
| lead_uppercut | 77.8% → 68.4% | 57.7% → 54.9% | 37.6% → 60.8% | 55.6% → 63.9% | 61.9% → 57.1% | 100.0% → 100.0% |
| rear_uppercut | 92.7% → 90.9% | 1.1% → 0.0% | 46.9% → 18.5% | — | 50.0% → 62.5% | 23.8% → 9.5% |

Per-class precision:

| Class | V5 main | V4 held out | V6 held out | V7 held out | V8 held out | V10 held out |
|---|---:|---:|---:|---:|---:|---:|
| jab | 98.3% → 97.7% | 26.8% → 20.0% | — | 75.0% → 87.5% | 80.0% → 90.9% | 86.7% → 70.0% |
| cross | 89.4% → 86.1% | 19.5% → 12.5% | — | 0.0% → 0.0% | 64.7% → 68.4% | 60.0% → 50.0% |
| lead_hook | 75.3% → 76.5% | 28.7% → 23.2% | 45.3% → 61.8% | 36.4% → 63.2% | 77.4% → 78.4% | 38.3% → 44.9% |
| rear_hook | 67.7% → 64.9% | 0.0% → 0.0% | 9.1% → 45.5% | 0.0% → 0.0% | — | 0.0% → 0.0% |
| lead_uppercut | 82.5% → 79.2% | 20.0% → 32.2% | 75.8% → 71.0% | 95.2% → 95.8% | 48.1% → 52.2% | 7.8% → 6.5% |
| rear_uppercut | 77.8% → 72.6% | 3.3% → 0.0% | 56.7% → 100.0% | — | 36.4% → 20.0% | 83.3% → 28.6% |

Rear hooks remain the clearest collapse: **0%** recall in held-out V4, the lone V7
example and V10, and only **6.25%/15.63%** in V6. V8 has no rear-hook annotations.
Rear uppercuts have partial transfer (V6 A 46.9%, V8 50–62.5%, V10 9.5–23.8%) but
nearly vanish on V4 (A 1.15%, B 0%). Lead uppercuts transfer better, including 100%
on V10, but that last figure is based on **four examples**, with substantial false
positives reflected in precision. It is not proof of solved uppercut recognition.

The confusions often preserve punch family while losing side: held-out V6 rear hooks
become lead hooks on **28/32 A** and **26/32 B** examples; V10 rear hooks become lead
hooks on **6/6 A** and **5/6 B**. V10 rear uppercuts become lead uppercuts on **16/21 A**
and **14/21 B** examples. On V4, A maps 68/87 rear uppercuts to lead uppercut; B maps
53/87 to lead hook. This supports a side/viewpoint/phase or label-semantics problem,
without identifying which cause is responsible or justifying any class remapping.

The [structured report](../reports/boxingvi/level1-multi-recording-v1/summary.json)
contains every seed's original six-class metrics, precision/recall/support and confusion
matrix. The [readable generated report](../reports/boxingvi/level1-multi-recording-v1/summary.md)
contains full/mask mean confusion matrices with explicit true/predicted axes, and
per-recording source absence/exclusion statuses. V7's absent *eligible* rear uppercuts
must not be confused with absent source labels.

## Shortcut diagnostics

Mask-only models are trained with the same sampler, seeds, epoch budget and mirror
decisions. Duration-only diagnostics fit a fixed-bin lookup from **training data only**:
frame bins 8/15/24 and exact-PTS span bins 0.3/0.6/1.0 seconds, one pseudo-count per
cell/class, and training-sampler-weighted exposure. No development or held-out labels
fit bins, smoothing, probabilities or class priors. No new recognition architecture
or timing feature is added to the TCN.

| Evaluation | Full A/B supported F1 | Mask A/B supported F1 | Duration supported F1 | Duration accuracy |
|---|---:|---:|---:|---:|
| V5 (combined, 3 seeds) | 0.7968/0.7629 | 0.3786/0.3792 | 0.0863 | 18.87% |
| V4 (held out, 1 seed) | 0.1637/0.1397 | 0.1494/0.1478 | 0.0473 | 12.52% |
| V6 (held out, 1 seed) | 0.4236/0.4798 | 0.1549/0.1789 | 0.1412 | 14.90% |
| V7 (held out, 1 seed) | 0.2920/0.4514 | 0.1695/0.1475 | 0.0436 | 9.09% |
| V8 (held out, 1 seed) | 0.5444/0.5753 | 0.2776/0.2597 | 0.1365 | 18.33% |
| V10 (held out, 1 seed) | 0.2683/0.2195 | 0.0989/0.0989 | 0.1666 | 36.17% |

Full models clearly exceed these controls on V5, V6 and V7/V8: there is useful pose
content beyond missingness and length. On V4, full F1 is approximately mask-only
(and B is lower), suggesting strong source-dependent failure. V10 duration-only
accuracy **36.17%** exceeds full A **34.04%** and B **30.50%**, although full F1 is better;
accuracy alone would overstate meaningful recognition. Duration strata/error counts
are saved per model, recording and true class so class-mix confounding is visible.

These controls do not identify the causal fraction of shortcut reliance in the full
network. No temporal-order destruction ablation was run, so pose-content transfer
must not be equated with proven learned motion dynamics. The fixed-frame receptive
field, native FPS, class-phase durations, projection and missing-arm patterns remain
potential source cues. Record IDs themselves are not TCN features.

## Overfitting and decision

Main models' final training F1 is approximately **0.87–0.92**, while final V5 F1 is
**0.66–0.75**. Selection often favors an earlier epoch. High in-sample scores on
recordings included in a fold are explicitly stored as training-source diagnostics;
they are not mixed with held-out generalization results. The low LOGO scores and
rear-hook collapse are stronger evidence of the remaining problem than V5's gains.

**Conclusion:** multi-source training helps, including several difficult classes on
V5, but does not establish transferable six-class recognition. It learns some shared
pose information while retaining substantial recording-specific behavior. Neither
unknown athlete identities nor provisional labels support boxer-independent claims.

**Recommended next modeling direction: a bounded RGB + pose hybrid feasibility study**,
with the same guarded folds and explicit RGB-only/pose-only controls, after a small
sanity check of the failed rear-class poses/labels. Good observed-arm coverage on V6/
V10 still accompanies weak rear transfer, while V4's occluded side is largely missing.
Improving a 2D pose estimator may reduce false-valid joints and jitter, but cannot be
assumed to recover hidden arm geometry or missing depth, glove/stance/object cues.
A small alternative-estimator comparison is useful diagnostic evidence; a wholesale
re-extraction is not yet supported as the primary next investment.

RGB/pose complementarity has precedent in general action recognition, e.g.
[Duan et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html).
That evidence is motivation, not a BoxingVI result or an instruction to adopt that
paper's architecture. Our next-direction recommendation is an inference from the
local results above, not a demonstrated superiority of RGB over another pose estimator.

This recommendation is a hypothesis, not proof that RGB will solve the problem.
RGB can introduce stronger background/appearance shortcuts, and no architecture fixes
incorrect lead/rear labels. Keep the next scope small, retain exclusions, verify a few
representative rear cases, and judge cross-group behavior rather than V5 alone. No
hybrid architecture, RGB pipeline, new feature family, annotation correction or further
training experiment is implemented here. Stop for review.

## Reproduction and preservation

```sh
uv run python -m boxing_motion_ai.training.multi_study \
  --config configs/level1_multi_recording.yaml \
  --output reports/boxingvi/level1-multi-audit-repeat --audit-only
uv run python -m boxing_motion_ai.training.multi_study \
  --config configs/level1_multi_recording.yaml \
  --output reports/boxingvi/level1-multi-smoke-repeat --smoke
uv run python -m boxing_motion_ai.training.multi_study \
  --config configs/level1_multi_recording.yaml \
  --output reports/boxingvi/level1-multi-repeat
uv run python -m boxing_motion_ai.training.multi_verify \
  --output reports/boxingvi/level1-multi-repeat
uv run pytest
uv run ruff check .
```

Use fresh ignored output paths. Existing local prepared inputs, accepted checkpoints/
reports and the approved readiness manifest are prerequisites; nothing is downloaded.
The study snapshots source/config/lock files, hashes source inputs, logs every sampled
index/mirror decision, and preserves selected/final checkpoints, curves and predictions.
Default legacy training gates remain unchanged; multi-source admission is explicit.

Reproducibility verification: **28 selected checkpoints reloaded**, **48 V5/held-out
prediction sets reproduced exactly**, **1,120 sampler/mirror epochs replayed**, and
**3,908 prior files verified unchanged**. Metric recomputation covers saved in-sample
and group-held-out predictions as well. V9 inputs were not read; V7/V8 and V1/V6 guards
remain intact. Datasets, generated reports, weights, media and virtual environments
remain excluded from Git. No push is performed.

Final checks: **256 tests passed** (`uv run pytest -q`, 25.42 s);
`uv run ruff check .` and whitespace checks passed. Tests cover source-group boundaries,
V9 rejection, train-only sampling/duration fitting, mask/feature handling, exact synthetic
training reproducibility, sampler replay, safe absolute-config snapshots and
structured/readable report behavior.
The milestone is saved in a descriptive local Git commit after these checks; no push.
