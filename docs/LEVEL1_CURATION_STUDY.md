# Conservative annotation-curation experiment

## Approved dataset and readiness

`configs/level1_curated_subset_v1.json` pins the original curation configuration,
reviewed version-19 configuration, verified review-backup manifest, exact selection
hash and approved class counts. The dataset is a versioned **selection manifest**
over existing tensors, not a copy or rewrite of source annotations/poses.

The approved selection is the intersection of the historical curation proposal and
the current reviewed proposal. It has **1,123** training examples versus the original
**1,156**. V6:264 and V6:266 remain partial human holds; their hand confirmations do
not certify a punch label or establish `no_punch`. The six confirmed disputes
V4:497, V6:613, V7:136, V8:82, V10:146 and V10:147 also remain excluded.

| Recording | Original | Curated | Removed source rows |
|---|---:|---:|---|
| V4 | 527 | 520 | 65, 135, 181, 252, 462, 493, 497 |
| V6 | 302 | 295 | 82, 264, 266, 579, 613, 633, 654 |
| V7 | 66 | 60 | 52, 136, 144, 168, 179, 191 |
| V8 | 120 | 115 | 81, 82, 95, 161, 173 |
| V10 | 141 | 133 | 146–153 |
| Total | **1156** | **1123** | **33** |

| Class | Original | Curated |
|---|---:|---:|
| Jab | 190 | 190 |
| Cross | 97 | 97 |
| Lead hook | 303 | 301 |
| Rear hook | 112 | 108 |
| Lead uppercut | 257 | 245 |
| Rear uppercut | 197 | 182 |

The scoped `approved_subset_ready` gate requires that the only outstanding blocker
is the explicitly approved pending queue, that each such case has a human
hold/exclusion, and that none enters training. Missing anchor coverage, unapplied
corrections, missing classes, a different pending queue, changed review history or
changed hashes fail preparation. Existing review contracts and dashboard statuses
are unchanged; no partial review becomes fully verified. Historical exclusions
cannot be restored by this intersection. The entire retained pool is not certified.

## Controlled protocol

The runner reuses `multi_study.fit`, `CausalTCN`, exact existing XY+validity-gated
velocity features, mirror transform, grouped sampler and metrics. There are no
architecture or feature changes. Settings come from the frozen
`configs/level1_multi_recording.yaml` and `configs/level1_initial.yaml`: 40 epochs,
batch 32, AdamW learning rate/weight decay 0.001, hidden 32, dropout 0.1, two CPU
threads, 50% training mirror, no visibility augmentation or pre-clip context.

Both arms are trained afresh with seeds 20261021/20261022/20261023. The combined
comparison and **each** of four group holdouts use all three seeds (30 full fits).
The holdout groups are V4, V6, V7+V8 and V10. V5 is the only checkpoint selector
(best macro-F1, earliest tie); no held-out group enters fitting or selection. V1/V6
and V7/V8 guards remain. V9 is not loaded. This is development/diagnostic evaluation,
not independent test performance or evidence of boxer independence.

Evaluation always uses the same original eligible examples for both arms, including
held-out examples that curation would remove from *training*. IDs and target labels
are checked pairwise. V5 remains the frozen 583 examples. Consequently some diagnostic
source-label errors remain in evaluation; this limitation is not hidden by cleaning
only one arm's test examples.

The unchanged training-only sampler assigns one sixth expected mass to each class,
then square-root mass across groups and recordings within a class. Thus class-prior
changes do not alter expected sampled class mass. Removal does change within-class
source allocation and individual-example exposure. Epoch draws remain pool size,
so the curated arm uses 1,123 rather than 1,156 draws per combined epoch (36 versus
37 batches). This follows the prior strategy, not an equal-gradient-step experiment.
Different pool sizes also mean actual draws and mirror assignments are not paired
example-for-example, although seeds and shared initialization are matched. Do not
attribute differences exclusively to semantic annotation quality.

Both arms log sampler probabilities/exposures, selected and final train/development
metrics, complete curves, per-recording metrics, prediction identities, confusion
matrices, checkpoints and train-only duration controls. No new mask-only neural
models are included in this annotation-only comparison; historical controls remain
preserved. Supported-class F1 excludes absent target classes only from averaging;
predictions into those classes still count as errors.

## Reproduce

From the repository root with existing approved local data and dependencies:

```sh
.venv/bin/python -m boxing_motion_ai.training.curation_study \
  --output reports/boxingvi/level1-curation-smoke-new --smoke
.venv/bin/python -m boxing_motion_ai.training.curation_study \
  --output reports/boxingvi/level1-curation-study-new
```

Use fresh destinations: existing runs are never overwritten. Smoke runs use the
full pinned pools, two epochs and the first seed for every fold and arm. Preparation
is mandatory before either mode. If the live ledger has advanced, preparation fails
rather than silently adopting later reviews. Update approval only after inspecting
the new decisions. Generated files, original reviews and backups stay excluded from
Git. `dataset/selection.json` records every removed row, reason and current review
state, along with exact selected IDs and provenance.

The initial smoke completed all ten fits and verified 4,633 preserved files. Full
Python suite: **436 passed**, one existing upstream Starlette/httpx deprecation
warning. Frontend: **42 passed**. Ruff, TypeScript, frontend formatting and production
build passed. Nine new synthetic tests cover gate restrictions, unresolved/manual
provenance, historical holds, unchanged development membership, exact tensor reuse
and paired evaluation identity/label mismatches.

## Results and decision

**Curation yields a modest V5 development gain, but does not establish reliable
cross-recording generalization.** Keep the conservative exclusions as data hygiene;
do not promote this as a solved six-class model or infer that the excluded labels
were all wrong. Six excluded disputes were actually human-confirmed, deliberately
kept out to preserve this experiment's approved membership.

All 30 full fits completed. Full artifacts are in
`reports/boxingvi/level1-curation-study-v1/`, smoke artifacts in
`reports/boxingvi/level1-curation-smoke-v1/`.
The [complete metric appendix](../reports/boxingvi/level1-curation-study-v1/metrics.md)
contains per-recording precision/recall/support, all mean confusion matrices and
per-seed results. Individual exact confusion counts, predictions, training curves
and checkpoints remain in the corresponding case/seed/arm directories.

### Aggregate performance

Mean ± sample standard deviation across three matched seeds. A is original;
B is curated. Held-out results are recording-group diagnostics; only V5 selects
checkpoints. These are not independent V9 test scores.

| Evaluation | A six-class F1 | B six-class F1 | A accuracy | B accuracy | F1 gains across seeds |
|---|---:|---:|---:|---:|---|

| V5 development | 0.7968 ± 0.0423 | 0.8134 ± 0.0345 | 0.8748 ± 0.0179 | 0.8696 ± 0.0206 | +0.0222, +0.0052, +0.0224 |
| V4 held out | 0.1552 ± 0.0074 | 0.1705 ± 0.0141 | 0.1815 ± 0.0121 | 0.1954 ± 0.0356 | +0.0210, +0.0060, +0.0191 |
| V6 held out | 0.2599 ± 0.0249 | 0.2459 ± 0.0314 | 0.4216 ± 0.0538 | 0.4128 ± 0.0670 | -0.0524, -0.0385, +0.0490 |
| V7+V8 held out | 0.4002 ± 0.0396 | 0.4685 ± 0.0296 | 0.4624 ± 0.0246 | 0.5520 ± 0.0204 | +0.0307, +0.0262, +0.1481 |
| V10 held out | 0.2291 ± 0.0367 | 0.1908 ± 0.0267 | 0.2955 ± 0.0391 | 0.2695 ± 0.0465 | -0.0528, -0.0293, -0.0330 |

Equal-weight mean across the four held-out groups is **0.2611 → 0.2689** six-class
F1. Per-seed gains for that equal-group mean are **−0.0134, −0.0089, +0.0458**.
This small average is inconsistent, not robust evidence of transferable improvement.
The V7/V8 group benefits in all three seeds; V10 regresses in all three.

V6 has no jab/cross examples: its supported-class F1 is **0.3898 → 0.3688**.
V7 has no eligible rear uppercuts and V8 no rear hooks. Individually, their
supported F1 is **0.3237 → 0.3905** and **0.4919 → 0.5618**, respectively.
Their combined held-out group contains all six classes, but only one rear hook
and eight rear uppercuts; interpret large recall shifts with those small supports
in mind. Other group and V5 supported F1 values equal six-class F1.

### V5 precision, recall and seed consistency

| Class | A precision | B precision | A recall | B recall |
|---|---:|---:|---:|---:|
| jab | 0.9826 | 0.9685 | 0.9189 | 0.8982 |
| cross | 0.8938 | 0.8651 | 0.9710 | 0.9758 |
| lead_hook | 0.7526 | 0.8219 | 0.7892 | 0.8137 |
| rear_hook | 0.6770 | 0.6403 | 0.4274 | 0.7009 |
| lead_uppercut | 0.8252 | 0.7606 | 0.7778 | 0.7179 |
| rear_uppercut | 0.7782 | 0.8446 | 0.9269 | 0.8037 |

V5 macro-F1 improves by **+0.0222/+0.0052/+0.0224**, but mean accuracy falls
**87.48% → 86.96%**. Rear-hook recall rises **42.74% → 70.09%** (improves in two
seeds, declines in one), while its precision falls **67.70% → 64.03%**. Lead-uppercut
recall declines in two seeds; rear-uppercut recall declines in **all three**.
Jab recall also declines in all three, despite no jab removal.

The confusion tradeoff is explicit: mean V5 rear-hook → rear-uppercut errors fall
from **18.0 to 9.33**, but rear-uppercut → rear-hook errors rise from **1.0 to 9.67**.
Thus the rear-hook gain partly reflects a changed decision boundary, not a uniform
improvement in recognizing both rear punches.

### Weak-class recall on held-out groups

| Held-out group | Rear hook A → B | Lead uppercut A → B | Rear uppercut A → B |
|---|---:|---:|---:|
| V4 | 0.0000 → 0.0000 | 0.4695 → 0.2300 | 0.0115 → 0.0536 |
| V6 | 0.0625 → 0.0312 | 0.3573 → 0.3227 | 0.3457 → 0.3457 |
| V7+V8 | 0.0000 → 0.0000 | 0.5965 → 0.6608 | 0.5000 → 0.8750 |
| V10 | 0.0000 → 0.0000 | 1.0000 → 0.9167 | 0.3175 → 0.1270 |

Rear hooks remain at zero recall on V4, V7/V8 and V10 for all seeds. V6 rear-hook
recall is only **3.13%** after curation. V4 lead uppercuts and V10 rear uppercuts
regress substantially. V7/V8 uppercuts improve, with sparse rear-uppercut support.
These results do not support a balanced six-class transfer claim.

### Overfitting and distribution effects

At the final combined epoch, mean training/development F1 is **0.9010/0.6953** for
A and **0.8895/0.7173** for B. The gap narrows but remains large; selected-epoch
scores above are better because V5 chooses checkpoints. Selected epochs are
**31/16/28** for A and **24/34/24** for B. Training metrics describe different-sized
training pools and are not an identical-example accuracy comparison.

Curation removes 2.85% of examples, disproportionately uppercuts (12 lead and 15
rear), but the unchanged sampler still assigns **16.67% expected mass per class**
in both arms. Combined expected recording mass shifts as follows:

| Recording | Original | Curated |
|---|---:|---:|
| V4 | 38.90% | 40.04% |
| V6 | 21.55% | 22.70% |
| V7 | 7.33% | 6.41% |
| V8 | 12.26% | 11.96% |
| V10 | 19.96% | 18.88% |

The training-only duration lookup has unchanged combined V5 F1 **0.0863** in both
arms. This does not show increased utility of that simple duration shortcut, but
cannot rule out temporal shortcuts in the TCN. Source allocation, repeated-example
exposure, different gradient draws and slightly fewer batches remain confounders.
This experiment therefore estimates the effect of the approved curation policy
**under the existing sampling scheme**, not a pure label-quality effect isolated
from all distribution changes. No distribution-matched third training arm was run.

### Next direction

Keep the 1,123-example manifest and its exclusions for future approved work, while
preserving both trained arms and the prior reference. Do not automatically promote
a new model on the strength of V5 alone. The next useful step is a bounded
**pose-estimation reliability comparison**, concentrating on active-arm localization
and high-confidence anatomical mistakes on the existing human-reviewed examples
and the hardest held-out viewpoints. Compare an alternative pose estimator against
the current one before another classifier experiment; preserve exact timing and
masks and avoid another round of feature additions. This is a recommendation, not
an implemented pipeline or a guarantee that pose quality is the only cause.

Large-scale relabeling is not justified by this limited study. The confirmed anchors
and held exclusions should be reused rather than requesting extensive fresh manual
review. Frozen diagnostic labels, sparse rear punches, unknown identities and
viewpoint differences still limit interpretation. No further training was started.

## Final preservation and reproducibility

- Both arms used identical eligible evaluation IDs and target labels for every seed
  and fold. All 30 checkpoint provenance records match their approved training rows;
  V5 alone selected checkpoints. All paired initial model tensor digests match.
- The three combined original runs and all existing one-seed original group-holdout
  reference metrics reproduce **exactly**, not approximately.
- All ten smoke curves exactly equal the first two epochs of their corresponding
  full runs, including losses and train/development metrics.
- All **4,633** protected prior files passed post-run hashes. An additional **12**
  raw video/workbook hashes match the historical source pins before and after the
  experiment. Review ledger version 19 and the byte-verified backups are unchanged.
- V9 was not opened, decoded, hashed, evaluated or used for selection. No original
  candidates, annotations, media, model results or review decisions were rewritten.
- `reproducibility.json`, `preservation.json` and `raw-source-preservation.json`
  record these checks. The local `verify_results.py` alongside them reproduces the
  result/preservation audit from the repository root. Generated outputs remain ignored.

Checks remain **436 Python tests**, **42 frontend tests**, Ruff, TypeScript,
formatting and production build passing. The implementation milestone is local
commit `b38a63b`; final results are documented in a separate local commit. No push.
