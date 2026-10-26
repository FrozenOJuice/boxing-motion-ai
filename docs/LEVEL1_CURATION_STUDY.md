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

Results will be recorded below after the full experiment and preservation checks.
