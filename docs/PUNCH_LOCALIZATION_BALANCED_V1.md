# Source/event-balanced punch localization v1

**Outcome: no consistent transfer improvement.** V5 pose gains one match, RGB loses its two,
and fused still matches none with either frozen decoder. Fused useful ending evidence remains
absent near all eight references. No model is promoted and no second loss trial is started.

Starting revision: `88d3461`. One controlled objective comparison, no architecture/feature
changes, production promotion, decoder tuning or new annotation. Historical artifacts are
read-only. V4 and the V1/V6 group are training-only; V5 is development-only. No V7/V8/V10
evaluation or V9 access is authorized or performed.

## Predeclared protocol and loss

The following protocol was written before fitting the alternative objective. It uses the
unchanged blue-reference snapshot and native feature caches. The existing 14,819-parameter
centered TCN, 65 pose plus 97 RGB/motion channels, source clocks, context, redaction and shared
observation gate remain unchanged. Three modes: pose, RGB/motion, fused. Seed 20261029,
CPU/two threads, deterministic PyTorch, 120 full-batch AdamW updates, LR .003, decay .0001,
gradient clip 1, dropout .1, final epoch only. An exact same-setting repeat verifies
reproducibility; it is not another objective/budget trial. Both existing boundary decoders
and all their thresholds remain fixed. No settings may be changed after outcome inspection.

Let `m_ih` be the existing supervision mask intersected with the existing common observation
gate, `y_ih` the unchanged soft target, and `z_ih` the model logit. `h` ranges over activity,
onset and end; `s` over V4 and V6. Define `b+(z)=softplus(-z)` and `b−(z)=softplus(z)`.
For each source/head, the negative term is

```text
N_sh = Σ(i∈s) m_ih (1−y_ih) b−(z_ih) / Σ(i∈s) m_ih (1−y_ih).
```

For activity, `P_s0 = Σ m_i0 y_i0 b+(z_i0) / Σ m_i0 y_i0` within source. For onset/end,
attribute each existing positive target to the human events whose original triangular target
attains that cached target value. Ties divide its mass equally. Sparse events contribute only
inside their saved range; censored/conflicted boundaries are unavailable. This preserves the
original max-triangle targets rather than summing overlapping targets. Let `r_eih` be that
ownership share (zero when unavailable). With `A_eh = Σ m_ih r_eih y_ih`, and `E_sh` the events
with positive observable mass for this source/head:

```text
P_sh = (1/|E_sh|) Σ(e∈E_sh) [Σ_i m_ih r_eih y_ih b+(z_ih) / A_eh], h∈{onset,end}.
L = (1/3) Σ_h (1/2) Σ_s [(1/2) P_sh + (1/2) N_sh].
```

Thus each source/head/sign has coefficient mass **1/12**; each supported boundary event has
positive coefficient mass `1/(12 |E_sh|)`. Sources total 1/2 and heads total 1/3. Actual losses
and parameter gradient norms need not be equal: errors and feature Jacobians differ. This
replaces the old pooled positive-weight/cap calculation; it does not layer another multiplier
on top. Activity positives and negative terms remain frame-mass normalized within source,
so this does not claim equal activity-event or negative-interval contribution.

`(1−y)` at a soft boundary is the existing BCE negative term, not a new no-punch annotation.
Only existing masks supply it. Unknown gaps, excluded manual motion, conflict support,
censored boundaries, absent observations and sequence padding have zero added weight.
Missing source/head/sign support fails closed. No target, mask, source frame or label is edited.

## Training-source supervision diagnosis

The unchanged batch has 1,688 activity and 1,630 onset/end supervised observable frames.
V4 contributes 874/841/841 frames and positive masses 159/23.008/23.016; V6 contributes
814/789/789 and 245/19.552/16.492. Old positive weights are 3.1782/37.2989/40.
V4's old weighted-denominator shares are **47.52%/52.79%/54.83%**, so the source imbalance
is modest, not evidence that one recording completely dominates the objective.

Supported event-positive counts are V4 **12 onset / 12 end**, V6 **13 / 11**. Original V4
has one left-censored onset and one right-censored end. V6's three conflicted ends stay masked;
V6-motion-102's first onset loses its target support to the existing conflict-proximity mask,
despite observable input frames. No weight may override that exclusion. Surviving positive
target masses per event range 1.500–2.008 for
V4 onsets, 1.496–2.008 for V4 ends, and 1.496–1.512 for V6 boundaries. Long punches therefore
do not automatically dominate boundary-positive learning; target geometry/visibility matter.
They do contribute more activity-positive and within-event boundary-negative terms.

The old loss already averages three separately normalized heads. At identical seeded initial
fused weights, activity/onset/end parameter-gradient norms are 2.378/1.949/.634. At the preserved
fused final checkpoint they are .904/1.246/.760. Activity is not consistently overwhelming the
other heads. Final fused V4/V6 end gradients have cosine −.211, indicating a local conflict,
not proof of unstable training. These are eval-mode diagnostic gradients; optimizer/dropout
training gradients and clipping are recorded separately. Sparse source-specific supervision
can still encourage shortcuts: no negative labels may be inferred in those Unknown contexts.

For the old loss, with `D_h = Σ_i m_ih [(1−y_ih)+w_h y_ih]`, the logit gradient is
`m_ih [(1−y_ih) sigmoid(z_ih) − w_h y_ih (1−sigmoid(z_ih))] / (3 D_h)`.
Positive and negative terms can oppose at soft targets; at `y=.5, w=40` their equilibrium
score is `40/41`, not .5. This is one reason not to interpret those scores as calibrated
boundary probabilities. Such competition is mathematically expected, not itself evidence
of numerical instability. Sparse masks alter each source's positive/negative mass and provide
no penalty for spurious peaks in Unknown gaps; balancing cannot invent that missing supervision.

The alternative explicitly controls these contributions but is a limited hypothesis, not an
assumption that weighting is the primary cause of weak cross-recording ending evidence.
Source normalization, event-positive normalization and sign pooling change together in this
one objective; their individual causal effects are not isolated by this experiment.

## Actual results

**Balanced supervision did not establish improved transfer.** With either decoder, V5 pose
improves 0 → 1 matched positive reference, RGB regresses 2 → 0, and fused stays 0. Fused near-end
evidence stays below .5 at all eight references. More fused candidates are unsupported by
those references; they are not evidence of improved individual-punch recognition. No model
or objective is selected/promoted, and no additional settings were tried.

The unchanged snapshot contains 13 V4 training references, 14 selected V6 events (11 fully
scorable; three conflicting endpoints remain unscored), eight sparse V5 development references
and the original 12 verified negative intervals. Excluded manual events and unreviewed gaps
remain Unknown. Evaluation is identical to the previous study: one-to-one IoU ≥ .5, historical
core-clipped matching for the original 11 V4 events, full untrimmed ranges for sparse sources.
Matched-boundary errors retain full ranges. V4/V6 results are training-source diagnostics;
V5 has incomplete background coverage and has influenced earlier classifier development.
No full event precision, full recall, false-positive rate or independent-test accuracy is claimed.

### Freeze and provenance

All training behavior, exact repeats/reloads and V4/V6 comparisons completed before V5 model
comparison. The training freeze was written at **2026-10-30 23:25:09 UTC (17:25:09 MDT)**:
`reports/boxingvi/punch-localization-balanced-v1/freeze.json`, SHA256
`aa4f5aa1408169b68e714f24693c43e2e62af264466c967930a34da936c5319b`.
It binds 191 artifacts, including checkpoints, scores, reports, configuration and source snapshots.
Development mode checks these hashes plus live policy/code/test hashes before loading V5 tensors.
The single comparison used both fixed decoder policies. A subsequent identical development replay
verified deterministic scores/results; it did not change any setting. Known V5 failures from the
prior reports informed the research question, not objective parameters or selection.

### Matches, misses and extra proposals

Each arrow is preserved pooled-loss baseline → balanced loss, using the same decoder.
`Extra` is unmatched proposals, **not false positives** on sparse references. V6 denominators
exclude the same three conflicting full instances; they are never relabeled background.


| Source | Arm | Decoder | Matched | Missed | Candidates | Extra |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| V4 original (11) | pose | v1 | 1 → 2 | 10 → 9 | 1 → 4 | 0 → 2 |
| V4 original (11) | pose | v2 | 1 → 0 | 10 → 11 | 2 → 1 | 1 → 1 |
| V4 original (11) | rgb_motion | v1 | 1 → 4 | 10 → 7 | 7 → 10 | 6 → 6 |
| V4 original (11) | rgb_motion | v2 | 1 → 3 | 10 → 8 | 3 → 6 | 2 → 3 |
| V4 original (11) | fused | v1 | 5 → 4 | 6 → 7 | 8 → 7 | 3 → 3 |
| V4 original (11) | fused | v2 | 5 → 3 | 6 → 8 | 6 → 5 | 1 → 2 |
| V4 sparse (2) | pose | v1 | 1 → 1 | 1 → 1 | 2 → 4 | 1 → 3 |
| V4 sparse (2) | pose | v2 | 0 → 0 | 2 → 2 | 0 → 4 | 0 → 4 |
| V4 sparse (2) | rgb_motion | v1 | 1 → 1 | 1 → 1 | 4 → 6 | 3 → 5 |
| V4 sparse (2) | rgb_motion | v2 | 1 → 1 | 1 → 1 | 6 → 5 | 5 → 4 |
| V4 sparse (2) | fused | v1 | 1 → 0 | 1 → 2 | 5 → 2 | 4 → 2 |
| V4 sparse (2) | fused | v2 | 1 → 0 | 1 → 2 | 3 → 1 | 2 → 1 |
| V6 blue (11) | pose | v1 | 6 → 3 | 5 → 8 | 13 → 14 | 7 → 11 |
| V6 blue (11) | pose | v2 | 6 → 3 | 5 → 8 | 14 → 13 | 8 → 10 |
| V6 blue (11) | rgb_motion | v1 | 2 → 2 | 9 → 9 | 30 → 30 | 28 → 28 |
| V6 blue (11) | rgb_motion | v2 | 2 → 2 | 9 → 9 | 21 → 28 | 19 → 26 |
| V6 blue (11) | fused | v1 | 8 → 5 | 3 → 6 | 12 → 9 | 4 → 4 |
| V6 blue (11) | fused | v2 | 9 → 6 | 2 → 5 | 12 → 11 | 3 → 5 |
| V5 dev (8) | pose | v1 | 0 → 1 | 8 → 7 | 2 → 6 | 2 → 5 |
| V5 dev (8) | pose | v2 | 0 → 1 | 8 → 7 | 2 → 7 | 2 → 6 |
| V5 dev (8) | rgb_motion | v1 | 2 → 0 | 6 → 8 | 14 → 3 | 12 → 3 |
| V5 dev (8) | rgb_motion | v2 | 2 → 0 | 6 → 8 | 14 → 2 | 12 → 2 |
| V5 dev (8) | fused | v1 | 0 → 0 | 8 → 8 | 0 → 2 | 0 → 2 |
| V5 dev (8) | fused | v2 | 0 → 0 | 8 → 8 | 0 → 3 | 0 → 3 |

On the original 48.006 seconds of verified training negatives, V1 proposal counts remain
0/1/0 for pose/RGB/fused and V2 remains 0/0/0. This is unchanged candidate behavior, not a
held-out background evaluation. The balanced RGB V1 proposal is in V6-motion-5, as before;
its exact new range is recorded in the per-case output. Predicted unresolved activity on
negatives grows in several arms even when no candidate survives.

### Boundary evidence and competing peaks

Near-boundary maxima use the unchanged ±1/12 s window and .5 threshold. Scores are uncalibrated;
a high nearby value is neither a verified boundary nor a calibrated probability. Peak counts
cover entire contexts, including Unknown motion, so they are not false-positive counts.

| Source | Arm | Onsets with nearby score ≥ .5 | Ends with nearby score ≥ .5 | Mean near-end score | Onset peaks | End peaks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V4 original (11) | pose | 11 → 11 | 11 → 11 | 0.967 → 0.966 | 43 → 56 | 33 → 36 |
| V4 original (11) | rgb_motion | 10 → 10 | 11 → 11 | 0.966 → 0.976 | 34 → 39 | 41 → 43 |
| V4 original (11) | fused | 11 → 11 | 11 → 11 | 0.967 → 0.940 | 28 → 35 | 30 → 37 |
| V4 sparse (2) | pose | 2 → 2 | 2 → 2 | 0.937 → 0.934 | 21 → 23 | 19 → 16 |
| V4 sparse (2) | rgb_motion | 2 → 2 | 2 → 2 | 0.937 → 0.978 | 15 → 14 | 15 → 16 |
| V4 sparse (2) | fused | 2 → 2 | 2 → 2 | 0.938 → 0.880 | 15 → 19 | 17 → 15 |
| V6 blue (11) | pose | 10 → 10 | 11 → 11 | 0.971 → 0.971 | 108 → 137 | 56 → 69 |
| V6 blue (11) | rgb_motion | 11 → 11 | 11 → 11 | 0.901 → 0.972 | 93 → 121 | 92 → 95 |
| V6 blue (11) | fused | 11 → 11 | 11 → 11 | 0.986 → 0.974 | 65 → 78 | 38 → 33 |
| V5 dev (8) | pose | 5 → 5 | 3 → 4 | 0.436 → 0.462 | 10 → 26 | 7 → 15 |
| V5 dev (8) | rgb_motion | 4 → 4 | 4 → 0 | 0.351 → 0.101 | 40 → 41 | 25 → 20 |
| V5 dev (8) | fused | 4 → 5 | 0 → 0 | 0.007 → 0.008 | 6 → 15 | 3 → 7 |

Training endpoint scores remain strong, but unique pairing becomes harder in several arms.
V6 pose competing-assignment records increase 12 → 18 and fused 8 → 11 under decoder V2;
V1 multiple-onset records change 17 → 21 and 16 → 16 respectively. These are overlapping
reason records, not counts of real punches. Dense boundary evidence can coexist with worse
instance localization. Fused original V4 also loses local supported-pair feasibility for one
reference (11 → 10), despite strong raw scores near all endpoints.

V5 fused ending maxima, in the eight source-reference orders, change:

```text
baseline: .0062 .0024 .0318 .0024 .0085 .0023 .0037 .0012
balanced: .0066 .0035 .0246 .0035 .0117 .0047 .0089 .0019
```

This is not useful recovered ending evidence. Fused has zero locally supported thresholded
pairs at IoU ≥ .5 under either objective. Pose local feasibility rises 4 → 5, but only one
actual match survives either decoder. RGB still has four locally feasible broader pairs,
even though none of its ending maxima within the true boundary window reaches .5; misplaced
endings and competing evidence can overlap an event sufficiently without being accurate
endpoints. This label-assisted local feasibility is not a jointly attainable oracle or a
source of decoder boundaries.

Concrete V5 examples:

- Balanced pose V1 matches human [8230,8247) with [8236,8252): IoU exactly .5, +6/+5 frames
  (+200/+167 ms). Pose V2 instead matches [8221,8229) with [8220,8232): IoU 267/400,
  −1/+3 frames (−33/+100 ms). The one-match totals refer to different punches.
- RGB loses both former references [8230,8247) and [8400,8413). Its V2 candidates become
  [8215,8220) and [8419,8425); neither matches a selected reference. Removal of former
  fragments does not establish better segmentation when their punches are now missed.
- Balanced fused V2 returns [8109,8137), [8192,8210), [8210,8216). None matches any of the
  eight references. Added proposals and displaced end peaks do not solve the combinations.
- No six-class prediction is made. Punch types in matched-reference records are human
  reference metadata, not automatically predicted class labels.

### Boundary error, fragmentation and unresolved motion

The following comparison uses decoder V2, baseline → balanced. Errors are medians of absolute
native-frame errors **conditional on matches**, so populations differ; `—` means no matches.
V1 full errors and per-reference matches/misses are retained in the versioned records.

| Source | Arm | Median start / stop error | Split-risk records | Unresolved seconds |
| --- | --- | --- | ---: | ---: |
| V4 original (11) | pose | 2 / 1 → — | 0 → 0 | 11.053 → 12.638 |
| V4 original (11) | rgb_motion | 1 / 0 → 0 / 1 | 0 → 1 | 10.762 → 10.385 |
| V4 original (11) | fused | 0 / 0 → 1 / 1 | 0 → 1 | 7.923 → 8.508 |
| V4 sparse (2) | pose | — → — | 0 → 0 | 4.713 → 3.502 |
| V4 sparse (2) | rgb_motion | 0 / 0 → 0 / 0 | 0 → 0 | 2.752 → 3.045 |
| V4 sparse (2) | fused | 0 / 0 → — | 0 → 0 | 4.170 → 4.295 |
| V6 blue (11) | pose | 0 / 1 → 0 / 1 | 1 → 0 | 25.235 → 29.777 |
| V6 blue (11) | rgb_motion | 4 / 4 → 0 / 6 | 2 → 3 | 24.527 → 27.151 |
| V6 blue (11) | fused | 0 / 1 → 0 / 1 | 0 → 0 | 21.065 → 22.731 |
| V5 dev (8) | pose | — → 1 / 3 | 0 → 0 | 5.071 → 7.441 |
| V5 dev (8) | rgb_motion | 1.5 / 5 → — | 1 → 0 | 10.146 → 13.246 |
| V5 dev (8) | fused | — → — | 0 → 0 | 3.771 → 5.102 |

V1 split risks change: original V4 pose/RGB/fused 0/1/0 → 0/2/1; sparse V4 stays zero;
V6 1/5/0 → 0/5/0; V5 0/1/0 → 0/0/0. Neither decoder records merge-risk flags for these
groups, which does not prove no merged/ambiguous punches. Broad ambiguous motion remains
unresolved rather than being emitted as a verified punch. Fused V6 maximum onset/stop
error rises 0/1 → 2/1 frames with V2, while matched references fall 9 → 6.

Unresolved durations union overlapping records per case and then sum cases; they are not
punching duration or independent exposure. Inactivity expiry changes these durations too.
Observation gates are unchanged: V6 has 2.253 s and V5 .033 s of common unobservable support
for each arm/objective. Missing observations never become verified background.

### Loss and gradient behavior

Different objective loss values cannot be compared directly. Both objectives were therefore
measured at both preserved and new final checkpoints:

| Arm | Old checkpoint: pooled / balanced loss | New checkpoint: pooled / balanced loss | Balanced initial → final | Updates clipped (>1), old → new | Max pre-clip norm, old → new |
| --- | --- | --- | --- | ---: | ---: |
| Pose | .086996 / .086464 | .098170 / .096928 | .919082 → .096928 | 98 → 94 | 9.052 → 11.201 |
| RGB/motion | .132575 / .133172 | .115964 / .116369 | .946035 → .116369 | 97 → 92 | 14.621 → 27.733 |
| Fused | .062844 / .061257 | .076049 / .074105 | .909178 → .074105 | 85 → 93 | 12.668 → 8.401 |

All losses, gradients and updates are finite. Eval loss rises on 38/26/36 of the 119 adjacent
new epoch transitions for pose/RGB/fused, versus 36/38/39 originally. Frequent clipping and
nonmonotonic loss already occur in the pooled baseline. Balancing has not established more
stable optimization: RGB's maximum gradient increases, and old pose/fused checkpoints even
score better under the new objective than their respective new fixed-budget fits. This is
one fixed-seed, final-epoch engineering result; it does not isolate optimizer causality or
justify retrospectively choosing an earlier epoch or another setting.

Actual coefficient audits confirm 1/12 per source/head/sign (floating-point tolerance).
V4 event-positive coefficients are 1/144 per onset/end; V6 are 1/156 per onset and 1/132 per
end. Zero-mass/censored/conflicted units retain zero weight. At the identical fused initial
parameters, end-gradient norms V4/V6 change .434/.241 → .327/.370 under the alternative loss.
The normalization affects the intended contribution without forcing equal gradients. At the
new final checkpoint the balanced end norms are .377/.114, and source-end gradient cosine
is +.621 versus −.102 at the old checkpoint measured under the same balanced objective.
That local alignment does not produce V5 ending transfer. Per-source/head losses, positive/
negative logit-gradient sums, head/source cosines and each clipped-update norm are retained.

Primary fit calls (including gradient diagnostics) took 15.65/17.15/7.53 seconds for pose/RGB/
fused. Verification ran concurrently on the same machine; these are observed engineering
costs, not isolated inference/training benchmarks. No new feature extraction was needed.

### Previously tested uploads

These are candidate counts without exhaustive reference inventories, baseline → balanced:

| Arm / decoder | V6 six-second | V4 eight-second | V4 29-second |
| --- | ---: | ---: | ---: |
| pose / v1 | 1 → 0 | 2 → 4 | 4 → 5 |
| pose / v2 | 1 → 0 | 4 → 3 | 1 → 6 |
| rgb_motion / v1 | 1 → 5 | 1 → 2 | 20 → 17 |
| rgb_motion / v2 | 0 → 6 | 0 → 0 | 16 → 13 |
| fused / v1 | 2 → 1 | 3 → 1 | 4 → 6 |
| fused / v2 | 2 → 1 | 2 → 2 | 4 → 7 |

These are same-recording diagnostics, not unfamiliar-recording validation or verified counts.
The blank control retains zero candidates and 25 unobservable frames in every case.

## Interpretation and next meaningful step

The objective change verifies its mathematical contribution contract but **does not demonstrate
reliable cross-recording localization improvement**. A single pose match is offset by regression
in RGB and fused/V6 fit, and fused ending evidence on V5 remains effectively absent. The initial
source normalization imbalance was only modest; the experiment does not support another blind
weighting change as the next step. It does not establish that balance is irrelevant in general.

The clearest demonstrated next data bottleneck is **recording and punch-motion diversity**.
A post-freeze audit of human-reference type metadata finds:

| Source | Scorable reference metadata |
| --- | --- |
| V4 training | 3 jab, 4 lead hook, 1 lead uppercut, 5 rear uppercut |
| V6 training | 3 lead hook, 4 rear hook, 2 lead uppercut, 2 rear uppercut |
| V5 development | 4 jab, 4 cross |

The three partial/conflicted V6 events are another lead hook and two lead uppercuts; including
their uncontested training terms supplies no straight-punch diversity. There are **no human
cross examples in this experiment's training positives**, and all three jab references come
from V4. This is a coverage confound, not a new six-class ground-truth claim: these fields are
existing human metadata and never enter model inputs, loss classes or event pairing. An
objective balancing two recordings cannot supply missing movement examples or disentangle
recording from punch-family differences.

The learned failure is recording-dependent ending evidence, not a decoder-only problem. The
present data cannot isolate context from visual representation or diversity causally.
Post-freeze timing diagnosis
finds median scorable event durations V4 .459 s, V6 .834 s, V5 .3505 s. Ranges are .250–1.001,
.376–.960 and .200–.567 s respectively. The same RF15 covers about .584 s at V4/V6's native rate
and .467 s at V5's. The sparse V5 sample is small and selective; this is a demonstrated time
and human-metadata coverage mismatch, not proof that longer context alone will solve localization.
High training near-end scores and missing V5 scores also leave coarse 4×4 redacted motion and
masked pose features as plausible representation limitations.

The next meaningful dataset decision should address **straight-punch ending coverage across
training recordings**, with truthful Unknown masks, before another objective trial. Start with
an evidence audit of already-reviewed training material; currently excluded non-blue events
must not be imported without a separately approved subset decision. Keep V5 development-only
and all reserved recordings excluded. No such subset change or additional labeling is requested
or performed here. Broader eligible recording diversity would still be necessary before a
reliability claim; better fit to the same V4/V6 clips cannot establish that.

In parallel, a bounded **physical-time, motion-phase representation audit** can use existing
authorized short/long instances to check whether ending motion survives the seven-frame context
and redacted motion grid. Only if justified, predeclare one time-normalized-context ablation
with fixed parameter count and objective, or an appearance-aware representation experiment
if the grid demonstrably discards the evidence. These are alternatives to be justified by
measurements, not new experiments launched here. This milestone does not demonstrate that a
larger temporal model or more detailed visual features would solve the coverage confound.

## Verification, preservation and reproduction

- **744 Python tests** pass, including 14 new objective/gradient/mask/freeze tests and a permitted
  V4/V6 real-fixture contract. Tests cover tied boundary ownership, duration-independent event
  mass, censored/conflict/Unknown masks, missing observations, source isolation, missing-sign
  failure, baseline-loss equivalence, analytic gradients and pre-V5 freeze enforcement.
- **111 frontend tests**, TypeScript, Prettier, Ruff (`src tests`), changed-Python formatting
  and production build pass. One original frontend test timed out during concurrent CPU-heavy
  verification; the complete isolated rerun passed. No frontend source was changed. The existing
  Python Starlette/httpx deprecation warning remains.
- All three same-budget repeats have identical parameter hashes, full fit diagnostics/curves
  and training-source scores. Reloaded checkpoints agree exactly. All six development model/
  input combinations replay with identical scores/results on a second unchanged invocation.
- **162 baseline decoder records** reproduce the prior study exactly. **285** new candidate
  endpoints across both decoders use unchanged native peaks and exact start/last/stop PTS.
  Native frame maps, source offsets, observation resets and original masks are validated.
- **9,385 pre-existing protected artifact files remain byte-identical**. Dataset/review provenance,
  original sources, all historical checkpoints, source-group exclusions and prior experiments
  remain unchanged. No reserved recording was decoded/evaluated and no V9 path was opened.

Versioned code/config: `localization_balanced_loss.py`, `localization_balanced_experiment.py`,
`configs/punch_localization_balanced_v1.json`. All research outputs remain Git-ignored:

- `reports/boxingvi/balanced-loss-preflight-20261030T231656Z/`: preservation inventory, old
  supervision/gradient audit, pre-training coefficient audit and predeclared protocol copy.
- `reports/boxingvi/punch-localization-balanced-v1/`: three new checkpoints, coefficient/event
  audits, curves, gradients, both decoders' baseline/new records, boundary evidence and freeze.
- `reports/boxingvi/punch-localization-balanced-development-v1/`: post-freeze V5 comparison.
- `reports/boxingvi/punch-localization-balanced-development-repeat-v1/`: exact replay verification.
- `reports/boxingvi/balanced-loss-timing-diagnosis-v1.json`: post-freeze duration audit, never used
  to change this experiment's settings.
- `reports/boxingvi/balanced-loss-coverage-diagnosis-v1.json`: post-freeze human-metadata coverage
  audit; types are never model inputs or predicted classes.
- `reports/boxingvi/punch-localization-balanced-preservation-v1.json`: preservation, exact baseline
  replay, repeat and native endpoint verification receipt.

Fresh output paths only; these commands run this one fixed objective and its repeat, never
modify baseline checkpoints or promote a model:

```sh
.venv/bin/python -m boxing_motion_ai.detection.localization_balanced_experiment \
  --phase training --output reports/boxingvi/balanced-training-REPEAT
.venv/bin/python -m boxing_motion_ai.detection.localization_balanced_experiment \
  --phase development --training reports/boxingvi/balanced-training-REPEAT \
  --output reports/boxingvi/balanced-development-REPEAT
```
