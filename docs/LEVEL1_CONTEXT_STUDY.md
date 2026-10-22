# Experiment 3: real causal pre-clip pose context

2026-10-21. Protocol fixed before full training. Mirror-only remains the reference. Optional
visibility augmentation is **disabled in both conditions**. This study adds real context and
changes only which encoded positions are pooled; it does not change the TCN architecture,
features, quality policy, annotated examples, class weights, training budget or recording split.

## Exact-time context and boundaries

`real-preclip-context-1.0` selects preceding decoded pose frames in
**[punch-start PTS − 0.65 seconds, punch-start PTS)**. Selection uses exact rational PTS/time
bases and binary search within a continuous run. No duration×FPS indexing, resampling,
interpolation, smoothing or position filling. A 0.65-second window exceeds the existing TCN's
14-frame history span at both native rates (about 0.584s in V4 and 0.467s in V5). The model still
uses native frames; its physical receptive field is not made identical across recordings.

Context stops at recording start, confirmed cuts, changes in scene/segment IDs, nonconsecutive
frame indices, time-base changes, nonpositive timestamp steps or a gap **greater than 100ms**.
The original full-inventory loader rejects missing extraction rows or incomplete scopes. A
missing pose detection with a stored source frame/time is retained with its original NaNs and
false validity; it is not an absent extraction frame. A newly identified boundary inside a
punch fails validation instead of crossing it or silently dropping that example. Unknown scene
cuts remain a limitation: the known cut inventory is not certified exhaustive.

The existing approved `PoseClips` selection is wrapped, not replaced. All **527 V4 training** and
**583 V5 development** examples remain identical. Source coordinates, visibility/presence,
validity, decoded frame indices, integer PTS and rational time bases are copied directly from
the existing full-video NPZ arrays, with source hashes rechecked. Punch features are asserted
**bit-for-bit identical** after adding context because normalization is frame-local. The raw
punch slice remains separately accessible. No new class/background targets are created.

A preflight audit of the actual inputs found:

- V4: **15** preceding frames for every example; V5: **19**. Every example has at least the
  14 real history frames needed before the first punch state. No examples are dropped or require
  context truncation in these specific prepared datasets.
- V5's confirmed cut at frame **10679** is a hard boundary, in addition to its scene/segment
  changes. Context is never taken across that boundary.
- There are 11 missing-detection frame appearances in V4 context (not necessarily unique frames),
  all preserved with invalid masks; V5 context has none. No such frames are filled.
- Context overlaps another bounded source annotation in 412 V4 / 406 V5 examples. Context
  overlaps an excluded annotation interval in 4 / 10 examples. These are **unlabeled inputs**,
  never reinstated supervised examples. Context can include unknown or questionable actions;
  only the original approved punch carries a target. Recording splits remain strict.

## Causal processing and punch-only pooling

The unchanged TCN receives context followed by the complete punch. Its existing left-padded
convolutions and channel-only LayerNorm process those frames causally. A separate pooling mask
is false on context and right padding, true **only on the original annotated punch interval**.
The same existing masked-mean pooling and linear classification head are used. For that linear
head, averaging encoded punch states is also equivalent to averaging punch logits. No context
frame is classified as the target punch or `no_punch`.

Training mirroring remains one Bernoulli(0.5) choice per example, applied consistently to the
whole context+punch sequence, including confidence and masks. Context windows and pool masks
are unchanged by reflection. Evaluation uses unaugmented inputs. A no-context configuration
retains the original source loader/collation and reproduces the previous mirror-only runs.
The model source file and original feature module remain unchanged and hash-checked.

## Matched experiment and shortcut probes

Seeds **20261021, 20261022, 20261023**, each with mirror-only versus mirror+context, full-feature
TCN and mask-only TCN: **12 runs**, all 40 epochs with the original batch/optimizer/weighting
settings. V4 trains, V5 selects the best macro-F1 epoch (earlier ties). V9 is reserved and unopened.
Initial weights, annotated row selections, batch order and mirror decisions match within each
pair. The same dropout seed is used; actual dropout tensors differ as sequence shapes change.
No claim is made that dropout masks match across different input lengths.

Frozen, post-training diagnostics use the same punch-only pooling:

1. Replace mask-only inputs with identical all-valid tokens, preserving true prefix/punch
   lengths. This is a synthetic input sensitivity test, not a pose-filling policy.
2. Use identical tokens with punch lengths 3–32 and prefix lengths 0/15/19 to isolate artificial
   clip-start/length signals. With at least 14 warm-up frames, all punched positions have full
   receptive fields; a constant stream should become length-invariant within numeric tolerance.
3. Neutralize only mask context to all-valid tokens, leaving punch masks original, to probe
   reliance on actual preceding visibility patterns versus warm-up alone.
4. Remove context from the frozen model and keep only original punch inputs, to test dependence
   on context. This is an out-of-distribution intervention, not an additional training condition.

Every frozen original prediction must reproduce its saved checkpoint probabilities. These
probes cannot establish that all duration shortcuts are removed, or that improved performance
reflects physical movement rather than scripted combinations/visibility history. Most contexts
contain prior actions, and V5 remains repeatedly used development data. Seeds reuse the same
examples and are not independent boxer groups. No boxer-independent generalization is claimed.

## Reproduction and artifacts

Run from the repository root with fresh ignored output directories:

```sh
uv sync --locked --extra pose --extra train
uv run --extra pose --extra train boxing-motion-ai train-context-study --help
uv run --extra pose --extra train boxing-motion-ai train-context-study --config configs/level1_context_study.yaml --output reports/boxingvi/level1-context-smoke-repeat --smoke
uv run --extra pose --extra train boxing-motion-ai train-context-study --config configs/level1_context_study.yaml --output reports/boxingvi/level1-context-repeat
uv run --extra pose --extra train pytest
uv run ruff check .
```

The smoke study uses two epochs, two clips/class and the first seed in all four combinations.
Its metrics are diagnostics only. Full outputs are under `reports/boxingvi/level1-context-v1/`:
study/derived configs, summary JSON/Markdown, per-class recall/confusion plots, and each seed/arm's
source snapshots/provenance (including exact context boundaries and PTS), source-row predictions,
per-class metrics, curves, best/final checkpoints, mirror logs and `shortcuts.json`. Previous
mirror-study files are hashed before/after and every mirror-only control must reproduce its
previous metrics exactly. Context duration is an opt-in version-1.3 experiment field; earlier
configs default to no context. Source data, generated reports, media and checkpoints stay out of Git.

## Measured results

Completed the smoke study and all **12** full runs. All mirror-only controls reproduced previous seed/model metrics **exactly**. Both conditions retained identical approved examples, initial weights, epoch/optimizer budget, shuffle order and mirror choices; every visibility event log was empty. No context duration, model or other hyperparameter was retuned.

Best-checkpoint V5 results, **mean ± sample standard deviation** across the same three seeds:

| Input | Condition | Accuracy | Macro-F1 |
|---|---|---:|---:|
| tcn | mirror_only | 71.87% ± 3.35% | 0.4912 ± 0.0188 |
| tcn | mirror_context | 65.87% ± 1.74% | 0.4496 ± 0.0306 |
| tcn_mask_only | mirror_only | 56.89% ± 2.53% | 0.3546 ± 0.0187 |
| tcn_mask_only | mirror_context | 52.49% ± 2.42% | 0.3493 ± 0.0210 |

Full-feature macro-F1 fell in **every paired seed**: −0.0968, −0.0230, −0.0049; mean **−0.0416** (sample SD 0.0487). Selected epochs were 37/7/16 with context versus 31/31/28 for mirror-only. These three seeds are not independent recording groups and do not establish a significance/generalization claim beyond this development comparison.

### Which classes benefit?

| Source class | Full mirror recall | Full context recall | Mask mirror recall | Mask context recall |
|---|---:|---:|---:|---:|
| jab | 91.4% | 85.3% | 92.3% | 88.3% |
| cross | 96.4% | 98.3% | 54.3% | 41.1% |
| lead_hook | 80.9% | 38.7% | 38.7% | 22.5% |
| rear_hook | 21.4% | 30.8% | 17.1% | 29.9% |
| lead_uppercut | 0.0% | 2.6% | 0.9% | 5.1% |
| rear_uppercut | 21.9% | 22.4% | 20.1% | 28.3% |

- **Lead uppercut is not rescued:** recall rises only from 0% to 2.6%, with 0/39, 2/39 and 1/39 correct per context seed. Correct source rows are none for 20261021, 118/219 for 20261022, and 126 for 20261023. Mean confusion remains 68.4% jab and 29.1% lead_hook. The context-aware mask control reaches 5.1% recall, so the few full-feature hits do not establish reliable uppercut movement recognition.
- **Rear hook improves on average but not consistently:** 21.4%→30.8%, with paired recalls 5.1→41.0%, 20.5→38.5%, 38.5→12.8%. Mean context confusion is 31.6% rear_uppercut, 27.4% cross, 10.3% lead_uppercut, and 30.8% correct.
- **Rear uppercut is effectively flat and highly seed-dependent:** 21.9%→22.4%, with paired recalls 46.6→6.8%, 15.1→19.2%, 4.1→41.1%. Context predictions remain 29.2% rear_hook, 26.0% cross, 18.7% lead_uppercut and 22.4% correct on average. This is no stable rescue.
- **Lead hook regresses sharply:** 80.9%→38.7%, falling in all seeds. With context, 55.4% become jab and 5.9% cross. Jab also falls (91.4%→85.3%); cross improves slightly (96.4%→98.3%). Mean macro recall drops from about 52.0% to 46.3%.

**This context configuration does not improve balanced six-class recognition or mean overall accuracy.** Some rear-hook gains and a few uppercut hits come with substantial class tradeoffs. No class is removed from the metrics or dataset.

### Does the artificial sequence-start shortcut diminish?

**Yes for the targeted constant-input boundary/length signal.** In trained mirror-only mask models, identical tokens with no history yield maximum logit changes of 2.89–6.28 when punch length changes from 3 to 32, and predictions vary among cross/jab/lead_hook. For trained context mask models with 15 or 19 warm-up tokens, the maximum change is about 1.2e-6 or less and the predicted class is invariant across those lengths. Every actual V4/V5 example has enough context to put the artificial input start outside the first punch state's receptive field.

Replacing the entire mask sequence with all-valid tokens produces mean V5 macro-F1 0.0931 for context models (all predict jab), versus 0.1624 for mirror-only models, where length/boundary differences still produce varying predictions. This removes the specific constant-stream start-padding cue, not all duration information or source-specific shortcuts.

Frozen V5 diagnostic macro-F1, means across seeds:

| Model / condition | Original | Remove context | Neutralize only mask context | All-valid mask stream |
|---|---:|---:|---:|---:|
| Full mirror | 0.4912 | 0.4912 | — | — |
| Full context | 0.4496 | 0.2863 | — | — |
| Mask mirror | 0.3546 | 0.3546 | 0.3546 | 0.1624 |
| Mask context | 0.3493 | 0.2646 | 0.2748 | 0.0931 |

The trained models use real preceding content: removing it reduces full/model and mask-only performance, and neutralizing prior masks also hurts. These are potentially out-of-distribution input interventions, not valid alternative trained-model results or causal estimates of context benefit. The matched training comparison remains the primary result and favors mirror-only.

Context-aware mask-only macro-F1 is nearly unchanged on average (0.3546→0.3493), with paired changes +0.0363/−0.0362/−0.0159. Its full-feature gap narrows from 0.1366 to 0.1003. Visibility/history and remaining timing cues still classify punches without movement coordinates. Many contexts contain previous scripted punches; temporal sequence and source-specific history can replace the removed start cue. The study does not isolate which alternative shortcut causes each error.

### Overfitting

At epoch 40, full-feature mean training/development macro-F1 changes from **0.8044/0.4248** (mirror-only) to **0.9365/0.3795** (context). The gap grows from 0.3796 to **0.5570**. All context final-epoch development scores are below their selected best scores. Mask-only final training/development changes from 0.5719/0.3229 to 0.7289/0.3148. Additional real history improves in-sample fit and increases cross-recording overfitting; it does not solve the dominant transfer problem in this setup.

### Interpretation and stopping decision

**Retain mirror-only as the reference; leave pre-clip context optional and disabled by default.** The time-safe context implementation works and the targeted synthetic boundary signal is suppressed, but the tested 0.65s configuration worsens development recognition in all full-feature seeds. It has not solved lead uppercuts or robust rear-punch recognition.

The result separates a mechanical artifact from the main recognition task: removing that artifact is insufficient to overcome view/pose/label uncertainty and source-specific correlations. As a possible contributor, most short annotated punches share receptive fields with earlier actions in the context. That could blur class distinctions or enable script memorization, but these measurements do not prove it. Native-frame timing mismatch and unverified annotation boundaries/stance remain unchanged. No relabeling, new data, visibility augmentation, architecture change or context-window tuning followed these results. Stop for review.

### Verification

**141 tests passed; Ruff and `git diff --check` passed.** New tests cover exact-time selection at different rates, inclusive time cutoffs, cuts/scene IDs/segments/frame gaps/time gaps/changed time bases, preserving missing detections, refusing incomplete extraction and cuts inside a punch, keeping boundary-limited examples, bit-identical punch features, original NaNs/confidence/masks/timestamps, context-aware collation, punch-only pooling, causality and right-padding invariance, constant-token length probes, disabled visibility and checkpoint/report writing. All original prepared data and prior mirror-study files remain unchanged; generated artifacts and weights remain ignored. V9 was not opened.
