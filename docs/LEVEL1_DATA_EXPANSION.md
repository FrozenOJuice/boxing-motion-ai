# V7/V8 provisional training expansion

Mirror-only remains the reference. Visibility augmentation and real pre-clip context
remain optional and disabled. This study adds recording groups without changing the
TCN, feature representation, mask policy, class meanings or development recording.

## Source alignment and annotation review

The unchanged workbooks contain 195 V7 rows and 199 V8 rows. Both have a single
`Sheet1`, A/B frame bounds and literal C punch names. V8's A/B headers have trailing
spaces. Neither workbook supplies timing helper columns, an annotation FPS,
zero/one-based indexing, an end convention or a verified trim offset.

Sources, as identified by the supplied metadata:

- V7: <https://www.youtube.com/watch?v=sqU3DmLppcQ&t=29s>
- V8: <https://www.youtube.com/watch?v=tSNybMUpMKk&t=681s>
- Dataset: <https://github.com/Bikudebug/BoxingVI>

The URL playback parameters are **not** annotation offsets. Original source media
are unchanged. Licensing and redistribution remain unverified.

The hashed decoded inventories contain 28,062 frames for V7 (3840×2160,
reported average 30000/1001 FPS) and 24,081 for V8 (1920×1080, reported average
18912/631 FPS). Both inventories contain strictly increasing PTS with 33/34 ms
steps. No workbook interval is out of video bounds. This does not certify an
unedited recording or prove that the author used either reported FPS.

The new local review contains 22 V7 and 20 V8 source-row sequences, chosen across
each available class and early/middle/late portions, with overlap examples. It
compares raw decoded indices, raw minus one, and annotation clocks of 30 and
30000/1001 FPS against actual PTS. Each source frame is identified by its exact
PTS, even when seeking is used to accelerate decoding. Contact sheets compare
raw and 30 FPS interpretations; the HTML player steps through consecutive source
frames and displays which hypothetical intervals contain each frame.

Assistant inspection supports **raw decoded indices provisionally**, rather than
automatically converting annotations through a 30 FPS clock:

| Recording / row | Raw interval | Evidence from decoded frames |
|---|---|---|
| V7 / 2, jab | 4541–4557 | Visible extension and recovery inside the raw interval. |
| V7 / 106, cross | 12777–12787 | Raw interval centers a straight punch; 30 FPS includes the preceding action. Row remains overlap-excluded. |
| V7 / 108, rear uppercut | 12790–12805 | Uppercut-like movement in raw interval; diagnostic only because of overlap. |
| V7 / 195, rear hook | 19996–20018 | Raw interval contains hook-like movement; 30 FPS is mainly guard/loading. Overlap exclusion remains. |
| V8 / 112, cross | 11400–11406 | Raw interval contains a straight punch; 30 FPS maps to 11389–11395, mainly guard. |
| V8 / 197, lead uppercut | 19117–19131 | Raw interval contains punch movement; the 30 FPS alternative ends before most movement. Exact class remains provisional in this oblique view. |
| V8 / 200, lead hook | 19378–19391 | Raw interval contains a sweeping punch; the 30 FPS alternative maps to 19359–19372, mainly preparation. |

The rate alternatives increasingly separate over time: roughly 5 to 20 frames at
V7's early/late annotations and 1 to 19 in V8. This is **hypothesis separation**,
not measured drift in the raw annotation alignment. Inspected anchors support raw
timing across the recording; they cannot resolve one-frame indexing differences,
biomechanical onsets, inclusive/exclusive ends or every label. No rate conversion,
constant offset, stance inference or label correction is applied.

The explicit decision and remaining uncertainties are versioned in
`configs/v7_v8_alignment_review.yaml`. Local evidence is under
`reports/boxingvi/v7-v8-expansion-gate-v2/`; open `index.html` for source review.
The first review directory is also preserved.
That initial inventory deliberately marks all rows pending: it is immutable
preflight evidence. A separate full-data training gate records any subsequent
provisional decision and exact selected dataset hashes.

## Conservative exclusions

| Class | V7 source | V7 structurally eligible | V8 source | V8 structurally eligible |
|---|---:|---:|---:|---:|
| jab | 29 | 8 | 25 | 17 |
| cross | 30 | 1 | 32 | 21 |
| lead_hook | 60 | 22 | 88 | 64 |
| rear_hook | 5 | 1 | 0 | 0 |
| lead_uppercut | 50 | 39 | 29 | 22 |
| rear_uppercut | 20 | 0 | 25 | 8 |
| unmapped literal `Jab ` | 1 | 0 | 0 | 0 |
| total | 195 | 71 | 199 | 132 |

These are **annotation inventory counts, before full pose filtering**. V7 has 62
strict overlap pairs and 16 shared-endpoint pairs, involving 124 rows. V8 has 29
and 8, involving 67 rows. Every implicated row stays excluded under the existing
conservative policy. Shared endpoints might be valid under an exclusive-end
convention; no such convention is silently assumed. Some strict overlaps may
describe adjacent actions with overlapping retraction, rather than wrong labels.
They still require a separate reviewed decision before inclusion.

V7 row 54 is out of workbook time order and overlaps another interval. Row 59
has the literal class `Jab ` and overlaps; its whitespace is not silently cleaned.
Row 53's three-frame lead uppercut (8128–8130) is fully nested inside row 54's
rear uppercut (8125–8138). This is a specific boundary/class ambiguity beyond
shared endpoints; neither row is corrected or included.
V8 rows 65/66 similarly label a lead hook at 8913–8921 inside a lead uppercut at
8913–8935. Both stay excluded and are included in the local visual review.
All 20 V7 rear uppercuts are overlap-flagged. V8 has no rear-hook labels, which is
missing class coverage, not an invented negative or a reason to rename another
class. Oblique uppercut/hook movements remain a class-review concern; pose
confidence does not establish label correctness.

## Preparation and evaluation contract

Full extraction uses independent IMAGE inference, width 960, original 0.5 joint
thresholds, and no frame dropping, tracking, smoothing, interpolation or filling.
Coordinates, visibility/presence, masks, decoded indices and integer PTS/rational
time bases are retained. No scene cuts are newly certified by this review; empty
cut inventories do not assert that there are no edits. The estimator carries no
state across frames. Existing known-cut and extraction-boundary exclusions remain.

The mask-aware70 policy is unchanged: reliable torso anchors on at least 95% of
frames, one complete arm on at least 80%, at least 70% valid core entries, and at
most 10% missing frames. Full reports distinguish annotation exclusions from
pose-policy rejections. Flagged uppercut poses can be assessed diagnostically
without approving their labels or moving them into training. Unlabeled and
excluded footage remains unknown; no `no_punch` examples are generated.

Use V4 plus V7/V8 for training, and V5 exclusively for development. V7 and V8 share
`precision-striking-unresolved`; V4 and V5 have separate conservative source
groups. Recording IDs, source groups and unknown boxer/session IDs are separate
fields. These groups prevent known source relationships crossing this split;
they do not establish boxer-independent evaluation. **V9 is not opened.**
Prepared candidates retain their existing recording-level `split_group` and
unassigned `split` fields. Those are not the approved cross-recording split plan.
The frozen gate and grouped loader add the authoritative `source_group_id` and
keep V7/V8 together; they do not rewrite earlier prepared artifacts.

Both arms use mirror probability 0.5, the original 41-channel representation,
the same causal TCN and 40-epoch budget, seeds 20261021/20261022/20261023, and matched
mask-only controls. Visibility augmentation and context are disabled. Class weights
are recomputed from training examples only using the existing inverse-frequency
formula. More examples mean more optimizer steps per epoch; the study measures
the combined data expansion, not an isolated viewpoint or frame-rate effect.
V5 selects epochs as before and remains repeatedly used development, not a test.

The expanded loader permits a recording to lack classes, but requires all six in
the combined training set. It does not discard or merge classes. Source row numbers
are qualified by recording in predictions and augmentation logs. The runner checks
unchanged feature/timing/collation functions and architecture, matched initial
weights and full/mask augmentation decisions, exact reproduction of the V4-only
reference metrics, and preservation of prior artifact hashes.

## Completed full preparation

Both extractions are complete: **28,062 V7 frames and 24,081 V8 frames**. The frozen
gate compares every integer PTS/rational time base against the hashed source frame
inventory; all match exactly. The new artifacts use fresh full-run/preparation paths,
preserving earlier sample poses. Extraction remains resumable under the existing
exclusive-writer and immutable-prefix checks.

V7 has 238 missing, 10,069 unreliable and 17,755 fully valid body detections. V8 has
1,301 missing, 12,338 unreliable and 10,442 fully valid body detections. None of the
annotation-eligible event intervals contains a wholly missing pose frame; many
contain unreliable joints. Missing detections elsewhere are retained as missing.
The preparation identifies 25,743 V7 frames and 21,742 V8 frames outside plausible
annotation bounds as unknown, not negative examples.

| Class | V4 retained | V7 retained | V8 retained | Expanded training | V5 development |
|---|---:|---:|---:|---:|---:|
| jab | 117 | 8 | 16 | 141 | 226 |
| cross | 39 | 1 | 21 | 61 | 138 |
| lead_hook | 140 | 20 | 54 | 214 | 68 |
| rear_hook | 73 | 1 | 0 | 74 | 39 |
| lead_uppercut | 71 | 36 | 21 | 128 | 39 |
| rear_uppercut | 87 | 0 | 8 | 95 | 73 |
| total | 527 | 66 | 120 | 713 | 583 |

V7 excludes 124 annotation rows, then rejects 5 of 71 eligible intervals on pose
quality (2 lead hooks, 3 lead uppercuts). V8 excludes 67 annotation rows, then
rejects 12 of 132 on pose quality (1 jab, 10 lead hooks, 1 lead uppercut). No flags,
labels, bounds or unreliable coordinates are corrected. The training increase is
186 examples, about 35.3%, concentrated in lead hooks and lead uppercuts. Rear-hook
diversity barely expands; the extra source group supplies only one retained example.

Frame-weighted invalid **left/right elbow mask percentages**, among annotation-eligible
intervals before pose filtering:

| Class | V4 | V5 | V7 | V8 |
|---|---|---|---|---|
| lead_uppercut | 0.4 / 58.9 | 1.1 / 0.0 | 24.5 / 15.7 | 7.7 / 26.3 |
| rear_hook | 0.0 / 99.1 | 76.9 / 0.0 | 0.0 / 13.3 (one event) | no source labels |
| rear_uppercut | 0.0 / 96.1 | 77.1 / 0.3 | no eligible events | 13.4 / 10.1 |

V7/V8 provide less consistently one-sided elbow failure than V4/V5, but turning
and self-occlusion remain visible in overlays. Invalid masks measure estimator
reliability, not independently verified joint accuracy. In a separate diagnostic
over **all 20 excluded V7 rear uppercuts**, all pass the unchanged mask policy;
left/right elbow invalidity is 35.0%/11.2%. Those 20 remain excluded from training.
Similarly, all 25 V8 rear-uppercut intervals pass diagnostically, but only eight
are annotation-eligible. Better poses cannot resolve overlapping labels.

`reports/boxingvi/v7-v8-full-v1/dataset-summary.json` retains per-class joint masks
and exact excluded rows; its Markdown companion provides the V4/V5 comparison.
`training-gate.json` freezes source hashes, review evidence, candidate selection,
identities, grouping and remaining uncertainties. Full-run skeleton contact sheets
and individual overlays remain beside each recording's pose artifacts.

## Three-seed experiment results

The real-data smoke study passed (one seed, two epochs, at most two examples per
available class per recording). The complete comparison then ran both full-feature
and mask-only TCNs for both arms and all three fixed seeds: 12 runs, 40 epochs each.
Every V4-only control reproduced the accepted mirror-only selected epochs, training
metrics, V5 metrics and final-epoch metrics **exactly**. Initial model weights match
between paired arms, full/mask shuffle and augmentation decisions match within each
arm, and all 367 prior mirror-study artifact hashes remain unchanged.

| Input / training | V5 macro-F1, mean ± sample SD | V5 accuracy, mean |
|---|---:|---:|
| Full poses, V4 only | 0.4912 ± 0.0188 | 71.87% |
| Full poses, V4/V7/V8 | 0.5091 ± 0.0200 | 71.07% |
| Mask only, V4 only | 0.3546 ± 0.0187 | 56.89% |
| Mask only, V4/V7/V8 | 0.3636 ± 0.0058 | 56.72% |

Full-feature paired macro-F1 changes are **−0.0252, +0.0318, +0.0470** (mean
+0.0179, sample SD 0.0380). Two seeds improve; one worsens. Mask-only paired gain
is +0.0090, so the mean full-feature gain beyond the mask-only gain is only
+0.0089, with substantial seed variation. This is not evidence that missingness
shortcuts have disappeared, or that movement alone caused the aggregate gain.
The full model remains better than the mask control, but both can exploit
class-dependent visibility, source conditions and clip-boundary/length patterns.

Mean development recall, using the same 583 V5 examples:

| Class | V4 only | Expanded | Change (percentage points) |
|---|---:|---:|---:|
| jab | 91.45% | 92.18% | +0.74 |
| cross | 96.38% | 91.06% | −5.31 |
| lead_hook | 80.88% | 87.25% | +6.37 |
| rear_hook | 21.37% | 23.93% | +2.56 |
| lead_uppercut | 0.00% | 14.53% | +14.53 |
| rear_uppercut | 21.92% | 8.22% | −13.70 |

Lead-uppercut recall is **2.56%, 5.13%, 35.90%** across expanded seeds, so the
improvement is inconsistent. Rear-hook recall is 15.38%, 38.46%, 17.95%; rear-uppercut
recall is 9.59%, 5.48%, 9.59%. These rear classes remain poorly recognized.

The expanded confusion matrices expose a tradeoff rather than balanced success.
On average per seed, 45 of 73 rear uppercuts are predicted as lead uppercuts,
19 of 39 rear hooks as lead uppercuts, and 26.7 of 39 lead uppercuts as lead hooks.
The extra uppercut examples change the decision boundaries but do not establish
reliable six-class separation. No confusion is used to relabel source annotations.

Overfitting persists. Mean final-epoch training/development F1 is **0.8299/0.4795**
for expanded training, versus **0.8044/0.4248** for V4-only. The first expanded seed
peaks at epoch 19 (V5 F1 0.4860), then falls to 0.4210 as training F1 reaches 0.8546.
The other expanded runs select epochs 26 and 38. Training F1 on different training
populations is not directly equivalent; V5 is the shared comparison. Its repeated
use for selecting epochs makes it an optimistic development measure, not a test.

Both arms retain 40 epochs, batch size 32 and the existing optimizer. This gives
680 optimization steps per V4-only model and 920 per expanded model. Added data,
viewpoints, class mixture, native frame rates and more optimization steps are all
part of this intervention; their individual effects are not isolated.

The frozen readiness gate passed for **provisional positive-clip experiments**,
not gold labels or deployment. Keep **V4-only mirror augmentation as the reference**;
retain the expanded arm and its 186 new examples as an experimental dataset rather
than promoting this result as a six-class improvement. V5 remains development and
V9 remains completely reserved. No architecture, anticipation, early-recognition,
event-detection or Level 2 feature is introduced.

Before further expansion or training, prioritize a small review of V7 rows 53/54,
V8 rows 65/66, and representative late uppercuts (V7 row 194; V8 rows 149/197)
using the local exact-frame player. The overlap exclusions can only be revised in
a separately versioned decision supported by review or author evidence. Do not
recover all V7 rear uppercuts merely because their poses pass the quality filter.
The rear-hook source-diversity gap also remains unresolved.

Local artifacts:

- `reports/boxingvi/level1-expansion-smoke-v1/`: real-data smoke results.
- `reports/boxingvi/level1-expansion-v1/summary.md` and `summary.json`: paired results,
  per-class recalls, mean confusion matrices and seed variation.
- Each seed/arm/model directory: full confusion matrix, predictions with recording
  and source row, probabilities, curves, configuration, source snapshot and checkpoints.
- `reports/boxingvi/v7-v8-expansion-gate-v2/index.html`: exact-frame source review;
  the earlier v1 evidence and all earlier experiments are preserved.

## Reproduction

All commands use existing local media, annotations and pose weights. They do not
download inputs. Use a fresh report directory when rerunning a completed study.

```bash
uv run python -m boxing_motion_ai.data.expansion_review \
  --output reports/boxingvi/v7-v8-expansion-gate-v2

uv run boxing-motion-ai extract-poses \
  --video data/raw/boxingvi/RGB_videos/V7.mkv \
  --workbook data/raw/boxingvi/Annotation_files/V7.xlsx \
  --recording configs/v7_recording.yaml --config configs/v7_pose_full.yaml \
  --model weights/pose_landmarker_full.task \
  --output data/processed/boxingvi/V7/pose-full-image-v1 --resume
# Repeat for V8 using its media, workbook, recording and full-pose configuration.

uv run boxing-motion-ai prepare-recording \
  --pose-run data/processed/boxingvi/V7/pose-full-image-v1 \
  --workbook data/raw/boxingvi/Annotation_files/V7.xlsx \
  --output data/processed/boxingvi/V7/candidates-full-v1
uv run boxing-motion-ai assess-pose-quality \
  --dataset data/processed/boxingvi/V7/candidates-full-v1 \
  --output reports/boxingvi/v7-v8-full-v1/quality/V7
# Repeat preparation and quality assessment for V8.

uv run python -m boxing_motion_ai.training.expansion_study \
  --config configs/level1_expansion_study.yaml \
  --freeze-gate configs/v7_v8_alignment_review.yaml \
  --output reports/boxingvi/v7-v8-full-v1/training-gate.json
uv run python -m boxing_motion_ai.training.expansion_study \
  --config configs/level1_expansion_study.yaml \
  --output reports/boxingvi/level1-expansion-smoke-v1 --smoke
uv run python -m boxing_motion_ai.training.expansion_study \
  --config configs/level1_expansion_study.yaml \
  --output reports/boxingvi/level1-expansion-v1

uv run pytest
uv run ruff check .
```

The native MediaPipe runtime required graphics access beyond the sandbox even
with the CPU delegate. The initial sandbox crash was recovered using the existing
exclusive writer lock and resume mechanism; original files were not changed.
Datasets, poses, review media, reports, environments and checkpoints remain ignored.

## Final verification

- `uv run pytest`: **154 passed**, including synthetic expansion/training, exact-PTS
  hypotheses, nested/endpoint exclusions, partial class coverage, provenance,
  changed-evidence rejection, report writing and V9 rejection before file access.
- `uv run ruff check .`: passed. `git diff --check`: passed. CLI help works.
- V7/V8 original video and workbook hashes still match the source evidence.
- All 367 mirror-reference files are unchanged; every matched V4-only metric
  reproduces exactly. Earlier studies and sample poses remain preserved.
- Git tracks no raw/processed data, generated reports, local environment or weights.
- V9 media, workbook, poses and evaluation artifacts were not opened or used.

Stopped after this initial expanded Level 1 comparison for review.
