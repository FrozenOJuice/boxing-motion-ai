# Cross-recording label, boundary and pose diagnosis

Prioritize a small independently reviewed label/phase contract before another model
experiment. There is concrete evidence of mixed-action windows and a possible
source-local lead/rear convention change. Pose visibility also limits recognition,
but it does not explain clear, well-tracked hooks being confused across recordings.
Keep pose-only XY + validity-gated velocity + mirror as the provisional reference.
No training, architecture change, correction, candidate restoration or new exclusion
was performed. V9 was not opened, decoded, hashed or evaluated.

## Scope and evidence limits

The diagnostic includes **79 cases**: 72 approved candidates (12 per recording) and
7 existing holds, across V4/V5/V6/V7/V8/V10. It prioritizes rear hooks and uppercuts:
5 jab, 5 cross, 12 lead hook, 13 rear hook, 22 lead uppercut and 22 rear uppercut.
V6 has no jab/cross annotations; V7 has no approved rear uppercuts; V8 has no rear
hooks. Those gaps are not fabricated examples.

Selection is deterministic within class, distributed through each recording, with
source-label agreement/disagreement alternation where available. It is deliberately
error-enriched, **not a random accuracy or annotation-error prevalence estimate**.
The frozen multi-recording XY+velocity TCN seed 20261021 supplies predictions:
non-V5 cases use their source-group-held-out fold; V5 uses the combined-training
model selected on development data. Predictions are comparison hypotheses, never
annotation truth. The preserved V4-only reference was not replaced or retrained.
V7/V8 stay grouped; V1/V6 guard remains; V1 is not loaded. Boxer/session identities
and source-camera reflection remain unverified, not inferred from resemblance.

All 79 raw/overlay sheets were visually inspected by the assistant. Each shows up
to eight native decoded frames: interval samples plus at most 0.2 seconds of context
on each side, bounded by scene/segment IDs and extraction continuity. Exact rational
PTS drive decoding and frame identity; nominal FPS is never used for a correction.
For ten disputed/comparison cases, additional pages render **every native frame**
around the interval; key dense pages were inspected to check phase and side claims.
This is a visual diagnostic, not a calibrated framewise anatomical benchmark.

Sheets hide expected labels and predictions; the gallery reveals them on request.
Original source HUDs remain visible and can disclose exercise labels, so review is
not fully blinded. Orange overlays display invalid estimates for diagnosis only;
those coordinates remain invalid in model inputs. L/R colors are estimator channels,
not certified anatomical identity or stance.

The [authors' README](https://github.com/Bikudebug/BoxingVI) and
[paper, dataset-creation section](https://arxiv.org/html/2511.16524v1#S3)
describe classes and manual start/end annotations. The inspected material does not
supply an operational stance-switch/mirroring rule, or reproducible biomechanical
onset/peak/end instructions. This absence in inspected sources is not proof that no
additional author guidance exists. Do not infer an undocumented global frame offset
or label swap from these sheets.

## Independent evidence classifications

Separate axes avoid treating a questionable side as proof of a wrong punch family.
`consistent` means visible evidence supports that axis; `ambiguous` means insufficient
or conflicting evidence. `visibly_incorrect` on the boundary axis means a visibly
mixed action/recovery window **for an isolated single-punch evaluation unit**. It does
not establish the authors' intended boundary convention or a replacement interval.

| Recording | Cases | Family consistent / ambiguous | Boundary consistent / ambiguous / mixed | Pose aligned / masked failure / ambiguous |
|---|---:|---:|---:|---:|
| V4 | 12 | 5 / 7 | 6 / 6 / 0 | 1 / 11 / 0 |
| V5 | 12 | 11 / 1 | 9 / 3 / 0 | 3 / 8 / 1 |
| V6 | 14 | 9 / 5 | 7 / 5 / 2 | 7 / 5 / 2 |
| V7 | 15 | 10 / 5 | 6 / 4 / 5 | 7 / 8 / 0 |
| V8 | 12 | 10 / 2 | 7 / 2 / 3 | 3 / 9 / 0 |
| V10 | 14 | 14 / 0 | 14 / 0 / 0 | 8 / 6 / 0 |
| Total | 79 | 59 / 20 | 49 / 20 / 10 | 29 / 47 / 3 |

Lead/rear certification remains ambiguous in all 79 because stance, anatomical
channel identity and reflection are not independently certified. **That does not
mean all 79 side labels are wrong.** No class label was declared visibly incorrect
solely from a prediction, an estimated side or a source exercise banner. The
versioned assistant observations are in `configs/level1_label_pose_review.yaml`;
human verification is explicitly pending.

## 1. Annotation semantics and action phases: first priority

| Evidence | Interpretation and limit |
|---|---|
| V4 jab 144 / cross 51 / lead hook 343 versus V5 jab 262 / cross 387 | V4 jab and lead hook use the displayed R chain; its cross uses L. V5's local straight punches use the opposite side pattern. Compatible with different stances or channel uncertainty; not proof of globally reversed labels. Never equate lead with anatomical left across sources. |
| V10 early rear uppercuts 44/47 versus late rear uppercut 146 and lead uppercuts 147/149/153 | Early RU and late LU use the same displayed R chain; late RU uses L. The adjacent late pair is mutually distinguishable, but its source naming reverses the earlier arm association. A stance/exercise change could explain it; this needs stance confirmation before claiming erroneous labels. Held LU 9 resembles early RU execution and remains held. |
| V6 retained RU 613, raw 41090–41104; held RU 685, raw 44325–44338 | Native frames show long L straight extension/retraction at the beginning and a different R compact rising phase later. These are mixed-phase clips rather than clean, independently certified RU units. The uppercut-like phase exists; no replacement timing is invented. |
| V7 LU 136, 168, 191 | Strong R straight/overhand-like extension followed by recoil/high guard; a clean lead-uppercut phase is not independently clear. Row 136 dense frames support the concern. Row 179 also contains consecutive actions. No automatic relabeling to cross/overhand. |
| V8 RU 82, 81, 173 | Opposite-arm straight or curved action/recovery followed by a compact R rise. Row 173 contains a visible L sweep before the R rising phase within the source window. Source phase inclusion is not uniform enough for a clean isolated-punch claim. |
| V4 LU 497 | Near arm rises beside the head into cover/guard; a block rather than an outgoing uppercut is plausible. Source combination text includes a block, but text is not the decision criterion. Bag occlusion and sparse motion keep the class ambiguous. |
| V5 LH 432; LU 196; RU 514 | Prior extension/rotation or recovery changes which action phase dominates the window. V5 itself is provisional, not clean ground truth against which all other sources must be corrected. |
| V7 sole approved RH 52 and held RU 5/54/132 | Body dip/retraction and alternating actions leave family/phase uncertain. The sole RH cannot support a broad class-generalization conclusion. All held RU examples remain excluded. |

Readable physical scene text offers no obvious evidence of a whole-frame flip in
V4/V5, but overlays can be added after reflection and readable text is not a camera
provenance guarantee. Do not apply a global anatomical-side or timing correction.
The strongest finding is **local uncertainty and inconsistent phase composition**,
not a verified dataset-wide six-class mapping error.

## 2. Pose reliability: important, but not a complete explanation

Confidence-audit counts cover the shoulder/elbow/wrist chains for both arms within
these 79 intervals, including held diagnostic cases. They count frame-joint slots,
not unique frames, and overlapping intervals can repeat slots.

| Recording | Valid arm slots / total | Visibility below 0.5 | Presence below 0.5 |
|---|---:|---:|---:|
| V4 | 446 / 600 | 154 | 0 |
| V5 | 665 / 768 | 103 | 0 |
| V6 | 1208 / 1284 | 76 | 0 |
| V7 | 1181 / 1302 | 121 | 0 |
| V8 | 816 / 924 | 108 | 0 |
| V10 | 1091 / 1134 | 43 | 0 |

All 605 invalid sampled arm slots coincide with low visibility, rather than low
presence or nonfinite scores. This describes estimator output, not a calibrated
occlusion detector. Raw imagery often supports occlusion: V4's bag/head/body hide
the far elbow; V8 turns hide the active arm; V10's raised elbow/wrist lose validity
near uppercut peaks. V5 rear-hook/uppercut examples often retain a well-tracked
punching R arm while the off-hand L elbow is masked. Do not interpret an all-joint
quality failure as complete loss of the punch trajectory.

No independently confirmed **high-confidence anatomically wrong** landmark was
established in this review. Confident projections behind an occluder cannot be
certified from the raw 2D view. Scores at least 0.8 on both channels are reported
only as a diagnostic bin, not a changed quality threshold or proof of correctness.
A suspected occluded localization error stays ambiguous; no coordinates are filled.

Of 22 approved cases with visually consistent family/boundary and aligned projected
arms, 18 still disagree with their source six-class label. These are biased diagnostic
counts, not measured accuracy; side truth is still provisional. Examples V6 RH 46/87
and V10 RH 142 are predicted as lead hooks despite visible, aligned curved punching
arms. V5 LU 58/381 are predicted as lead hooks with clear rising bent-arm motions.
This argues against blaming every failure on missing landmarks or poor labels.

## 3. Other causes after the label/phase check

In the 72 approved diagnostic cases, 22 predictions agree with the source label,
20 disagree only within straight/hook/uppercut family, and 30 disagree across
families. This diagnostic regrouping does not change the six training classes.
There is no basis to assume a single lead/rear label swap will solve recognition.

V8 cross 51 points toward the camera: substantial real motion has a small 2D
projection even with reasonable tracking. V4 bag work, V6 shadowboxing, V7/V8
combination drills and V10 fitness execution differ in stance, target depth, arm
path and phase. Torso translation/scale normalization does not remove these
viewpoint/execution differences. Neither the negative RGB pilot nor clear-pose
failures prove that a different pose estimator alone will fix them.

Priority order:

1. **Verify a small label/phase evaluation contract first.** Review the six disputes
   below, then certify the 12 anchors. Record observed arm, stance, family and
   boundary evidence separately; allow unknown. Ask original authors for their
   operational side and boundary rules if needed. No bulk correction follows from
   this diagnosis.
2. **Measure active-arm localization on those verified examples.** Distinguish
   occlusion/low visibility from visible-but-mislocalized joints, especially at punch
   peak. Only then decide whether a targeted alternative pose estimator comparison
   is justified. Do not lower validity thresholds to manufacture coverage.
3. **Reassess view/phase transfer using frozen predictions.** If clear reviewed
   labels and reliable poses still fail, viewpoint/execution and phase mismatch
   deserve attention before another feature/model trial. More RGB training is not
   justified by this evidence. No new experiment is implemented here.

## Small human-review queue

Start with the [18-case human review queue](../reports/boxingvi/level1-label-pose-assessment-v2/human-review.html),
which links the native-frame pages. Open the local [blind gallery](../reports/boxingvi/level1-label-pose-review-v1/index.html)
and [case-by-case assessments](../reports/boxingvi/level1-label-pose-assessment-v2/REVIEW.md).
Inspect raw frames first, then overlay, then reveal source label and frozen output.
For disputed motion, use consecutive pages from
`reports/boxingvi/level1-label-pose-dense-v1/` (page order/frame IDs in `dense.json`).
The initial human request is **18 existing cases**, not all 79:

- Six priority disputes: **V4:497, V6:613, V7:136, V8:82, V10:146, V10:147**.
  Resolve guard-versus-punch, preceding-action inclusion, and stance/side convention.
  Compare V10:146/147 against V10:44 (already in the anchor set).
- Twelve candidate anchors, two per source class: jab **V4:144 / V7:8**; cross
  **V8:51 / V10:71**; LH **V4:343 / V6:7**; RH **V6:46 / V10:142**;
  LU **V6:264 / V8:159**; RU **V6:266 / V10:44**.

Anchors have clearer families/phases, not automatically certified six-class labels.
Some have masked off-hand joints or uncertain stance. A qualified reviewer should
record family, observed anatomical arm (or unknown), stance/lead-rear evidence,
whether the intended punch phase is inside the interval, and visible localization
errors. Accept, hold or reject the proposed anchor explicitly; do not silently replace
source values. The generated `human-review.json` contains blank pending fields.
**Zero cases are claimed human-verified.**

After certification this is a small **diagnostic evaluation subset**, not a fresh,
unbiased final test: cases were chosen after seeing errors and results must remain
tied to their frozen held-out-group models. V5 remains development-only and is not
in this candidate subset; V7/V8 remain one group. No boxer-independent claim.

## Reproduce and verify

Existing optional pose/train dependencies suffice; no new dependency or weight
was downloaded. Run from the repository root, using fresh output directories:

```sh
uv run --extra pose --extra train python -m boxing_motion_ai.training.forensic_review \
  --output reports/boxingvi/level1-label-pose-review-replay
uv run --extra pose --extra train python -m boxing_motion_ai.training.forensic_review \
  --review reports/boxingvi/level1-label-pose-review-v1/cases.json \
  --dense-cases V4:497 V6:613 V6:685 V7:136 V8:82 V8:173 V10:44 V10:146 V10:147 V10:9 \
  --output reports/boxingvi/level1-label-pose-dense-replay
uv run --extra pose --extra train python -m boxing_motion_ai.training.forensic_report \
  --cases reports/boxingvi/level1-label-pose-review-v1/cases.json \
  --assessments configs/level1_label_pose_review.yaml \
  --dense reports/boxingvi/level1-label-pose-dense-v1/dense.json \
  --output reports/boxingvi/level1-label-pose-assessment-replay
uv run --extra pose --extra train pytest
uv run ruff check .
```

The normal case manifest is bound to the versioned assessment by SHA256; it includes
exact PTS, raw intervals, literal source class, existing flags/holds, original fold,
unknown identities and frozen prediction hashes. Replaying the selection reproduced
all 72 eligible IDs exactly; seven holds remain diagnostic-only. The source video
hash is checked before decoding. Preservation checks verified **4,896 existing files**,
including original media/workbooks, prepared candidates/poses and previous study
artifacts, without changes. Generated reports, images, data, caches and weights
remain ignored. Only code, synthetic tests, review decisions and this note are
versioned. The synthetic tests do not access real datasets or download models.

Validation: **273 tests passed** with pose/train extras enabled; `uv run ruff check .`
passed. These checks include synthetic selection, exact-PTS context boundaries,
missing scores, immutable pose arrays, reserved-recording rejection, portable report
links, pending human-review fields and rejection of automatic corrections.
