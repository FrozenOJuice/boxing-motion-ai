# Full-body conditional punch classification V1

## Outcome and scope

Starting commit: `1ba39da`. This milestone **trained and evaluated** the existing small temporal
classifier with the 209-channel full-body representation. It did **not** improve V5 recognition
under the frozen comparison: development macro-F1 was **66.17%**, versus **70.86%** for the
matched 65-channel model. All three paired seeds favored 65 channels. Training fit increased,
so a larger feature inventory is not, by itself, evidence of better transfer.

A separately versioned person-matte/targeted-OCR preprocessing experiment also **failed** its
predeclared retention and identifier checks. Consequently **no new R3D-18 training-clip feature
cache, video classifier or fusion classifier was fitted**. Existing R3D features and weights
remain unchanged. There was no repeated tuning against V5 and no production promotion.

This is six-class classification **conditional on supplied known-punch clips**. It neither
finds punches nor establishes their boundaries, occurrence, count or end-to-end accuracy.
All primary class labels and clip boundaries remain **provisional source annotations**, not
independently verified ground truth. These scores measure agreement with those labels.

## Evidence, split and immutable selection

The new user authorization permits the **existing 812 eligible weak clips** for this experiment;
it does not authorize an expanded localization dataset or override historical source holds.
`configs/fullbody_classification_v1.json` records that distinction. The older foundation pin
and its blocked-*expanded*-training status were not edited.

The input pin remains `configs/video_foundation_snapshot_v1.json`: dataset manifest SHA-256
`e54b8aecd37bfbddfcd52c7eeb6a2c9cd37891f82a6ce627883572a050393f3d`, exclusion overlay
`134ea69d51b7e827011207a621080f3965103f3931e3f998061a5e6a5a6fef98`.

| Recording / role | Jab | Cross | Lead hook | Rear hook | Lead uppercut | Rear uppercut | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 training | 117 | 39 | 140 | 70 | 70 | 84 | 520 |
| V6 training | 0 | 0 | 62 | 32 | 121 | 77 | 292 |
| V5 development | 226 | 138 | 68 | 39 | 39 | 73 | 583 |

Training membership is the pinned 815-clip curated inventory **minus V6 rows 7, 315 and 316**
that intersect excluded manual evidence. The 18 excluded non-blue events, three conflicted
strong V6 references, source annotations and every review revision remain intact. None becomes
a new localization/background label. No additional human labeling was requested.

V4 and V6 source groups are training-only; the distinct V5 recording/source group is development
only, with its unchanged 583-clip legacy membership and existing holds. There is no random
adjacent-frame train/development split, V5 training, or access to V7/V8/V10 evaluation tensors
or V9 data. V5 influenced historical classifier development and is **not an independent test**.
Its provisional label/boundary errors remain possible, including known disagreement between
some source types and saved human events. Labels were not silently corrected or selected by
current model agreement. The eight sparse V5 localization positives are not an exhaustive
classification or detection benchmark and were not used to tune these models.

## Controlled training contract

All views use exactly the same eligible clips, labels, source sampler, sampled indices and
reflection decisions. Models never receive recording IDs, row IDs or absolute source times.
Raw native timestamps determine derivatives; sequences are neither held to 10 Hz nor resampled.
V4 clips contain 3–19 frames, V6 9–28, and V5 4–22. Duration is an available shortcut; it is not
claimed to have been eliminated.

- Existing `CausalTCN`: hidden 32, three causal kernel-3 blocks with dilations 1/2/4, dropout
  0.1, masked mean pooling over all real clip frames, six-class head. Receptive field: 15 native
  frames; complete-clip pooling is offline conditional classification.
- Views: existing **65-channel** core XY/visibility/presence/validity/velocity;
  **209-channel** `full-body-native-2d-1.1`; and **mask209**, retaining only explicit validity
  columns as a shortcut control. Missing joints remain masked sentinels, never interpolated.
- 40 epochs; batch 32; AdamW learning rate 0.001, weight decay 0.001; gradient clipping 1;
  CPU two threads; seeds **20261021, 20261022, 20261023**. No scheduler or architecture search.
- Reuse existing class-uniform / square-root source-group and recording sampler, replacement
  812 draws/epoch; unit cross entropy avoids a second class weighting. No fitted scaling.
- Physical reflection probability 0.5: swap anatomical sides and reflect coordinates on copies,
  then re-encode 209 features. Tactical lead/rear class labels do not change. The original
  65-channel mirror function remains the matched core view. No visibility augmentation.
- Shared 41→65 initialization uses the existing zero-initialized velocity expansion. The same
  block/head weights and semantically corresponding columns initialize 209; extra weights start
  at zero. Width changes parameter counts: **11,814** versus **16,422**. This is not a parameter-
  count-matched architecture ablation, but retains the requested compact architecture/budget.
- Both views reset derivative evidence at the clip's first frame. Every eligible 209 clip
  exactly reproduces the pinned full-recording cache after that explicit first-frame reset.
  Shared 65-feature maximum drift is **3.82e-6** in training and V5; zero clips exceed 1e-4.
  Thus 209 does not receive additional pre-clip motion context.
- **Final epoch 40 only**, no best-on-V5 selection. All nine runs, curves, input hashes,
  predictions and checkpoints were frozen before `development_views()` was called.
  Every run was fitted twice with exact state/curve/output/sampling agreement; reloading its
  checkpoint also reproduced training outputs exactly. Same-device/library reproducibility
  was verified; arbitrary cross-device equality is not promised.

Run files live outside Git in `reports/boxingvi/fullbody-classification-v1/`. Committed
`configs/fullbody_classification_results_v1.json` pins all nine checkpoints and metric/prediction
files plus the pre-development freeze. Scores are uncalibrated conditional softmax scores;
they are not probabilities that a punch occurred. Checkpoints are new `final.pt` files, never
replacements for production or historical `best.pt` artifacts.

## Actual recognition results

Percentages below are means ± population SD across the three fixed seeds; these are descriptive
seed variations, not independent-recording confidence intervals.

| View | Training macro-F1 | V5 macro-F1 | V5 accuracy |
| --- | ---: | ---: | ---: |
| Core 65 | 91.54 ± 1.47 | **70.86 ± 4.77** | 78.73 |
| Full body 209 | **92.71 ± 0.85** | 66.17 ± 2.59 | 74.21 |
| Validity masks 209 | 54.88 ± 1.79 | 34.93 ± 1.99 | 55.17 |

| Seed | Core V5 macro-F1 | Full-body V5 macro-F1 | Full-body minus core |
| --- | ---: | ---: | ---: |
| 20261021 | 71.98 | 66.67 | -5.31 points |
| 20261022 | 64.54 | 62.78 | -1.76 points |
| 20261023 | 76.06 | 69.05 | -7.00 points |

V4 per-recording training macro-F1 means: core **90.77%**, body **90.32%**. V6 supports only
four types: its six-class macro-F1 means are **60.79% / 64.52%**; F1 averaged only over those
four supported types is **91.18% / 96.78%**. Absent jab/cross classes are not evidence of V6
recognition. Per-recording matrices and supports are retained for every run.

For historical context, replaying the **unchanged pinned production-reference 65-channel
checkpoint** (`bd314c2f608a0f73b5b13bca912510a513e7491c6062a8c73446404f146bdbfa`, epoch 32)
on the same V5 pool yielded **65.97% macro-F1 / 80.10% accuracy**. It used V4-only training
and historical V5 epoch selection. The new 65 mean is higher in macro-F1 but lower in accuracy;
this is **not** an isolated full-body benefit or a fair causal estimate of a data-expansion gain.
The matched fresh 65 runs above are the primary comparison. No seed was selected for deployment.

### Per-class V5 recall

| Type (provisional support) | Core 65 | Body 209 | Mask209 |
| --- | ---: | ---: | ---: |
| Jab (226) | 77.73 ± 11.32 | 64.01 ± 3.34 | 87.61 ± 8.14 |
| Cross (138) | 86.47 ± 3.47 | 86.96 ± 9.85 | 59.18 ± 14.66 |
| Lead hook (68) | 78.92 ± 5.00 | 86.76 ± 4.80 | 37.75 ± 0.69 |
| Rear hook (39) | 29.91 ± 17.43 | **11.97 ± 5.27** | 9.40 ± 5.27 |
| Lead uppercut (39) | 84.62 ± 7.25 | 87.18 ± 2.09 | 0.00 ± 0.00 |
| Rear uppercut (73) | 89.95 ± 5.52 | 96.35 ± 0.65 | 17.35 ± 16.60 |

### Confusion matrices

Fixed first seed **20261021**, rather than selecting a favorable seed or counting three
predictions per clip as independent examples. Rows = source label; columns = prediction.
Order: **J, C, LH, RH, LU, RU**. All three seeds' full metrics are preserved in the report.

```text
Core65                      Body209                     Mask209
202  10  11   0   3   0       150  12  51   7   6   0     212   5   5   0   1   3
  1 126   5   0   2   4         0 101   0  31   0   6       2  71   0   7  10  48
  5   0  55   1   4   3         0   0  63   2   2   1      40   0  25   0   1   2
  0   0   1   5   2  31         0   0   1   5   0  33       0  14   1   1   8  15
  5   0   3   0  31   0         0   0   3   0  35   1      30   0   9   0   0   0
  1   0   0   0   4  68         0   0   0   0   2  71       2  31   0   0  11  29
```

The largest full-body failures are jab→lead-hook (51 first-seed clips), cross→rear-hook (31)
and persistent rear-hook→rear-uppercut (33). It fixes 23 first-seed core errors but introduces
85 errors where core matched the provisional label. Examples with exact half-open native
bounds: V5 row 4 `[71,77)` jab→lead-hook; row 150 `[2988,2994)` cross→rear-uppercut;
row 172 `[3558,3569)` cross→rear-hook. These are reference disagreements, not newly verified
human corrections. Row-level scores/bounds make them auditable.

Observation patterns differ strongly across recordings: core left/right wrist validity is
V4 **99.2/53.3%**, V6 **94.9/88.9%**, V5 **66.3/99.0%**; elbow validity shifts similarly.
This is thresholded observation availability, **not landmark accuracy**. The substantial mask-only
result and absence of V6 jabs/crosses expose source/class/visibility confounds. Full-body feet
are mostly available, but availability did not yield better overall development recognition.
The comparison does not isolate which added geometry/velocity columns cause the regression;
there was no post-V5 ablation search or arbitrary rescaling to improve these scores.

## RGB preprocessing investigation and gate

Original `source-fixed-identifiers-fullbody-1.1` removes broad fixed HUD/source rectangles and
a whole-context chest-text union. The union covers the moving glove's path even when no text
occupies that location in the current frame. The V6 footer overlays the feet. Prior foundation
measurements found wrist-center mask intersections of **42.9% / 72.8% / 88.8%** in V4-bg-1 /
V6-motion-101 / V6-motion-105 and **100%** feet intersections in V6. Static masks also leave
source background cues. These are geometric mask intersections, not visual recovery accuracy.

New experimental policy `person-matte-targeted-ocr-1.0`, separately configured in
`configs/person_preserving_rgb_v1.json`:

1. Use the already-local MediaPipe full task's **person segmentation output**, independently
   per native frame. Threshold 0.5 and a fixed five-source-pixel dilation; background becomes
   constant gray. New pose landmarks are not used or substituted into original cached poses.
2. Run local Tesseract PSM11 and redact nonempty word rectangles with two-pixel margins,
   including text overlapping the person. Do not protect an overlay simply because it covers
   a foot, and do not inpaint or fabricate the underlying body.
3. Before outcomes, require ≥95% retained valid-joint neighbourhood pixels, ≥80% retained
   adjacent-frame motion in those neighbourhoods, and **no visibly retained source text/logos
   or identifiers**. Geometric retention is only a proxy, never proof of accurate landmarks,
   useful features or source invariance. OCR returning no text is not proof of no identifiers.

Sample: **96 real native frames**, eight consecutive frames at each of four uniform positions
in each of the same three authorized V4/V6 contexts. Exact PTS/frame mappings and source hashes
were checked before decoding. All 96 returned a segmentation mask; that alone is not success.
Three comparison sheets show original / old preprocessing / new preprocessing; native source
and processed pairs were inspected, including fast hands and foot overlays. No V5 pixels were
used to adjust this preprocessing.

| Case (32 frames each) | Retained joint-neighbourhood pixels | Retained original adjacent motion | Segmentation ms/frame | OCR ms/frame |
| --- | ---: | ---: | ---: | ---: |
| V4-bg-1 | 89.23% | 85.60% | 47 | 937 |
| V6-motion-101 | 30.92% | 5.99% | 51 | 667 |
| V6-motion-105 | 30.61% | 15.25% | 31 | 505 |

Neighbourhoods are source-coordinate discs around valid in-frame wrists, elbows, ankles, heels
and foot tips (radius max(3px, 1.2% of shorter image dimension)). Pixel retention is the summed
surviving-disc area divided by original disc area. Motion retention sums original absolute
adjacent-RGB difference within those discs and divides surviving motion (pixels retained in
**both** frames) by original motion. It excludes jumps between the four sample windows. It
measures retained original evidence, not motion created by moving matte/text-mask edges. No
claim of a complete body-visibility benchmark or learned source-invariance test is made.

**Failed visual/integrity checks:** V4 frame 17150 retains shorts lettering and recognizable
clothing/glove/foot details while OCR creates small false holes. V6 frame 588 loses the visible
lower body. V6 frame 30550 has a large rectangular glove/torso hole while **HOOK and number 5
remain visibly over the shoes**, directly exposing source annotation information. OCR has both
false-positive rectangles and false negatives: frame 588 falsely recognizes `Je` across native
rectangle `[269,547,1398,988]`, and frame 30550 recognizes `oR` across `[793,296,1088,424]`; foreground matting cannot remove text composited
on the person by itself. Matte-edge background remnants also remain. Narrower rectangles cannot
be assumed safe without a new check. No attempt was made to tune OCR/retention settings until
this failed pipeline passed.

Costs above are measured component means, not production latency: complete case loops took
92.83 / 43.66 / 31.06 seconds, including masks, rendering and file writes, excluding the earlier
bulk source decode/hash. OCR and large-image mask construction dominate; segmentation is not
the only cost. Peak memory was not instrumented. First sandbox attempt aborted because macOS
MediaPipe graphics services were unavailable; the first host attempt exposed its H×W×1 mask
shape. The corrected, separately retained host run squeezed only that singleton channel. No
source/cache was deleted to recover. Environment: Python 3.11.17, torch 2.14.1, NumPy 2.4.6, MediaPipe 0.10.35 and
Pillow 13.0.0 on Apple M1. Dependencies are existing local MediaPipe, Pillow and Tesseract; no extra model or external dataset was downloaded.

Artifacts: `reports/boxingvi/person-preserving-rgb-v1-native-2/`; committed
`configs/person_preserving_rgb_results_v1.json` pins the protocol, measurements, failed gate and
private previews. Old/new visual sheets are useful qualitative comparisons, **not a numerically
matched segmentation-accuracy benchmark**. The old wrist-intersection metric and new disc/motion
retention metric should not be subtracted. The gate fails numerically and visually, so R3D
and fusion training are correctly skipped rather than fed leaked or erased movement.

## Implications and next experiment

The 209-channel representation is usable for experiments but has **no demonstrated transfer
advantage** at the fixed budget. Preserve these checkpoints as a negative result. Do not
replace the current six-class model, V2 analyzer, activity detector or individual localizer.
A separate authorized training-source ablation could isolate lower-body XY versus derived
geometry/root/coordination channels using source-held-out **supported-class** checks and
predeclared scaling. The present results do not identify a winning subset. Recording diversity
and opposite-arm visibility are material constraints; more epochs or another loss sweep on the
same two sources is not an evidenced solution.

For RGB, a subsequent preprocessing experiment should localize actual text/graphic ink with
stable masks and validate the **on-person footer intersection** explicitly. It must demonstrate
both body retention and removal of overlaid punch labels before any pretrained-head fit.
A person silhouette alone is insufficient. Do not relax redaction or regard extra retained
pixels as useful evidence by default.

For automatic localization, weak class clips cannot teach complete onset/end/background
supervision. The existing reviewed V4/V6 boundaries and Unknown/conflict/censoring masks remain
the only approved strong evidence. Full-body features could be compared in the unchanged
localizer in a **separate frozen, source-separated experiment**, but classification fit cannot
justify that promotion or manufacture missing arm evidence. Reliable end-to-end claims still
require lawfully available recording-diverse clips and completely reviewed continuous sequences
with background and combinations, evaluated outside development sources. No such acquisition,
new annotation request or next experiment was begun here.

## Verification and reproduction

The final full Python suite passed **774 tests**, including 11 focused new initialization,
reflection, clip-reset, mask-control, role, deterministic-fit, freeze and RGB-mask regressions. Existing 111 frontend tests, TypeScript, formatting and production
build pass. Ruff across `src tests` and formatting of all changed Python files pass. No UI was
modified, so no browser behavior change is claimed. Existing httpx/Starlette deprecation and
a benign tensor-copy warning remain.

The original validator again verified **815 clip joins and 580 native video windows**. Live
review/snapshot comparison still preserves 18 excluded events and all holds/conflicts.
**9,717 protected historical files are byte-identical** to the preflight receipt, including
reviews, source artifacts, datasets, models, earlier experiments and pinned R3D weights.
Receipts: `reports/boxingvi/fullbody-classification-foundation-check-v1/` and
`reports/boxingvi/fullbody-classification-preservation-v1/`. No screenshots, media, feature
arrays or checkpoints are committed. No push or model promotion occurs.

```sh
# Use new, nonexisting ignored directories; never overwrite preserved results.
.venv/bin/python -m boxing_motion_ai.training.fullbody_classification train \
  --output reports/boxingvi/fullbody-classification-REPEAT
.venv/bin/python -m boxing_motion_ai.training.fullbody_classification development \
  --output reports/boxingvi/fullbody-classification-REPEAT
.venv/bin/python -m boxing_motion_ai.vision.person_rgb_study \
  --output reports/boxingvi/person-rgb-REPEAT
```

The second command verifies training/code/input hashes before loading V5. Changing the
implementation between phases intentionally invalidates that freeze. RGB requires local
Tesseract at the recorded Homebrew path and host graphics availability on this Mac; no implicit
download fallback exists. All compared model seeds and failures are retained, not selected away.
