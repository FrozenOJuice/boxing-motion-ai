# Body feature groups and RGB integrity audit V1

Starting commit: `c89d1f6`. This is one bounded diagnostic comparison, not model promotion,
a feature search or an automatic-detection accuracy result. Source class labels and supplied
clip boundaries remain provisional. No dashboard, production model, historical dataset role,
review, source annotation or checkpoint was changed.

**Outcome:** no additional group shows reproducible overall benefits. V5 macro-F1 is
70.86% for core65, 63.85% with upper, 65.73% with lower and 63.97% with torso features.
Native-frame checks demonstrate leg-identity flips and an unregistered camera cut.
RGB masking still leaks text or removes genuine body evidence; video training remains blocked.

## Frozen comparison

Reuse exactly the previous **812 eligible V4/V6 clips**, exclusion overlay, native clocks,
clip-boundary derivative resets, masks and physical-reflection pipeline. V4 supplies 520 clips
and V6 292. V6 has **no eligible jabs or crosses**; this source/class confound remains.
V6 rows 7/315/316 and all 18 non-blue manual events stay excluded. The previous strong
localization snapshot is not expanded, and no weak boundary or background label is invented.

Protocol: `configs/body_feature_groups_v1.json`, predeclared before any fitting. Each addition
is tested **alone** against exact core65, using unchanged feature units and masks. These groups
partition all 144 extra channels; original common features remain byte-identical core65 values.

| View | Added source columns of fullbody209, half-open | Input width | Parameters |
| --- | --- | ---: | ---: |
| Core reference | None | 65 | 11,814 |
| Upper | `[48,96)` finger/hand observations and `[176,188)` extension/elbow geometry/rates | 125 | 13,734 |
| Lower | `[112,176)` knees/ankles/heels/foot tips and `[188,192)` knee geometry | 133 | 13,990 |
| Torso | `[192,208)` shoulder/hip axes, alignment, root velocity and wrist/foot coordination | 81 | 12,326 |

The architecture remains the existing hidden-32, three-block causal TCN (15-native-frame
receptive field, dropout 0.1, complete-clip mean pooling, six outputs). Extra input weights
start at zero; core weights, downstream weights and random-number state match. Only input
width changes parameter count. There is no fitted scaling, clipping, smoothing, joint filling,
feature combination search or changed representation version.

Same seeds **20261021/22/23**, **40 final-epoch-only epochs**, AdamW lr/decay 0.001/0.001,
batch 32, gradient clip 1, two CPU threads, 50% matched mirror coins, unit CE. Pooled sampling
is the existing uniform-class/square-root-group sampler. Each epoch draws its training-pool
size with replacement. For the single V6-only fold, sampling is uniform across its **four
actually supported classes**; no missing-class examples or targets are manufactured. The six-way
head is retained, and predictions outside the supported reference classes still count as misses.

There are exactly **36 final models**: four views × three seeds × three source splits. Every fit
is repeated with identical settings and requires exact state, curve, sampling and prediction
agreement; checkpoint reloads must reproduce outputs. Repeats verify determinism rather than
supply extra seeds or model selection. Raw feature scales, seed policy and all groups stay
fixed after looking at source-held-out results.

1. **V4→V6:** train all 520 V4 clips; compare all 292 V6 clips, four supported target classes.
2. **V6→V4:** train 292 V6 clips; compare 364 supported-class V4 clips. The 156 jab/cross
   references are explicitly outside this fold's recognition support, not silently scored as
   learned classes. Macro-F1 averages the four supported target classes; outputs remain six-way.
3. **Pooled V4+V6:** train all 812, then freeze all runs, source comparisons and input/code
   hashes **before one V5 pass across the twelve pooled checkpoints**. No held-out model is
   selected for V5. All groups/seeds are reported.

The predeclared benefit rule requires a positive mean paired delta and ≥2/3 positive paired
seeds in **both** source folds, plus all three positive paired V5 deltas. This is a consistency
rule, not a statistical-significance or independent-generalization test. V4/V6 diagnostics are
training-source experiments. V5 remains the previously used development source, never training
or epoch selection here. V7/V8/V10 evaluation data and V9 data are not used.

## Source-held-out result, inspected before V5

Supported-class macro-F1, percent mean ± population seed SD:

| View | V4→V6 | Positive paired seeds | V6→V4 | Positive paired seeds |
| --- | ---: | ---: | ---: | ---: |
| Core65 | 9.38 ± 2.02 | — | 21.24 ± 1.31 | — |
| + Upper | 11.93 ± 2.14 | 2/3 | 19.00 ± 1.44 | 0/3 |
| + Lower | 8.15 ± 2.81 | 1/3 | 19.80 ± 3.48 | 1/3 |
| + Torso | 10.32 ± 0.88 | 2/3 | 20.08 ± 1.12 | 1/3 |

**None passes both directions.** Upper improves V4→V6 mean by 2.55 points but regresses
V6→V4 in every seed. Lower regresses in both means. Torso's small V4→V6 gain reverses in the
other direction. Absolute scores are poor; gains here cannot establish useful recognition.
Rear-hook recall is **zero in every source-held-out run**. V4→V6 predictions concentrate on
lead hook and unsupported jab/cross types; V6→V4 transfers some lead-hook/uppercut evidence
but not rear-hook recognition. Exact matrices, per-class recalls and supported counts are
retained for all seeds. There is **no supported cross-source jab/cross comparison** because
V6 lacks these training/reference types. Their V5 results cannot repair that missing evidence.

The source findings were written to `source_findings_before_v5.json` before the training freeze
and V5 access. No feature policy was revised in response.

## One frozen V5 development pass

After the source comparison, all 36 models and input/code/output hashes were frozen. V5 was
then loaded once and each of the twelve **pooled** final checkpoints evaluated once. All three
new core65 states, sampling hashes, training predictions and V5 scores exactly reproduce
`c89d1f6`'s reference runs. The reference did not change underneath this ablation.

Percent mean ± population seed SD. These are provisional-label agreement metrics on **583
supplied punch clips**, not independent-test or automatic-detection accuracy. Class scores are
uncalibrated conditional softmax scores, never proof that a punch occurred.

| View | Pooled training macro-F1 | V5 macro-F1 | V5 accuracy | Positive paired seeds |
| --- | ---: | ---: | ---: | ---: |
| Core65 | 91.54% | **70.86 ± 4.77%** | 78.73% | — |
| + Upper | 91.27% | 63.85 ± 3.64% | 68.78% | 0/3 |
| + Lower | 93.49% | 65.73 ± 6.05% | 71.24% | 1/3 |
| + Torso | 92.21% | 63.97 ± 5.74% | 75.07% | 0/3 |

| Seed | Core65 | + Upper | + Lower | + Torso |
| --- | ---: | ---: | ---: | ---: |
| 20261021 | 71.98 | 64.32 | 66.31 | 66.06 |
| 20261022 | 64.54 | 59.19 | 72.83 | 56.13 |
| 20261023 | 76.06 | 68.06 | 58.06 | 69.72 |

There is **no qualifying added group**. Lower's one-seed gain (+8.29 points) is offset by
-5.67 and -18.00 points in the other seeds. It is not a basis for selecting that seed or model.
For context only, the preserved joint209 comparison averaged **66.17%**. Isolated group effects
are not additive and do not establish how an untested leave-one-group-out model would behave.

### Per-class tradeoffs

V5 recall, percent mean ± seed SD; support order remains unchanged:

| Type / source support | Core65 | + Upper | + Lower | + Torso |
| --- | ---: | ---: | ---: | ---: |
| Jab / 226 | 77.73 ± 11.32 | 50.15 ± 4.41 | 55.46 ± 26.29 | 73.16 ± 18.90 |
| Cross / 138 | 86.47 ± 3.47 | 91.79 ± 3.94 | 88.89 ± 9.64 | 86.23 ± 3.13 |
| Lead hook / 68 | 78.92 ± 5.00 | 88.73 ± 1.39 | 89.71 ± 4.16 | 85.78 ± 4.85 |
| Rear hook / 39 | 29.91 ± 17.43 | 39.32 ± 28.58 | 44.44 ± 27.32 | 10.26 ± 9.13 |
| Lead uppercut / 39 | 84.62 ± 7.25 | 84.62 ± 4.19 | 73.50 ± 11.90 | 76.92 ± 16.35 |
| Rear uppercut / 73 | 89.95 ± 5.52 | 71.69 ± 17.77 | 82.65 ± 17.84 | 83.56 ± 11.62 |

Mean **per-class F1 deltas** versus core65, percentage points, show that higher recall can
come with extra wrong-type predictions:

| Class | + Upper | + Lower | + Torso |
| --- | ---: | ---: | ---: |
| Jab | -18.83 | -19.71 | -3.87 |
| Cross | +2.84 | -1.76 | -0.13 |
| Lead hook | -15.96 | -3.78 | +4.13 |
| Rear hook | +6.39 | +10.31 | -18.31 |
| Lead uppercut | -9.64 | -14.54 | -16.62 |
| Rear uppercut | -6.84 | -1.28 | -6.54 |

**Jab:** upper hurts F1 in all seeds, with recall deltas -38.05/-18.14/-26.55 points. Its
first-seed confusion matrix sends **94/226 jabs to lead hook**, versus 11/226 for core65.
Lower also hurts the mean substantially but is unstable: seed-3 jab recall falls by 62.39
points, while seed-2 rises by 4.87. This is not a stable lower-body benefit.

**Cross:** upper improves average recall/F1, but its F1 deltas are **-1.67/+3.13/+7.07**
points; no cross improvement is supported in both source-held-out directions because V6 lacks
cross examples. Lower improves average recall while worsening F1, exposing wrong-type tradeoffs.

**Rear hook:** torso reproduces the earlier joint209 collapse: F1 drops in **every** seed
(-13.33/-19.71/-21.88 points), and first-seed **32/39** rear hooks become rear uppercuts.
Upper/lower improve the mean but with large variance and no source-held-out rear-hook recall.
Their per-seed rear-hook F1 deltas are **+9.36/-0.97/+10.78** and **-4.98/+30.55/+5.36**.
Do not describe those averages as reproduced transfer or promote a subgroup winner.

All six-way confusion matrices, per-class precision/recall/F1, exact source bounds and
uncalibrated scores are retained for all models. `configs/body_feature_groups_results_v1.json`
pins the training freeze, source findings, one V5 comparison, summaries, reference replay and
all 36 checkpoints. Nothing was refitted or reselected after these results.

## Concrete evidence behind the feature failures

### Confident joint identity changes create false lower-body velocity

Native V4 frames **8074→8075**, source row **60**, PTS **336.753→336.795 s**: source feet
barely change location while left/right knee, ankle, heel and foot-tip indices exchange physical
sides. Left ankle normalized source x changes **0.3950→0.5390**; right ankle **0.5454→0.3941**.
The cached left-ankle velocity becomes **40.67 torso lengths/s** despite valid observations.
This is a visible correspondence failure, not evidence of exceptionally fast footwork.

Across eligible native adjacent pairs with both ankles valid, the descriptive cross-assignment
check flags **278/3,467 V4 pairs (8.02%)**, versus **1/4,575 V6 pairs (0.022%)**. It flags
same-identity displacement >0.1 source-height units (aspect corrected) and >4× displacement
under crossed correspondence. This is a reproducible diagnostic, **not 278 newly verified
anatomical swaps**, and is never used to relabel, fill joints, alter training or declare punches.
Actual leg crossing can also trigger it; the frame pair above was visually checked.

On valid derivative observations, left-ankle horizontal speed p99 is **31.48** in V4 versus
**2.50** in V6 (right ankle **31.22 / 2.48**). Availability masks alone do not reject an
incorrect but high-confidence joint identity. Lower-body channels consequently carry large,
recording-dependent false-motion tails. The training-source audit records every derivative's
p50/p95/p99/max, using its explicit validity column rather than counting masked zeros as stillness.

### Source edits contaminate root movement

V6 row **631**, native frames **41929→41930**, PTS **1748.789→1748.830 s**: the original
video visibly switches camera viewpoint. Native time is continuous, but the boxer position
jumps and cached root vx becomes **36.79 torso lengths/s**. Current preparation's explicit
cut list is empty for both recordings; scene/gap handling therefore cannot reset this
unregistered edit. This is a limitation of the existing evidence, not a new normalization fix.
The comparison preserves the frozen inputs rather than removing the clip after seeing results.
Any future derivative-quality fix must be separately versioned and revalidated.

### Missingness, geometry and source/class shortcuts remain

V4 right-hand landmark validity is about **49.7–50.2%**, versus **87.8–88.4%** in V6.
Right elbow-geometry validity is **47.1% / 87.4%**, and right extension-rate support
**42.0% / 82.2%**. Repeated finger observations and derived validity features add redundant
recording-specific availability patterns. Torso axes encode projected camera/stance geometry,
not measured 3D rotation; root movement includes camera effects; coordination inherits
unreliable foot and wrist derivatives. The previous mask-only experiment already demonstrated
that observation patterns carry provisional class information.

V4 is bag work while V6 is shadowboxing, with different framing, movement, arm visibility and
class inventories. The absence of V6 jabs/crosses, short V4 clips and longer V6 clips remain
source shortcuts. Source-held-out failures show that excellent pooled training fit does not
resolve them. These diagnostics identify concrete defects and associations, **not an exhaustive
causal decomposition** of the 209-channel regression. Individual-addition experiments do not
identify interactions among all extra groups, nor prove that full-body information is inherently
unhelpful. No extra scaling or subgroup model was fitted after this diagnosis.

## RGB: original pixels, clean-source availability and one bounded test

`configs/rgb_overlay_integrity_v1.json` fixes the audit before running it. Inputs are the exact
original, hash-verified authorized V4/V6 `.mkv` sources, with their saved download metadata.
The corresponding local assets contain one encoded video stream and audio, not a separate
clean picture/overlay layer. No alternate clean master is mapped to either recording locally.
The original author upload identifiers are retained in the audit provenance. No external
acquisition, rights assumption or reconstructed clean version is introduced.

Real QA consists of the same **96 native diagnostic frame requests** plus one center frame
from the first/middle/last eligible clip of every supported class in each source: **30 further
requests, 126 total / 125 unique source frames**. This is a bounded sample, not certification
of all 812 clips. No V5 imagery was used to change preprocessing. All exact frame indices,
PTS, original/processed previews and source hashes are preserved outside Git.

Three fixed comparisons use the existing person matte (threshold 0.5, five-source-pixel
dilation), without the previous indiscriminate OCR:

- **Matte only:** suppress background, retain on-person text as an explicit leakage control.
- **Targeted bright ink:** fixed V6 bottom-20% label region; V4 fixed HUD removal plus observed
  torso/upper-thigh garment region. Remove neutral bright ink (minimum RGB channel 180,
  channel spread ≤55, one-pixel dilation). No box is chosen from a class label, no OCR rectangle
  can erase an arbitrary body region, and no missing torso is fabricated.
- **Whole-region removal control:** erase the entire V6 footer or V4 garment region, plus HUD.
  This measures the price of avoiding text-shaped holes. Cropping away the footer would discard
  the same original foot pixels; this control retains the canvas rather than distorting aspect.

The gate requires ≥95% retained pixel neighbourhoods and ≥80% retained adjacent motion for
**each arm and the feet separately**, plus no visible source labels/digits/logos. Original
pixels survive unchanged wherever retained; elsewhere constant gray is an erasure, never
recovered anatomy. As before, valid-joint discs are an evidence-retention proxy, **not proof
of landmark accuracy or complete silhouette preservation**. Motion is original adjacent-RGB
absolute difference surviving both masks, not new edges generated by the masks. Gaps between
sample windows are excluded. Camera edits or misplaced landmarks can also contribute to
this proxy; it is not a pure limb-motion or optical-flow ground truth. No threshold or
alternative was repeatedly adjusted to pass.

| Diagnostic case / policy | Left arm pixels / motion | Right arm pixels / motion | Feet pixels / motion |
| --- | ---: | ---: | ---: |
| V4-bg-1, matte only | 99.97 / 92.57% | 96.41 / 82.55% | 87.61 / 90.17% |
| V4-bg-1, targeted ink | 75.63 / 64.09% | 96.41 / 82.34% | 87.61 / 90.17% |
| V4-bg-1, full region | 25.91 / 10.44% | 87.02 / 74.04% | 87.61 / 90.17% |
| V6-101, matte only | 99.83 / 86.07% | 99.72 / 96.20% | 99.88 / 85.29% |
| V6-101, targeted ink | 99.83 / 86.07% | 99.72 / 96.20% | 98.53 / 80.14% |
| V6-101, full footer | 99.83 / 86.07% | 99.72 / 96.20% | 23.84 / 30.13% |
| V6-105, matte only | 99.84 / 98.47% | 98.60 / 95.81% | 93.02 / 93.72% |
| V6-105, targeted ink | 99.84 / 98.47% | 98.60 / 95.81% | 63.51 / 78.63% |
| V6-105, full footer | 99.84 / 98.47% | 98.60 / 95.81% | 6.43 / 16.65% |

**The gate fails.** The new method avoids the old OCR's large false rectangles but does not
solve the underlying integrity problem:

- Native V6 frame **30550** shows instruction text `HOOK` and digit `5` directly over shoes.
  Targeted ink removal leaves perfectly readable **gray letter/digit silhouettes** on the
  shoes. The static legend is source metadata; its causal influence on class predictions was
  not measured, and it is not assumed to reveal the current punch type. A pixel mask can itself preserve the label's shape. Erasing the surrounding region
  removes that shape but also deletes original foot evidence. Original pixels behind the
  composite are unavailable; there is no clean layer to recover, and no generative filling.
- V4 frame **17150** still contains shorts lettering and glove/shoe identifiers; the bright-ink
  garment mask erases genuine white glove pixels as the hand crosses that region. Garment
  text and glove details cannot be distinguished reliably by brightness in this test.
- V6-101 has genuinely more usable movement and no footer legend at the sampled early frames;
  it passes the numeric retention proxy. It still retains glove logos and matte-edge background
  remnants. Later V6 eligible samples contain legends, changed layouts and `BACK/FRONT` text;
  the per-class galleries confirm that an early clean-looking interval cannot certify the pool.

Both original-versus-targeted galleries (18 V4 / 12 V6 clips), all three diagnostic alternatives
and representative native-resolution pairs were visually inspected. Some imagery is useful for
inspection and potentially a future restricted experiment; **the complete existing RGB pool is
not approved as reliable label-free recognition input**. No conclusion that all RGB learning is
impossible is implied. A validated clean subset or permitted pre-overlay masters could change
that decision, but neither is fabricated here. Further pretrained-video training is not authorized
by this failed gate and none was launched.

The 33 QA jobs took approximately **404.5 seconds** in their measured loops, including decode,
full-resolution masks/metrics and preview writes, while other work ran. This intentionally
redundant audit is not a production latency benchmark; peak memory was not measured. Existing
local MediaPipe weights and Pillow were reused; no OCR, new visual model or external dataset
was downloaded. The host graphics environment was required. Source/pose/timing artifacts remain
unchanged. Further redaction iterations stopped after this explicit failed comparison.

## Decision and evidence required for localization

1. **Keep core65 as the reference.** None of the added groups has consistent cross-source
   benefits under the predeclared rule. Specific V5 class tradeoffs must not become a post-hoc
   feature/seed selection. The current experiment does not establish that dropping one group
   from the joint209 model would reproduce the single-addition result.
2. **Do not approve full-pool pretrained-video learning.** Some V6 introductory material has
   clean-looking motion, but source legends physically cover body pixels elsewhere and targeted
   redaction preserves glyph geometry. Acquire a permitted pre-overlay master or validate a
   genuinely clean subset in a separate evidence milestone. Do not continue trying to reconstruct
   invisible foot motion or lower the gate to fit a video model.
3. **Fix observation integrity before another body-feature experiment.** A future version could
   flag ambiguous anatomical correspondence, reset derivatives at visually verified edits and
   expose missing derivative support without filling positions. First verify those interventions
   on native images; do not auto-swap legs, silently patch the old cache or add more loss sweeps.

Reliable individual localization requires evidence the current weak clip pool cannot supply:

- **Continuous native-rate, pre-overlay recordings**, covering the whole boxer, with useful
  views of both arms during extensions, recovery, hooks, uppercuts and overlapping combinations.
  Known camera edits and failures must create explicit discontinuities/Unknown, not punch evidence.
- **Several distinct recording/boxer/session groups with within-source positive and negative
  examples**, including jab/cross outside V4. Stance, distance, attire, bag/shadowboxing and
  viewpoint must not identify the class or whether activity exists. A handful of extra clips
  from one source does not provide this separation.
- **Human-verified individual start/end ranges with exhaustive local inventory** for selected
  continuous training and development sequences, including footwork, guard, defense, recovery,
  background and difficult combinations. Censored/occluded/ambiguous regions stay Unknown or
  unresolved; no class scores, provisional rows, excluded manual events or pose absence become
  boundary/background truth.
- A **recording-separated development protocol and untouched final evaluation role**, reporting
  matches, misses, boundary errors, merges/splits and extras only where reference completeness
  supports them. V5's eight selected positives and incomplete background do not support full
  event precision/false-positive rates or independent-test claims.

Current strong supervision remains **13 V4 + 14 V6 event records, 24 fully scorable**, with three
conflicted boundaries and the historical verified-negative inventory. The 812 weak clips provide
conditional type labels, not hundreds of verified localization examples. V5 remains development;
V7/V8/V10 and V9 restrictions remain intact. These are requirements for a future approved data
milestone, **not a request for more labeling now**. No new review queue, training experiment,
R3D extraction, classifier replacement or localizer promotion follows this diagnosis.

## Artifacts, verification and review

- Model run: `reports/boxingvi/body-feature-groups-v1/`; immutable result bindings:
  `configs/body_feature_groups_results_v1.json`. All source-fold/pooled models, seed curves,
  metrics, native frame/PTS bounds, source diagnostics and prediction scores are retained.
- RGB audit: `reports/boxingvi/rgb-overlay-integrity-v1/`; bindings:
  `configs/rgb_overlay_integrity_results_v1.json`. `gate.json` records **failed**, not approval;
  `summary.json` contains per-arm/per-foot metrics for all 33 QA jobs.
- **786 Python tests pass**, including 12 new group-partition, initialization/RNG, unchanged
  core, supported-class sampling/metrics, role/freeze, deterministic-fit and pixel-integrity
  regressions. Existing missing-pose, timestamp and historical dataset checks also pass.
- **111 frontend tests pass**, TypeScript, frontend formatting and production build pass.
  Initial frontend attempts during concurrent CPU-heavy fitting/QA had asynchronous UI lookup
  and 5-second timeout failures. With heavy jobs finished, all tests pass using one worker;
  no frontend code or timeout threshold was changed. This is regression testing, not a claim
  of new browser/automatic-upload functionality.
- Ruff across `src tests`, changed-Python formatting and Git whitespace checks pass. Existing
  Starlette/httpx deprecation and the earlier reference trainer's tensor-copy warning remain.
- Original foundation validation again verifies **815 clip joins and 580 video windows** and
  unchanged live review evidence. **9,814 protected historical files** are byte-identical to
  preflight, including the previous classification/RGB experiments, original models, datasets,
  source artifacts, reviews and weights. No cache deletion or historical rewrite was used.
  Receipts: `reports/boxingvi/feature-group-preflight-v1/`,
  `reports/boxingvi/body-feature-groups-preservation-v1/` and
  `reports/boxingvi/body-feature-groups-foundation-check-v1/`.

Generated media, previews, features and checkpoints stay outside Git. Only versioned experimental
code, focused tests, protocol/result pins and this report are committed. No push occurs.
Review the source comparison before the V5 section; native-frame tail previews distinguish
false joint/camera motion from real footage, and the original/targeted galleries demonstrate
the RGB masking tradeoff. Peak process memory was not measured; timings under concurrent work
must not be treated as deployment benchmarks.

Reproduction uses existing authorized private inputs and **fresh ignored output directories**:

```sh
.venv/bin/python -m boxing_motion_ai.training.body_feature_groups train \
  --output reports/boxingvi/body-feature-groups-REPEAT
# Inspect the source comparisons; this command refuses a changed training/input/code freeze.
.venv/bin/python -m boxing_motion_ai.training.body_feature_groups development \
  --output reports/boxingvi/body-feature-groups-REPEAT
.venv/bin/python -m boxing_motion_ai.vision.rgb_overlay_integrity \
  --output reports/boxingvi/rgb-overlay-integrity-REPEAT
```

RGB requires the existing local MediaPipe task and host graphics access on this Mac, with no
implicit download. Source edits/joint-assignment flags are diagnostics only, not a new corrected
dataset. Reproduction commands do not authorize further experiments or production promotion.
