# Bounded frozen RGB + pose feasibility study

This milestone compares the accepted multi-recording XY+velocity pose model with a
frozen visual encoder and a small temporal head. V4-only XY+velocity+mirror remains
the provisional reference pending review. No source labels, exclusions, pose quality
thresholds, original media or annotation files are changed. V9 is never opened.

## Targeted inspection before modeling

Nineteen exact-PTS contact sheets compare raw frames against pose overlays, covering
first/last available approved rear-hook and rear-uppercut annotations across
V4/V6/V7/V8/V10, plus excluded V6 row 685 and V10 row 9. When no approved examples
exist, excluded examples are displayed for context only. The review generator is
`boxing_motion_ai.training.rgb_review`; the versioned decision and uncertainties are
in `configs/level1_rgb_review.yaml`. Local sheets are under
`reports/boxingvi/level1-rgb-review-v1/` and are not distributed.

| Source | Observed evidence | Remaining limitation |
|---|---|---|
| V4 RH 3/527; RU 63/561 | Punch/action phases overlap raw intervals; bag/head hide far arm; some landmarks attach to occluded anatomy. | Sparse sheets cannot independently certify every subtype, stance, onset or recovery boundary. |
| V6 RH 41/336; RU 8/674 | Hook-like sweeping motions visible; some RU intervals include substantial recovery. | Phase and anatomical-side conventions remain provisional. |
| V6 held RU 685 | Straight extension and mixed phases support the existing hold. | No correction or restoration. All prior holds remain. |
| V7 RH 52 | The sole approved RH includes retraction/body dip; phase is ambiguous. | One provisional example cannot establish reliable RH generalization. |
| V7 excluded RU 5/132 | Multiple consecutive action phases and overlap concerns remain. | No RU examples restored. |
| V8 RU 8/173 | Compact rising bent-arm movements, with far-arm visibility failures. | No source RH annotations; side semantics remain unverified. |
| V10 RH 7/144; RU 41/152 | Curved/rising arm movements and torso rotation are more visible. | Some elbow errors; anatomical side and boundaries remain provisional. |
| V10 excluded LU 9 | Resembles the side/execution of an approved RU example. | Existing side-mismatch hold retained; no label swap. |

No new confirmed pervasive label contradiction was established that prevents a
**provisional** pilot. This is not independent certification of all source labels.
The retained V7 rear hook and late/recovery RU phases remain explicit caveats.

Source overlays are a serious visual shortcut: V4/V7 combination codes, V6 punch
legends, V10's “Uppercut” banner, and V5's large punch-number overlay. Preflight crop
QA found incomplete redaction, then overly broad global masking. Neither preflight
cache was used for training. Final policy `rgb-feasibility-1.2` uses reviewed, fixed
source-specific rectangles, constant across classes/times. All six per-class preview
sheets were inspected before training. The rectangles remove visible text/digits and
banner tips while preserving upper-body anatomy. They can obscure some peripheral
pixels; they do not guarantee removal of every source-specific visual cue.

## Frozen protocol

- Same approved pool as the accepted multi-recording study: training pool 1,156
  examples from V4/V6/V7/V8/V10; V5 development 583. No unlabeled intervals become
  `no_punch`. Original holds, quality masks, provenance and uncertainty stay intact,
  including V6 321/598/620/675–686, V10 123/130–137/9–10, and excluded V7 rear uppercuts.
- Seed **20261021** only. Combined pool plus four group-held-out folds: V4, V6,
  V7+V8 together, V10. V5 alone selects epochs within every fold. Held-out groups
  never train or select checkpoints. V1/V6 guard remains; V1 is not loaded. Unknown
  boxer/session identities remain unknown; this is not boxer-independent evaluation.
- Pose-only and compatible mask-only checkpoints are reused from the accepted study,
  with exact prediction replay. The preserved V4-only reference is not retrained.
- [Torchvision MobileNetV3-Small ImageNet1K V1](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html)
  supplies 576 pooled visual features. Its classifier is removed; encoder parameters
  are frozen and batch normalization stays in evaluation mode. Official preprocessing:
  resize 256, center crop 224, ImageNet mean/std. Official weight SHA256:
  `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.
  Torchvision 0.29.1 / torch 2.14.1 are locked. The encoder runs on local Apple MPS;
  temporal heads train on CPU. No encoder fine-tuning or model/setting search.
- Each clip samples its first decoded frame, then the next native frame at least
  0.1 seconds after the preceding sample. Exact rational PTS, not nominal FPS,
  determines sampling. Native pose frames causally hold the latest sampled RGB
  embedding. Sampling resets at scene/segment changes and missing frame boundaries.
  No synthetic image/joint interpolation or future-frame selection is used. Very short punches
  may contain only one to three distinct RGB samples: a material feasibility limit.
- Boxer-centered square: 4.5 torso lengths around shoulder/hip midpoints, using only
  reliable anchors. Gray padding; full-source centered-square fallback when anchors
  are unreliable. No invented joints. RGB-only uses these pose-guided crops, so it is
  not independent of the pose cropper. Crops retain some scene context.
- Each sampled crop and its physical horizontal reflection are encoded separately.
  One training-only 50% clip coin selects RGB reflection and the corresponding
  anatomical pose/velocity reflection. Lead/rear labels stay unchanged. Evaluation
  uses unreflected inputs. No visibility augmentation, pre-clip context or geometry.
- Pose head: 65 inputs. RGB: 576 inputs. Hybrid: concatenated 65+576 inputs. The same
  causal TCN (hidden 32, three dilated blocks, RF 15 native frames, dropout 0.1,
  annotated-clip mean pooling) is used. Shared hidden initialization follows the
  original seed; input begins with the reference's 41-column initialization and
  remaining columns zero. For RGB those initial columns are arbitrary encoder
  channels, not pose channels. No learned encoder or embedding normalization.
- Same 40 epochs, batch 32, AdamW lr 0.001, decay 0.001, gradient norm 1. Same
  training-only class/group/recording sampler, replacement draws and mirror coins
  are checked against the preserved pose run. Unit CE avoids double balancing.
- No IDs, filenames, absolute timestamps, source rows, annotation metadata or drawn
  review text enter tensors. Exact timing/identities remain audit metadata. Labels
  are training targets and inform the training-only sampler; they are never features. Duration and source appearance remain possible indirect cues.
- Static background control: one fixed redacted, conservatively boxer-erased source
  embedding per recording, repeated throughout every clip. Reflected counterpart
  follows the same training coin. This avoids leaking a moving erasure-box trajectory;
  it tests static scene/recording identity plus duration, not all dynamic backgrounds.
  Existing mask-only and training-only duration controls are also reported.

Caches contain original/reflected crops and background features, source indices and
exact PTS; inputs, encoder hash/device, policy and cache bytes are verified on reuse.
No source media is rewritten. Decoding refuses unmatched PTS. Completed recording
caches resume; an interrupted recording restarts. Reports contain configurations,
source-code snapshots, curves, selected/last checkpoints, example-level predictions,
confusion matrices, per-class metrics, epoch sampling/mirror logs and preservation
manifests. Generated artifacts remain excluded from Git. Licensing/redistribution
rights for source material and pretrained weights are **unverified** here; this study
does not redistribute them or infer permission from public hosting.

## Reproduction

From the repository root, with the existing approved local datasets and earlier
experiment artifacts present:

```sh
export UV_CACHE_DIR=/private/tmp/boxing-motion-ai-uv-cache
export MPLCONFIGDIR=/private/tmp/boxing-motion-ai-mpl
export TORCH_HOME="$PWD/weights/torch"
uv sync --extra pose --extra train
uv run python -m boxing_motion_ai.training.rgb_review --output reports/boxingvi/rgb-review-repeat
uv run python -m boxing_motion_ai.training.rgb_review --output reports/boxingvi/rgb-crop-review-repeat --preprocessing
# Inspect review sheets and source-specific preprocessing previews before training.
uv run python -m boxing_motion_ai.training.rgb_study --output reports/boxingvi/rgb-smoke-repeat --cache reports/boxingvi/rgb-features-repeat --smoke
uv run python -m boxing_motion_ai.training.rgb_study --output reports/boxingvi/rgb-study-repeat --cache reports/boxingvi/rgb-features-repeat
uv run python -m boxing_motion_ai.training.rgb_verify --output reports/boxingvi/rgb-study-repeat --cache reports/boxingvi/rgb-features-repeat
uv run python -m boxing_motion_ai.training.rgb_report --output reports/boxingvi/rgb-study-repeat
uv run pytest
uv run ruff check .
```

MPS extraction requires local GPU access. CPU users can change only `encoder_device`
to `cpu` in a copied configuration and use a fresh cache. This changes the numerical
extraction backend and is recorded; it is not promised bitwise identical across
hardware. Selected frozen-head predictions must replay exactly on the same backend;
re-encoded features allow small floating-point batch-size differences.

## Results

**Pose-only provides the strongest evidence in this pilot.** It wins six-class
macro-F1 on V5 and every held-out source group. RGB-only and hybrid show severe
class collapse; neither configuration merits a three-seed confirmation unchanged.
This conclusion applies to this frozen ImageNet encoder, crop/sampling scheme and
fixed-budget temporal head. Other visual encoders and motion representations remain
untested. Pose-only's own held-out scores remain inadequate for reliable six-class
recognition; this is not a Level 1 completion claim.

Local results: `reports/boxingvi/level1-rgb-study-v1/REPORT.md` and `summary.json`.
The report contains every fold's V5 results, individual held-out recording metrics,
per-class precision/recall/F1, and full confusion matrices. JSON also contains all
mask/background controls and source-level predictions. Previous V4-only and
multi-recording results are preserved. This comparison uses matched seed 20261021;
the prior three-seed mean is not substituted for the paired pose result.

### Balanced performance

Cells are **accuracy / six-class macro-F1 / supported-class macro-F1**. Unsupported
classes remain in the six-class score with zero F1; supported F1 averages only classes
with eligible ground truth, and predictions into absent classes still count as errors.


| Evaluation | Pose | RGB | Hybrid |
|---|---:|---:|---:|
| V5 development, combined pool | 0.8542 / 0.7532 / 0.7532 | 0.3859 / 0.2543 / 0.2543 | 0.7307 / 0.5047 / 0.5047 |
| V4 held out | 0.1954 / 0.1637 / 0.1637 | 0.2657 / 0.0700 / 0.0700 | 0.2657 / 0.0747 / 0.0747 |
| V6 held out | 0.4801 / 0.2824 / 0.4236 | 0.0033 / 0.0050 / 0.0075 | 0.0033 / 0.0050 / 0.0075 |
| V7 held out | 0.4091 / 0.2433 / 0.2920 | 0.0000 / 0.0000 / 0.0000 | 0.0758 / 0.0802 / 0.0963 |
| V8 held out | 0.5000 / 0.4536 / 0.5444 | 0.1750 / 0.1213 / 0.1455 | 0.2333 / 0.1531 / 0.1837 |
| V7+V8 group | 0.4677 / 0.4230 / 0.4230 | 0.1129 / 0.0891 / 0.0891 | 0.1774 / 0.1330 / 0.1330 |
| V10 held out | 0.3404 / 0.2683 / 0.2683 | 0.0284 / 0.0093 / 0.0093 | 0.0709 / 0.0508 / 0.0508 |

V6 has no source jab/cross annotations. V7 has no eligible rear uppercuts (all remain
excluded); V8 has no source rear hooks. Their absent classes are not scored as observed
recall failures. The V7+V8 group has six supported classes, but only one provisional rear hook.

The unweighted mean over four held-out source groups is:

| Input/control | Accuracy | Six-class F1 | Supported F1 |
|---|---:|---:|---:|
| pose | 0.3709 | 0.2843 | 0.3196 |
| rgb | 0.1026 | 0.0433 | 0.0440 |
| hybrid | 0.1293 | 0.0659 | 0.0665 |
| mask | 0.2396 | 0.1382 | 0.1511 |
| background | 0.1012 | 0.0578 | 0.0655 |

These are descriptive fold means, not uncertainty estimates across seeds or athletes.

### Rear punches and lead uppercuts

Cells are recall for **rear hook / lead uppercut / rear uppercut**.

| Evaluation | Pose | RGB | Hybrid |
|---|---:|---:|---:|
| V5 combined | 0.2308 / 0.7692 / 0.9315 | 0.0513 / 0.0256 / 0.1507 | 0.0769 / 0.0256 / 0.3836 |
| V4 | 0.0000 / 0.5775 / 0.0115 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0141 / 0.0000 |
| V6 | 0.0625 / 0.3760 / 0.4691 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 |
| V7+V8 | 0.0000 / 0.5789 / 0.5000 | 0.0000 / 0.2456 / 0.3750 | 0.0000 / 0.3684 / 0.1250 |
| V10 | 0.0000 / 1.0000 / 0.2381 | 0.0000 / 1.0000 / 0.0000 | 0.0000 / 1.0000 / 0.0000 |

RGB and hybrid have **zero held-out rear-hook recall in every fold**. Neither improves
rear-uppercut recall on any held-out group. V10 lead-uppercut recall of 1.0 is
misleading in isolation: only four examples exist, and RGB predicts lead uppercut
for 139 of 141 clips. Its overall accuracy is 2.84%.

In the combined-pool V5 comparison, hybrid slightly increases jab/cross recall
(0.9027→0.9336 / 0.9638→0.9710), while lead-hook recall declines 0.7941→0.7206
and all three weak classes above regress substantially. There is no balanced benefit.

### Overfitting and shortcuts

At each V5-selected checkpoint:

| Fold | RGB epoch | RGB training/V5 F1 | Hybrid epoch | Hybrid training/V5 F1 |
|---|---:|---:|---:|---:|
| combined | 27 | 0.9160 / 0.2543 | 31 | 0.9597 / 0.5047 |
| hold-V4 | 31 | 0.8618 / 0.2503 | 28 | 0.9388 / 0.4449 |
| hold-V6 | 38 | 0.9142 / 0.2149 | 33 | 0.9047 / 0.3782 |
| hold-V7-V8 | 17 | 0.9160 / 0.2120 | 10 | 0.8450 / 0.3749 |
| hold-V10 | 12 | 0.8383 / 0.2072 | 28 | 0.9152 / 0.6291 |

The combined RGB/hybrid final-epoch training F1 reaches 0.9537/0.9755 while V5 F1
falls to 0.2171/0.4853. Strong training fit and severe held-out failure support
overfitting/domain dependence, rather than failure to optimize the training labels.

- Held-out V4: RGB predicts lead hook for **527/527** examples. Accuracy rises
  above pose (0.2657 vs 0.1954) while macro-F1 falls (0.0700 vs 0.1637).
- Held-out V6: RGB and hybrid predict absent jab/cross classes for **299/302**
  examples. Supported F1 is only 0.0075; this is not explained by six-class averaging.
- Held-out V10: RGB predicts lead uppercut for 139/141 examples; hybrid for 131/141.
- Combined V5 mask-only F1 is 0.3782, static-background F1 0.1755 and duration
  F1 0.0863. RGB-only (0.2543) is weaker than mask-only. In every held-out group,
  both visual configurations are weaker than the compatible mask-only model.
- Group-held-out duration F1 is V4 0.0473, V6 0.0941, V7+V8 0.0959, V10 0.1666.
  RGB/hybrid are often comparable to, or worse than, simple shortcut controls.
- Static background alone cannot explain every error or isolate changing camera
  views, bag appearance, clothing, stance or foreground texture. Those remain
  confounds in RGB crops. The control is not proof of background invariance.

### Input-quality and implementation checks

Final caches contain **6,670 sampled source frames**, with zero torso-crop fallbacks.
Across selected RGB samples there are **zero reliable shoulder/elbow/wrist points
outside the crop or inside a HUD-redaction rectangle**. This is a geometric check
against source validity masks, not independent truth about landmark correctness.

| Source | Cached frames | Median distinct RGB samples/clip | Clips with ≤2 RGB samples |
|---|---:|---:|---:|
| V4 | 1527 | 3 | 164 |
| V6 | 1819 | 6 | 0 |
| V7 | 311 | 4 | 0 |
| V8 | 551 | 4 | 0 |
| V10 | 613 | 4 | 0 |
| V5 | 1849 | 3 | 149 |

V4 sampled-arm validity is particularly weak for rear hooks (786/1170 arm slots,
67.2%) and rear uppercuts (945/1398, 67.6%). V6 has much higher source-valid
rear-arm coverage (96.1% RH / 94.6% RU), yet visual transfer collapses there.
Visibility alone therefore does not account for the observed model failures.

Checks found no label-order, feature-join, PTS-selection, pooling-mask or reflection
error. Original labels and accepted provisional intervals were used unchanged.
The most plausible remaining explanations are generic ImageNet appearance features
with little motion specialization, strong foreground/camera domain shifts, and sparse
visual observations in short punch intervals. Their individual causal effects were
not separated in this pilot. Existing phase/side uncertainty and physical occlusion
remain limitations for both modalities.

### Verification and decision

- Two-epoch smoke test passed before all 15 new 40-epoch head runs.
- 60 preserved pose/mask prediction sets matched their original files exactly;
  90 new frozen-head prediction sets replayed exactly from checkpoints.
- All **600** new sampling/mirror audit epochs match the paired pose run.
- Six source frames were decoded and re-encoded; maximum absolute embedding
  difference was **3.5763e-6** (rtol/atol 1e-4 allowed for different batch sizes).
- **4,507 protected files**, including six original videos and six workbooks, are unchanged.
- Automated tests: **266 passed**. Ruff and lock consistency checks pass.

**Keep the existing pose reference. Do not promote RGB/hybrid or run a three-seed
confirmation of this configuration.** The pilot provides no consistent visual benefit,
including for rear hooks or rear uppercuts. A future RGB investigation should first
validate motion-suitable frozen features and sufficient temporal observations in a
small prespecified check. Targeted annotation/arm-tracking review remains useful;
no additional review, encoder change, training study or Level 2 work is started here.
Dataset identities, source grouping and all original exclusions remain unchanged.
