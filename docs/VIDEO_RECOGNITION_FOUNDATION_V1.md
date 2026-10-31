# Full-body and pretrained-video foundation v1

Starting revision: `ffaf8a1`. **Foundation delivered; the requested expanded-data training
experiment is blocked, not completed.** No new recording or verified punch boundary was
acquired. Available releases do not establish the required new-data rights/contents, local
V1–V3 alignment holds remain unresolved, and current redacted RGB inputs fail recognition
training-readiness QA. Training another head on these same recordings would not demonstrate
the requested materially expanded, cross-recording result. No such experiment was fabricated.

Implemented and tested: a versioned evidence inventory, exclusion-only weak-label selection,
209-channel full-body features, a real frozen R3D-18 native-frame extractor, and read-only
source/label/timing/cache validation. All output features are real; no punch predictions or
recognition statistics are invented. The dashboard, automatic analyzer, binary detector,
six-class classifier, historical checkpoints, datasets and human-review records are unchanged.

## Actual data audit

Original workbooks were read with the existing inspected-schema reader for the nine permitted
recordings. **5,302 source rows** were counted, including structurally suspect rows; these are
provisional annotations, not verified human instances. V9's workbook, media and poses were not
opened. The old 5,451-row ten-recording report is not substituted for this audit.

| Recording | Actual source rows | Existing assets / current disposition | This foundation |
| --- | ---: | --- | --- |
| V1 | 1,866 | Video/workbook; sampled poses/candidates; unresolved alignment, V1/V6 guard | No tensors/labels admitted |
| V2 | 232 | Video/workbook; sampled poses/candidates; unresolved alignment | No tensors/labels admitted |
| V3 | 815 | Video/workbook; sampled poses/candidates; unresolved alignment | No tensors/labels admitted |
| V4 | 560 | Full prepared poses/candidates; training | 520 weak clips; existing 13 localization references |
| V5 | 598 | Full preparation; development only | Existing eight localization references retained; no new model/feature evaluation |
| V6 | 685 | Full preparation; training, grouped with V1 | 292 eligible weak clips; existing 14 selected localization events |
| V7 | 195 | Full preparation; localization evaluation only | Metadata audit only; no feature extraction/training/evaluation |
| V8 | 199 | Full preparation; grouped with V7, evaluation only | Metadata audit only; no feature extraction/training/evaluation |
| V10 | 152 | Full preparation; evaluation only | Metadata audit only; no feature extraction/training/evaluation |

V1/V2/V3 raw-index hypotheses are already contradicted by earlier visual checks; confidence
scores cannot resolve that mapping. No guessed FPS/offset or unreviewed annotation is promoted.
See [baseline readiness](BASELINE_READINESS.md). The new manifest also records the conservative
V2/V3/V4 publisher guard, alongside V1/V6 and V7/V8, without changing any historical registry.
Publisher IDs are not verified boxer/session identities. No boxer-independent claim is made.

### Existing usable supervision, not new ground truth

Reused the existing approved curated-subset builder and blue-snapshot validator. Historical
raw policy selections contain 527 V4 + 302 V6 clips. Existing curation holds remove 14, leaving
**815 inventoried provisional clips** (520 + 295). A new exclusion-only overlay removes another
three V6 clips, rows **7, 315, 316**, because their ranges intersect the non-blue manual events
excluded from this experiment. No class is rewritten and no review is changed. The overlay
also rejects an exact-range human/provisional type conflict if one is present.

The resulting **812 eligible weak classification clips** have:

| Provisional class | Clips |
| --- | ---: |
| Jab | 117 |
| Cross | 39 |
| Lead hook | 202 |
| Rear hook | 102 |
| Lead uppercut | 191 |
| Rear uppercut | 161 |

These are existing footage, not 812 new independently verified punches. Every weak clip has
`class_verified=false`, `instance_boundaries_verified=false`, `localization_loss=false` and
`background_loss=false`. Its exact source-frame/PTS window is the physically extracted clip,
not a claim that those frames are the true punch onset/end. Overlapping training clips share
one native feature cache rather than becoming independent examples or random frame splits.
There are **9,058 unique cached source frames** across the 815-clip inventory. The three
newly excluded clips remain in the inventory/features for audit; the pinned eligibility overlay
is mandatory for any later weak-label experiment.

The unchanged localization snapshot retains **13 V4 + 14 V6 training event records**, of which
**24 have fully scorable boundaries** and three have conflicting ends. Its original **12
verified negative intervals / 48.006 seconds** remain unchanged. The eight V5 positive references
remain development-only with incomplete background coverage. All 18 excluded non-blue human
events, their revisions, conflicts and previous completion attestations are preserved. Excluded
motion/gaps remain Unknown; weak labels supply neither boundary targets nor background evidence.

### New-source availability and permission gate

Checked public author repositories, folder listings and bounded metadata responses. Their
bytes/hashes are retained privately under `reports/boxingvi/video-foundation-discovery-v1/`.
No new dataset media, annotation package or external pickle was downloaded or executed.

| Source | What was actually verified | Admission decision |
| --- | --- | --- |
| BoxingWeb | Author repository and public Tsinghua share exist; share lists `boxingweb.rar`, **5,190,972,710 bytes**. README refers to LICENSE, but repository root has none and the direct LICENSE URL returns 404. | No verified dataset terms; archive exceeds observed free space even before extraction. No acquisition/role assigned. |
| Remaining BoxingVI | Public annotation folder visibly contains V1.xlsx–V10.xlsx; RGB folder currently exposes `Meta_data.ods`, not raw video files. No V11–V20 annotation/media source was verified. Repository has no LICENSE file. | Published 20-recording claim is not an available licensed local dataset. No new source admitted. |
| BoxMAC | Author paper describes multi-label boxing footage; no usable licensed release was established from it. Multi-boxer labels would require an identity/event adapter. | Not acquired; paper claims are not actual validated annotations. |
| Olympic boxing data | Public Kaggle metadata lists a **3,339,592,607-byte** dataset with license name “Other (specified in description)”. Authors release fight-analysis code; action labels are hand/outcome categories rather than the requested six techniques. | Terms, contents and a valid task/track mapping were not established; no forced six-class conversion or acquisition. |
| Boxing IMU alternatives | The cited Zenodo deposits contain sensor/force data, not synchronized native RGB/MediaPipe video evidence. | Not imported as fabricated visual or instance supervision. |

Primary sources: [BoxingWeb repository](https://github.com/gouba2333/BoxingWeb),
[share metadata](https://cloud.tsinghua.edu.cn/api/v2.1/share-links/c435311fecda4566ae76/dirents/?path=/),
[BoxingVI repository](https://github.com/Bikudebug/BoxingVI),
[annotation folder](https://drive.google.com/drive/folders/1hcW_IeHXeGQogWL5iG5cyNpfWsPd5ltn),
[RGB folder](https://drive.google.com/drive/folders/1_wTc2X4CtdufBzB3QfZfKh7IyxU5EnWx),
[missing-release issue](https://github.com/Bikudebug/BoxingVI/issues/1),
[BoxMAC paper](https://arxiv.org/html/2412.18204v1),
[Olympic author repository](https://github.com/piotr-stefanski/boxing-fight-video-analysis),
[Kaggle metadata](https://www.kaggle.com/api/v1/datasets/list?search=olympic-boxing-punch-classification-video-dataset),
[IMU deposit](https://zenodo.org/records/14965635).

This does not establish that additional material exists nowhere; it records what this run could
verify. Public hosting, article licenses and code licenses are not treated as dataset-media
permission. Existing V4/V6 reuse stays within the user's existing local research authorization;
this document does not certify redistribution/commercial rights. No author was contacted and
no additional manual labeling is requested.

## Full-body representation

`vision/full_body.py`, version **full-body-native-2d-1.1**, outputs **209 channels**:

- All 22 shoulder-to-foot MediaPipe landmarks, including fingers, knees, ankles, heels and foot
  indices: aspect-corrected hip-centered XY divided by torso length, visibility/presence, valid
  flag, exact-time XY velocity and velocity-valid flag: 176 channels.
- Per-arm extension, projected elbow-angle cosine, extension-rate proxies and their masks: 12.
- Projected knee-angle cosines/masks: 4. Shoulder/hip projected axes and alignment/masks: 9.
- Image-plane root velocity/mask: 3. Wrist–foot velocity alignment/masks: 4. Normalization mask: 1.

Estimator Z is unused. These are **2D motion/coordination proxies**, not measured 3D rotation,
impact, force or weight transfer. No stance/lead-side inference or verified movement phase is
created. A large velocity can be tracking/camera error rather than genuine movement.

Original confidence/validity and pose status gate each observation. Off-image/nonfinite joints
are unavailable. Missing torso anchors invalidate normalization; zeros are explicit masked
sentinels, not recovered positions or confirmed stillness. Velocities require consecutive valid
observations and exact rational elapsed source time. Missing indices, timestamp gaps, scene/
segment changes and explicit known cuts reset derivatives. No smoothing, interpolation or
cross-gap position filling occurs. Existing poses remain byte-identical.

Confidence-valid joints are not independently verified accurate joints, particularly during
occlusion or when finger locations are estimated through gloves. Additional feature channels
do not demonstrate recovered observations or improved punch recognition.

The cache retains source frame/PTS/time-base arrays and native joins. Labels/IDs remain audit
metadata, never feature columns. These 209 channels are **incompatible with the existing
65-channel classifier checkpoint**; no existing model is silently fed the new representation.

## Frozen pretrained video path

Selected [Torchvision R3D-18 Kinetics400 V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html),
a practical available spatiotemporal CNN with a ~127 MiB checkpoint, rather than introducing
another dependency stack or a large transformer search. It supplies **512 pooled features**;
the pretrained classification layer is removed. All parameters and batch-normalization buffers
stay frozen/eval, verified unchanged before/after extraction. New classifier/localizer heads
were **not trained**. Full-body/video concatenation would have 721 channels before any separately
declared head-side scaling or masks.

Official weight SHA256:
`b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d`.
The extractor refuses missing/changed weights and never downloads implicitly. Torchvision code
has [BSD-3-Clause terms](https://raw.githubusercontent.com/pytorch/vision/main/LICENSE);
that is not represented as clearance for source media or commercial deployment of every
pretrained component. Weights/features remain outside Git. No pretrained-data overlap audit
against future evaluation videos is claimed.

Each center uses **16 genuine consecutive native frames**, seven before and eight after it.
Incomplete context is unavailable, not padded with repeated images. There is one encoded center
per native frame where the full context exists; no 10 Hz sampling/holding. Windows do not cross
known source resets. Every decoded image must match its exact PTS. The physical 15-gap input span
is ~.625 s at 24 fps and .5 s at 30 fps: native-rate fidelity is not rate invariance. Synthetic
24/30 fps tests verify clocks/indexing, **not semantic recognition at both real camera rates**.

Whole-context reliable body points choose a fixed crop. The complete crop is letterboxed into
the central 112×112 model view; official spatial transforms cannot crop wrists/feet just to fill
the view. Whole-context crop estimation itself uses future poses, so this is explicitly **offline**,
not a causal or bounded-live-latency pipeline. Fixed HUD/identifier rectangles and a static chest
text union remove known text without adding moving mask edges. No annotations are drawn on pixels.
These 16-frame windows are **encoder context, not proposed punch ranges**; 580 encoded centers
are not a punch count. Pretrained category logits are not exposed as boxing predictions.

### Actual native checks and negative visual QA

Final cache: `reports/boxingvi/video-foundation-features-v1-2/`.

| Sequence | Native frames | Encoded centers | Body-normalization available | Extraction/verification seconds |
| --- | ---: | ---: | ---: | ---: |
| V4-bg-1 | 143 | 128 | 143 | 35.81 |
| V6-motion-101 | 241 | 226 | 193 | 62.43 |
| V6-motion-105 | 241 | 226 | 241 | 66.15 |

All **580** centers use real frozen outputs and exact source-window boundaries. Fifteen context
edge frames per sequence remain unavailable. `video_valid` means a complete encoded context,
**not certified person/arm visibility or reliable punch evidence**. Missing body normalization
remains a separate mask and is not background.

Pixel review rejected the initial preflight because shirt text remained visible. Static chest
redaction removes that text, but loses important visual evidence too. Geometry QA against valid,
in-frame landmark centers finds:

| Sequence | Wrist centers intersecting masks | Ankle centers intersecting masks | Foot-index centers intersecting masks |
| --- | ---: | ---: | ---: |
| V4-bg-1 | 42.9% | 0% | 0% |
| V6-motion-101 | 72.8% | 69.7% | 100% |
| V6-motion-105 | 88.8% | 99.2% | 100% |

These are mask-intersection diagnostics, not landmark accuracy or percent of all arm pixels.
Gloves returning toward the chest and feet underneath the footer are particularly affected.
Pose feet can remain available; **the RGB path does not recover obscured feet or ending phases**.
The static union is particularly broad when context includes viewpoint/framing changes.
The source scene inventories list confirmed cuts only and are not exhaustive; V6-101 context
also visibly changes viewpoint. Respecting known reset metadata does not certify continuity
through every unmarked edit. These RGB caches have `training_ready=false`. Background, garments,
face appearance, mask footprint and unresolved small marks remain possible source shortcuts;
no invariance or adequate end-evidence claim is made.

For one real end-centered window per sequence, explicitly synthetic reversed-order and held-center
controls were encoded separately, never used as training footage:

| Sequence / reviewed range | Reverse cosine distance | Held-center cosine distance |
| --- | ---: | ---: |
| V4 [17164,17179) | .00616 | .02077 |
| V6 [541,550) | .00524 | .02089 |
| V6 [30594,30616) | .00928 | .03352 |

The features are not numerically identical to static-image controls, but their modest order
sensitivity **does not demonstrate accurate endings, combination separation or useful classifier
transfer**. Global temporal pooling can blur phases. Re-encoded normal windows equal cached
features in this run (maximum absolute difference zero). No encoder setting was chosen by punch
accuracy or V5 outcomes.

Preflights `video-foundation-features-v1/` and `...-v1-1/` are preserved. The final pass adds
explicit-cut handling to pose derivatives as well as video windows. Data inventory preflights
are also preserved; final whole-body dataset is `video-foundation-dataset-v1-1/`. These are
engineering corrections before any head training, not repeated model-performance searches.

## Training/evaluation decision

| Question | Actual result |
| --- | --- |
| New eligible recordings | **0** |
| New verified individual boundaries | **0** |
| Existing weak clip inventory / eligible labels | **815 / 812**, two existing training recordings |
| Frozen-video / full-body / fused localization comparison | **Not run:** expanded trustworthy data and suitable RGB evidence are not ready |
| Six-class/per-class recognition performance | **Not measured:** no new classifier trained; provisional source counts are not accuracy |
| New individual matches, misses, extras, boundary errors, splits | **Not measured:** no new localization head/candidate decoder output |
| New independent unseen-recording performance | **Not established** |

Existing benchmarks stay available: frozen arm-cycle V2, pooled/balanced localization models,
both offline boundary decoders and the pinned six-class classifier. The last balanced fused
model matched 6/11 V6 training references and 0/8 V5 references with decoder V2; the preserved
pooled model matched 9/11 and 0/8. Those are historical development/training diagnostics from
[the balanced study](PUNCH_LOCALIZATION_BALANCED_V1.md), not new feature performance.
V5 was not run through another classifier/localizer or used to tune this pipeline. No reserved
evaluation group supplies training tensors and V9 data remains untouched.

The next step requires **a verified release/permission path for additional recordings and
annotations**, plus sufficient storage for permitted media and extraction. BoxingWeb additionally
needs actual event schema, human-vs-generated provenance, boxer-track identity, native timing and
stance/hand mapping verification. Do not equate physical left/right labels with lead/rear or
turn model-generated events into human ground truth. V11–V20 need actual accessible source assets
and timing/provenance, not assumed subject IDs from the paper.

Once those gates pass, freeze new recording/source/publisher groups and a real unseen-reference
protocol before fitting. Use separate tasks: class-agnostic activity/onset/end heads with the
frozen decoders on verified boundaries; clip-level six-class loss on eligible weak labels only.
Classification scores never create event boundaries or punch occurrence. Compare full-body-only,
frozen-video-only and fused inputs under one predeclared lightweight-head budget. At fixed data,
compare old/new representations; at fixed representation, compare old/expanded supervision.
Do not call their combined change a measured data benefit. Unknown/conflict masks remain intact.
Include mask/duration/background controls and do not select settings by repeatedly revisiting
the eight V5 references.
Resolve the demonstrated RGB redaction/continuity problem before treating those features as
stronger evidence. No additional annotation work or training experiment is started here.

## Verification, resources and review

Final immutable pins: `configs/video_foundation_snapshot_v1.json`. It binds the dataset,
eligibility overlay and final pretrained-feature manifest, and explicitly marks the expanded
training experiment **blocked**. Source code, source/media/pose hashes, weight provenance,
exact timings, masks, source groups, weak flags and exclusion reasons are retained. The read-only
validator verifies **815 native clip joins and 580 video windows**, recomputes full-body features,
checks both source clocks, and rejects changed labels, frames, code or caches. Earlier artifacts
are never overwritten.

The backbone file is **133,546,016 bytes**. Final all-body weak features plus metadata occupy
about **5.5 MiB**; all new report/preflight artifacts were about **18 MiB** before final logs.
Extraction used local Apple MPS, batch four, torch 2.14.1 / torchvision 0.29.1. GPU access required
the host environment; sandbox-only execution reported MPS unavailable. Feature extraction enforces
a **2.5 GiB free-space reserve** and 64 MiB new-feature budget. Observed free space fluctuated
between ~2.8 and 3.8 GiB; no cache/data was deleted. Timings include repeated forward verification
and decoding/preprocessing, not an isolated production latency benchmark. Three engineering passes
were retained. No model-head training/storage budget was consumed.
Peak GPU/RSS memory was not instrumented and is not inferred from checkpoint size.

**763 Python tests and 111 frontend tests pass**, including 19 new feature/role/weak-label/cache
contracts. Ruff across `src tests`, changed-Python formatting, TypeScript, frontend formatting
and production build pass. One existing Starlette/httpx deprecation warning remains. Automated
preservation checks confirm **9,638 historical artifact files remain byte-identical**, including
all original reviews, datasets, source artifacts and checkpoints; live blue-review validation
still reports the same 18 excluded manual events and three conflicted V6 references. All nine
retained preflight/final feature-cache files also match their recorded hashes. Receipts/logs
are in `reports/boxingvi/video-foundation-preservation-v1/`. No generated
screenshots, media, features or checkpoint is committed. Review the pinned manifests, validation
receipts, temporal controls and private pixel previews; the production application remains unchanged.

Reproduce with **fresh ignored destinations** and the existing private inputs/official pinned
weights; there is no automatic external dataset acquisition or training command:

```sh
.venv/bin/python -m boxing_motion_ai.data.video_foundation \
  --output reports/boxingvi/foundation-dataset-REPEAT
.venv/bin/python -m boxing_motion_ai.vision.video_foundation \
  --output reports/boxingvi/foundation-features-REPEAT --device mps
.venv/bin/python -m boxing_motion_ai.data.video_foundation_validation \
  --dataset reports/boxingvi/foundation-dataset-REPEAT \
  --features reports/boxingvi/foundation-features-REPEAT \
  --output reports/boxingvi/foundation-validation-REPEAT
```

CPU extraction is supported with `--device cpu`; exact cross-device feature equality is not
promised. No new source can bypass existing exclusions by changing a command-line recording ID.
