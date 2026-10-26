# Bounded pose-estimator investigation

2026-10-26. The conservative 1,123-example training pool remains the provisional pool. No classifier was trained, no full recording was re-extracted, and no review or annotation was changed. V9 was not accessed.

## Decision

Retain MediaPipe in the production pipeline for now. Pose errors are a demonstrated contributor: high-confidence elbows and wrists sometimes land on the opposite arm or empty space. RTMPose improves several of these specific failures, particularly in V6/V7, but introduces other errors in V5/V10 and does not resolve hidden limbs or depth ambiguity. This comparison supports a targeted localization validation before replacing the estimator; it does **not** establish that replacing it would fix classifier generalization.

The next useful step is a small visible-joint reference set from these already decoded frames, including successful frames and counterexamples. Mark only visible elbow/wrist locations, allow occluded joints to remain unknown, and compare normalized localization error and arm-identity mistakes. A reviewer can prioritize the eleven enlarged frames below rather than repeat annotation review. Calibrate an alternative estimator's validity policy separately before any downstream experiment. Do not undertake full extraction yet.

## Sample and safeguards

The pinned configuration selects 34 intervals: all 18 previously human-reviewed cases plus the first available historically approved diagnostic case for each missing recording/class combination. V6 has no jab/cross annotations; V7/V8 each lack some individual class coverage. This is a purposive diagnostic sample, not a random prevalence estimate. Historical candidates later held by curation remain diagnostic examples only.

There are 537 exact decoded frames, including at most two context frames on either side; 401 are within source annotation intervals. Context cannot cross scene/segment boundaries or extraction gaps. Frame indices and rational source PTS are checked by the existing exact-frame decoder. All 34 raw/MediaPipe/RTMPose contact sheets and eleven enlarged disagreement frames were inspected. The new visual judgments are **automated assistant observations**, not human-certified labels or landmark ground truth.

The audit verifies 4,647 pinned file hashes before/after processing: prior protected inputs, the six relevant source videos/workbooks, and audit configuration/model. Review ledger v19 remains pinned at `f5d8ef1e437fa7f50beec40e81b9bb526b1a3422dd2d5b7d4f7426e30f90ce5c`. All original exclusions remain in force, including the six confirmed disputes. V6:264/266 remain held: their human notes say no punch was observed; hand-only confirmations do not establish valid punch labels. V1/V6 and V7/V8 grouping precautions remain unchanged.

## Reliability by recording and class

Numbers below are percentages passing the **existing MediaPipe validity mask**, within annotation bounds. They measure available features, not anatomical accuracy. Shoulders and hips pass on every sampled frame; this does not prove their positions correct. L/R mean anatomical channels, not lead/rear hand. Viewpoint descriptions are visual observations of this sample, not verified identities or stance labels.

| Recording | Cases / frames | Left elbow | Right elbow | Left wrist | Right wrist | Viewpoint and recurring issue |
|---|---:|---:|---:|---:|---:|---|
| V4 | 6 / 42 | 100.0 | 42.9 | 100.0 | 45.2 | Side/oblique bag work; far arm hidden by body, glove or bag |
| V5 | 6 / 63 | 60.3 | 100.0 | 61.9 | 95.2 | Oblique ring view; left arm disappears during turns |
| V6 | 5 / 70 | 100.0 | 94.3 | 100.0 | 95.7 | Frontal/oblique shadowboxing; overlapping gloves create confident wrong chains |
| V7 | 5 / 77 | 89.6 | 97.4 | 89.6 | 100.0 | Frontal/oblique; crossing arms and sleeves obscure elbow location |
| V8 | 5 / 61 | 100.0 | 42.6 | 100.0 | 49.2 | Rear/oblique outdoor view; far right arm hidden, depth compressed |
| V10 | 7 / 88 | 96.6 | 93.2 | 98.9 | 93.2 | Frontal fitness footage with rotation; arm overlap confuses both estimators |

| Source class | Cases / frames | Left elbow | Right elbow | Left wrist | Right wrist |
|---|---:|---:|---:|---:|---:|
| Jab | 5 / 54 | 100.0 | 66.7 | 98.1 | 66.7 |
| Cross | 5 / 56 | 75.0 | 91.1 | 75.0 | 91.1 |
| Lead hook | 6 / 88 | 92.0 | 88.6 | 93.2 | 89.8 |
| Rear hook | 5 / 59 | 86.4 | 88.1 | 86.4 | 91.5 |
| Lead uppercut | 6 / 68 | 100.0 | 75.0 | 100.0 | 75.0 |
| Rear uppercut | 7 / 76 | 90.8 | 81.6 | 94.7 | 84.2 |

Uppercut source-class bins include the two held V6 intervals. These bins must not be interpreted as verified punch ground truth. Viewpoint, recording and class are confounded in this small sample; no independent viewpoint accuracy estimate is available.

## Exact-frame evidence

[Generated comparisons](../reports/boxingvi/pose-estimator-audit-v1/index.html) contain raw frames and both skeletons. Cyan is anatomical left, magenta right; orange MediaPipe edges fail its existing mask. RTMPose finite coordinates are displayed without asserting validity. Different confidence scores are never equated.

| Case / exact frame | Observation | Comparison |
|---|---|---|
| V6:7 / 493–494 | MediaPipe's right elbow is near overlapping gloves rather than the visible lower elbow. At 493 visibility is .957 and presence .9999. | RTMPose closer to visible elbow; active-arm blur still limits certainty |
| V6:46 / 8370, 8375 | MediaPipe's left wrist is near the elbow/crossing forearm instead of the raised glove; visibility .894/.915, presence >.999. | RTMPose better aligns the offhand wrist/elbow |
| V6:613 / 41095 | MediaPipe's right wrist floats above the visible lowered glove; visibility .820, presence .9999. | RTMPose closer to glove |
| V7:51 / 7993 | Elevated right elbow placed near opposite lower arm; visibility .980, presence .9988. | RTMPose closer to elevated elbow |
| V7:38 / 6510 | Right wrist in empty space instead of raised hand; visibility .993, presence .9999. | RTMPose closer to hand |
| V10:78 / 5811–5812 | Overlapping arm identities confuse both; at 5812 RTMPose duplicates the left chain onto the raised right arm. | Counterexample: MediaPipe better at 5812; no evidence for a global L/R correction |
| V5:387 / 8149 | Extended elbow visible; RTMPose shifts it toward shoulder. | Counterexample: MediaPipe better |
| V4:343 / 18201 | Far elbow occluded. | Disagreement is ambiguous, not proof either estimator is correct |

A trigger requiring MediaPipe visibility **and** presence ≥.8 and inter-estimator distance >.3 torso lengths identified 13 joint slots on eleven frames across eight cases. This is a review trigger, **not an error rate**. The counterexamples above are why disagreement cannot be treated as truth. Details are recorded separately in `configs/pose_estimator_observations_v1.json`; none enter the human ledger.

V10:146/147 show broadly similar visible uppercut trajectories in both estimators, with uncertainty on the offhand. V10:142 shows a visible hook despite the frozen classifier predicting lead rather than rear hook. In the existing curated group-held-out results for seed 20261021, 12 of the 16 fully human-verified sampled cases are wrong. This error-enriched selection is not a population accuracy estimate. It shows that annotation confirmation alone has not solved transfer; visible movement can also be tracked reasonably yet classified incorrectly. No RTMPose classifier predictions were generated.

## Temporal behavior

Projected steps exceeding half the fixed median torso length, on identical MediaPipe-valid consecutive joint pairs:

| Recording | MediaPipe / RTMPose large steps | Joint pairs |
|---|---:|---:|
| V4 | 1 / 2 | 397 |
| V5 | 5 / 3 | 571 |
| V6 | 4 / 1 | 669 |
| V7 | 4 / 1 | 713 |
| V8 | 0 / 0 | 522 |
| V10 | 3 / 5 | 843 |
| Total | 17 / 12 | 3,715 |

These counts include context. Speed uses exact elapsed PTS; pairs across gaps, cuts, segments or nonpositive/over-.1-second deltas are excluded. A large step may be genuine punch motion, and lower variation can reflect quantization rather than better tracking. RTMPose has fewer large steps overall but more in V4/V10; there is no verified temporal-stability win. Neither output was smoothed or corrected. The stored MediaPipe extraction uses image mode, so its coordinates are not a temporally tracked ground truth.

## Alternative suitability and performance

[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) is a practical top-down alternative. This trial uses its body7 medium ONNX model with 17 COCO joints at 192×256 input. [rtmlib](https://github.com/Tau-J/rtmlib/blob/main/README.md) supports a lightweight OpenCV backend, avoiding a full MMPose/MMCV installation. Boxing-specific performance and occlusion accuracy are not established by general pose benchmarks.

[MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) supplies 33 landmarks and visibility/presence information. RTMPose's 17-joint scores are a different contract: it cannot be substituted for the existing 33-landmark confidence/mask arrays without explicit adaptation and calibration. We compared only shoulders, elbows, wrists and hips, retained raw scores separately, and chose no RTMPose validity threshold.

RTMPose received one fixed crop per case from the union of valid MediaPipe landmarks plus 15% margin; the wrapper adds its own affine padding. This is assisted localization using the whole interval, not a causal deployable detector. It may favor RTMPose, inherit MediaPipe crop errors, and hides the cost of detection. Both estimators still face inherently ambiguous occluded joints and 2D projection; neither establishes boxer stance or lead/rear semantics.

RTMPose preprocessing/inference/postprocessing measured **22.0 ms median, 30.9 ms p95** across 537 frames on this local CPU with two OpenCV threads, excluding decoding and person detection. The attempted fresh MediaPipe timing run aborted during macOS graphics-service initialization; no comparable MediaPipe speed number is available. Existing saved poses remained readable. This deployment issue needs a separate fix before any re-extraction and is not evidence of landmark quality. OpenCV/PyAV also emitted duplicate macOS AVF class warnings without affecting the successful RTMPose run.

## Reproduction and artifacts

No production dependency declarations changed. Isolated ignored dependencies: `rtmlib==0.0.14`, `tqdm==4.67.1`; existing OpenCV 5.0.0, NumPy, Pillow and PyAV were reused. The exported model pipeline explicitly specifies RGB normalization; Pillow RGB input was used (the wrapper does not convert colors).

Official model archive: [OpenMMLab body7 RTMPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip).
Local ONNX SHA256: `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`.
The module requires a local model and never downloads implicitly. After preparing those optional dependencies/model:

```sh
PYTHONPATH=.cache/pose-audit-deps .venv/bin/python -m boxing_motion_ai.vision.estimator_audit \
  --config configs/pose_estimator_audit_v1.json \
  --model weights/pose-audit/rtmpose-m-body7.onnx \
  --output reports/boxingvi/pose-estimator-audit-v1-rerun
```

Use a fresh output directory; existing outputs are refused. The original audit is under `reports/boxingvi/pose-estimator-audit-v1/`: per-case coordinates/masks/scores, exact-frame sheets, eleven enlarged frames, pinned inputs, JSON summaries and supplementary rendering/timing scripts. Original timing logs record the failed benchmark honestly. All generated images, arrays, weights and reports remain ignored by Git.

Synthetic tests cover common-joint correspondence, confidence/mask preservation, exact timing, gaps/cuts, invalid intervals and crop bounds. Verification: 442 Python tests passed (one existing Starlette/httpx deprecation warning); Ruff check passed; both new Python files pass Ruff formatting checks. The 44 targeted audit/curation regressions also passed. All 4,647 hashes and sampled human verification states were rechecked successfully after the audit. No changes were made to model architecture, classifier inputs, quality thresholds, the curated pool, human decisions or existing model results.
