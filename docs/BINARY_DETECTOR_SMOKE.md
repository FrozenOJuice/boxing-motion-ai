# First causal binary detector — training-only smoke test

2026-10-28. **The engineering smoke test passed.** The fixed 30-epoch CPU fit reduced weighted training loss from **0.931830 to 0.111951**. Gradients and parameter updates are nonzero/finite, checkpoint reload preserves logits exactly, and reviewed-frame inference agrees with the streaming reference within **2.98e-7 probability**. A second identical run reproduced feature hashes, every curve value, weights and checkpoint bytes exactly. These are training-fit/reproducibility checks, **not generalization results**.

## Frozen data and scope

The accepted [continuous dataset v1](CONTINUOUS_DATASET_V1.md) remains unchanged: 3 completed V4 punch intervals with 11 full human event records, plus 12 verified V4/V6 no-punch intervals. Both V4-bg-4 and V4-bg-5 are intentionally held cardio intervals. `configs/detector_smoke_v1.json` explicitly authorizes this bounded binary smoke test; it does not change the frozen dataset's historical training-authorization/readiness flags or authorize a six-class/evaluation study.

Optimization uses **V4 only**: the 3 completed punch intervals and 6 V4 background intervals. V6 background records are revalidated as part of the accepted dataset, but supply **zero optimization frames/features**. V5 remains development-only, V7/V8/V10 remain evaluation-only (V7/V8 grouped), V1/V6 grouping is preserved and V9 is never opened.

| Training inventory | Actual value |
|---|---:|
| Reviewed V4 intervals | 9 |
| Reviewed footage | 35.994 s |
| Known punch-active frames | 142 |
| Known no-punch frames | 721 |
| Total supervised frames | 863 |
| Unreliable-torso/missing-pose reviewed frames | 0 |
| Training windows | 27 |

The 721 negative frames include 576 in the six explicitly verified V4 negatives and 145 non-event frames inside explicitly exhaustive punch reviews. No source-workbook gap or unrelated footage becomes background. The rear uppercut ending at frame 22982 retains its full human bounds, while loss/inference stop at that reviewed interval's exclusive frame 22978. The four extra context frames are not supervised. All annotations, original review history and the full 12-background dataset remain intact.

## Implementation

`boxing_motion_ai.detection` is a small separate package reusing existing feature, mirror and temporal-model functions. The historical clip classifiers, their configurations/results and dashboard are unchanged. No new dependency, extraction, dataset candidate or annotation is introduced.

- **Source-bound loader:** replays the frozen dataset validator, verifies live human/source hashes, reads the original pose NPZ and validates its 33-joint order, confidence/validity/timing contracts. Source dimensions are read from the hash-verified source video's metadata (V4: 3840×2160), not an assumed or preview aspect ratio. Exact frame indices, integer PTS, rational timebases and source provenance remain available as metadata and never enter model inputs.
- **65-channel features:** existing hip-relative, torso-scaled 2D core joints, gated confidence/masks and backward-difference joint velocities using exact elapsed PTS. Invalid zeros are masked sentinels, never invented joints. Derivative pairs must be reliable and consecutive in one scene/segment. The existing source-clock gap rule is frozen from source timing (V4: 21/250 s); it is not estimated from labels or learned movement. No smoothing, velocity clipping, interpolation, future statistics, geometry or visibility augmentation is added.
- **Temporal boundaries:** windows contain up to 32 supervised core frames plus up to 14 real past warm-up frames. Loss is disabled for warm-up, unknown footage and right padding. Warm-up is required causal history, not the earlier optional clip-context augmentation. Missing/unreliable torso, missing pose, source cuts, segment/scene changes or timing gaps reset history; partly missing arms retain their remaining usable features and masks. Unobservable reviewed frames remain accounted for and receive null/abstained inference, never automatic No punch.
- **Model:** existing 32-channel `CausalTCN` encoder, three left-padded blocks with dilations 1/2/4, and one per-frame binary logit instead of clip pooling. It has **11,649 parameters** and a 15-frame receptive field. A bounded past-only `step()` reference retains at most 14 previous feature vectors. Future-perturbation, warm-up/chunk, padding and streaming tests verify causality.
- **Fit:** seed 20261021, CPU/two threads, deterministic Torch operations, 30 epochs, batch size 6, AdamW learning rate .001/weight decay .0001, dropout .1, gradient-norm clipping at 1, and the existing physical mirror transform at 50% during training only. Weighted BCE uses the observed V4 negative/positive ratio **721/142**; warm-up/padding/unknown labels have no loss. No recording IDs, absolute timestamps, event types or filenames enter features. No validation source, best-epoch selection or threshold search is used: the final fixed-budget epoch is saved.

Source joint-validity thresholds and historical clip policies are unchanged. The clip-level mask-aware70 aggregate rule is not silently repurposed as a continuous-window rule: this pilot explicitly uses the per-frame torso-normalization/missing-pose observability gate described above. It masks unobservable frames rather than borrowing labels from neighbors or rejecting a whole punch because one joint is hidden. There happened to be no such torso-level abstentions in these selected V4 reviewed frames; the missing-arm handling remains active and is tested synthetically.

## Actual smoke diagnostics

| Diagnostic | Result |
|---|---:|
| Optimizer steps | 150 |
| Initial weighted training loss | 0.9318304926 |
| Final weighted training loss | 0.1119507721 |
| Maximum gradient norm before clipping | 24.7796 |
| Parameter-change L2 norm | 1.82489 |
| Checkpoint reload maximum logit difference | 0 |
| Reviewed inference rows | 863 |
| Streaming/window maximum probability difference | 2.9802322e-7 |

At the fixed .5 threshold, the final **training-frame** confusion matrix is:

| Reviewed target | Predicted No punch | Predicted Punch |
|---|---:|---:|
| No punch | 685 | 36 |
| Punch | 5 | 137 |

These 36 false-positive **frames** are not 36 false-alarm events or a false-positive-per-minute result. The tiny, same-recording pool is readily memorized, and masks/confidence or movement-style shortcuts remain possible. No six-class performance, detection latency, event recall or held-out accuracy is claimed. Validation/test performance is unavailable, not zero. Current training progress demonstrates fitting and correct engineering, not a solved Level 1 detector.

Runtime: Python 3.11.17, Torch 2.14.1, NumPy 2.4.6, CPU. The captured training provenance includes the prior Git revision/dirty flag plus exact implementation hashes; the local commit records the completed implementation. No result metadata is rewritten after commit to pretend a different source snapshot trained the model.

## Reproduce and inspect

Existing project extras `dashboard`, `pose` and `train` supply the dependencies; none was downloaded for this task. Run from the repository root with the prepared local data/reviews intact:

```sh
.venv/bin/python -m boxing_motion_ai.detection --help

.venv/bin/python -m boxing_motion_ai.detection smoke \
  --config configs/detector_smoke_v1.json \
  --output reports/boxingvi/detector-smoke-repeat

.venv/bin/python -m boxing_motion_ai.detection infer \
  --run reports/boxingvi/detector-smoke-v1-verified \
  --output reports/boxingvi/detector-smoke-inference-repeat
```

Output directories must be new and under ignored `reports/`; previous runs cannot be overwritten. `uv run python -m ...` is equivalent in the existing synchronized environment. This module CLI leaves the existing `boxing-motion-ai` CLI unchanged.

The verified run is `reports/boxingvi/detector-smoke-v1-verified/`: `config.json`, `curves.json`, `summary.json`, `checkpoint.pt`, `checkpoint.sha256`, and `inference-v4/{frames.jsonl,receipt.json}`. The checkpoint SHA256 is `b24f527e6f50a883a1c2bd6ae8c054f029aefba07ce2f23ee8d2acbe1655fd63`. A fresh repeat under `detector-smoke-v1-replay/` reproduced its checkpoint and curves byte-for-byte. The earlier pre-verification run under `detector-smoke-v1/` is preserved; its older checkpoint lacks the finalized feature-binding metadata and is not the command's reference checkpoint.

Each inference row carries source recording/interval/frame ID, integer PTS/timebase, exact rational source time, Punch probability or explicit abstention, fixed-threshold decision and a separately named reviewed target for fit diagnostics. Absolute metadata and targets are not model inputs. Checkpoint digest, feature-version and frozen-dataset binding are verified before weights-only loading. Registered inference scope is V4/V6 reviewed intervals only; the optional V6 command is diagnostic inference, never training. V6 inference was not run here. This is cached-pose inference, not a new RGB/pose extraction service or measured live-video throughput.

## Event metric infrastructure

`detection.metrics` accepts exact source-time reference events, predicted events with true first-emission times, and declared exhaustive scored intervals. It implements deterministic **maximum-cardinality then maximum-total-IoU** one-to-one assignment at a versioned exact IoU threshold (default 1/2). Duplicate alarms count as false positives; unmatched references count as misses. It reports event precision/recall/F1, FP per fully reviewed minute, matched latency median/p90 and explicit match details.

It refuses zero/overlapping exposure, duplicate IDs, missing emission times and events crossing scored boundaries or unknown regions. Empty precision/recall/latency are undefined (`None`), not invented perfect scores. Unknown time is never included in the denominator. A future scorer must explicitly declare its edge/ignore policy for partial events and must not discard missed references during pose abstention. Backdating an onset estimate cannot backdate first-emission latency.

Synthetic tests verify assignment (including greedy traps), duplicates, misses, cross-recording separation, IoU preference, latency and exposure errors. **No meaningful event evaluation was run.** Frame probabilities still need a development-fixed causal alarm decoder and a frozen edge policy before event scoring; the training .5 frame threshold is only a transparent diagnostic default.

## Next blockers and bounded next step

The smoke implementation is complete. Further engineering can continue without asking for labels on V4-bg-4/5 or re-reviewing the accepted backgrounds. The genuine blockers to a meaningful detector study remain a trustworthy continuous V5 development reference, exhaustive V7/V8/V10 evaluation references, more positive recording diversity and missing cross/rear-hook training support. These are limitations to report, not work requested from the user now.

Next, after review, define the small causal event decoder and its first-emission/end/censoring protocol, preserving the binary model. Do not tune thresholds on training clips or randomly split V4 frames into a supposed independent validation set. Keep the V6 negative-only pool outside initial optimization. Before claiming generalization, use the reserved recording groups with a separately authorized reference/evaluation protocol; V9 stays completely untouched. No additional experiment or model improvement is started automatically.

## Verification and preservation

**529 Python tests**, **76 frontend regression tests**, Ruff lint, formatting of all changed Python files, module CLI help and diff whitespace checks pass. The existing Starlette/httpx deprecation warning remains. No frontend code or build configuration changed.

Twenty-six new synthetic tests cover exact timing and source-array preservation, unknown/warm-up/padding loss masks, missing joints/torso, scene/segment/cut/gap resets, causality, streaming/chunk equivalence, binary gradients, event assignment/exposure/latency edge cases, rejected training-source expansion, checksum-bound weights-only checkpoint inference and null abstentions. Real V4 loading, training, reload, inference and deterministic replay were verified separately.

All **4,770 pre-change protected dataset, review, backup and historical-artifact hashes** remain unchanged. The protection inventory and replay receipt are under ignored `reports/boxingvi/detector-smoke-validation-v1/`. Dataset v1, all accepted human reviews/backgrounds, original media/annotations/poses and earlier experiments/checkpoints were preserved. V9 was not opened. Trained weights, features/metadata receipts, curves and inference outputs remain excluded from Git. No model training beyond this fixed engineering smoke/replay was performed.
