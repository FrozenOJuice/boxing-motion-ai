# Causal binary punch-event decoding

2026-10-28. Implemented timestamp-based streaming decoding around the accepted frozen binary detector. **No training, architecture changes, new datasets or human reviews.** The fixed engineering demonstration produced **4 V4 activity-event predictions and 0 on the six selected V6 negative clips**. These counts are not meaningful precision/recall or false-positive-rate estimates, and activity bouts are not guaranteed individual punch instances.

## Fixed policy and causality

`configs/decoder_v1.json` was set before the demonstration; no sweep or optimization against training labels was performed:

| Setting | Value |
|---|---:|
| Activation probability | ≥ 0.65 |
| Deactivation probability | ≤ 0.35 |
| Minimum activation | 2/25 s (80 ms) |
| Sustained-low termination | 2/25 s (80 ms) |
| Maximum adjacent observation gap | 1/10 s (100 ms) |

Durations are exact rational **seconds**, applied to original integer PTS/timebases. There is no assumed annotation FPS or fixed number of frames. Sampling can delay a decision by up to the next observed frame; this does not make the learned model frame-rate invariant. Synthetic 24/25/30/60 FPS tests verify the elapsed-time policy.

`CausalEventDecoder.push()` consumes one observation and immediately returns any activation/finalization updates:

1. A candidate starts at the first high-threshold crossing. It must persist above the low threshold for the minimum activation time; an earlier low crossing cancels it. Middle-band probabilities retain the candidate/activity state.
2. Activation is emitted on the **current observation** when the duration is met. The estimated start remains the earlier candidate frame. `first_emission` and `activation_delay_seconds_exact` preserve this distinction; the onset is never substituted for emission time.
3. Once active, a low run shorter than the termination duration is tolerated. If the signal recovers early, that observed dip becomes part of the same activity span. A recovery at/after the deadline ends the old event before considering a new candidate. Short observed gaps can therefore merge; sufficiently long valleys separate consecutive events.
4. End is estimated at the first low frame, excluding the trailing low run. The later actual decision observation is retained in `finalized_at`. The last included frame, its exact PTS, the exclusive stop frame and known exclusive timestamp are separate fields.

No future probability is read, and emitted activation updates are immutable snapshots. Final confidence summarizes observations available by finalization; `confidence_at_emission` in the activation log records what was available at first emission. Confidence mean/peak are raw model-score summaries, explicitly **uncalibrated**.

Batch decoding is an adapter over the same streaming state machine. I/O/model chunk boundaries never flush state. The inference demonstration independently replays arbitrary 13-row chunks and requires exact equality of all decoder events/updates with the batch adapter. Frozen model streaming/window probability agreement is also retained as a separate tolerance check.

## Discontinuities, scope and edges

Missing/unreliable pose, unknown/out-of-scope observations, explicit model resets, scene/segment changes, skipped frame indices and excessive timestamp gaps interrupt and censor active events immediately. Candidates without sufficient observed activation are discarded, not converted into negative labels. Repeated/backward observations are rejected; a new stream/scope must be explicitly started. Nothing is interpolated or silently joined across a discontinuity.

The inference harness remains restricted to the existing registered reviewed intervals. The `reviewed` scope flag is checked, while punch labels, source annotation types and `reviewed_target` are ignored by decoding. Unknown scope interrupts the bounded demonstration; it is not evidence of background or an automatically reviewed negative. This is not a new full-video inference service.

- `left_censored=true` indicates activation at an observation boundary or after an unresolved middle-band prefix without a prior inactive observation. It does not certify the true physical onset.
- `right_censored=true` indicates an interrupted/unresolved end. At EOF, an explicit end-of-coverage marker may supply the registered exact exclusive frame/PTS; this is metadata/control, **not a future video observation** or proof that the punch ended there.
- Without a known exclusive endpoint (for example, a timestamp/frame gap), `stop_timestamp_exact` can remain null. The last actually included frame/time is retained. No nominal-FPS extrapolation fills it.
- `finalized_at.clock_basis` distinguishes a current source observation, an explicit coverage-end control and a last-observed source-clock anchor. Source clocks from different recordings are never combined. These are source-time decisions, not measured wall-clock capture/compute latency.

The event includes recording/interval IDs, estimated source bounds, first emission, finalization/censoring information and confidence. Prediction records explicitly have `source=model_prediction`. They are stored separately and never submitted to the manual annotation/review system.

## Integration and reproduction

Run from the prepared repository environment, choosing fresh ignored output directories:

```sh
.venv/bin/python -m boxing_motion_ai.detection infer \
  --run reports/boxingvi/detector-smoke-v1-verified \
  --recording V4 \
  --decoder-config configs/decoder_v1.json \
  --output reports/boxingvi/decoder-repeat-v4
```

The same command with `--recording V6` runs only the already reviewed negative intervals. V5 is development-only, V7/V8/V10 evaluation-only and V9 remains inaccessible. Neither V6 inference nor its results enter optimization or threshold selection. Both V4-bg-4/5 remain excluded. The CLI defaults to the tracked decoder policy; the Python API can explicitly provide a config path or use identical typed defaults.

Outputs:

- `frames.jsonl`: the existing frozen frame probabilities and .5 diagnostic decisions, extended with reviewed scope, scene/segment and discontinuity metadata. The .5 frame decision is **not** the event decoder's hysteresis policy.
- `events.json`: final structured predictions, causal activation/finalization update history, discarded-candidate count, exact decoder configuration/code hash, checkpoint and dataset hashes. It does not contain new human labels or evaluation metrics.
- `receipt.json`: existing inference checks plus decoder config hash, event count, exact streaming/batch agreement and event-file hash.

The authoritative demonstration outputs are `reports/boxingvi/causal-decoder-v1-v4/` and `reports/boxingvi/causal-decoder-v1-v6/`. Earlier verification outputs remain preserved under `decoder-demo-v1-*`. The decoder's finalized policy/source, not the model, changed while synthetic edge-case tests were completed; no thresholds were tuned using these outputs.

Checkpoint SHA256 remains `b24f527e6f50a883a1c2bd6ae8c054f029aefba07ce2f23ee8d2acbe1655fd63`; dataset v1 SHA256 remains `c7993318ee8f8a16e2cd89f04ad0acaf7ce95c0cad8aabad0c70079b683b2d59`. Decoder config SHA256 is `e485fd922144311c73b034a2353e927eb2ee4aa62d09b6df220940b9b41db28f`. All original V4 frame-probability fields compare exactly equal with the preserved smoke inference. No fitted model or dataset was rewritten.

## Demonstration observations, not evaluation

V4 covers its 9 reviewed training intervals (863 frames). V6 covers 6 selected negative-only intervals (575 frames). No model-observability gaps occurred in these selected rows; discontinuity/missing/unknown behavior is exercised synthetically.

| V4 interval | Estimated native [start, stop) | First emission frame | End handling | Peak / mean score |
|---|---|---:|---|---:|
| V4-bg-1 | [17121,17180) | 17123 | Left censored; sustained-low end | .984 / .850 |
| V4-bg-1 | [17201,17217) | 17203 | Right censored at interval end | .919 / .842 |
| V4-bg-2 | [22926,22978) | 22928 | Right censored at interval end | .990 / .948 |
| V4-bg-3 | [11438,11487) | 11440 | Right censored at interval end | .994 / .915 |

There are no predictions on the six V4 verified-background intervals, and none on the six selected V6 negatives. Do not turn this small, selected sample into a specificity, false-positive-rate or generalization claim.

Obvious failure patterns:

- **Merged punches:** the first V4-bg-1 prediction spans its three human events; V4-bg-2 and V4-bg-3 each produce one broad event over four annotated punches. Back-to-back binary-positive targets can have no inactive frames between punch annotations. A single activity probability plateau cannot uniquely identify individual punch boundaries, and hysteresis alone cannot recover that missing signal. No arbitrary maximum-duration split is invented to force the predicted count to eleven.
- **Extra activation:** the second V4-bg-1 event is in an exhaustively reviewed non-event tail. Its high raw score demonstrates why confidence summaries are not calibrated event correctness.
- **Early broad activation:** V4-bg-3 activates at frame 11438, twelve frames before its first human punch at 11450. This is an erroneous/overbroad activity start, **not an anticipation result**.
- **Unresolved ends:** three predictions remain active at the review boundary and are right-censored. In particular V4-bg-2's final human punch extends beyond that boundary; no unreviewed extension is scored or assumed inactive.

No event matcher was run to advertise precision, recall, FP/minute or latency on these training examples. Synthetic tests validate the machinery; the demonstration validates engineering behavior and exposes limitations.

## Remaining evaluation requirements

Before meaningful event scoring, freeze what counts as an individual punch versus an activity bout, the emission/finalization protocol and a declared policy for left/right-censored and unknown regions. Never silently drop difficult edge predictions or pose-abstained references from denominators. Then use separately authorized continuous development/evaluation references, keeping V5 for development and V7/V8/V10 recording-group-separated for evaluation; V7/V8 remain together. Those sources have already informed previous development, so do not call them a pristine independent test or boxer-independent evaluation. V9 stays untouched.

Separating touching punches may require a distinct onset/instance-boundary signal in a later controlled experiment, rather than retuning these thresholds on the current tiny training set. Existing human event boundaries can inform a later approved target design; no new labels, targets, models or dataset versions are generated now. No further manual annotation, including either rapid cardio interval, is requested. Stop here for review.

## Verification and preservation

**561 Python tests** pass, including 32 new synthetic decoder tests and the existing frozen-checkpoint inference/event-metric regressions. Ruff, changed-Python formatting, CLI help and whitespace checks pass. The existing Starlette/httpx deprecation warning remains; no frontend code changed.

Tests cover source-time durations across frame rates, activation/termination thresholds, recoveries at the termination deadline, brief fluctuations, short/long gaps, causal prefix immutability, confidence history, missing/unreliable/unknown observations, source cuts/segments/frame gaps/time gaps, recording/interval changes, censoring, invalid inputs, exact scope endpoints and arbitrary chunk replay. The real frozen-model demonstration requires exact decoder streaming/batch equality; model streaming/window differences stay below 3e-7.

All **4,787 protected file hashes** match, including the accepted model checkpoints, frame outputs, dataset/review files and previous experiment artifacts. The snapshot is under ignored `reports/boxingvi/decoder-validation-v1/protected-inputs.json`. Generated demonstrations remain ignored. No model was retrained, no architecture changed, no annotation or dataset was created/modified, and no V9 file was accessed. V4-bg-4/5 stay held. The implementation stops here for review.
