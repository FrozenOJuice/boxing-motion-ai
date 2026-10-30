# Learned localization: frozen-score boundary decoder study

Starting revision: `13e57d7`. **Outcome: both learned evidence and pairing are limiting.**
V2 corrects a demonstrated stale-onset error, raising fused V6 matching from 8/11 to 9/11.
It does not improve V5 matching: pose/RGB/fused remain **0/8, 2/8, 0/8**. Fused has no supported
thresholded boundary pair overlapping any V5 reference at IoU ≥ .5. Decoder changes alone,
with the existing evidence gates, cannot repair that failure. Conversely, available training
boundary evidence is often competing rather than absent. This is a negative development result,
not a model selection or deployment recommendation.

No training, model inference, feature extraction, dashboard change, six-class inference or
production promotion was performed. The existing decoder, segmentation V2, models, scores,
datasets, reviews and checkpoints remain intact. This experimental **boundary decoder V2** is
different from the earlier frozen **arm-motion segmentation V2**.

## Evidence and controlled protocol

Replayed the exact saved activity/onset/end arrays from all three blue-reference checkpoints
in `reports/boxingvi/punch-localization-blue-v1/`. Checkpoint bytes were verified against
their saved hashes; weights were not deserialized or executed. Inputs remain the same 27 native
contexts: original diagnostics, three previously tested uploads, blank control, two sparse V4
anchors, five V6 blue intervals and one V5 development interval.

Reference scope is unchanged from [the blue-reference experiment](PUNCH_LOCALIZATION_BLUE_V1.md):

- Original V4: 11 events in three exhaustively reviewed cores, with historical core-clipped
  matching and full matched-boundary errors retained for comparability.
- Sparse V4: two uncontested anchors, with full candidate ranges scored without clipping.
- V6: 11 fully scorable selected blue events. Three other selected events retain conflicting
  endpoints and remain unscored full instances. Excluded manual events and gaps stay Unknown.
- V5: eight selected positive references only, development-only, incomplete background inventory.
  This is recording-separated development, not an independent final test.
- Twelve original explicitly reviewed V4/V6 negative intervals: 48.006 seconds, training-source.

V4 and the V1/V6 source group remain training-only. V7/V8/V10 were not decoded or evaluated;
V9 was not accessed. Inherited preservation inventories can hash existing reserved artifacts,
without interpreting their contents as development evidence.

Policy development used synthetic contracts and V4/V6 scores only. The previous report already
described V5 failures; that context was known, but no new V5 score inspection or trial informed
policy design. The replay tool skips V5 before opening its cache metadata, references or scores
in training mode. The single fixed policy, code, synthetic tests and training reports were frozen
at **2026-10-30 23:06:33 UTC (17:06:33 MDT)**, before running the V5 comparison.

Freeze: `reports/boxingvi/boundary-decoder-v2-freeze-v1.json`, SHA256
`e45ee9dc9e66ebf6360a2a7f4ac088e32e2dd93b1742092d605de129ff3a768b`.
Development mode verifies that binding before opening V5. No policy adjustment followed V5.

## Diagnosis of decoder V1

V1 accumulates onset peaks until the next end, then clears the entire pending list, even when
the pair is invalid. There is no onset expiry or predicted-inactivity reset. Consequently:

1. A stale onset can persist through quiet footage and make a later valid pair ambiguous.
2. A too-short first ending can consume an onset before a compatible ending arrives.
3. Multiple onsets are rejected together without checking whether duration/activity gates rule
   out one competitor. Conversely, an accepted first ending need not be uniquely supported.
4. Noisy boundary heads can generate short fragments; treating every alternating peak as a
   punch cannot solve this. Missing endings cannot be repaired by pairing alone.

Across the 26 training-source/control cases, V1 reports 51/46/36 `multiple_onsets_before_end`
records for pose/RGB/fused. These are overlapping diagnostic records, not punch counts. Among
them, 1/5/5 spans cross a qualifying predicted quiet interval or exceed the two-second pair
limit. Dense spurious peaks and weak boundaries remain separate model-evidence issues.

For each reviewed reference, the study also enumerates unchanged thresholded peaks within the
same observable run and applies the original duration/activity gates. At least one such pair
can locally overlap each of the 13 V4 and 11 V6 scorable references at IoU ≥ .5 for every arm.
This is **label-assisted local feasibility**, not a jointly attainable oracle, decoder input,
or evidence that all boundaries are learned correctly. Several references can compete for
the same peaks; accidental matches can exist in noisy predictions. It establishes that the
training misses cannot all be attributed to a complete absence of possible endpoint pairs.

## Separately versioned offline policy

Implementation: `localization_decode_v2.py`; fixed contract:
`configs/localization_decoder_v2.json`, version `offline-boundary-pair-decoder-2.0`.
The original `localization_decode.py` is unchanged.

- Reuse the original peaks and shared observation/reset policy exactly. Activity and boundary
  thresholds stay .5; peak radius .08 s; pair span .08–2 s; duration-weighted activity support
  at least .5. No thresholds are lowered and no new peaks are manufactured.
- Reset pairing across predicted low activity lasting at least .25 s, borrowing the existing
  short-pause scale from activity grouping. Record these as predicted inactivity, **not verified
  background**. Onsets cannot pair beyond the existing two-second span; unsupported evidence
  is explicitly expired, interrupted or censored.
- Group consecutive end peaks into ending phases separated by actual onset evidence. For
  each onset, consider its first compatible ending phase. An invalid early end does not consume
  the onset. Competing feasible edges form a bipartite graph: only isolated one-onset/one-end
  components emit candidates. There is no maximum-count or score-based assignment optimization.
- Coincident onset/end peaks, competing assignments and overlapping unique pairs remain
  unresolved. No midpoint, fixed-duration subdivision or classifier score invents a boundary.
- Preserve unobservable ranges, source discontinuities, observation-edge censoring and uncovered
  predicted activity. Audit rejected edges and quiet resets separately.

This is deliberately conservative and **offline**: it inspects ending-phase alternatives and
predicted quiet spans using future evidence. The unchanged model has centered RF15 context,
and peak suppression is future-aware too. No causal first-emission or live latency claim is
made. End peaks refer to the last included frame; ranges are native **[start, stop)** with
stop at the next source PTS. Context frame indices remain local; stored provenance supplies
the recording source-frame offset. No assumed FPS or guessed frame offset is used.

## Exact replay comparison

Matching uses the existing one-to-one IoU ≥ .5 policy. `Extra` means unmatched proposals:
on sparse V4/V6/V5 it does **not** mean false positives. `Split/merge` are inherited overlap
risk flags, not manually established error counts. Zero flags do not prove good segmentation.
Arrows below are original decoder → experimental decoder on identical scores.

| Source / arm | Candidates | Matched | Missed | Extra | Split risks | Merge risks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original V4 / pose | 1 → 2 | 1 → 1 | 10 → 10 | 0 → 1 | 0 → 0 | 0 → 0 |
| Original V4 / RGB | 7 → 3 | 1 → 1 | 10 → 10 | 6 → 2 | 1 → 0 | 0 → 0 |
| Original V4 / fused | 8 → 6 | 5 → 5 | 6 → 6 | 3 → 1 | 0 → 0 | 0 → 0 |
| Sparse V4 / pose | 2 → 0 | 1 → 0 | 1 → 2 | 1 → 0 | 0 → 0 | 0 → 0 |
| Sparse V4 / RGB | 4 → 6 | 1 → 1 | 1 → 1 | 3 → 5 | 0 → 0 | 0 → 0 |
| Sparse V4 / fused | 5 → 3 | 1 → 1 | 1 → 1 | 4 → 2 | 0 → 0 | 0 → 0 |
| V6 / pose | 13 → 14 | 6 → 6 | 5 → 5 | 7 → 8 | 1 → 1 | 0 → 0 |
| V6 / RGB | 30 → 21 | 2 → 2 | 9 → 9 | 28 → 19 | 5 → 2 | 0 → 0 |
| V6 / fused | 12 → 12 | 8 → 9 | 3 → 2 | 4 → 3 | 0 → 0 | 0 → 0 |
| V5 / pose | 2 → 2 | 0 → 0 | 8 → 8 | 2 → 2 | 0 → 0 | 0 → 0 |
| V5 / RGB | 14 → 14 | 2 → 2 | 6 → 6 | 12 → 12 | 1 → 1 | 0 → 0 |
| V5 / fused | 0 → 0 | 0 → 0 | 8 → 8 | 0 → 0 | 0 → 0 | 0 → 0 |

### Concrete corrections and regressions

- Fused V6-motion-101 previously kept onset 551 through inactivity and rejected the later onset
  566 with ending 575. Resetting stale evidence recovers **[566,576)** against human [566,577),
  IoU 139/153, onset error zero, stop error −1 frame/−42 ms. [541,550) remains exact.
  Fused's remaining misses are [6468,6485) and [30594,30616); ambiguity is not resolved by fiat.
- Pose V6 gains the [541,550) reference but loses [30509,30532); its unchanged aggregate match
  count hides this exchange. Original V4 pose also exchanges one matched event for another.
  Per-event recovered/lost IDs and source ranges are retained in diagnostics/records.
- Pose V4:144 loses the old [10407,10411) match to human [10404,10411). Multiple compatible
  endpoints now make that component unresolved. This is a genuine matching regression.
- RGB's verified-background proposal [20914,20922) in V6-motion-5 becomes unresolved because
  there are two compatible end alternatives. The 12 negative intervals now have 0/0/0 proposals
  for pose/RGB/fused, versus 0/1/0 previously. This is training-background behavior only.
- RGB V6 split risks fall 5 → 2, but nine selected references remain missed. Fewer proposals
  cannot be interpreted as better recall or precision on incomplete reference inventories.

### Boundary accuracy and unresolved evidence

Errors below are conditional on matched instances, so changing the matched set changes the
population. `Median start / stop` and `max start / stop` are absolute errors in native frames.

| Source / arm | Median start / stop, V1 → V2 | Max start / stop, V1 → V2 | Unique unresolved seconds, V1 → V2 |
| --- | --- | --- | ---: |
| Original V4 / pose | 0 / 4 → 2 / 1 | 0 / 4 → 2 / 1 | 12.054 → 11.053 |
| Original V4 / RGB | 1 / 0 → 1 / 0 | 1 / 0 → 1 / 0 | 9.013 → 10.762 |
| Original V4 / fused | 0 / 0 → 0 / 0 | 1 / 1 → 1 / 1 | 7.171 → 7.923 |
| Sparse V4 / pose | 3 / 0 → unavailable | 3 / 0 → unavailable | 4.130 → 4.713 |
| Sparse V4 / RGB | 0 / 0 → 0 / 0 | 0 / 0 → 0 / 0 | 2.876 → 2.752 |
| Sparse V4 / fused | 0 / 0 → 0 / 0 | 0 / 0 → 0 / 0 | 3.461 → 4.170 |
| V6 / pose | 1 / 1 → 0 / 1 | 6 / 4 → 6 / 4 | 28.573 → 25.235 |
| V6 / RGB | 4 / 4 → 4 / 4 | 8 / 7 → 8 / 7 | 22.150 → 24.527 |
| V6 / fused | 0 / .5 → 0 / 1 | 0 / 1 → 0 / 1 | 24.190 → 21.065 |
| V5 / pose | unavailable | unavailable | 4.671 → 5.071 |
| V5 / RGB | 1.5 / 5 → 1.5 / 5 | 2 / 7 → 2 / 7 | 9.244 → 10.146 |
| V5 / fused | unavailable | unavailable | 9.942 → 3.771 |

Unresolved seconds union overlapping records within each case, then sum cases; they are not
independent exposure or punching duration. Expiry changes how long unsupported evidence is
retained, so fused V5's reduced unresolved duration is **not** recovered punches or evidence
of verified background. Unobservable support is identical: V6 2.253 s; V5 .033 s per arm.

### V5 evidence limits: why pairing is insufficient

Boundary support is diagnosed within the existing supervised radius ±1/12 s of the true
first/last included frame, using the unchanged .5 threshold. Scores are uncalibrated.

| Arm | References with some supported pair at IoU ≥ .5 | Weak near-onset evidence | Weak near-end evidence | Actual matches V1 → V2 |
| --- | ---: | ---: | ---: | ---: |
| Pose | 4/8 | 3/8 | 5/8 | 0 → 0 |
| RGB | 4/8 | 4/8 | 4/8 | 2 → 2 |
| Fused | 0/8 | 4/8 | 8/8 | 0 → 0 |

Tags overlap and are not mutually exclusive missed-punch attribution. Pose also has one onset
and two ending peak-displacement/suppression tags. A weak nearby score can coexist with a
broader pair meeting IoU; these are distinct diagnostics. The local feasibility calculation
does not impose V2 phase uniqueness and cannot be interpreted as a promised decoding ceiling.

Fused's eight near-end maxima are .006, .002, .032, .002, .009, .002, .004 and .001 in source
reference order. Its global end maximum .842 occurs elsewhere and does not supply useful
end evidence for these punches. A pairing rewrite cannot synthesize the absent endings.
Pose/RGB retain some locally feasible but competing/displaced evidence that conservative
pairing still cannot resolve. This is not solely missing-person abstention: all selected V5
boundaries pass the common observation gate; validity does not establish landmark accuracy.

RGB's same two matches remain [8231,8240) against [8230,8247), with +1/−7 frame errors, and
[8402,8410) against [8400,8413), with +2/−3. The first combination still fragments into
[8231,8240), [8241,8245), [8246,8255). Fused emits no merged punch to conceal its unresolved
combinations. V5 extras are unlabeled, not established false positives; no full precision,
full recall, false-positive rate or independent-test accuracy is reported.

### Previously tested uploads

Candidate counts only; no exhaustive upload reference inventories. V6 uploads are from the
training recording, not unfamiliar-recording validation.

| Arm | V6 six-second upload | V4 eight-second upload | V4 29-second upload |
| --- | ---: | ---: | ---: |
| Pose | 1 → 1 | 2 → 4 | 4 → 1 |
| RGB | 1 → 0 | 1 → 0 | 20 → 16 |
| Fused | 2 → 2 | 3 → 2 | 4 → 4 |

The blank control retains zero candidates and 25 unobservable frames for every arm.
No counts are presented as verified individual punches.

## Verification and retained artifacts

- **730 Python tests**, including 14 focused synthetic/contract tests, and **111 frontend tests**
  pass. Ruff across `src tests`, changed-Python formatting, TypeScript, frontend Prettier and
  production build pass. One existing Starlette/httpx deprecation warning remains.
- Synthetic cases cover stale/expired onsets, inactivity, invalid first endings, competing
  onsets/ends, continuous-high-activity combinations, coincident and overlapping evidence,
  missing observations, discontinuities, censoring, weak/no evidence, exact rational clocks,
  deterministic replay and the pre-V5 freeze gate. They establish mechanics, not accuracy.
- All **81** original-decoder case/arm results equal historical saved outputs exactly.
  Both decoders use identical peak arrays. New decoding repeats exactly; scores are unchanged.
- Original source clocks and offsets were checked against raw pose caches, receipts and native
  frame maps. Every emitted V2 candidate endpoint is an original peak with exact native PTS.
- **9,363 pre-existing artifact files remain byte-identical**, including reviews, datasets,
  checkpoints, sources and prior experiments. Live blue-subset provenance is revalidated.

Git-ignored audit artifacts:

- `reports/boxingvi/boundary-decoder-preflight-20261030T225836Z/`: protected inventory and initial
  V4/V6-only diagnosis.
- `reports/boxingvi/boundary-decoder-v2-training-v1/`: both replay versions, per-reference evidence,
  unmatched proposals, pairing reasons, boundary errors, summary and input hashes.
- `reports/boxingvi/boundary-decoder-v2-freeze-v1.json`: policy/code/test/training-output binding.
- `reports/boxingvi/boundary-decoder-v2-development-v1/`: the post-freeze V5 comparison.
- `reports/boxingvi/boundary-decoder-v2-verification-v1.json`: preservation and timing checks.

Reproduce with fresh output paths; commands replay saved scores, never fit a model:

```sh
.venv/bin/python -m boxing_motion_ai.detection.localization_decoder_study \
  --phase training --output reports/boxingvi/decoder-training-REPEAT
.venv/bin/python -m boxing_motion_ai.detection.localization_decoder_study \
  --freeze-training reports/boxingvi/decoder-training-REPEAT \
  --output reports/boxingvi/decoder-freeze-REPEAT.json
.venv/bin/python -m boxing_motion_ai.detection.localization_decoder_study \
  --phase development --freeze reports/boxingvi/decoder-freeze-REPEAT.json \
  --output reports/boxingvi/decoder-development-REPEAT
```

## Next controlled experiment

Prioritize **source/event-balanced boundary supervision with the unchanged representation and
TCN**, under a predeclared budget. Compare the current loss with per-source/per-event-normalized
head contributions, preserving truthful Unknown/conflict masks. Report V4/V6 separately and
measure near-boundary end evidence, spurious peaks and instance matches, rather than only
activity fit. Use the frozen decoders as diagnostics, not a V5 parameter-search loop. Keep the
fixed final-epoch policy and do not select a model solely on this training fit or eight V5 hits.

The demonstrated reason is weak/misplaced development end evidence alongside dense competing
training peaks and V4 regression after adding V6. This study does not establish that representation
is adequate or that loss balancing will work. Rate-dependent receptive field and redaction's
removal of useful motion remain specific representation hypotheses for a later controlled
ablation if boundary supervision still fails. Another decoder search or larger architecture is
not justified by these results alone. No new annotation or training is initiated here.
