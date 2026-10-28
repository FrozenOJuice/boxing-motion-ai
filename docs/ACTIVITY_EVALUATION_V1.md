# Continuous punching-activity evaluation v1

Status: protocol frozen before held-out binary predictions; dashboard ready for six human
reference reviews with unverified source-annotation suggestions. No detector training, held-out
inference, threshold selection or final scoring was performed for this milestone. All six references are currently incomplete.

## What is being detected

An **activity bout** is a visible period of committed punching movement, potentially containing
several consecutive punches. It is not an individual punch and has no six-class punch type.
Guarding, preparatory movement, footwork and defense alone are not punching activity. Mark the
first visible committed punching-action frame, and the last included frame after the final
punch excursion/return. Keep pauses **shorter than 0.25 seconds** within one bout; pauses of
0.25 seconds or longer separate bouts. Use exact source PTS, not nominal FPS, for this rule.

When a boundary or movement is unclear, mark an **uncertain section**, including the unclear
boundary region if necessary. It may overlap an activity interval. Do not force a precise
boundary or infer background. A completed review can retain uncertainty. For this bounded v1,
a completed block containing any uncertain section is omitted **entirely** from primary
scoring, with its duration and reason reported. This conservative rule avoids turning unclear
boundary neighborhoods into negative labels or silently changing the matching exposure.

Intervals are zero-based, half-open `[start, stop)` in storage. The viewer's End control marks
the last included frame, so stored stop is that frame plus one. Activity already underway at
block start is left-censored; activity still underway at block end is right-censored. Mark
only its visible support inside the block, without guessing its physical onset or ending.

## Fixed evaluation rules

[configs/activity_eval_protocol_v1.json](../configs/activity_eval_protocol_v1.json) pins the
interval manifest, smoke checkpoint, decoder configuration and existing matching implementation
by SHA-256. Each manual reference revision is bound to this protocol hash. Changing a pinned
input fails closed; revisions never silently rebind to a different protocol.

- Use the frozen detector and decoder settings; no tuning against these blocks. Causally group
  decoder events with gaps shorter than 0.25 seconds into activity bouts, within the same
  contiguous block/segment only. Never bridge unknown, cut, missing or unreliable-observation
  resets. Preserve the first segment's estimated onset and first emission, and the last
  segment's end. Finalize after the silence rule or censor at the block boundary. This grouping
  is specified for the future evaluator; this milestone does not run it or change the decoder.
- Clip references and predictions to each scored block's visible support. Include censored
  events in windowed event precision/recall; do not discard censored predicted false alarms.
  This measures visible bouts, not complete physical punch instances.
- Match one-to-one with timestamp intersection-over-union **at least 0.5**, maximizing matching
  cardinality first, then total IoU, with deterministic ties. Overlap alone is insufficient.
  Duplicate alarms cannot all match the same reference. Matching never crosses blocks.
- Report event precision, recall, F1, matched/unmatched counts and **false positives per minute**
  of full clean, exhaustively reviewed block duration. Pose abstentions remain in the denominator
  and missed reference events remain misses. Unreviewed/uncertain blocks contribute no exposure.
- Report signed **first-detection latency** = first source-time emission minus reference onset,
  only for matched references with uncensored starts. Include sample counts and misses; report
  median and range. Negative values are early alarms, not evidence of anticipation. Estimated
  onset is separate from emission; neither measures wall-clock processing latency.
- Empty denominators produce undefined (`null`) metrics, never invented perfect scores.
  Report boundary-censored counts and excluded duration alongside primary results.
- Report pooled results and separate V7/V8 and V10 groups. No boxer-independent claim is valid
  while identities remain unknown. These recordings were used diagnostically in earlier
  classifier studies: this is held-out **binary detector pilot** evaluation, not a pristine
  independent final project test.

The six blocks are V7-eval-1/2, V8-eval-1/2 and V10-eval-1/2: **179.526 seconds** total.
V7/V8 stay in one source group. They and V10 remain evaluation-only; V5 remains development-only,
V1/V6 grouping restrictions remain, and V9 stays inaccessible and untouched.

## Dashboard workflow

Launch the dashboard as described in [DASHBOARD_V01.md](DASHBOARD_V01.md). Open **Review →
Video intervals → Review pilot → Activity evaluation · V7/V8/V10**.

1. Watch the block with buffered playback. Dashed timeline periods are unverified source
   suggestions. Click one (or **Review period**) to open its proposed boundaries.
2. Use **Confirm period** if correct. After full visible block playback an unchanged suggestion
   can be confirmed without individually inspecting every frame. Alternatively, inspect its
   start/end using the clickable **Proposed start/end** markers. Adjust timing with exact-frame
   controls and **Set start/end here**; changed boundaries require exact-frame inspection.
   **Remove suggestion** records a rejection without changing the source annotation.
3. Use **Add activity** to label source omissions, mark its start and last included frame, then
   **Save section**. No punch-type identification is required.
4. Use **Mark uncertain section** for anything unclear. Saved regions are clickable timeline
   segments; Edit and Remove append revisions and invalidate any previous completion.
5. After watching the entire block, explicitly confirm all activity and uncertainty has been
   marked, then **Save complete & Next**. For an empty block the confirmation instead explicitly
   states no punching activity was observed. Nothing is automatically certified.

Model predictions remain hidden. At the user's request, source-annotation activity suggestions
are now visible; this replaces the initial source-blind presentation, not the frozen matching
protocol. Source-assisted review can introduce anchoring bias, so review unmarked footage too.
There is no held-out prediction endpoint in this workflow. Existing playback
coverage, foreground/visible-player checks and exact inspection evidence are reused: watching
normally is distinct from inspecting every frame. A saved completion requires complete viewing
coverage and explicit manual attestation. It cannot prove attention; the human attestation is
still required. Playback gaps/seeks/background-tab time do not count as uninterrupted viewing.
No real human confirmations were created during implementation testing.

Records live separately under ignored `reports/boxingvi/activity-evaluation-reviews-v1/`.
Original source labels, historical human punch reviews and previous experiments are preserved.
Unlabeled footage outside explicitly completed reference blocks remains unknown. Reference
completion does not enable training or add background examples to any training dataset.

## Readiness and future reference freeze

Read-only readiness check (no predictions, scoring or reference export):

```sh
uv run python -m boxing_motion_ai.data.activity_reference
```

After all six human reviews, explicitly freeze the reference snapshot **before any evaluation
predictions**:

```sh
uv run python -m boxing_motion_ai.data.activity_reference --freeze
uv run python -m boxing_motion_ai.data.activity_reference --check
```

Freezing refuses incomplete reviews or overwriting a previous freeze. It copies the exact review
ledger bytes and protocol/source-manifest receipts into ignored
`reports/boxingvi/activity-reference-v1/`, verifies source endpoint decoding and records immutable
hashes. Validation rejects later changes to reviews, sources, checkpoint, decoder or protocol.
A freeze can preserve unresolved blocks, but primary scoring remains blocked unless clean
reference exposure exists in both source groups and at least one clean activity bout exists.
Do not manufacture labels to satisfy this gate. The current inference CLI remains restricted
to reviewed V4/V6 material. A later, separately approved evaluation runner must require a valid,
scoring-ready frozen reference and implement the activity grouping/latency rules above before
opening held-out predictions. No final performance claims are ready yet.

## Initial milestone verification (before source suggestions)

567 Python tests and 80 frontend tests passed. Ruff, changed-Python formatting checks,
TypeScript, Prettier and production build passed. Chrome verification used actual V7 (~29.97 FPS)
and V10 (25 FPS) footage: buffered playback, exact-frame activity drafts, uncertain-section
controls, draft cancellation, and a complete V10 playback reaching Interval viewed while the
human confirmation stayed unchecked. No manual reference was saved by automated testing.
Original video may contain burned-in instructional text; hiding dashboard metadata cannot
remove that potential human-labeling cue without changing the footage.

The existing continuous dataset still validates. All 4,793 protected source/review/experiment
files matched their previous hashes; the model, decoder and earlier results are unchanged.
Private validation receipts and the browser screenshot are under ignored
`reports/boxingvi/activity-evaluation-validation-v1/`. There are zero completed reference blocks
and no frozen activity reference yet. The next step is the six human block reviews, followed by
an explicit reference freeze; performance evaluation remains deferred.


## Source-assisted review v1

Suggestions use only the existing BoxingVI source rows and their **provisional** mapped half-open
frame intervals, including the adapter's unverified inclusive-end convention. They do not
establish new offsets, class mappings, timing corrections, or punch validity. Source bounds,
labels, flags, member rows, mapping assumptions and input hashes are retained in every saved
suggestion snapshot. Timing uses exact extracted source PTS. Overlaps and gaps strictly shorter
than 1/4 second merge; exactly 1/4 second remains separate. Periods are clipped to the reviewed
block. Unmappable source rows are flagged, never repaired. Exclusions in training datasets are
unchanged; a source suggestion is not a restored training example.

| Block | Suggested activity periods |
| --- | ---: |
| V7-eval-1 | 8 |
| V7-eval-2 | 0 |
| V8-eval-1 | 15 |
| V8-eval-2 | 0 |
| V10-eval-1 | 20 |
| V10-eval-2 | 0 |

There are 43 suggestions in total. V10's reversed source row 123 remains unmapped and is listed
as a recording-level issue, not silently shifted into a block. Zero suggestions means no
source marks overlap that block; it does **not** mean no punches occur.

Suggestions have stable source-derived IDs and `source=automated`, `verified=false`. They are
never placed into reference `events` automatically. Confirmation, correction, dismissal and
subsequent removal are explicit manual ledger revisions with the original suggestion snapshot.
A dismissed or corrected suggestion can be revisited under **Labeling history & details →
Original source suggestions**. Confirmed/added human regions and dashed unverified source
regions stay distinct on the shared timeline. Source suggestions share one compact lane.

Every suggestion must be resolved before a new exhaustive completion can be saved. Full viewing
coverage and explicit completion confirmation remain necessary, including blocks with zero
suggestions. Removed proposals do not certify their surrounding footage as background. Prior
human ledger revisions remain replayable; no old decision is rewritten or auto-certified.
Completion and reference freeze still follow the unchanged evaluation protocol. No detector
predictions, threshold selection, training or scoring are used in this workflow.

The versioned proposal generator is `services/activity_suggestions.py`. New review revisions
persist the complete original suggestion set and issues, bound to source-input hashes and the
unchanged protocol hash. An additional ignored pre-review snapshot is preserved at
`reports/boxingvi/activity-suggestions-validation-v1/source-suggestions.json`.


Source-assisted verification: 575 Python tests and 83 frontend tests passed, alongside Ruff,
changed-Python formatting, TypeScript, Prettier and production build. Real Chrome checks used
V7 exact source/proposed boundary navigation and V10 buffered playback. A full V10 replay reached
Interval viewed while all 20 source suggestions remained unverified and the human-label count
remained zero. Drafts were canceled; automated testing saved no human confirmations. The frozen
protocol, previous reviews, source files and results remain unchanged.
