# Rear-punch data quality and source priorities

V10 is the recommended **next preparation source**. It has the clearest sampled
alignment, all six source classes and a different source/viewpoint/execution style.
V6 offers the **largest potential rear-hook/uppercut gain**, but should remain behind
a targeted alignment/class-review gate. Recovering V7 uppercuts is a separate
annotation-review task, not an automatic way to grow the training set.

Follow-up: [V10 preparation](V10_PREPARATION.md) now records the full extraction,
additional conservative label quarantines and 141 actual retained candidates.
The investigation counts below remain historical structural/planning estimates.

V4-only mirror augmentation remains the reference model. V5 remains development;
V9 media, annotations, poses and evaluation artifacts were not accessed. No real
models were retrained, no architecture/augmentation changed, no new pose inference
or full extraction was run, and no annotation was restored or corrected in this step.

## Evidence and limits

Original workbooks/media are read only and match the existing source hashes.
The new review decodes 914 V7, 968 V6 and 1,156 V10 source frames across 94 short
windows, matched by exact PTS. Windows are decoded separately even when timing
hypotheses are far apart; no resampling or modified source clips are produced.
The local HTML player steps every decoded review frame and plays with the source
timestamp differences. Dense sheets show expected source labels and neighboring
intervals. Class/hand semantics and biomechanical boundaries are still provisional.

Evidence: `reports/boxingvi/rear-punch-priorities-v1/index.html`, `review.json` and
`pose-comparison.json`. Curated observations and the proposed grouping/quarantines
are in `configs/rear_punch_data_review.yaml`. This is **not a training approval gate**;
it has no annotation overrides or newly approved examples.

Original sources identified by the supplied metadata:

- Dataset: <https://github.com/Bikudebug/BoxingVI>
- V7: <https://www.youtube.com/watch?v=sqU3DmLppcQ>
- V6: <https://www.youtube.com/watch?v=cQW-h8yGbRc>
- V10: <https://www.youtube.com/watch?v=OfZFdwW322E>

No source was downloaded in this investigation. Licensing/redistribution remain
unverified; generated source imagery stays local and excluded from Git.

## V7: what the 20 excluded rear uppercuts contain

All 20 intervals were visually inspected with their overlapping neighbors and
context. The overlap geometry separates into three useful review queues:

| Queue | Rear-uppercut source rows | Finding and disposition |
|---|---|---|
| Endpoint only, 2 | 5, 96 | Distinct preceding punch and uppercut-like movement are visible. Only a shared endpoint triggers exclusion. Review author/end conventions and both adjacent events; still excluded. |
| Partial overlap, 17 | 10, 42, 45, 49, 99, 102, 105, 108, 110, 112, 114, 117, 120, 123, 126, 129, 132 | Plausible consecutive punch combinations, with retraction/loading and the next action sharing annotated frames. Boundaries remain ambiguous; still excluded. |
| Nested conflicting labels, 1 | 54 | Row 53's lead uppercut, 8128–8130, lies inside row 54's rear uppercut, 8125–8138. This suggests a boundary/label error, rather than an endpoint-convention issue. Keep both quarantined. Row 55 also overlaps the recovery/next action. |

For example, row 5 shows a preceding horizontal action, followed by a distinct
lowering/rising glove sequence around frames 4688–4701. Row 108 similarly follows
the preceding action's recovery with a rising bent-arm action around 12796–12800.
Rows 114–132 often form hook-like / uppercut-like / hook-like combinations. Those
are observations of motion, not independent certification of lead/rear labels.

Partial overlaps here span 1–6 frame-coordinate units (2–7 shared frames under the
current inclusive assumption), approximately 33–200 ms between the overlap's first
and last timestamps. That can be a large part of these short actions. An event can
be genuine while its positive training clip includes another punch. Do not restore
it merely because a plausible uppercut is visible somewhere in the interval.

The previous full-pose diagnostic found all 20 pass mask-aware70. Pose quality is
therefore not the principal blocker. All remain excluded under the unchanged
annotation policy. The first efficient manual review is **rows 4/5 and 95/96**;
then inspect the 17 combination boundaries if needed. The nested 53/54 case needs
qualified review or author evidence. Any later decisions require a new sidecar
with evidence, reviewer and approved event intervals, leaving original cells intact.
No exact replacement bounds are proposed from interval arithmetic alone.

Recovering V7 adds execution examples within the existing V7/V8 source group; it
does not add an independent boxer/session or source family. Currently usable new
rear uppercuts from this review: **zero**. At most 2 simple endpoint cases and 17
more involved cases are recovery opportunities, not promised training examples.

## V6 and V10 source coverage and alignment

| Property | V6 | V10 |
|---|---|---|
| Source rows | 685 | 152 |
| Structurally eligible under existing exclusions | 317 | 151 |
| Available classes | Hooks and uppercuts only | All six |
| Overlap pairs | 214 strict, 22 shared-endpoint | None among valid intervals |
| Other problems | Row 3 helpers; row 379 order; suspected class/phase issue at 685 | Row 123 reversed lead-hook bounds |
| Source frames / duration | 46,705 / 1948.021 s | 26,917 / 1076.701 s |
| Decoded timing | 41/42 ms PTS steps | Exact 40 ms steps |
| Existing pose evidence | 346 frames, 5 events per available class | 380 frames, 5 events per class |

**V6:** A/B/C supply start/end/class. D is an end-minus-start helper; E/F usually
subtract 157. Row 3 instead contains numeric helper values 13/22. These helpers
do not verify a 157-frame trim of the source. There are also 119 auxiliary-only
rows; they are not punch examples. Jabs/crosses occur visibly in combinations but
have no workbook labels, so absence of a label must never become `no_punch`.

V6 raw intervals at rows 2, 49, 288, 321, 338, 411, 598, 642 and 686 contain
plausible punch motion across the recording, with changing views and glove use.
Some include neighboring movement or recovery. **Row 685, rear uppercut at
44325–44338, is concerning:** its visible interval is dominated by a straight
extension and recovery, not a clearly isolated uppercut. This is a suspected
class/phase mismatch, not proof of a replacement class. Propose keeping this row
quarantined in any future reviewed dataset, in addition to existing exclusions.

Raw PTS and a 24 FPS annotation-clock hypothesis separate by 0.006 s at row 2,
1.068 s at row 321 and 1.848 s at row 686. This is **hypothesis separation**, not
measured raw-label drift. At row 686 the 30 FPS interpretation falls in instruction,
while the raw interval includes punch movement; the 24 FPS and `−157` alternatives
sample other phases/combinations and do not establish a global correction. Reported
stream FPS is approximately 23.98, but the exact source PTS inventory is authoritative
for decoding, not evidence of the annotation FPS. No rate/offset is applied.

**V10:** Source bounds are A/C, class E; the intervening columns are blank. All
151 valid intervals are in video bounds. The remaining row 123 has lead-hook
bounds 9760–9723 and stays excluded; do not reverse them automatically.

Inspected raw intervals show straight, hook-like and uppercut-like movements at
early, middle and late anchors. V10's 25 FPS clock matches native decoded indices
mathematically. A 30 FPS interpretation separates by 11.47 s at row 2 and 139.07 s
at row 153, so assuming 30 FPS is inappropriate. Raw timing is supported
provisionally; zero/one indexing, exact endpoints and lead/rear semantics remain
unverified. The mostly frontal, sometimes square stance deserves a short semantics
review before treating alternating arms as independently verified lead/rear classes.

The workout includes kicks, floor exercises and transitions. Inspected rows
131/132/135/137 are **standing** uppercut-like movements under a “Plank + Uppercut”
banner; the banner alone is not grounds to reject them. Row 137 ends before the
nearby descent to the floor. A future preparation must keep that transition and
other unlabeled exercise footage unknown, and flag any event crossing a posture
transition or edit. Continuous PTS do not establish visual continuity.

## Pose visibility and diversity

The existing quality report input hashes were rechecked. V6/V10 saved masks were
recomputed against original coordinates/confidence at the unchanged 0.5 threshold,
and every saved sample index/PTS matches the source inventory. No new poses were
estimated, filled or modified. Both sources retain all five sampled events in
each available class under mask-aware70. These are small **ordinally selected,
nonrandom samples**, not measured full-video retention or proof of accurate labels.

Invalid left/right elbow fractions, event-frame weighted before policy filtering:

| Class | V4 full eligible | V5 full eligible | V6 sample | V10 sample |
|---|---|---|---|---|
| rear_hook | 0.0% / 99.1% | 76.9% / 0.0% | 0.0% / 8.4% | 19.0% / 0.0% |
| lead_uppercut | 0.4% / 58.9% | 1.1% / 0.0% | 12.5% / 45.0% | 0.0% / 17.7% |
| rear_uppercut | 0.0% / 96.1% | 77.1% / 0.3% | 0.0% / 13.5% | 11.5% / 1.6% |

V7's all-20 diagnostic rear-uppercut elbow invalidity was 35.0%/11.2%, with
zero annotation-eligible rows. Its poses can help after annotation review, but
cannot bypass it. V6/V10 offer potentially much more complete rear-arm observations
than V4/V5. Confidence/masks measure estimator reliability, not independently
measured anatomical accuracy; view-dependent patterns remain possible shortcuts.

Viewpoint and execution diversity improve, but are **not yet sufficient for a robust
generalization claim**:

- V7 has indoor oblique/front/back views while turning through repeated combinations;
  it remains linked to V8's source family. Recovering labels does not add a group.
- V6 adds gloved demonstrations, frontal/side views and camera changes, with repeated
  scripted technical combinations. Its jabs/crosses remain unlabeled.
- V10 adds a mostly fixed frontal workout view, compact exercise-style punches and
  alternating uppercuts around mixed exercise phases. It adds domain variation,
  which may not transfer directly to technical boxing execution.
- Publisher provenance differs, but **boxer/session IDs remain unknown**. No person
  identities or stance labels are inferred from faces, channel names or clothing.
  We cannot report a verified unique-boxer count. Multiple recordings and mirror
  augmentation do not establish multiple independent boxers or balanced stances.
- The pool is still narrow: scripted workouts, few source families, limited verified
  stance/session diversity and little evidence of varied competitive execution.

Keep V7/V8 in `precision-striking-unresolved`, V6/V1 in
`natebowerfitness-unresolved` and V10 in `utah-lee-unresolved`. The V1 relationship
comes from prior publisher evidence; V1 was not opened in this investigation.
Recording identity remains separate from these guards and from unknown boxer/session
identity. V5 stays exclusively development and V9 is not used in these decisions.

## Estimated expansion capacity and recommended order

The table separates exact structural ceilings from a practical **80% capacity
scenario** with a 20% contingency for quality/phase review. This fraction is an
explicit planning assumption, **not** an observed yield, confidence interval or
inference from 5/5 retention. Actual counts can be lower, including zero, until the
timing, class semantics and unchanged full-pose policy gates pass. No examples are
newly approved now.

| Class | V6 structural ceiling | V6 review-constrained ceiling | V6 80% scenario | V10 ceiling | V10 80% scenario |
|---|---:|---:|---:|---:|---:|
| jab | 0 | 0 | 0 | 49 | 39 |
| cross | 0 | 0 | 0 | 36 | 28 |
| lead_hook | 64 | 64 | 51 | 25 | 20 |
| rear_hook | 32 | 32 | 25 | 6 | 4 |
| lead_uppercut | 133 | 133 | 106 | 10 | 8 |
| rear_uppercut | 88 | 87 | 69 | 25 | 20 |
| total | 317 | 316 | 251 | 151 | 119 |

V6's review-constrained estimate reserves suspicious row 685; this changes only the
proposed plan, not its original annotation or existing dataset. V10 reserves the
existing reversed row 123. Counts are rounded down separately per class. V7 recovery
is excluded from these estimates because no restored intervals are approved.

1. **Prepare V10 next after a small review:** verify representative lead/rear uppercut
   semantics and all six rear-hook labels, preserve row 123's exclusion, then run
   full IMAGE pose extraction and mask-aware preparation without changing thresholds.
   Freeze per-event standing/exercise scope and exact source timing before training.
   Expect an order of magnitude of 119–151 conditional examples, including only
   4–6 rear hooks and 20–25 rear uppercuts. This is the lowest-risk next all-class
   source addition, not a solution to rear-hook scarcity.
2. **Resolve V6 before a full run:** review rows 2/321/685/686 against raw indices,
   24 FPS, 30 FPS and helper hypotheses; check early/mid/late rear-hook sequences.
   Clarify row 3 and preserve row 379/overlap exclusions. If class/phase alignment
   remains inconsistent, request the authors' annotated media/version, frame clock
   and origin/edit details instead of guessing a conversion. Only after this gate
   consider its 251–316 conditional examples, especially 25–32 rear hooks and
   69–87 rear uppercuts. This is the greatest potential volume for the weak classes.
3. **Review V7 independently:** start with the two endpoint pairs, then the 17 partial
   overlaps. Require a reviewed sidecar; never clear all overlap flags at once.
   Keep the nested conflict quarantined. All approved new examples remain zero now.
4. **Close the actual diversity gap:** if these sources cannot supply reliable rear
   punches, prioritize consented footage with documented boxer/session IDs, stance,
   two viewpoints and both rear hooks/uppercuts across multiple performers/sessions.
   Target roughly 30–50 reviewed examples per weak class per new source group rather
   than hundreds of repetitions from one routine. This is a collection target, not
   an estimate of files already present; no collection/download was initiated.

After preparation is reviewed, evaluate additions as separate group-aware data
interventions against the preserved V4-only mirror reference, always keeping V5
development and V9 reserved. Do not silently combine every experimental source or
change the policy to meet a target count. No training is authorized by this report.

## Reproduce this read-only investigation

From the repository root, using already installed pose dependencies and existing
local inputs:

```sh
uv run python -m boxing_motion_ai.data.source_review \
  --output reports/boxingvi/rear-punch-priorities-repeat
uv run pytest
uv run ruff check .
```

Use a fresh report directory. This command reads only V7/V6/V10 source media and
workbooks, plus existing V4/V5/V7 quality evidence and V6/V10 pose samples. It performs
no network download, pose inference, annotation write or model training. Generated
media/reports remain ignored. Original data, model artifacts and accepted studies
are preserved. Stop here for review before running the proposed preparation work.

Validation: `uv run pytest -q` passed all **161 tests**; `uv run ruff check .`
and `git diff --check` passed. SHA-256 checks confirmed preservation of all six
reviewed source media/workbooks and all 367 files in the preserved mirror-reference
manifest. Raw/processed data, generated reports and the virtual environment remain
excluded from Git, with no tracked files in those directories.

Follow-up: [V6 readiness](V6_READINESS.md) investigates the frame/phase concerns and
recommends provisional extraction with conservative quarantines. Its 302-event
review-constrained ceiling supersedes the earlier V6 capacity estimate for planning;
it is not measured full-pose retention or approval to train.
