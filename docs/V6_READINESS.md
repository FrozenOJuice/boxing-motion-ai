# V6 readiness investigation

Approved follow-up: [V6 full preparation](V6_PREPARATION.md) now records actual
retention and full-recording quality. The original review evidence and proposed
holds below remain preserved; provisional mapping is not certified label truth.

V6 is worth **provisional full extraction with conservative quarantines**. It offers
potentially useful hook/uppercut poses and viewpoints, but is **not training-ready**:
full-policy retention is unknown, labels remain provisional, and some intervals mix
consecutive actions. No full extraction, pose inference or training was performed.
V4-only mirror augmentation remains the reference; V5 remains development-only and
V9 was not opened or used. No original file, exclusion or prior experiment was changed.

## Evidence and scope

Sources: `data/raw/boxingvi/Annotation_files/V6.xlsx` and
`data/raw/boxingvi/RGB_videos/V6.mkv`. Original provenance is the
[BoxingVI repository](https://github.com/Bikudebug/BoxingVI) and the metadata's
[V6 YouTube source](https://www.youtube.com/watch?v=cQW-h8yGbRc). Online availability
was not rechecked. Licensing/redistribution remains unverified; keep review imagery local.

Final ignored evidence: `reports/boxingvi/v6-readiness-v3/`:

- `review.json`: source hashes, all 685 events, helpers, overlap pairs, timing
  hypotheses, saved-pose diagnostics and conditional capacity.
- `review.html`: exact-frame player showing source row/class, index, exact PTS and
  hypothesis; `V6-row-<row>-<hypothesis>.jpg` are surrounding-frame sheets.
- `-tight.jpg` sheets include every target frame for selected cases. There are
  125 review windows over 52 distinct event rows and 2,711 unique decoded frames.

The assistant inspected representative raw sheets and temporal alternatives, plus
four saved-pose overlays. This is provisional broad movement/phase evidence, not
expert certification of all labels or precise onset. Existing 346-frame sample
poses were reassessed without inference. `configs/v6_alignment_review.yaml` records
recommendations and hashes; it does not change source/training configurations.

## Workbook and frame conventions

The populated extent is 805 rows × 8 columns on `Sheet1`:

| Cells | Observed content | Interpretation limit |
|---|---|---|
| A/B/C, rows 2–686 | Start, end, class; 685 events | Author frame base, end convention and annotation FPS unverified |
| D | `=B[row]-A[row]` for all 685 events | Coordinate difference, not verified inclusive/exclusive count |
| E/F | `=A[row]-157`, `=B[row]-157` for 684 events | Helper offset, not proof the current video should shift |
| E3/F3 | Numeric 13/22, with A3/B3 = 311/320 | Contradicts surrounding formulas; row 3 remains excluded |
| G | Empty | No identity information |
| H | Worksheet row minus 2 throughout | Ordinal 0–803, not verified frame/boxer/session ID |
| Rows 687–805 | 119 auxiliary-only rows, H populated | Not extra punch annotations |

The source has 46,705 frames, 1920×1080, PTS 0 through 1947.946 s; container duration
1948.021 s. PTS increase strictly in 41/42 ms steps. Reported rate `27266/1137`
(approximately 23.98) does not establish annotation FPS. The complete existing source
timestamp inventory fits `index / (24000/1001)` within 1 ms; `index / 24` differs by
up to 1.946 s. Source cadence is not evidence of the authors' annotation clock.
Continuous timestamps do not certify visual continuity across camera edits.

Exact-time alternative review targets (not proposed corrected annotations):

| Row / class | Raw indices | 24 FPS clock | Helper −157 | 30 FPS clock |
|---|---:|---:|---:|---:|
| 2 / lead uppercut | 157–171 | 157–171 | 0–14 | 125–137 |
| 288 / lead uppercut | 23274–23287 | 23251–23264 | 23117–23130 | 18601–18611 |
| 336 / rear hook | 30535–30557 | 30504–30526 | 30378–30400 | 24404–24421 |
| 642 / rear uppercut | 42486–42500 | 42444–42458 | 42329–42343 | 33955–33966 |
| 685 / rear uppercut | 44325–44338 | 44281–44294 | 44168–44181 | 35425–35435 |
| 686 / lead uppercut | 44352–44372 | 44308–44328 | 44195–44215 | 35446–35462 |

A 24000/1001 clock maps selected anchors to the same indices as raw coordinates.
Raw 288/642 show upward bent-arm action; their 24 FPS alternatives mostly show
preparation/guard before it. Raw 336 shows the rear-hook-like curved action, whereas
its 24 FPS target mostly shows the preceding action/recovery. Row 2's helper target
is title/black footage. No global helper shift or FPS conversion is supported.

Retain offset-zero raw indices and a provisional inclusive end as the working
hypothesis. A one-frame base difference is about 42 ms and remains unresolved.
Author trimming, exact onset/end and lead/rear semantics remain unverified. Growing
clock-hypothesis differences are not measured annotation drift. No constant,
accumulated or abrupt annotation correction has been established.

## Row 685 and phase contamination

Row 685 is labeled rear uppercut at 44325–44338. Its tight strip contains pronounced
straight-arm extension around 44325–44327 and retraction, followed by compact
rising/recovery movement of the other glove around 44329–44334. This is not a clean
isolated rear-uppercut example. It could be a class problem or a boundary spanning
consecutive actions. The 24 FPS target shows earlier bent-arm rise/recovery but does
not independently establish an author-intended replacement. Helper/30 FPS targets
also do not justify correction. **Keep 685 held; do not relabel or retime it.**

Row 686 similarly begins with straight movement before bent-arm rise; 680/683 show
related phase contamination. Some neighboring rows, including 675/678, look more
uppercut-like. Quarantine the entire final block 675–686 conservatively, rather than
cherry-pick guessed boundaries; this does not claim all 12 labels are wrong.
Additional proposed holds 321/598/620 each include preceding extension/recovery and
rising movement. These **15 proposed holds** are review-sidecar recommendations,
not changes to original candidate files. Pose-policy passes do not validate labels.

## Overlaps

There are 236 unique pairs: 22 shared endpoints, 213 partial overlaps and one nested
interval, involving 367 rows. No pairs have identical boundaries, but the nested
pair is effectively a near duplicate.

| Rows / raw bounds | Exact-frame evidence | Decision |
|---|---|---|
| 5/6: 391–402 / 402–415 | Distinct consecutive hooks; shared loading/recovery at 402 | End convention unresolved; both remain excluded |
| 327/328: 29972–29998 / 29998–30025 | Distinct hook-like actions with nearby preceding recovery | Shared endpoint/phase ambiguity; neither restored |
| 339/340: 30735–30758 / 30756–30771 | Consecutive hook-like actions with shared phases | Boundary ambiguity, not automatically two wrong labels; no trimming |
| 331–334 | Fast combination with shared loading/recovery across labels | Not isolated clean clips; exclusions retained |
| 101/102: 12844–12869 / 12845–12869 | One observed rear-hook action covered twice | Likely near-duplicate annotation; neither selected/restored |
| 379: 31001–31020 | Out of row-time order, overlaps 347, much straight recovery/guard | Suspected coordinate/order problem; do not guess 32001 |

Eight rows have shared-endpoint flags only (5, 6, 327, 328, 345, 352, 353, 478).
None is restored: distinct consecutive punches do not establish boundary ownership
or clean positive clips. All original overlap/helper/order flags remain active.

## Pose quality and diversity

The existing nonrandom sample has 20 events, five per available class, with 346
frames: 216 `ok`, 130 `unreliable`, no missing detections. All 20 pass mask-aware70,
including label-questionable rows. After proposed holds, the in-memory diagnostic
leaves 15 events:

| Class | Remaining sample / policy passes | Invalid elbow frames, left/right | Invalid wrist frames, left/right |
|---|---:|---:|---:|
| Lead hook | 5 / 5 | 0.0% / 14.1% | 1.2% / 14.1% |
| Rear hook | 5 / 5 | 0.0% / 8.4% | 0.0% / 7.5% |
| Lead uppercut | 2 / 2 | 0.0% / 31.0% | 0.0% / 24.1% |
| Rear uppercut | 3 / 3 | 0.0% / 13.0% | 0.0% / 10.9% |

Only four pass strict whole-body quality, all lead hooks. Elbow failures here are
saved visibility scores below 0.5, without nonfinite/out-of-image/low-presence
failures. This identifies the threshold mechanism, not physical occlusion or
coordinate accuracy. Glove overlap/body turns are visible; reviewed overlays broadly
follow the body but cannot certify every estimate. No joint was filled or made valid.

The unchanged policy requires anchors in 95% of frames, one complete reliable arm
in 80%, 70% valid core entries, and at most 10% missing detections, using original
0.5 masks. All saved sample indices and exact rational PTS match the source inventory.

The remaining sample does not show V5's dominant left-elbow failure. Rear-class
right-elbow invalidity is also much lower than the previously measured V4 problem.
These are sample-versus-full-recording comparisons, not matched population estimates.
Two surviving sampled lead uppercuts cannot establish all 125 potential candidates.

V6 adds gloved combination work, turns and frontal/oblique/side views with camera
edits, compared with V4's bag viewpoint, V5's ring viewpoint and V10's more frontal
fitness presentation. This may help rear hooks/uppercuts with better arm visibility.
Repetitive workout execution limits independence; viewpoint variety does not prove
boxer diversity. Boxer/session identities remain unknown. Keep V1/V6 together under
`natebowerfitness-unresolved`, a source-leakage guard rather than verified identity.

## Conditional capacity

| Class | Source events | Existing structural ceiling | Ceiling after proposed holds | Assumed 80% capacity scenario |
|---|---:|---:|---:|---:|
| Jab | 0 | 0 | 0 | 0 |
| Cross | 0 | 0 | 0 | 0 |
| Lead hook | 125 | 64 | 64 | 51 |
| Rear hook | 95 | 32 | 32 | 25 |
| Lead uppercut | 259 | 133 | 125 | 100 |
| Rear uppercut | 206 | 88 | 81 | 64 |
| **Total** | **685** | **317** | **302** | **240** |

Original 368 exclusions comprise 367 overlap-flagged rows plus helper-anomaly row 3;
row 379's order flag is already included. Proposed holds remove eight lead and seven
rear uppercuts. **302 is a ceiling, not a trustworthy-example count.** The 80% column
is an explicit planning assumption, rounded down per class, not a confidence interval
or extrapolation of the sample. Actual usable yield could be lower, including zero,
after full filtering and further review. No new examples are approved for training.
Visible straight punches lack source jab/cross labels; these and unlabeled gaps
remain unknown, never automatically `no_punch`.

## Next step and small manual review

After review, provisional full extraction is worthwhile with all original flags and
15 proposed holds. Preserve raw indices, exact PTS/time bases, coordinates,
confidences and masks. Use the existing independent IMAGE pipeline and thresholds;
no smoothing/interpolation across cuts, discontinuities or missing boundaries.
A complete cut inventory is not certified. Freeze actual class retention and hashes
before any later training proposal. The sidecar itself authorizes no extraction.

Open `reports/boxingvi/v6-readiness-v3/review.html` for a short targeted review:

1. Row 685 raw tight strip versus 24 FPS/helper/30 FPS alternatives: confirm the
   mixed-action concern without guessing replacement boundaries.
2. Rows 101/102 together: check that one action is covered twice.
3. Shared-boundary pairs 5/6 and 327/328: distinct actions alone do not validate ends.
4. Rows 288/642 raw versus 24 FPS targets: check support for provisional raw indices.

Author confirmation is still needed for clock/base/end, helper provenance and
lead/rear semantics. No extensive relabeling is required to decide whether provisional
extraction is worthwhile; no full-label correctness claim is made.

## Reproduce and validation

With existing local source/sample artifacts and installed pose dependencies:

```sh
uv run python -m boxing_motion_ai.data.v6_review \
  --output reports/boxingvi/v6-readiness-repeat
uv run pytest -q
uv run ruff check .
git diff --check
```

Use a fresh output directory. The tool verifies frozen source/sample hashes and
decodes selected frames only. It requires prior inventory/quality artifacts and
does not download, infer poses, alter originals or train. Four new synthetic tests
cover helper anomalies, overlap geometry, alternative-target labels and capacity
accounting. **180 tests passed; Ruff and `git diff --check` passed.** Generated
reports/media, raw/processed data, environments and weights remain excluded from
Git. Stop here for review.
