# V4 second-recording validation — 2026-10-20

V4 is suitable for a provisional full IMAGE-mode extraction. The fixed mask-aware policy
transfers as a usable retention rule on this small sample, not as proof of label/pose accuracy
or model generalization. No full V4 extraction or model training was performed.

## Verified from local inputs

`Meta_data.ods` pairs V4 with
[the original recording](https://www.youtube.com/watch?v=IVvOmoB88FM&t=345s).
The URL's `t=345s` is not an annotation offset. The local full-source MKV matches the prior
acquisition hash: 3840×2160 AV1, 24000/1001 FPS, 35,363 decoded frames, time base 1/1000,
duration 1474.961 seconds. Original media, workbook, metadata and archive hashes are unchanged.

The workbook is one `Sheet1`, with header row `Start, End, Class, Difference, Start_norm,
End_norm`; 560 events occupy rows 2–561. A/B are integral-valued source coordinates spanning
6437–24731. D is `=B{row}-A{row}`, E/F subtract 6437 from A/B. All inspected helper formulas
match that pattern. Source vocabulary/counts: Jab 120, Cross 39, Lead hook 144, Rear Hook 75,
Lead Uppercut 76, Rear Uppercut 106. `Lead hook` has an explicit provisional lexical mapping;
no stance or biomechanical interpretation is inferred from capitalization.

Rows 117/118, 163/164 and 513/514 share endpoints. All six remain excluded under the existing
conservative overlap check; an exclusive-end convention might explain these contacts but is
unverified. No reversed intervals, duplicate events or helper-formula anomalies were found.

## Sample results

Five structurally eligible events per class were selected by ordinal position across each
class's occurrence range, without using pose quality. All 229 event frames were processed:
70 strict body-valid, 158 partially unreliable, one entirely missing (frame 18667).
All 229 integer PTS/rational-time-base pairs agree with the earlier source frame inventory.

| Class | Assessed | Strict body retained | Mask-aware70 retained | Left elbow invalid frames | Right elbow invalid frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| jab | 5 | 1 | 5 | 0.0% | 21.3% |
| cross | 5 | 0 | 5 | 0.0% | 79.3% |
| lead_hook | 5 | 0 | 4 | 2.1% | 42.6% |
| rear_hook | 5 | 0 | 5 | 0.0% | 100.0% |
| lead_uppercut | 5 | 0 | 4 | 2.7% | 64.9% |
| rear_uppercut | 5 | 0 | 5 | 0.0% | 93.5% |

The unchanged policy retains 28/30. Row 352 has a complete arm on only 7/11 frames; row 358
has one missing detection among seven frames, failing anchor coverage, core coverage and the
10% missing limit. Neither was automatically corrected or removed from source annotations.

The matched five-per-class V5 subset retains 30/30 mask-aware and 9/30 strict; its left elbow
fails on 77.1%, 78.0%, 86.7% of cross/rear-hook/rear-uppercut frames respectively. V5's full
eligible reference is 583/584 mask-aware and 249/584 strict. Samples have different event
lengths and viewpoints; five per class is not a population estimate.

V4 does **not** repeat the dominant left-elbow failure. Instead, right-arm visibility is
poor, especially for rear punches. Right-elbow invalidity in cross/rear-hook/rear-uppercut
frames is mechanically due to visibility below 0.5, with high presence. Overlays show partly
hidden limbs and implausible unreliable raw joint positions. Camera view, self-occlusion and
bag obstruction are plausible contributors, not established causes for every failure.
High-confidence landmark geometry also needs review; scores are not accuracy guarantees.

Mask/class association remains visible within V4 (descriptive right-wrist MI about 0.985 bits
in this 30-event sample). The reversal of the failing side across recordings argues against
hardcoding a left-arm fix. Preserve both sides' original masks, scores and coordinates;
missingness shortcuts remain a concern. No mask-only classifier or recognition model was fit.

## Mapping and manual review still required

Raw A/B → zero-based decoded frame indices, offset 0, inclusive end is a **provisional review
configuration**, not a verified author convention. Prior broad source-FPS stills and current
sample overlays contain plausible bag punches at raw indices; they do not certify exact
classes, boundaries, indexing or edit alignment. The 6437 helper rebase is not applied.

At the first start, decoded PTS is 268.477 seconds; A/30 would be 214.567 seconds. At the final
start, decoded PTS is 1031.239 seconds; A/30 is 824.167 seconds. Using exactly 24 rather than
24000/1001 creates about 0.268–1.031 seconds of growing clock discrepancy over these anchors.
These are alternative timing calculations, not measured annotation errors. A one-frame base
change is about 41.7 ms; the helper subtracts about 268.477 seconds. Source rate, author rate,
trim origin, first-frame base and inclusive/exclusive end remain distinct unknowns.

Review the six class sheets under `reports/boxingvi/v4-mask-aware/visual-review/`; each shows
three sampled events (worst/median/best core visibility) at start/mid/end with exact frame IDs,
PTS and raw scores. All 30 events also have start/mid/end overlays in
`data/processed/boxingvi/V4/pose-sample-image-v1/overlays/`.

Prioritize row 352 (frames 18488–18498), row 358 (18661–18667, missing final pose), right-arm
geometry in rear-hook rows 151/250/527 and cross rows 15/35/95, then early/middle/late class
and boundary agreement. Inspect shared-endpoint pairs separately in the original video.
`reports/boxingvi/v4-validation/alignment-table.md` lists all 30 raw-index and alternative-rate
anchors. Stills cannot establish exact punch onset/end; use short sequential playback when
reviewing those boundaries. Review transitions before any temporal feature construction.

## Pipeline/readiness

See [RECORDING_PIPELINE.md](RECORDING_PIPELINE.md) for commands. Explicit recording YAMLs
separate V4/V5 layouts, offsets, endpoint assumptions and identities. V1–V10 share the same
code but require individually inspected configurations. Sparse arrays retain source frame
IDs separately from compact array offsets; unextracted events are unassessed, not missing.
All unlabeled footage remains unknown. Recording IDs are not boxer IDs: boxer/session remain
null and splits unassigned. Licensing and redistribution remain unverified.

Proceed with full V4 extraction for further validation using `configs/v4_pose_full.yaml`;
keep the mapping provisional and inspect additional views/transitions. IMAGE mode has no
tracking, interpolation or smoothing across edits. V4's cut inventory is not certified and
V5's cut list is not reused. Training should wait for alignment/pose review, a full V4 quality
assessment and a defensible split plan with identity uncertainties retained.
