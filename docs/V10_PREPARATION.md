# V10 provisional training-source preparation

V4-only mirror augmentation remains the reference. V5 remains development; V9 inputs
are not accessed. This step prepares V10 without training, architecture changes,
augmentation changes or confidence/quality-threshold changes. Earlier source files,
sample poses and experiment results are preserved.

**Completed:** all 26,917 source frames are extracted and exact-PTS verified.
V10 supplies **141 provisional candidates**, including **21 rear uppercuts**.
It is ready for review as an additional positive-clip source, not certified labels
or evidence that six-class recognition is solved. Eleven annotations are excluded;
the unchanged mask policy rejects no further candidates. No training was run.

Subsequent approved work: the [V10 controlled study](LEVEL1_V10_STUDY.md) evaluates
this frozen preparation. The preparation evidence and exclusions below are preserved.

## Targeted annotation review

Original dataset: <https://github.com/Bikudebug/BoxingVI>.
V10 source identified by the supplied metadata:
<https://www.youtube.com/watch?v=OfZFdwW322E>.
Licensing and redistribution remain unverified; source imagery stays local.

The workbook has literal A/C/E start/end/class cells, with 152 event rows across
all six classes. Its original bounds and labels are never rewritten. The source
inventory contains 26,917 frames, 1920×1080, with strictly increasing exact PTS at
40 ms intervals (25 FPS); the format duration is 1,076.701 seconds. A nominal FPS
does not establish the annotation clock. Raw decoded indices remain a provisional
mapping, with offset zero and an inclusive end assumption. No timing correction,
stance inference, zero/one-base correction or reversed-bound swap is applied.

The new exact-PTS review is `reports/boxingvi/v10-targeted-review-v1/review.html`.
It includes existing time-spread six-class anchors, all six rear hooks, all lead
uppercuts, adjacent opposite-hand labels and wider transition context. Review
selection does not depend on pose quality. The generated JSON is pre-decision
evidence; the subsequent sidecar `configs/v10_alignment_review.yaml` records the
additional two introductory-label quarantines. Neither is expert certification.

| Focus | Source rows | Observation / decision |
|---|---|---|
| Straight punches | 2, 6 and earlier class-spread anchors | Raw bounds contain extension/recovery; no offset supported. |
| Hooks | 5, 7, 8, 33, 120, 138–145 | Curved bent-arm actions; opposite-hand late repetitions visible. All six rear-hook intervals inspected. Lead/rear meanings remain source-provisional. |
| Uppercuts | 9, 10, 41, 79, 83, 146–153 | Upward bent-arm actions, including a change in visible side between workout blocks. Retain literal labels provisionally except explicit quarantines. |
| Introductory lead uppercuts | 9, 10 | Resemble later rear-uppercut execution without a clearly established stance change. Suspected label/stance inconsistency, not a verified error or a justification to swap classes. Exclude pending short review. |
| Reversed interval | 123, lead hook | Original 9760–9723 remains reversed and excluded. |
| Floor exercise transition | 130–137 | Raw 18703–18777 shows standing uppercuts after rising from the floor and before descending again. Near-square stance makes lead/rear semantics uncertain; row 137 also borders descent. Quarantine all eight conservatively. |

The floor-exercise banner alone is not evidence of an incorrect punch label. The
quarantine addresses this particular phase/stance ambiguity, not all workout
footage. The floor motion before/after remains unknown, never `no_punch`.
Other samples support proceeding provisionally; they cannot certify every event,
biomechanical onset, end convention, source edit or handedness meaning. The smallest
later manual review is rows 9/10 against 41 and 79, and the short 130–137 sequence;
no extensive relabeling is required for this conservative preparation.

## Extraction and candidate contract

Use the unchanged full MediaPipe model in independent IMAGE mode at maximum width
960, original detection/visibility/presence threshold 0.5, and one pose maximum.
Every decoded source index and integer PTS/rational time base is retained, including
missing detections. No smoothing, tracking, filling, interpolation, frame dropping
or resampling occurs. The original normalized coordinates, raw confidence scores
and masks are packed as float64 arrays without replacing unreliable coordinates.
Missing poses retain missing values and false masks.

No complete scene-cut inventory is certified. IMAGE mode carries no state across
edits; it does not make unknown cuts safe for future context or interpolation.
Known source and extraction boundaries remain authoritative. This preparation
does not enable pre-clip context.

Extraction embeds the existing row-123 and 130–137 review exclusions. The later
preparation explicitly adds rows 9/10 using `PreparationConfig`; this does not
rewrite the extraction or its annotation provenance. Original recording YAML and
prior sample artifacts also remain unchanged.

The unchanged `mask_aware70` policy requires reliable torso anchors on at least
95% of frames, one complete arm on at least 80%, at least 70% valid core entries,
and no more than 10% missing frames. It never promotes invalid joints, changes
labels or restores quarantined annotations. Reports distinguish annotation
exclusions from pose-policy rejections.

## Actual counts and pose quality

| Class | Source | Annotation excluded | Pose-policy rejected | V10 retained | Proposed V4+V10 |
|---|---:|---:|---:|---:|---:|
| jab | 49 | 0 | 0 | 49 | 166 |
| cross | 36 | 0 | 0 | 36 | 75 |
| lead_hook | 26 | 1 | 0 | 25 | 165 |
| rear_hook | 6 | 0 | 0 | 6 | 79 |
| lead_uppercut | 10 | 6 | 0 | 4 | 75 |
| rear_uppercut | 25 | 4 | 0 | 21 | 108 |
| total | 152 | 11 | 0 | 141 | 668 |

All 141 eligible intervals have complete extracted frame coverage and no wholly
missing detections. **113** contain at least one unreliable body joint; only **28**
meet the strict all-body-joints-on-every-frame rule. All six rear hooks and all four
retained lead uppercuts contain unreliable joints. Their retention reflects the
approved partial-pose policy, not confidence-threshold relaxation or filled poses.

Across the entire recording, there are **19,195** fully valid-body detections,
**7,255** unreliable-body detections and **467** missing detections. These are frame
statuses, not event labels or independent accuracy measurements. **25,111** frames
outside plausible source annotation bounds remain unknown; excluded intervals also
remain unavailable as training positives or verified negatives.

Exact source timing matches at **every one of the 26,917 indices**, including the
original integer PTS/rational time bases. The full run exactly reproduces all **380**
previously extracted sample frames' coordinates, confidence scores, validity masks,
timing and pose statuses. Source/workbook hashes and the earlier sample are unchanged.

Artifacts are under `reports/boxingvi/v10-full-v1/`: `quality/V10/quality.json`,
`quality/V10/policy-index.json`, `readiness/readiness.json` and `readiness.md`.
The readiness JSON lists each retained/excluded row, source hashes and uncertainty,
and includes full-recording quality for all five comparison recordings. An independent
fresh-process `readiness-verification/` report matches the persisted readiness report
exactly, including the final mapping/provenance checks. `sample-reproduction.json`
records the full-versus-sample equality checks. Skeletons are beside the new pose
run in `overlays/` and `skeleton-contact-sheet.jpg`.

## Viewpoint, arm visibility and diversity

V4's reviewed bag punches are predominantly directed toward screen left in a side
view. V5's reviewed ring punches are directed toward screen right, with the opposite
arm often unreliable. V7 has technical bag/combination footage with turning views;
V8 includes an outdoor oblique/front view with self-occlusion. V10 adds a frontal
garage workout view, frequent torso turns, compact fitness punches and wrist weights.
These are source/view observations, not verified boxer identities or stance labels.
`viewpoint-comparison/viewpoints.jpg` illustrates these differences using existing
overlays; its different classes/intervals are not a matched punch-quality experiment.

Frame-weighted **left/right elbow invalid-mask percentages**, among all
annotation-eligible intervals before pose-policy filtering:

| Class | V4 | V5 | V7 | V8 | V10 |
|---|---|---|---|---|---|
| rear_hook | 0.0 / 99.1 | 76.9 / 0.0 | 0.0 / 13.3 (one event) | no labels | 17.0 / 0.0 |
| lead_uppercut | 0.4 / 58.9 | 1.1 / 0.0 | 24.5 / 15.7 | 7.7 / 26.3 | 0.0 / 42.9 |
| rear_uppercut | 0.0 / 96.1 | 77.1 / 0.3 | none eligible | 13.4 / 10.1 | 6.5 / 3.2 |

V10's **21 retained rear uppercuts**, compared with V7's zero eligible and V8's eight
retained, provide useful additional arm-visibility and execution variation. Both
elbows are much more frequently valid than in V4/V5. This supports a provisional
data intervention, not a claim that poses or rear labels are independently correct.
Rear hooks remain scarce at six. Only four lead uppercuts survive the annotation
quarantines, all in one late sequence (rows 147/149/151/153); their right-elbow
invalidity is **42.9%**, worse than the earlier mixed five-event sample suggested.
Class-dependent masks and adjacent repetitions remain potential shortcuts. Retain
the proposed mask-only control and do not claim identity-independent diversity.

The smaller lead-uppercut count is an explicit quality tradeoff, not a silent class
drop. V10 does not close the broader diversity gap in independently identified
boxers, sessions, viewpoints or realistic punch execution.

## Proposed next experiment — not run

`configs/level1_v10_proposed.yaml` proposes V4/V10 training versus the preserved
V4-only mirror reference, with V5 development and the same three fixed seeds,
40-epoch budget, causal TCN, features, mask-only control and mirror probability
0.5. Visibility augmentation and context remain disabled. V7/V8 stay a separate
experimental arm to isolate V10's contribution.

The proposal is deliberately not accepted by the existing training runners: their
frozen grouped loader currently supports V4/V7/V8. Extending its explicit V4/V10
gate requires the next reviewed training task; no architecture change is needed.
The preparation-readiness report is not training authorization.

V10 has its own conservative `utah-lee-unresolved` source group. V7/V8 remain
together and V1/V6 retain their publisher guard. Recording identity, source group,
boxer identity and session identity remain separate. Unknown people/sessions stay
null; no boxer-independent claim follows from a new publisher or appearance.

## Reproduce using existing local inputs

```sh
uv run python -m boxing_motion_ai.data.v10_preparation review \
  --output reports/boxingvi/v10-targeted-review-repeat
uv run boxing-motion-ai extract-poses \
  --video data/raw/boxingvi/RGB_videos/V10.mkv \
  --workbook data/raw/boxingvi/Annotation_files/V10.xlsx \
  --recording configs/v10_recording_provisional_v1.yaml \
  --config configs/v10_pose_full.yaml --model weights/pose_landmarker_full.task \
  --output data/processed/boxingvi/V10/pose-full-image-v1 --resume
```

Resume validates the extraction configuration, source hashes and immutable prefix;
a complete matching run is reused. MediaPipe required native graphics access beyond
the sandbox even with its CPU delegate; the initial initialization failure was
recovered with the existing resume mechanism.

Preparation adds the explicit final review exclusions:

```python
from pathlib import Path
import yaml
from boxing_motion_ai.data.preparation import PreparationConfig, prepare_recording

review = yaml.safe_load(Path("configs/v10_alignment_review.yaml").read_text())
prepare_recording(
    Path("data/processed/boxingvi/V10/pose-full-image-v1"),
    Path("data/raw/boxingvi/Annotation_files/V10.xlsx"),
    Path("data/processed/boxingvi/V10/candidates-full-v1"),
    PreparationConfig(excluded_source_rows=review["excluded_source_rows"]),
)
```

```sh
uv run boxing-motion-ai assess-pose-quality \
  --dataset data/processed/boxingvi/V10/candidates-full-v1 \
  --output reports/boxingvi/v10-full-v1/quality/V10
uv run python -m boxing_motion_ai.data.v10_preparation finish \
  --output reports/boxingvi/v10-full-v1/readiness
uv run pytest
uv run ruff check .
```

Preparation/quality/readiness require fresh destinations. Their reproduction uses
the original evidence paths frozen in the sidecar; new review evidence needs a
separately versioned decision/hash rather than overwriting accepted evidence.
All generated media, datasets, reports, virtual environments and weights stay
excluded from Git. No source download or model training is part of these commands.

## Validation and disposition

`uv run pytest -q`: **167 passed**. `uv run ruff check .` and `git diff --check`:
passed. Synthetic tests cover report writing, exact-PTS mismatch/frame-drop rejection,
changed-evidence rejection, disallowed offset changes and exclusion preservation.
All original V10 media/workbook hashes, readiness input hashes, the earlier V10
sample hashes and **367** preserved mirror-reference files match. Generated data,
media/reports, weights and the environment remain ignored and untracked.

V10 is ready as a **provisional** additional source for the proposed reviewed
positive-clip experiment, especially rear uppercuts. The six-class imbalance,
lead/rear uncertainty, four-event lead-uppercut cluster and unknown identities remain
material limitations. No model was retrained, and V9 inputs were not opened.
Stop here for review before extending the training loader or running the proposal.
