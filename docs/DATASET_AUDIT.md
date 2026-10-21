# Phase 0 dataset audit status

Status date: **2026-10-18**. **Foundation implemented; real dataset gate not verified.**

## Verified in this workspace

- The repository initially contained `docs/BOXING_MOTION_AI_SPEC.md`, README, code license and
  `.gitignore`; no `pyproject.toml`, lockfile, Python package or dataset was present. The spec remains
  unchanged; `PROJECT_SPEC.md` points to it.
- Phase 0 now inspects local files only, with synthetic tests. No external workbook, video or
  skeleton was obtained or inspected. There are no measured real-data class totals or alignment
  results. Test fixtures are synthetic and do not verify the BoxingVI release schema.
- Python 3.11 and project/test dependencies were installed with the user's explicit approval.
  `ffprobe` was not installed in the execution environment. Missing probe tools are reported honestly.

## Verified documentation, distinct from verified dataset contents

The [original BoxingVI README](https://github.com/Bikudebug/BoxingVI) describes XLSX punch intervals
and class labels, with annotation, RGB and skeleton folders. It does not establish actual Excel
headers, sheet layout, numeric class codes, row-to-video correspondence, or frame indexing and end
inclusivity. Its published totals have **not** been used as local file or class counts.
The [paper](https://arxiv.org/abs/2511.16524) is the original research reference.

[Issue 1](https://github.com/Bikudebug/BoxingVI/issues/1) raises missing-video availability concerns.
That report is not proof of what files a new download will contain. Audit the files actually
obtained and ask the authors about any missing recordings. Do not assume the advertised release
is complete. Links/documentation were reviewed; no hosted dataset contents were downloaded.

## Outstanding manual checks

- Real XLSX fields/sheets, class code definitions, source recording correspondence, frame base,
  inclusive/exclusive end, annotation FPS and any trimming/resampling remain **unverified**.
- Full-video/label alignment, real class counts and annotation completeness remain **unverified**.
- Continuity of untrimmed footage and reviewed negatives/feints for Level 2 remain **unverified**.
  Positive clips alone cannot satisfy the Level 2 evaluation requirement.
- Stable boxer and session IDs, repeated-athlete leakage and a frozen grouped split plan remain
  **unverified**. Filename stems are not reliable identity evidence.
- Biomechanical onset/peak labels, stance/hand interpretation and representative visual checks
  remain outstanding. No source start frame is automatically treated as validated onset.
- Footage and annotation licensing, permitted research/portfolio use and redistribution are
  explicitly **unverified**. Retain documented evidence from the authors/rightsholders; this
  project's code license does not license third-party media.

## Local files needed next

Obtain these manually through the authors' link in the
[original repository](https://github.com/Bikudebug/BoxingVI), retaining original filenames:

1. **Original `.xlsx` files from `Annotation_files`** (the README also uses singular naming),
   including every event sheet and any accompanying label dictionary/instructions. Place under
   `data/raw/boxingvi/Annotation_files/`. Start with one workbook plus its corresponding full video
   to establish the schema; obtain all workbooks to verify dataset-wide coverage.
2. **The exact matching full, untrimmed source recordings from `RGB_videos`**, in original resolution,
   frame rate and edit/version. Place under `data/raw/boxingvi/RGB_videos/`. Obtain every recording
   referenced by the annotations; trimmed punch clips or alternate web encodes cannot establish
   full alignment. Ask authors for any absent source files.
3. **Author-provided workbook/sheet/row-to-video correspondence, class dictionary and timing notes**,
   including frame base, end convention, FPS/resampling and original recording identifiers/URLs.
   Keep these alongside the local release. If not supplied, these require author clarification and
   visual validation before creating the explicit mapping.
4. **Boxer/session provenance and license/permission/attribution documents** for both annotations
   and footage, retained alongside the local release. Missing identity information must stay null.

Exact author basenames cannot be listed honestly until the release is available locally. No
skeleton/pose files, extracted frames or weights are required to verify Phase 0 alignment.
Install FFmpeg separately, then follow the README's first-pass audit and verified mapping workflow.
Inspect the resulting `reports/boxingvi/audit.json` and `audit.md` with the project lead before
selecting the dataset or starting a pose/model pipeline.

Original alternatives for a later dataset decision:
[BoxingWeb](https://github.com/gouba2333/BoxingWeb),
[Punch_DL](https://github.com/balezz/Punch_DL). Their availability, contents and rights are not
verified here, and no adapters/downloads were added for them.

## V5 pose investigation authorized — 2026-10-19

The project lead authorized an initial Phase 1 pose-quality investigation using **V5 raw A/B
frame numbers provisionally against decoded source indices**, without applying the `−24` helper
rebase. This does not resolve author indexing, source end convention, FPS or trim provenance.
Rows 272, 323, 442 and 536 remain questionable; no corrections or canonical class mappings
were adopted. Unlabeled footage remains unknown. Split and rights verification are still pending.

The [pose extraction contract and reproduction guide](POSE_EXTRACTION.md) documents the modular
MediaPipe/PyAV implementation, exact PTS preservation, missing masks and local visual samples.
Generated measurements/media live under ignored `data/processed/` and `reports/` directories.
This authorization covers pose investigation only; it does not approve model training.

## Full V5 extraction and preparation — 2026-10-20

Full-frame independent pose extraction and provisional candidate preparation are implemented;
see [the preparation guide](DATASET_PREPARATION.md). Confirmed cut boundaries are recorded,
all missing/unreliable observations retain masks, and suspect annotations remain excluded.
The generated report distinguishes annotation-eligible from strict pose-clean examples. Empty
pose-clean classes and single-video split limitations remain explicit training-readiness blockers;
no classes, thresholds, source frame coordinates or annotations were silently changed.

## V4 second-recording sample (2026-10-20)

Inspected the distinct V4 header/helper layout; selected 30 source events across six classes
and extracted 229 exact-index/PTS frames. The unchanged provisional mask-aware policy retains
28/30, compared with 1/30 strict all-body retention. V4's dominant failure is the right arm,
while V5's was the left; pose confidence does not establish coordinate correctness. Full V4
extraction is a reasonable next validation step, with raw-index mapping still provisional.
See [V4_POSE_VALIDATION.md](V4_POSE_VALIDATION.md) for evidence, exclusions, comparison limits
and manual review priorities. Boxer/session identity, author timing conventions and licensing
remain unverified. No training or full V4 extraction was performed in this step.

## V1–V10 pose-validation expansion (2026-10-20)

Full V4 extraction is complete: 35,363 source frames, exact PTS matched to the prior source
inventory; 527/554 structurally eligible positives retained under the unchanged mask-aware70
policy (strict all-body: 69/554). Retained counts by class are jab 117, cross 39, lead hook
140, rear hook 73, lead uppercut 71, rear uppercut 87. V5 remains unchanged at 583/584,
providing a combined **1,110 provisional positives** across all six classes.

V1–V3 and V6–V10 samples cover 211 selected annotations and 2,507 exact-index/PTS frames;
206 events are structurally assessable, of which 184 pass the same mask-aware policy.
Five V7 rear-uppercut samples remain annotation-flagged and diagnostic only. No source
rear hooks exist in V2/V8; V6 contains no jab/cross labels. Total actual event rows: 5,451.

New visual comparisons confirm mislabeled-context concerns in raw-index V1/V2/V3 hypotheses
(slipping/title cards, rest/sit-ups, warm-up/rest respectively). V6 retains all 20 assessed
poses but needs class/phase and overlap review. V9 is the strongest next full-extraction
candidate; V7/V8 need boundary/visibility review, and V10 needs mixed-exercise segment review.
All author FPS/base/end/trim conventions and boxer/session identities remain unverified.

Interrupted IMAGE extraction now validates/reuses completed pose records; synthetic VFR/cut
restart tests match uninterrupted output exactly. The completed V4 resume and eight-recording
batch rerun reused existing artifacts without inference. Decode replay is still required for
interrupted prefixes. Original media/workbooks/metadata/archive, V5 poses and weights are
unchanged. Generated media/reports and datasets remain ignored; no model training occurred.

See [EXTRACTION_RESUME.md](EXTRACTION_RESUME.md) for commands and
[BASELINE_READINESS.md](BASELINE_READINESS.md) for source gates, manual checks and the proposed
recording/publisher-group split strategy. The authoritative local result index is
`reports/boxingvi/expansion/README.md`; no final train/validation/test split is assigned.
