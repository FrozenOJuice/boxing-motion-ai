# Boxing Motion AI

Computer vision and ML for punch recognition and early detection. Phase 0 provides local dataset
discovery, annotation contracts, defensive BoxingVI XLSX inspection, validation and JSON/Markdown
audit reports. Phase 1 adds recording-aware pose extraction and provisional clip preparation.
The first Level 1 experiment uses V4 for training and V5 for development validation. Level 2 is the core target; Level 2.5 analytics follows the core.
See [the project specification](docs/PROJECT_SPEC.md).

**Current workflow:** [versioned annotation curation](docs/ANNOTATION_CURATION.md).
The proposed selection retains 1,125 provisional training examples; the 18 targeted
human reviews remain pending. Original annotations and the frozen V5 development
benchmark are preserved. No curated training experiment has been run.

Preparation for the FastAPI backend and React development dashboard begins with the
[repository cleanup assessment](docs/CLEANUP_ASSESSMENT.md). It records the current
check baseline and preservation requirements. P0 and the read-only part of P1 are
now implemented; see [dashboard service interfaces](docs/DASHBOARD_SERVICES.md) for
queries, artifact status semantics and the intentionally limited integration scope.
Dashboard v0.1 now supplies the read-only UI. Other cleanup and review writes remain deferred.

The provisional Level 1 reference is now **V4-only XY + validity-gated velocities +
50% mirror augmentation**, selected on V5 development. The
[frozen transfer study](docs/LEVEL1_VELOCITY_TRANSFER.md) finds that its V5 improvement
is not consistent on V6/V7/V8/V10. See [the reference registry](configs/level1_reference.yaml).
V9 remains reserved. Earlier study/reference descriptions below are historical.

The [arm-geometry study](docs/LEVEL1_LIMB_INVESTIGATION.md) improves lead-uppercut
recognition and mean F1 on V5/V6/V7/V8, but regresses on V10 and leaves rear punches
unresolved. Geometry remains optional; the accepted reference stays unchanged.

The [multi-recording study](docs/LEVEL1_MULTI_RECORDING.md) substantially improves
V5 development recognition, but group-held-out rear-punch failures remain. It compares
both existing representations with training-only balancing and mask/duration controls.
The accepted V4-only XY+velocity+mirror reference remains unchanged.

## Local development dashboard

The read-only **[Dashboard v0.1](docs/DASHBOARD_V01.md)** provides annotation review,
exact-frame MKV previews with existing pose overlays, dataset statistics and frozen
experiment/model metrics. Original annotations and V9 protections remain unchanged.

```sh
uv sync --locked --extra dashboard
npm --prefix web ci
npm --prefix web run build
uv run --locked --extra dashboard python -m boxing_motion_ai.api
```

Open **http://127.0.0.1:8000**. For frontend development, use
`npm --prefix web run dev` at port 5174 with the backend running. See the dashboard
notes for dependencies, tests, cache behavior and current limitations.

## Setup and checks

Install [uv](https://docs.astral.sh/uv/) and run from this directory:

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run boxing-motion-ai --help
uv run boxing-motion-ai audit --help
```

The base install includes YAML for curation. Pose/training tests intentionally skip when
optional dependencies are absent. To exercise the complete synthetic test suite:

```sh
uv sync --locked --extra pose --extra train
uv run --locked --extra pose --extra train pytest --require-no-skips
```

CI runs both installations; the full job rejects any collection or test skips.
See [dependency and CI verification](docs/CI.md).

The project pins Python 3.11 via `.python-version` and locks dependencies in `uv.lock`.
Install FFmpeg using your system package manager to provide `ffprobe` (e.g. `brew install ffmpeg`
on macOS). The audit still runs without it, but video metadata/alignment checks remain incomplete.
It never estimates exact frame count from rounded duration × FPS. Tests use synthetic Excel files
and mocked media metadata; they need neither FFmpeg nor external datasets.

## Obtain data yourself from the original authors

Start at the original [BoxingVI repository](https://github.com/Bikudebug/BoxingVI) and follow its
authors' dataset link. Obtain permission and retain the source readme, class definitions, file
correspondence and rights documentation. No dataset or footage is downloaded by this project.
Do not infer permission from this repository's code license or public availability.

Place the original release locally, keeping filenames and folder structure intact, for example:

```text
data/raw/boxingvi/
  Annotation_files/   # Original .xlsx workbooks, not CSV conversions or screenshots
  RGB_videos/         # Matching full, untrimmed recordings in their original versions
```

The scanner recurses, so the authors' singular/plural folder spelling is not assumed.
Skeleton data is unnecessary for Phase 0. Keep any later poses under `data/processed/`.
Raw/processed data, `.venv`, reports, videos, poses and weights are ignored by Git.
Never commit third-party data or rights-restricted media. Alternative original sources, if a
later dataset decision requires them: [BoxingWeb](https://github.com/gouba2333/BoxingWeb) and
[Punch_DL](https://github.com/balezz/Punch_DL). Neither has an adapter in Phase 0.

## Audit workflow

```sh
uv run boxing-motion-ai audit --dataset boxingvi --root data/raw/boxingvi --output reports/boxingvi
```

This works even when the root is absent or empty: it writes `audit.json` and `audit.md`, reports
missing inputs, and exits **1**. Output must be outside the dataset root to avoid counting reports
as dataset inputs. Symlinks are skipped. Paths in reports are relative to the dataset root.
Reports are deterministic for unchanged inputs, mapping and probe results, and contain actual
inventories/counts rather than advertised dataset totals.

With XLSX files present, the first pass records sheet names, nonempty row counts and the first
five nonempty rows (up to 20 columns, each preview cell limited to 200 characters). It does **not**
guess headers, numeric class encodings, video names, handedness or frame conventions. Check the
workbooks and the author's documentation, then create your own mapping as described in
[the annotation guide](docs/ANNOTATION_GUIDE.md):

```sh
uv run boxing-motion-ai audit --dataset boxingvi --root data/raw/boxingvi --output reports/boxingvi --mapping /path/to/verified-mapping.json
```

Reports include source row diagnostics, raw mapped class counts, canonical parsed-row balance,
interval-union coverage, media metadata, candidate identity/session/video groups and outstanding
manual checks. Parsed class counts include duplicate and out-of-bounds rows; they are explicitly
**not usable training sample totals**. Invalid/unmapped rows remain diagnostics, not fabricated
events. Unlabeled intervals are never automatically labeled `no_punch`.

Exit **2** means CLI usage/output failure. Exit **1** means missing/incomplete input or pending
manual verification. Even structurally consistent files yield `needs_review` and exit 1: this
Phase 0 tool cannot certify licensing, footage continuity, identity independence or visual label
alignment. `structural_checks_passed` describes only automated checks, and `phase_0_gate` stays
`not_verified`. Review the [status note](docs/DATASET_AUDIT.md) before approving any modeling work.
The audit itself never trains models. The opt-in Level 1 experiment is documented below;
no inference server or round analytics are implemented.

## V5 pose-quality investigation

See [pose extraction](docs/POSE_EXTRACTION.md) for the optional `pose` dependencies, reproducible
YAML configuration, CLI, versioned keypoint/timestamp contract and private skeleton overlays.
V5 raw annotation frame numbers are an explicitly provisional correspondence; questionable
rows remain flagged. Pose extraction does not approve annotation accuracy or dataset readiness.

Full V5 extraction and provisional punch candidates are documented in
[dataset preparation](docs/DATASET_PREPARATION.md), including per-class quality counts, confirmed
scene boundaries and explicit unknown footage. No model training or independent split is performed.

[Partial-pose quality assessment](docs/MASK_AWARE_POSES.md) compares six retention policies,
per-joint score failures and mask/class association without training. The recommended development
selection preserves original masks and coordinates; retained low-confidence joints are not promoted.

For the second-recording validation, explicit V4/V5 timing and layout configurations,
class-balanced sampling and cross-recording comparison commands, see
[the per-recording pipeline guide](docs/RECORDING_PIPELINE.md).

Full V4 extraction, V1–V10 layouts, restartable sample batches and recovery guarantees are
documented in [EXTRACTION_RESUME.md](docs/EXTRACTION_RESUME.md). Generated dataset-wide
readiness reports distinguish full-video candidates from diagnostic samples and keep
boxer/session identity unknown unless supported by evidence.

## Initial Level 1 experiments

Use the prepared full V4/V5 datasets and the approved `mask_aware70` policy indices.
Run from the repository root; outputs must be fresh directories under ignored `reports/`.

```sh
uv sync --locked --extra pose --extra train
uv run --extra pose --extra train pytest
uv run ruff check .
uv run --extra pose --extra train boxing-motion-ai train-level1 --help
uv run --extra pose --extra train boxing-motion-ai train-level1 --config configs/level1_smoke.yaml --output reports/boxingvi/level1-smoke-repeat
uv run --extra pose --extra train boxing-motion-ai train-level1 --config configs/level1_initial.yaml --output reports/boxingvi/level1-initial-repeat
```

Both commands run the non-temporal linear baseline, a small causal TCN and matched mask-only
ablations. Configuration, source snapshots/hashes, clip selections, predictions, metrics, curves
and checkpoints remain local. V9 is reserved and never loaded. See
[the experiment protocol and initial findings](docs/LEVEL1_EXPERIMENTS.md) for interpretation.
These are provisional six-class positive-clip experiments, without `no_punch`, event detection,
early recognition or claims of boxer-independent generalization.

The controlled mirror-only TCN study uses three matched seeds and mask-only controls:

```sh
uv run --extra pose --extra train boxing-motion-ai train-mirror-study --config configs/level1_mirror_study.yaml --output reports/boxingvi/level1-mirror-repeat
```

Use `--smoke` for a short end-to-end check. See [the mirror study](docs/LEVEL1_MIRROR_STUDY.md)
for reflection semantics, matched-control checks and measured per-class results. Other
augmentations and pre-clip context are not part of this comparison.

The visibility comparison preserves mirror augmentation and tests temporary class-independent
wrist/forearm occlusions, with reliable torso anchors protected:

```sh
uv run --extra pose --extra train boxing-motion-ai train-visibility-study --config configs/level1_visibility_study.yaml --output reports/boxingvi/level1-visibility-repeat
```

Use `--smoke` for a short check. See [the visibility study](docs/LEVEL1_VISIBILITY_STUDY.md)
for the lead-uppercut preflight, frozen masking policy and per-class findings.

The context comparison keeps mirror-only as its control and visibility masking disabled:

```sh
uv run --extra pose --extra train boxing-motion-ai train-context-study --config configs/level1_context_study.yaml --output reports/boxingvi/level1-context-repeat
```

Use `--smoke` for a short check. [The context study](docs/LEVEL1_CONTEXT_STUDY.md) documents the
exact-time window, hard continuity boundaries, punch-only pooling and shortcut diagnostics.

The [V7/V8 expansion study](docs/LEVEL1_DATA_EXPANSION.md) documents per-recording alignment
evidence, conservative annotation exclusions, full-pose preparation and a frozen readiness gate.
It compares V4-only against V4/V7/V8 training with the same mirror-only TCN and three seeds.
V7/V8 share one unresolved source group; V5 remains development and V9 remains reserved.
Visibility augmentation and pre-clip context stay disabled.

After preparation and the documented gate check:

```sh
uv run python -m boxing_motion_ai.training.expansion_study --config configs/level1_expansion_study.yaml --output reports/boxingvi/level1-expansion-repeat --smoke
```

Omit `--smoke` for the matched 40-epoch comparison. Use a fresh output directory; prior
studies, original data and existing pose artifacts are preserved.

The [rear-punch data expansion plan](docs/DATA_EXPANSION_PLAN.md) investigates V7's
excluded uppercuts and compares V6/V10 without inference or model training. It keeps
V4-only mirror augmentation as reference, recommends V10 as the next preparation
source, and separates V6's larger potential yield from its unresolved alignment/class
concerns. Run the read-only source review with a fresh ignored output directory:

```sh
uv run python -m boxing_motion_ai.data.source_review --output reports/boxingvi/rear-punch-priorities-repeat
```

[V10 preparation](docs/V10_PREPARATION.md) documents the targeted transition/label
review, explicit quarantines, full IMAGE extraction and unchanged mask-aware policy.
Its exact-PTS readiness report freezes retained examples without authorizing training.
`configs/level1_v10_proposed.yaml` is a review-only proposal for V4/V10 versus the
preserved V4-only mirror reference; existing training runners reject that proposal
schema. V5 stays development and V9 remains reserved.

The accepted proposal is now supported through the separate frozen V4/V10 gate in
`python -m boxing_motion_ai.training.v10_study`. The [V10 controlled study](docs/LEVEL1_V10_STUDY.md)
records smoke/full reproduction commands and three-seed results. V10 improves
rear-uppercut recall but regresses lead hooks and does not improve aggregate macro-F1;
V4-only mirror augmentation remains the reference. The proposal file is preserved;
the runner translates it into a recorded executable study without enabling visibility
augmentation or context.

The [V6 readiness investigation](docs/V6_READINESS.md) reviews exact-frame alignment,
overlaps, helper columns and existing pose samples without inference or training.
It recommends provisional extraction with quarantines; the 302-event ceiling is
not measured full-policy retention. `configs/v6_alignment_review.yaml` records
recommendations without altering source or training configurations.

```sh
uv run python -m boxing_motion_ai.data.v6_review --output reports/boxingvi/v6-readiness-repeat
```

[V6 full preparation](docs/V6_PREPARATION.md) now documents all 46,705 frames and
302 retained provisional hook/uppercut examples under the unchanged mask policy.
All exclusions and V1/V6 grouping remain; no new training was run. The
[representation assessment](docs/POSE_REPRESENTATION_ASSESSMENT.md) proposed a
separate causal velocity-feature experiment, evaluated below.

The [velocity study](docs/LEVEL1_VELOCITY_STUDY.md) implements optional
masked backward velocities and compares V4-only mirror XY against XY + velocities.
All three seeds improve macro-F1, especially rear uppercuts; rear hooks and lead
uppercuts remain weak, and some confident pose errors generate large derivatives.
V6 is excluded from this comparison and V9 remains untouched.

```sh
uv run python -m boxing_motion_ai.training.velocity_study --config configs/level1_velocity_study.yaml --output reports/boxingvi/level1-velocity-repeat --smoke
```

Omit `--smoke` for the three-seed 40-epoch study. Use fresh ignored destinations;
the original XY mirror controls must reproduce the preserved reference exactly.

Frozen transfer evaluation (no training), using the existing accepted checkpoints:

```sh
uv run python -m boxing_motion_ai.training.velocity_transfer --config configs/level1_velocity_transfer.yaml --output reports/boxingvi/level1-velocity-transfer-repeat
```

See the [transfer findings](docs/LEVEL1_VELOCITY_TRANSFER.md) for class coverage,
confusions, mask controls, velocity tails and the recommendation before V6 expansion.

Read-only relative-limb/joint-angle coverage investigation (no training):

```sh
uv run python -m boxing_motion_ai.training.limb_investigation --config configs/level1_limb_investigation.yaml --output reports/boxingvi/level1-limb-readiness-repeat
```

Controlled additive arm-geometry study (same V4-only training budget and three seeds):

```sh
uv run python -m boxing_motion_ai.training.limb_study --config configs/level1_limb_study.yaml --output reports/boxingvi/level1-limb-study-repeat --smoke
```

Omit `--smoke` for the full comparison and frozen external evaluations. Source datasets,
old checkpoints and the frozen readiness report must already exist locally.

Multi-recording study and guarded group-held-out pilot (fresh ignored outputs required):

```sh
uv run python -m boxing_motion_ai.training.multi_study --config configs/level1_multi_recording.yaml --output reports/boxingvi/level1-multi-smoke-repeat --smoke
```

Omit `--smoke` for three-seed combined training and the one-seed four-fold pilot.
Use `--audit-only` for pool/sampling inspection without training. Verify a completed
run with `uv run python -m boxing_motion_ai.training.multi_verify --output <run>`.

The bounded [RGB + pose feasibility pilot](docs/LEVEL1_RGB_FEASIBILITY.md) compares
frozen MobileNetV3-Small crop features with the existing XY+velocity TCN, using the
approved multi-recording pool and grouped holdouts. It includes static-background,
mask-only and duration controls. Source HUD/punch-number overlays are redacted;
visual features, media and checkpoints remain local and ignored. This is an
exploratory Level 1 study, with V5 development-only and V9 reserved.

```sh
uv sync --extra pose --extra train
TORCH_HOME="$PWD/weights/torch" uv run python -m boxing_motion_ai.training.rgb_study --help
```
