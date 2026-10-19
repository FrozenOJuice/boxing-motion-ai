# Boxing Motion AI

Computer vision and ML for punch recognition and early detection. **Phase 0 only is implemented:**
local dataset discovery, annotation contracts, defensive BoxingVI XLSX inspection, validation and
JSON/Markdown audit reports. Level 2 is the core target; Level 2.5 analytics follows the core.
See [the project specification](docs/PROJECT_SPEC.md).

## Setup and checks

Install [uv](https://docs.astral.sh/uv/) and run from this directory:

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run boxing-motion-ai --help
uv run boxing-motion-ai audit --help
```

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
No models, pose pipeline, training, inference server, UI or round analytics are implemented.
