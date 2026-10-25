# Repository cleanup assessment before the development dashboard

Assessment date: **2026-10-25**. Baseline commit: **ff6bb5e**.
Status at assessment: **plan for approval; no cleanup, refactor, archive or dashboard implementation performed**.

**Implementation update (2026-10-25):** approved P0 and read-only P1 are complete; see
[CI verification](CI.md) and [dashboard service boundary](DASHBOARD_SERVICES.md).
The assessment below remains the historical baseline. No archive/removal, review-write
API or broader P2/P3 refactor has been performed.
The only changes in this assessment are this document and a current-status/navigation
addition to the README.

The repository has a useful, tested research core. The strongest improvements are
clearer current-state documentation, reliable dependency/CI coverage, lightweight
interfaces for reading artifacts and reviewing annotations, and explicit separation
of historical experiments from dashboard services. A wholesale reorganization or
artifact purge would put reproducibility at risk without being necessary for the dashboard.

## Baseline and assessment scope

| Check | Observed result |
|---|---|
| `uv run pytest` in the existing environment | **289 passed**, no skips, 9.76 seconds |
| `uv run ruff check .` | **Passed** |
| `uv run boxing-motion-ai --help` | **Passed**, 17 commands |
| Working tree before assessment | Clean |
| Tracked inventory before these documentation additions | 175 files: 66 package Python files, 31 test/support files, 41 configurations, 30 documentation files, plus root/CI files |
| Python package size | 15,116 lines; training 31 modules, data 17, vision 12, package root 6, including `__init__.py` files |
| Static references | No tracked production module or undecorated top-level function established to be unused; see limits below |
| Local Markdown target scan | No missing source/document targets found in README/docs; external URLs, ignored artifact links and illustrative paths excluded |
| Git data exclusions | No tracked files under `data/`, `reports/`, `weights/` or `.venv/`; existing ignore rules cover these |

Reviewed source, test structure, configuration references, docs, CI, entrypoints,
selected report manifests and local utility source text. The local artifact inventory
uses file names/sizes and prunes V9-named subtrees/files before reading their payloads
or taking file-size measurements. Raw dataset directories/ZIP contents were not
traversed, and no video was decoded, annotation edited or V9 payload opened/hashed.
Sizes below therefore exclude V9 and all raw data. Historical documents may mention
V9; that does not authorize new inspection or use of the reserved recording.

Static reference checks considered source, tests, docs and configuration text, plus
24 standalone `python -m` entrypoints. This is evidence against immediate deletion,
not a proof that every branch is live. Some names are generic, decorators and dynamic
loading complicate call graphs, and users may invoke documented commands externally.
Do not use an automated dead-code report alone to delete experiment code.

### CI finding requiring attention first

`.github/workflows/checks.yml` still has the title **Phase 0 checks** and runs
`uv sync --locked` without pose/train extras. PyYAML is declared only in those extras,
but `data/curation.py:run_curation` imports it and two new curation tests call that
function without an optional-dependency skip. A process-local import-isolation probe
that made **only PyYAML unavailable** produced **2 failures / 14 passes** in
`tests/test_annotation_curation.py`:

- `test_proposal_is_reproducible_portable_immutable_and_cli_honestly_blocks`
- `test_case_fingerprints_and_pinned_ledgers_prevent_silent_changes`

This was an intentional dependency probe, not a regression in the installed full
environment and not a remotely observed CI run. The dependency declaration and
workflow nevertheless expose a real fresh-install gap. In addition, **23 of 30 test
modules** use `pytest.importorskip` somewhere, so a base-only CI pass cannot establish
full pose/training coverage.

**Proposed fix after approval:** declare the lightweight YAML dependency for the
supported curation runtime (prefer a base dependency while curation is a core CLI),
keep a base-install check, and add a locked pose/train CI check with an explicit
expected collection/skip policy. Preserve optional ML dependencies for lightweight
API installations. Do not hide the missing curation dependency by skipping its core
behavior tests. No dependency files or workflow were changed during this assessment.

## Keep / Refactor / Archive / Remove decisions

“Archive” means initially **classify as historical while retaining files in place**.
Physical relocation requires the separate preservation procedure below. “Remove”
items are candidates for a later approved cleanup, not instructions executed here.

| Classification | Area / concrete examples | Rationale and next action | Risk |
|---|---|---|---|
| **Keep** | Python 3.11, `uv`, `pyproject.toml`, `uv.lock`, `src/boxing_motion_ai`, pytest/Ruff | Good project foundation; no need for a second Python project or toolchain before FastAPI. | Low |
| **Keep** | `models.py`, `vision/contracts.py`, exact PTS/index handling, masks, scene/segment guards, extraction resume | These are correctness contracts, not incidental complexity. Preserve their semantics and tests. | High if altered |
| **Keep** | `data/annotation_review.py`, `data/curation.py`, both versioned curation JSON configurations | Active review provenance and exclusion-only proposal. Never reinterpret assistant observations as human confirmations. | High if altered |
| **Keep** | `level1_reference.yaml`, velocity/mirror controls, all important checkpoints, frozen predictions/configs/source snapshots | Current pose-only reference and evidence needed for comparisons. Negative results are still reproducible results. | High if removed |
| **Keep** | Original datasets/annotations, approved pose runs/candidates/policy indexes, all prior review decisions; V9 exclusion | Non-negotiable preservation set. No “old data” cleanup by age, filename or size. | High |
| **Keep** | Decoder subprocesses and extraction locks | Sequential/sparse decoding and crash isolation have distinct behavior. Persistent lock files are not disposable merely because they are small or empty. | High if casually changed |
| **Refactor** | Dependency declarations and `.github/workflows/checks.yml` | Resolve the demonstrated YAML gap and make full test coverage observable. Rename the stale workflow title. | Low–medium; CI evidence required |
| **Refactor** | README navigation and docs status labels | README is a chronological study catalogue and omitted the active curation milestone. Separate “current workflow” from historical reproduction. Minimal links/status added now; broader rewrite remains proposed. | Low |
| **Refactor** | `cli.py` (459 lines, 17 commands) versus 24 standalone module entrypoints | Document one command catalogue first; later use thin CLI adapters to common services, keeping old commands working. A study script is not automatically an API endpoint. | Medium |
| **Refactor** | Hashing/storage helpers in `vision/pipeline.py`, `training/experiment.py`, `data/annotation_review.py` | Move future shared infrastructure into a lightweight module with compatibility exports. Preserve atomic replace, exclusive-create and ordinary report-write semantics separately. | Medium |
| **Refactor** | Taxonomy/group constants in `models.py`, `data/preparation.py`, `training/data.py`, `training/groups.py`, `training/multi_recording.py`, `data/curation.py` | Reduce drift through a versioned shared definition for new services, while keeping frozen study-specific allowed-source gates. `unknown`, six positive classes and background state are intentionally different concepts. | Medium–high |
| **Refactor** | Review rendering/decoding spread over `data/*review.py`, `training/rgb_review.py`, `training/forensic_review.py` | Reuse small exact-frame/asset interfaces for future review tools. Keep differences in sequential decode, sparse decode, exact-PTS seek and context limits. | Medium–high |
| **Refactor** | `training/forensic_report.py` imports training data/experiment and `forensic_review`; the latter imports RGB helpers and `load_pool` | Metadata/review presentation should not need broad model/feature dependencies. Separate report reading from rendering/computation before using it in request handlers. | Medium |
| **Refactor** | `multi_recording.load_pool` constructs both velocity and limb datasets for every source | Request only the representations a job needs. Avoid loading pose tensors to list datasets or show existing metrics. Preserve numerical outputs when changing computation paths. | Medium |
| **Refactor** | Related training loops in `experiment.py`, `multi_study.py`, `rgb_study.py`; study orchestration/aggregation | Candidate for small shared primitives, but loss weights, samplers, augmentation coins and initialization differ. Defer this until after the dashboard boundary; no large trainer framework needed. | High for reproducibility |
| **Refactor** | Fixed `reports/boxingvi/...` paths throughout study/review modules, current-working-directory assumptions | Introduce an explicit artifact catalogue/root for new services, with read adapters for existing layouts. Preserve old path-based runners. | Medium–high |
| **Archive** | Initial, visibility, context, V7/V8 expansion, V10, geometry and RGB study entrypoints/configs/results | Mark as historical/optional in navigation and a future experiment catalogue. Do not delete tested features or checkpoints. Geometry remains optional; RGB crop/decoder helpers are still reused. | Low for classification; high for moving |
| **Archive** | Historical readiness/preparation docs and milestone-specific V6/V10 tools | Preserve their evidence and commands; label scope/date and point to current curation. V6/V10 modules contain gates and source-specific exclusions that a generic preparer must not discard. | Low for labels; high for replacement |
| **Archive** | Smoke runs, superseded review versions, one-off utilities under ignored `reports/` | Preserve provenance bundles first. Some smoke runs are transitively hash-pinned by the active review workflow. Hide historical entries by default in the future dashboard. | Medium–high |
| **Remove** | `.pytest_cache`, `.ruff_cache`, ordinary unpinned `__pycache__`/`.pyc` | Regenerable local caches; eligible for explicit cleanup after approval and while relevant processes are idle. None were pinned by the inspected current preservation manifest. | Low |
| **Remove** | Empty `level1-rgb-features-v1/` and `level1-rgb-features-v2/` output trees | Both currently contain no files. Confirm no active job/reference needs the directories before removing empty scaffolding. | Low |
| **Remove** | Duplicate replay output only after reference/hash checks, e.g. the repeated curation proposal | `annotation-curation-v2/curated-selection.json` and the replay copy have identical SHA256. Keep a verified canonical copy and replay receipt; do not generalize this to every file/version. | Medium; approval needed |
| **Remove** | Production modules or optional feature implementations | **No verified deletion candidate identified.** Low-reference configs and test-only helpers are not enough evidence. | Not recommended now |

## Concrete duplication and complexity boundaries

- `vision.pipeline.sha256` and `annotation_review.file_hash` implement the same
  streaming file-hash responsibility. Their placement is a coupling issue.
  `pipeline.write_json` uses temporary-file replacement; `experiment.write_json`
  writes directly; `annotation_review.write_new` uses exclusive creation and path
  restrictions. These writers must not be collapsed into one behavior accidentally.
- `v10_preparation.check_timing` and `v6_preparation.check_arrays` both validate
  native indices/exact rational timestamps. Their surrounding quarantine and review
  rules differ. Extract narrow validated helpers, not an undifferentiated “prepare all” command.
- `prepare_v5` is a compatibility wrapper around recording preparation. Keep the
  public alias until an explicit deprecation plan exists; its small size is not a
  useful storage or complexity saving.
- Event Pydantic models at package root and neural network models under `training/`
  are different responsibilities despite the shared filename. Prefer explicit import
  names/API schemas over moving checkpoint-sensitive modules solely for naming style.
- `experiment.py` is 591 lines, `multi_study.py` 542 and `diagnostics.py` 736, but
  size alone is not a removal criterion. Separate computation, serialization and
  presentation along tested boundaries instead of creating a large new framework.
- Zero textual references to `v8_pose_full.yaml` and `v5_recording.yaml` outside
  their own files in the tracked-content scan are **investigation candidates**, not
  proof of obsolescence. Configs can be supplied directly on the command line or
  expanded from a recording ID. Preserve them until provenance/use is established.

## Documentation and configuration organization

The historical specification remains valuable, but its initial Gradio/no-API stack,
Phase 0-only first task and illustrative schedule are not the current development
instruction. The requested **FastAPI backend + React development dashboard** should
be recorded in a short architecture amendment after cleanup approval. It does not
establish that early recognition, Level 2 or Level 2.5 analytics are implemented.
Do not rewrite past experiment conclusions to make the project look complete.

| Document/config family | Recommendation |
|---|---|
| `PROJECT_SPEC.md` pointer and `BOXING_MOTION_AI_SPEC.md` | **Keep**; add a current architecture amendment later rather than erasing initial scope/history. |
| `ANNOTATION_CURATION.md`, `LEVEL1_LABEL_POSE_DIAGNOSIS.md`, annotation ledger and curated proposal | **Keep active**, prominently linked. Curation remains blocked pending human review. |
| `ANNOTATION_GUIDE.md`, `EXTRACTION_RESUME.md`, `RECORDING_PIPELINE.md` | **Keep operational**; later distinguish current commands from historical examples and document supported optional dependencies. |
| `DATASET_AUDIT.md`, `POSE_EXTRACTION.md`, `DATASET_PREPARATION.md`, `V4_POSE_VALIDATION.md`, `MASK_AWARE_POSES.md` | **Archive in navigation as dated milestones**, retain links and evidence. Old “no training yet” statements need scope labels, not deletion. |
| `BASELINE_READINESS.md`, `DATA_EXPANSION_PLAN.md`, `V6_READINESS.md`, `V6_PREPARATION.md`, `V10_PREPARATION.md`, `POSE_REPRESENTATION_ASSESSMENT.md` | **Archive/historical status**. `BASELINE_READINESS.md` already has a history notice, but its old “V9 next extraction” section must never become a current task recommendation. |
| All `LEVEL1_*STUDY`, experiments, failure/transfer, limb, multi-recording, RGB findings | **Keep reproducible / archive in navigation** according to study status; preserve full protocols and negative results. |
| Recording, pose, alignment-review and study configs | **Keep paths and versions** initially. Separate active, proposal, diagnostic and historical categories in an index; defer directory moves. |
| Root package description and CLI summary (“audit/pose extraction”) | **Refactor metadata later** to include review/curation and dashboard-facing services. These are runtime/config edits and were not made here. |

README now points to active curation and this plan. No historical spec, result,
configuration or review record was rewritten.

## Local artifacts and retention plan

Metadata-only inventory, excluding raw data and V9:

- `reports/boxingvi/`: **24,565 files, 1,292.71 MiB** apparent file size.
- `data/processed/boxingvi/`: **1,021 files, 1,135.15 MiB** apparent file size.
- Pose estimator weight: **8.96 MiB**; cached visual encoder weight: **9.83 MiB**.

These are not filesystem allocated-size measurements or promised reclaimable space.
The raw dataset was deliberately excluded, and no pretrained/checkpoint file was
loaded to measure these totals.

| Artifact family | Observed size | Retention decision |
|---|---:|---|
| `level1-velocity-v1`, `level1-mirror-v1` | 15.68 / 11.81 MiB | **Keep** current reference and matched historical control, including per-seed best/last checkpoints, predictions, curves, source snapshots and provenance. |
| `level1-multi-recording-v1`, `level1-rgb-study-v1` | 73.34 / 42.75 MiB | **Archive logically**, retain complete comparison/holdout evidence. |
| `level1-rgb-features-v3` | 41.17 MiB | **Keep/archive cache bundle** with encoder/preprocessing hashes and source correspondence; not disposable merely because RGB was negative. |
| `v6-readiness-v1/v2/v3` | 100.37 / 120.37 / 120.40 MiB | **Archive earlier views after dependency audit**; similar sizes do not establish identical images/evidence. |
| `rear-punch-priorities-v1`, `v10-targeted-review-v1` | 103.69 / 47.81 MiB | **Keep review provenance**, archive presentation copies only after manifest/link review. |
| `v7-v8-expansion-gate-v1/v2` | 28.33 / 30.35 MiB | **Archive older view**, preserve source grouping/exclusion evidence and all live references. |
| 79-case gallery, dense pages, assessment-v2, curation-v2 and review ledger | Gallery 30.08 MiB; dense pages 10.49 MiB | **Keep active** for human review. The ledger/configs remain versioned in Git; media/reports stay ignored. |
| Initial, context, visibility, expansion, V10 and smoke outputs | Individual measured runs roughly 1–29 MiB | **Archive logically**, preserve exact study receipts. Many smoke outputs are pinned, so size/“smoke” name is not deletion evidence. |
| Raw data, prepared source pose JSONL/NPZ, original annotations and reserved V9 | Not part of reclaimable estimate | **Keep**; no cleanup operation proposed on these. |

The current review preservation manifest references **4,896 paths**. It includes
entire older study trees and smoke runs. `multi_study.preserve_inputs` reads frozen
source snapshots and compares architecture hashes and feature ASTs; other runners
also compare code hashes. Therefore renaming a folder, formatting model code or
removing “duplicate” snapshots can break reproduction even when weights and metrics
appear unchanged. This assessment read manifest keys; it did not rehash all protected
payloads or claim to rerun every historical study.

Before any physical archive operation:

1. Build a non-V9 dependency manifest of relative paths, sizes, checksums, incoming
   references, run status, configurations, checkpoints, cached features and source
   snapshots. Preserve all review decisions and author/source evidence.
2. Mark historical status in a separate catalogue without changing old run files.
3. Copy a complete bundle to the approved archive destination; verify its manifest
   and perform a restore/read-verification test. Avoid shared writable hard links.
4. Keep existing paths available to frozen runners, or introduce a separately versioned
   resolver with compatibility tests. Do not edit old hashes to make moved data pass.
5. Remove a redundant working copy only with explicit approval after verifying the
   retained bundle and reference resolution. No such movement/deletion is approved
   by this assessment itself.

### Temporary utilities

Seventeen local Python utilities exist under ignored `reports/boxingvi/`, outside
saved run source snapshots. Examples include `inspect_annotations.py`,
`development/download_sources.py`, `development/probe_recordings.py`,
`v5-alignment-review/{prepare_review,extract_frames,build_review,write_findings}.py`,
`write_v5_mask_findings.py`, and the V5 `finish_report.py` helpers.

**Archive with their evidence first.** Several execute top-level work, write fixed
output paths, or preserve milestone-specific test receipts. The acquisition utility
can download footage; the original inspection script traverses the raw dataset and
includes V9. Their source was read, but none was executed. Do not auto-discover these
as dashboard jobs or import them into a server. If a capability is still needed,
port a narrow guarded function into the package with tests; preserve the original
utility as historical provenance.

## Proposed structure for FastAPI and React

Keep the existing Python package and its public paths. Add only the pieces required
by an approved first dashboard slice; no empty scaffold was created here:

```text
src/boxing_motion_ai/
  api/                 # future FastAPI routes and response/request schemas
  services/            # future artifact queries and explicit review operations
  data/                # existing preparation, policy and review contracts
  vision/              # existing exact-frame extraction; compute invoked explicitly
  training/            # retained reference models and frozen study entrypoints
web/                   # future React project and its own package lock
configs/               # keep existing paths; add an index before moving anything
reports/               # ignored local artifacts, accessed through a catalogue
```

Recommended initial boundary: React → FastAPI adapters → lightweight artifact/review
services → current contracts and explicitly selected manifests. Avoid wrapping
`run_study`, `load_pool` or a whole preservation scan inside every listing request.
Read existing summaries and versioned statuses; run expensive verification/extraction
as explicit jobs if that capability is approved later. File-based storage is adequate
for the first local dashboard; no database or distributed job infrastructure is
justified by this assessment.

Specific preparation work after approval:

- Provide typed adapters for dataset/run/review summaries with schema-version checks,
  stable IDs and explicit missing/stale/blocked status. Preserve separate source
  annotations, assistant findings, human decisions and proposed selection.
- Keep API imports independent of training/PyTorch/visual-encoder imports for metadata
  queries. Consider a separate optional RGB dependency group later; retain frozen
  dependency snapshots for historical replay.
- Serve only registered local assets by ID. Do not expose the entire `reports/` or
  dataset tree as a static folder, or copy private media into React's public assets.
  V9 stays outside the development asset/job registry.
- Preserve exact frame index and rational PTS/timebase in API contracts; display
  seconds are derived values. Keep invalid coordinates/masks explicit when serializing
  arrays, without promoting unavailable positions to valid points.
- Put explicit human attribution and stale-parent/concurrent-review checks at the
  review-write boundary. Existing exclusive file creation is useful but is not a
  complete multi-request transaction service. Never auto-check human attestation.
- Preserve extraction locking and job resource ownership. `pipeline.py` uses `fcntl`,
  so portability beyond the current macOS/Linux workflow must be evaluated explicitly.
- Keep existing frozen replay routes operational. A development dashboard does not
  make Level 1 solved or authorize Level 2/analytics or a new training experiment.

## Prioritized implementation plan for approval

| Order | Proposed work | Safe boundary / completion evidence |
|---|---|---|
| **P0 — before backend work** | Fix YAML/base-install and full-extras CI coverage; catalogue supported commands and active docs. | No data/model changes; clean base + locked full dependency checks, explicit skip reporting, CLI help. Documentation-only navigation is the only part completed now. |
| **P1 — enable dashboard integration** | Add lightweight artifact/review service interfaces and an explicit non-V9 catalogue; plan API schema adapters. | Read-only metadata queries need no pose loading/training. Synthetic tests for stale/missing artifacts and source-group guards; keep current CLI compatibility. |
| **P1 — before dashboard review writes** | Adapt review append operations for explicit human attribution and concurrent stale-parent rejection. | Preserve histories and original exclusions; test two competing revisions, failure recovery and no source writes. Requires separate implementation approval. |
| **P2 — small, isolated refactors** | Neutral hashing/storage helpers, targeted render/decode reuse, lazy representation loading, better CLI organization. | One responsibility per change; contract/PTS/mask/golden-feature tests and compatibility exports. Confirm impact on code-hash guards before touching frozen functions. |
| **P2 — organize history** | Catalogue studies/docs as active, optional, historical or archived-in-place; inventory snapshot dependencies. | Every retained run has config, seed, inputs, metrics and checkpoint references. No path move or output mutation. |
| **P3 — optional storage cleanup** | Remove approved caches/empty directories; copy/verify selected old bundles; deduplicate only proven redundant copies. | Explicit target list, no V9/raw/source-pose actions, verified backup/restore and intact incoming links. Potentially destructive; separate approval. |
| **Defer** | Generic trainer rewrite, broad `features/models/evaluation` package relocation, removing RGB/geometry/context code, new database/queue stack. | Not needed to start the dashboard; risks frozen experiments and exceeds cleanup's immediate value. |

Approval recommendation: start with **P0 and the read-only part of P1**. Keep archive
classification logical until a dependency/restore manifest exists. Do not approve a
blanket “delete unused files” or “move old experiments” operation. The next milestone
should implement only the selected cleanup slice before dashboard construction.
